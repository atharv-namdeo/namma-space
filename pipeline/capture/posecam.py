"""PoseCam RAW session (folder or .zip) -> capture folder. METRIC, gravity-aligned.

Port of pipeline/mapping/ingest_posecam.py (that file imports torch + open3d at module
level, so it cannot be imported on torch-blocked machines). Session layout:

    frames/<frame_index:06d>_<timestamp_ns>.jpg
    poses.csv        frame_index,timestamp_ns,tracking_state,tx,ty,tz,qx,qy,qz,qw
    intrinsics.json  fx,fy,cx,cy[,width,height]
    imu.csv          optional, copied through
    depth/           optional ARCore Depth API images, uint16 mm (DEPTH16 ok),
                     named <frame_index:06d>_<timestamp_ns>.png or <frame_index:06d>.png

Poses: ARCore Camera.getPose() (OpenGL camera, world +Y up, metres) -> OpenCV by
right-multiplying diag(1,-1,-1,1). Relocalisation jumps split the track; the longest
clean run is kept.

Depth, in order of preference:
  * --depth-model none + ARCore depth present: ARCore depth is written directly (metric).
  * --depth-model indoor|relative (torch): monocular depth, then scaled to metres by
    ARCore depth / range CSV (fusion.py) or, failing that, by triangulating ORB matches
    with the metric poses.
  * --depth-model none, no ARCore depth: no depth/ folder (allowed by the contract).
"""
from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import tempfile
import zipfile

import cv2
import numpy as np

from . import common as cm

JUMP_SPEED_M_S = 3.0
JUMP_RATE_RAD_S = 10.0


def open_session(path):
    """Folder or zip -> (session_dir, tmpdir_to_cleanup_or_None)."""
    if os.path.isdir(path):
        root, tmp = path, None
    elif zipfile.is_zipfile(path):
        tmp = tempfile.mkdtemp(prefix="ns_posecam_")
        with zipfile.ZipFile(path) as z:
            for m in z.namelist():  # zip-slip guard
                dst = os.path.realpath(os.path.join(tmp, m))
                if not dst.startswith(os.path.realpath(tmp)):
                    raise SystemExit(f"[posecam] unsafe path in zip: {m}")
            z.extractall(tmp)
        root = tmp
    else:
        raise SystemExit(f"[posecam] not a folder or zip: {path}")
    hits = glob.glob(os.path.join(root, "**", "poses.csv"), recursive=True)
    if not hits:
        raise SystemExit(f"[posecam] no poses.csv inside {path}")
    return os.path.dirname(min(hits, key=len)), tmp


def load_session_poses(sdir):
    rows = []
    with open(os.path.join(sdir, "poses.csv"), newline="") as f:
        for r in csv.DictReader(f):
            tracked = r.get("tracking_state", "TRACKING").strip() == "TRACKING"
            pos = np.array([float(r[k]) for k in ("tx", "ty", "tz")]) if tracked else np.full(3, np.nan)
            q = np.array([float(r[k]) for k in ("qx", "qy", "qz", "qw")]) if tracked else np.full(4, np.nan)
            rows.append({"fi": int(r["frame_index"]), "ts": int(r["timestamp_ns"]),
                         "tracked": tracked, "pos": pos, "quat": q})
    return rows


def longest_clean_segment(rows):
    runs, cur = [], []
    for i, r in enumerate(rows):
        brk = not r["tracked"]
        if r["tracked"] and cur:
            p = rows[cur[-1]]
            dt = (r["ts"] - p["ts"]) / 1e9
            if dt > 0:
                step = np.linalg.norm(r["pos"] - p["pos"]) / dt
                ang = 2 * np.arccos(min(1.0, abs(float(np.dot(r["quat"], p["quat"]))))) / dt
                if step > JUMP_SPEED_M_S or ang > JUMP_RATE_RAD_S:
                    brk = True
        if brk:
            if cur:
                runs.append(cur)
            cur = []
        if r["tracked"]:
            cur.append(i)
    if cur:
        runs.append(cur)
    if not runs:
        raise SystemExit("[posecam] no tracked frames in poses.csv")
    best = max(runs, key=len)
    print(f"[posecam] {len(rows)} rows -> {len(runs)} clean run(s); longest = {len(best)}")
    return best


def pose_cv(row):
    T = np.eye(4)
    T[:3, :3] = cm.quat_to_mat(*row["quat"])
    T[:3, 3] = row["pos"]
    return T @ cm.GL_TO_CV


def _find_frame(sdir, fi, ts):
    p = os.path.join(sdir, "frames", f"{fi:06d}_{ts}.jpg")
    if os.path.exists(p):
        return p
    hits = glob.glob(os.path.join(sdir, "frames", f"{fi:06d}*.jpg"))
    return hits[0] if hits else None


def _find_arcore_depth(sdir, fi, ts):
    for name in (f"{fi:06d}_{ts}.png", f"{fi:06d}.png"):
        p = os.path.join(sdir, "depth", name)
        if os.path.exists(p):
            return p
    return None


def triangulation_scale(frames, depths, poses, K, max_pairs=25, min_matches=40):
    """Global s so depth*s is metric, from ORB matches triangulated with metric poses."""
    orb = cv2.ORB_create(1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ratios = []
    step = max(1, len(frames) // max_pairs)
    for i in range(0, len(frames) - 1, step):
        g0 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        k0, d0 = orb.detectAndCompute(g0, None)
        k1, d1 = orb.detectAndCompute(g1, None)
        if d0 is None or d1 is None:
            continue
        m = sorted(bf.match(d0, d1), key=lambda x: x.distance)[:200]
        if len(m) < min_matches:
            continue
        p0 = np.float32([k0[x.queryIdx].pt for x in m]).T
        p1 = np.float32([k1[x.trainIdx].pt for x in m]).T
        T0, T1 = np.linalg.inv(poses[i]), np.linalg.inv(poses[i + 1])
        X = cv2.triangulatePoints(K @ T0[:3], K @ T1[:3], p0, p1)
        X = X[:3] / X[3:4]
        zc = (T0[:3, :3] @ X + T0[:3, 3:4])[2]
        u, v = p0[0].astype(int), p0[1].astype(int)
        h, w = depths[i].shape
        ok = (zc > 0.1) & (zc < 20) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if ok.sum() < min_matches // 2:
            continue
        dz = depths[i][v[ok], u[ok]]
        g = dz > 1e-3
        if g.sum() >= 10:
            ratios.append(np.median(zc[ok][g] / dz[g]))
    return float(np.median(ratios)) if ratios else 1.0


def ingest_posecam(src, out, stride=2, size=640, max_frames=0, depth_model="none",
                   near=0.3, far=8.0, range_csv=None, range_offset_ns=0):
    sdir, tmp = open_session(src)
    try:
        cm.ensure_dirs(out)
        with open(os.path.join(sdir, "intrinsics.json")) as f:
            intr = json.load(f)
        rows = load_session_poses(sdir)
        seg = longest_clean_segment(rows)[::stride]
        if max_frames:
            seg = seg[:max_frames]

        frames, poses, ts, arc = [], [], [], {}
        sc = None
        for idx in seg:
            r = rows[idx]
            fp = _find_frame(sdir, r["fi"], r["ts"])
            bgr = cv2.imread(fp) if fp else None
            if bgr is None:
                continue
            h0, w0 = bgr.shape[:2]
            s = min(1.0, size / max(h0, w0))
            if s < 1.0:
                bgr = cv2.resize(bgr, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA)
            if sc is None:
                sc = s
                H, W = bgr.shape[:2]
                # intrinsics.json is for the native frames; rescale with the frames
                sx, sy = W / intr.get("width", w0), H / intr.get("height", h0)
                cm.save_intrinsics(out, intr["fx"] * sx, intr["fy"] * sy,
                                   intr["cx"] * sx, intr["cy"] * sy, W, H)
            name = f"{len(frames):05d}.jpg"
            cv2.imwrite(os.path.join(out, "rgb", name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            ap = _find_arcore_depth(sdir, r["fi"], r["ts"])
            if ap:
                arc[name] = ap
            frames.append(bgr)
            poses.append(pose_cv(r))
            ts.append(r["ts"])
        if not frames:
            raise SystemExit("[posecam] no frames ingested (check frames/ names)")
        cm.save_poses(out, ts, np.asarray(poses))
        cm.save_timestamps(out, ts)
        if os.path.exists(os.path.join(sdir, "imu.csv")):
            shutil.copy(os.path.join(sdir, "imu.csv"), os.path.join(out, "imu.csv"))
        meta = dict(source=f"posecam:{os.path.basename(os.path.normpath(src))}", metric=True,
                    gravity=[0.0, -1.0, 0.0], poses="arcore", frames=len(frames))
        # gravity: ARCore world is +Y up, so gravity points along -Y of the capture world.

        H, W = frames[0].shape[:2]
        if depth_model == "none":
            if arc:
                for name, p in arc.items():
                    a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                    a = (a.astype(np.uint32) & 0x1FFF).astype(np.uint16)
                    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(cm.depth_path(out, name), a)
                meta.update(depth="arcore-depth-api")
                print(f"[posecam] wrote {len(arc)} ARCore depth maps (metric)")
            else:
                os.rmdir(os.path.join(out, "depth"))
                meta.update(depth=None)
            cm.update_meta(out, **meta)
            return out

        # monocular depth (torch) + metric scaling
        import sys
        mapping = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mapping")
        sys.path.insert(0, mapping)
        try:
            from depth import DepthEstimator
        except Exception as e:
            raise SystemExit(f"[posecam] --depth-model {depth_model} needs torch ({e!r}); "
                             "use --depth-model none") from e
        from PIL import Image
        est = DepthEstimator(model=depth_model, near=near, far=far)
        depths = [est.infer_depth(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                  for f in frames]
        for i, d in enumerate(depths):
            cm.write_depth_m(os.path.join(out, "depth", f"{i:05d}.png"), d)
        cm.update_meta(out, **meta, depth=f"mono:{depth_model}")
        from . import fusion
        if arc or range_csv:
            fusion.fuse(out, range_csv=range_csv, arcore_depths=arc or None,
                        offset_ns=range_offset_ns, scale_poses=False)
        elif not est.metric:
            s = triangulation_scale(frames, depths, np.asarray(poses), cm.K_of(cm.load_intrinsics(out)))
            fusion.apply_scale(out, s, scale_poses=False)
            cm.update_meta(out, scale=s, scale_sources=["triangulation"])
        return out
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
