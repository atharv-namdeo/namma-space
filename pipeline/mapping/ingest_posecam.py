"""Android pose source: PoseCam RAW session -> common capture format (METRIC).

PoseCam (github.com/Krish-Kheni/PoseCam) records raw ARCore output on-device. We
read the RAW session (NOT the "Export for pipeline" MP4, which rotates the video
90 deg, drops intrinsics, and is gripper-training specific). A raw session folder:

    frames/<frame_index:06d>_<timestamp_ns>.jpg   native-orientation JPEGs
    poses.csv     frame_index,timestamp_ns,tracking_state,tx,ty,tz,qx,qy,qz,qw
    intrinsics.json   fx,fy,cx,cy (+ width,height) for the frames above
    imu.csv       sensor,timestamp_ns,x,y,z,bias_x,bias_y,bias_z   (carried forward)

Conventions (PoseCam docs/COORDINATES.md, verified):
  * world is gravity-aligned, +Y up, metres; origin at session start.
  * each pose is Camera.getPose() = T_world_camera, camera frame +X right / +Y up /
    -Z forward (OpenGL). We convert to OpenCV via  T_world_cv = T_world_cam . diag(1,-1,-1,1).
  * quaternion is (qx,qy,qz,qw), Hamilton, scalar-last.
  * ARCore relocalizes mid-recording -> poses jump while still "TRACKING". We split
    at tracking loss and at >3 m/s / >10 rad/s jumps and keep the longest clean run.

PoseCam has no depth sensor, so we estimate monocular depth (Depth Anything) and
scale it to real metres using the known metric poses (feature triangulation).

Usage:
    python src/ingest_posecam.py --in android/PoseCam/data/capture-XXXX \
        --out capture/scan1 --stride 2 --size 640
"""
import argparse
import csv
import json
import os
import shutil

import cv2
import numpy as np
from PIL import Image

import capture_format as cf
from depth import DepthEstimator

JUMP_SPEED_M_S = 3.0
JUMP_RATE_RAD_S = 10.0


def load_session_poses(indir):
    """Read poses.csv -> list of dicts with frame_index, timestamp_ns, tracked,
    position (3,), quat (xyzw) for every row (untracked rows have NaN pose)."""
    rows = []
    with open(os.path.join(indir, "poses.csv"), newline="") as f:
        for r in csv.DictReader(f):
            tracked = r.get("tracking_state") == "TRACKING"
            pos = np.array([float(r[k]) for k in ("tx", "ty", "tz")]) if tracked else np.full(3, np.nan)
            quat = np.array([float(r[k]) for k in ("qx", "qy", "qz", "qw")]) if tracked else np.full(4, np.nan)
            rows.append({"fi": int(r["frame_index"]), "ts": int(r["timestamp_ns"]),
                         "tracked": tracked, "pos": pos, "quat": quat})
    return rows


def longest_clean_segment(rows):
    """Split at tracking loss and at relocalization jumps; return the longest run
    of consecutive tracked rows (ARCore poses across a jump are not one frame)."""
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
    print(f"[posecam] {len(rows)} rows -> {len(runs)} clean run(s); "
          f"using longest = {len(best)} frames")
    return best


def pose_cv(row):
    """Build T_world_camera in the OpenCV convention from a tracked row."""
    T = np.eye(4)
    T[:3, :3] = cf.quat_to_mat(*row["quat"])
    T[:3, 3] = row["pos"]
    return T @ cf.GL_TO_CV


def estimate_metric_scale(frames, depths, poses, K, max_pairs=25, min_matches=40):
    """Fit one scale s so depth*s is metric, via triangulation from the metric poses."""
    orb = cv2.ORB_create(1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ratios = []
    step = max(1, len(frames) // max_pairs)
    for i in range(0, len(frames) - 1, step):
        g0 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
        k0, d0 = orb.detectAndCompute(g0, None)
        k1, d1 = orb.detectAndCompute(g1, None)
        if d0 is None or d1 is None:
            continue
        m = sorted(bf.match(d0, d1), key=lambda x: x.distance)[:200]
        if len(m) < min_matches:
            continue
        pts0 = np.float32([k0[x.queryIdx].pt for x in m]).T
        pts1 = np.float32([k1[x.trainIdx].pt for x in m]).T
        Tcw0 = np.linalg.inv(poses[i]); Tcw1 = np.linalg.inv(poses[i + 1])
        P0 = K @ Tcw0[:3, :4]; P1 = K @ Tcw1[:3, :4]
        X = cv2.triangulatePoints(P0, P1, pts0, pts1)
        X = X[:3] / X[3:4]
        zc = ((Tcw0[:3, :3] @ X) + Tcw0[:3, 3:4])[2]
        u = pts0[0].astype(int); v = pts0[1].astype(int)
        h, w = depths[i].shape
        ok = (zc > 0.1) & (zc < 20) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if ok.sum() < min_matches // 2:
            continue
        dz = depths[i][v[ok], u[ok]]
        good = dz > 1e-3
        if good.sum() >= 10:
            ratios.append(np.median(zc[ok][good] / dz[good]))
    if not ratios:
        print("[posecam] WARN: metric scaling found too few points; using scale=1.0")
        return 1.0
    s = float(np.median(ratios))
    print(f"[posecam] metric depth scale = {s:.4f}  (from {len(ratios)} frame pairs)")
    return s


def run(indir, out_dir, stride=2, size=640, max_frames=0, near=0.3, far=8.0,
        model="indoor"):
    cf.ensure_dirs(out_dir)
    with open(os.path.join(indir, "intrinsics.json")) as f:
        intr = json.load(f)

    rows = load_session_poses(indir)
    seg = longest_clean_segment(rows)
    seg = seg[::stride]
    if max_frames:
        seg = seg[:max_frames]

    est = DepthEstimator(model=model, near=near, far=far)
    frames, depths, poses, timestamps = [], [], [], []
    scale_xy = None
    for n, idx in enumerate(seg):
        r = rows[idx]
        fp = os.path.join(indir, "frames", f"{r['fi']:06d}_{r['ts']}.jpg")
        if not os.path.exists(fp):
            continue
        bgr = cv2.imread(fp)
        if bgr is None:
            continue
        h0, w0 = bgr.shape[:2]
        sc = min(1.0, size / max(h0, w0))
        if sc < 1.0:
            bgr = cv2.resize(bgr, (round(w0 * sc), round(h0 * sc)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if scale_xy is None:
            scale_xy = sc
            fx, fy = intr["fx"] * sc, intr["fy"] * sc
            cx, cy = intr["cx"] * sc, intr["cy"] * sc
            H, W = rgb.shape[:2]
            cf.save_intrinsics(out_dir, fx, fy, cx, cy, W, H)
        frames.append(rgb)
        depths.append(est.infer_depth(Image.fromarray(rgb)))   # metres
        poses.append(pose_cv(r))
        timestamps.append(r["ts"])
        print(f"[posecam] frame {len(frames)} (row {idx}, fi {r['fi']})")
    if not frames:
        raise SystemExit("[posecam] no frames ingested (check frames/ names)")

    if est.metric:
        s = 1.0   # metric depth model already outputs metres
        print("[posecam] metric depth model -> no triangulation scaling needed")
    else:
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        s = estimate_metric_scale(frames, depths, np.asarray(poses), K)

    for i, (rgb, depth) in enumerate(zip(frames, depths)):
        cv2.imwrite(os.path.join(out_dir, "rgb", f"{i:05d}.jpg"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        depth_mm = np.clip(depth * s * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(out_dir, "depth", f"{i:05d}.png"), depth_mm)

    cf.save_poses(out_dir, timestamps, np.asarray(poses))
    imu = os.path.join(indir, "imu.csv")
    if os.path.exists(imu):
        shutil.copy(imu, os.path.join(out_dir, "imu.csv"))   # for the rover VIO bridge
    print(f"[posecam] DONE  {len(frames)} metric frames -> {out_dir}"
          + ("  (+imu.csv)" if os.path.exists(imu) else ""))
    print("[posecam] poses are metric + gravity-aligned; depth is mono scaled by one "
          "global factor. Upgrade depth -> Video Depth Anything for per-frame consistency.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True, help="PoseCam RAW session folder")
    ap.add_argument("--out", default="capture/scan")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--near", type=float, default=0.3)
    ap.add_argument("--far", type=float, default=8.0)
    ap.add_argument("--model", default="indoor",
                    choices=["indoor", "outdoor", "relative"],
                    help="indoor metric model is best for room scans")
    args = ap.parse_args()
    run(args.indir, args.out, args.stride, args.size, args.max_frames,
        args.near, args.far, args.model)
