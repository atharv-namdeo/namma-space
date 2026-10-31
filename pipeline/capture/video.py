"""Plain phone video -> capture folder.

1. Frame extraction (ffmpeg if on PATH, else OpenCV) at `oversample x fps`, then keep the
   sharpest frame of every `oversample` consecutive candidates and drop frames whose
   sharpness (variance of Laplacian) is below `blur_rel` x the rolling median. This is
   cheap and removes the motion-blurred frames that wreck both odometry and splats.
2. Poses, pluggable:
   * `monomap` (default) - Depth Anything V2 + Open3D RGB-D odometry, ported from
     pipeline/mapping/make_capture.py. Needs torch + transformers + open3d (cloud/Colab,
     or any machine where torch imports). Poses/depth are relative-scale unless fused
     with a range sensor (see fusion.py).
   * `none` - frames + intrinsics only (poses = identity placeholders flagged in meta);
     use when COLMAP will solve poses downstream.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from . import common as cm


def sharpness(bgr, size=320):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = size / max(g.shape)
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _resize(bgr, size):
    h, w = bgr.shape[:2]
    s = size / max(h, w)
    if s < 1.0:
        bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return bgr


def _candidates_ffmpeg(video, fps, size, tmp):
    ff = shutil.which("ffmpeg")
    vf = f"fps={fps},scale='if(gt(iw,ih),min({size},iw),-2)':'if(gt(iw,ih),-2,min({size},ih))'"
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-i", video, "-vf", vf,
           "-q:v", "2", os.path.join(tmp, "%06d.jpg")]
    subprocess.run(cmd, check=True)
    files = sorted(glob.glob(os.path.join(tmp, "*.jpg")))
    for k, f in enumerate(files):
        yield k / fps, cv2.imread(f)


def _candidates_cv2(video, fps, size):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"[video] cannot open {video}")
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(vfps / fps)))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield i / vfps, _resize(fr, size)
        i += 1
    cap.release()


def extract_sharp_frames(video, fps=4.0, oversample=3, size=640, blur_rel=0.5,
                         window=15, max_frames=0, use_ffmpeg=None):
    """-> list of (t_seconds, bgr, sharpness) of the kept frames."""
    use_ffmpeg = shutil.which("ffmpeg") is not None if use_ffmpeg is None else use_ffmpeg
    cand_fps = fps * max(1, oversample)
    tmp = tempfile.mkdtemp(prefix="ns_frames_")
    try:
        gen = (_candidates_ffmpeg(video, cand_fps, size, tmp) if use_ffmpeg
               else _candidates_cv2(video, cand_fps, size))
        group, kept = [], []
        for t, fr in gen:
            if fr is None:
                continue
            group.append((t, fr, sharpness(fr)))
            if len(group) == max(1, oversample):
                kept.append(max(group, key=lambda g: g[2]))
                group = []
        if group:
            kept.append(max(group, key=lambda g: g[2]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not kept:
        raise SystemExit(f"[video] no frames decoded from {video}")
    s = np.array([k[2] for k in kept])
    keep = []
    for i in range(len(kept)):
        lo, hi = max(0, i - window // 2), min(len(kept), i + window // 2 + 1)
        if s[i] >= blur_rel * np.median(s[lo:hi]):
            keep.append(kept[i])
    print(f"[video] {len(kept)} candidates (sharpest of {oversample} @ {cand_fps:g} fps) -> "
          f"{len(keep)} after blur rejection (rel {blur_rel})"
          + (" [ffmpeg]" if use_ffmpeg else " [opencv]"))
    if max_frames:
        keep = keep[:max_frames]
    return keep


def _monomap_modules():
    mapping = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mapping")
    if mapping not in sys.path:
        sys.path.insert(0, mapping)
    try:
        import open3d as o3d  # noqa: F401
        from depth import DepthEstimator  # pipeline/mapping/depth.py (torch)
        import capture_format as mcf      # pipeline/mapping/capture_format.py (open3d)
    except Exception as e:  # torch DLL blocks, missing open3d, ...
        raise SystemExit(
            f"[video] pose backend 'monomap' needs torch+transformers+open3d ({e!r}).\n"
            "        Run this step on Colab/a GPU box, or use --poses none and solve poses "
            "with COLMAP.") from e
    return o3d, DepthEstimator, mcf


def ingest_video(video, out, fps=4.0, oversample=3, size=640, blur_rel=0.5, max_frames=0,
                 poses="monomap", f_rel=0.85, near=0.3, far=8.0, max_step=0.5,
                 model="relative", start_ns=0, use_ffmpeg=None):
    frames = extract_sharp_frames(video, fps, oversample, size, blur_rel,
                                  max_frames=max_frames, use_ffmpeg=use_ffmpeg)
    cm.ensure_dirs(out, depth=(poses == "monomap"))
    h, w = frames[0][1].shape[:2]
    fx = fy = f_rel * w
    cm.save_intrinsics(out, fx, fy, w / 2.0, h / 2.0, w, h)
    ts = [int(start_ns + t * 1e9) for t, _, _ in frames]
    for i, (_, fr, _) in enumerate(frames):
        cv2.imwrite(os.path.join(out, "rgb", f"{i:05d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cm.save_timestamps(out, ts)

    if poses == "none":
        cm.save_poses(out, ts, np.repeat(np.eye(4)[None], len(frames), 0))
        cm.update_meta(out, source=f"video:{os.path.basename(video)}", metric=False,
                       poses="placeholder-identity", note="solve poses with COLMAP")
        print(f"[video] {len(frames)} frames -> {out} (poses: placeholders)")
        return out

    # ---- monomap RGB-D odometry (port of pipeline/mapping/make_capture.py) ----
    o3d, DepthEstimator, mcf = _monomap_modules()
    est = DepthEstimator(model=model, near=near, far=far)
    intrinsic = mcf.make_intrinsic(w, h, f_rel)
    opt = o3d.pipelines.odometry.OdometryOption()
    jac = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
    pose, prev, P, fails = np.eye(4), None, [], 0
    from PIL import Image
    for i, (_, fr, _) in enumerate(frames):
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        depth = est.infer_depth(Image.fromarray(rgb))
        gray = mcf.to_rgbd(rgb, depth, far, intensity=True)
        if prev is not None:
            ok, trans, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                gray, prev, intrinsic, np.eye(4), jac, opt)
            if ok and np.linalg.norm(trans[:3, 3]) > max_step:
                ok = False
            if ok:
                pose = pose @ trans
            else:
                fails += 1
        prev = gray
        cm.write_depth_m(os.path.join(out, "depth", f"{i:05d}.png"), depth)
        P.append(pose.copy())
        print(f"[video] odometry {i + 1}/{len(frames)} fails={fails}")
    cm.save_poses(out, ts, P)
    cm.update_meta(out, source=f"video:{os.path.basename(video)}", metric=bool(est.metric),
                   poses="monomap-rgbd-odometry", depth_model=model, odometry_fails=fails)
    return out
