"""Pose source (fallback / test harness): plain video -> common capture format.

Runs the existing stack (extract frames -> Depth Anything -> RGB-D odometry) and
writes a capture/ folder the backbone can consume. The poses are metric *up to
an unknown scale* (monocular), so this is the non-metric path we use to validate
the output stages on the clips we already have. The real metric path is
ingest_posecam.py (ARCore poses).

Usage:
    python src/make_capture.py --video data/clip1.mp4 --out capture/clip1 \
        --stride 2 --size 640
"""
import argparse
import glob
import os

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

import capture_format as cf
from capture_format import make_intrinsic, to_rgbd
from depth import DepthEstimator
from extract_frames import extract


def run(video, out_dir, stride=2, size=640, max_frames=0, f_rel=0.85,
        near=0.3, far=8.0, max_step=0.5, model="relative"):
    cf.ensure_dirs(out_dir)
    frames_dir = os.path.join(out_dir, "_frames")
    extract(video, frames_dir, stride, size, max_frames)
    files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not files:
        raise SystemExit(f"No frames extracted from {video}")

    est = DepthEstimator(model=model, near=near, far=far)
    intrinsic = None
    opt = o3d.pipelines.odometry.OdometryOption()
    jac = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    pose = np.eye(4)
    prev_gray = None
    poses, timestamps, fails = [], [], 0

    for i, fpath in enumerate(files):
        color_np = np.asarray(Image.open(fpath).convert("RGB"))
        h, w = color_np.shape[:2]
        if intrinsic is None:
            intrinsic = make_intrinsic(w, h, f_rel)
            fx = fy = f_rel * w
            cf.save_intrinsics(out_dir, fx, fy, w / 2.0, h / 2.0, w, h)

        depth_np = est.infer_depth(Image.fromarray(color_np))   # metres
        gray = to_rgbd(color_np, depth_np, far, intensity=True)

        if prev_gray is not None:
            ok, trans, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                gray, prev_gray, intrinsic, np.eye(4), jac, opt)
            if ok and np.linalg.norm(trans[:3, 3]) > max_step:
                ok = False
            if ok:
                pose = pose @ trans
            else:
                fails += 1
        prev_gray = gray

        # write rgb + depth(mm) into the capture folder
        cv2.imwrite(os.path.join(out_dir, "rgb", f"{i:05d}.jpg"),
                    cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR))
        depth_mm = np.clip(depth_np * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(out_dir, "depth", f"{i:05d}.png"), depth_mm)

        poses.append(pose.copy())
        timestamps.append(i)
        print(f"[capture] {i+1}/{len(files)}  fails={fails}")

    cf.save_poses(out_dir, timestamps, poses)
    # tidy the scratch frames
    for f in files:
        os.remove(f)
    os.rmdir(frames_dir)
    print(f"[capture] DONE  {len(poses)} frames, {fails} odometry dropouts -> {out_dir}")
    print("[capture] NOTE: poses are relative-scale (monocular). Use ingest_posecam "
          "for metric ARCore poses.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="capture/clip")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--f-rel", type=float, default=0.85)
    ap.add_argument("--near", type=float, default=0.3)
    ap.add_argument("--far", type=float, default=8.0)
    ap.add_argument("--max-step", type=float, default=0.5)
    ap.add_argument("--model", default="relative",
                    choices=["relative", "indoor", "outdoor"])
    args = ap.parse_args()
    run(args.video, args.out, args.stride, args.size, args.max_frames,
        args.f_rel, args.near, args.far, args.max_step, args.model)
