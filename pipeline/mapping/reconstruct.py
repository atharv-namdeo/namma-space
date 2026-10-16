"""Stage 3 - Reconstruct: estimate the camera path and fuse a 3D map.

For each frame:
  1. get relative depth (Depth Anything V2, computed inline or loaded from .npy)
  2. convert normalised disparity -> relative metric depth (near/far mapping)
  3. RGB-D odometry (Open3D) vs the previous frame -> incremental camera motion
  4. accumulate the camera-to-world pose  (this IS "the path travelled")
  5. back-project the frame into the world and add it to the global point cloud

Outputs:
  output/map.ply          fused coloured point cloud (the 3D map)
  output/trajectory.npy   (N,4,4) camera-to-world poses
  output/recording.rrd    Rerun recording (open with: rerun output/recording.rrd)
and (unless --no-view) opens the Rerun 3D viewer live.

Usage:
    python src/reconstruct.py --frames data/frames
"""
import argparse
import glob
import os

import numpy as np
import open3d as o3d
import rerun as rr
from PIL import Image

from depth import DepthEstimator


def disparity_to_depth(disp, near, far):
    """disp in [0,1] (1=near) -> depth in [near,far] via inverse-depth mapping."""
    inv = disp * (1.0 / near - 1.0 / far) + 1.0 / far
    return (1.0 / inv).astype(np.float32)


def make_intrinsic(w, h, f_rel):
    f = f_rel * w
    return o3d.camera.PinholeCameraIntrinsic(w, h, f, f, w / 2.0, h / 2.0)


def to_rgbd(color_np, depth_np, far, intensity):
    color = o3d.geometry.Image(np.ascontiguousarray(color_np))
    depth = o3d.geometry.Image(np.ascontiguousarray(depth_np))
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=1.0, depth_trunc=far,
        convert_rgb_to_intensity=intensity)


def run(frames_dir, out_dir, depth_dir=None, f_rel=0.85, near=0.3, far=8.0,
        voxel=0.02, max_frames=0, max_step=0.5, view=True):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")) +
                   glob.glob(os.path.join(frames_dir, "*.png")))
    if max_frames:
        files = files[:max_frames]
    if not files:
        raise SystemExit(f"No frames in {frames_dir}")

    est = None if depth_dir else DepthEstimator()

    rr.init("monomap", spawn=view)
    if not view:
        rr.save(os.path.join(out_dir, "recording.rrd"))
    rr.log("world", rr.ViewCoordinates.RDF, static=True)  # x-right, y-down, z-fwd

    intrinsic = None
    opt = o3d.pipelines.odometry.OdometryOption()
    jac = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    pose = np.eye(4)                 # camera-to-world of current frame
    prev_gray = None
    global_pcd = o3d.geometry.PointCloud()
    traj, cam_centers = [], []
    fails = 0

    for i, fpath in enumerate(files):
        color_np = np.asarray(Image.open(fpath).convert("RGB"))
        h, w = color_np.shape[:2]
        if intrinsic is None:
            intrinsic = make_intrinsic(w, h, f_rel)

        if depth_dir:
            disp = np.load(os.path.join(depth_dir, f"{i:05d}.npy"))
        else:
            disp = est.infer(Image.fromarray(color_np))
        depth_np = disparity_to_depth(disp, near, far)

        gray = to_rgbd(color_np, depth_np, far, intensity=True)
        colr = to_rgbd(color_np, depth_np, far, intensity=False)

        if prev_gray is not None:
            ok, trans, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                gray, prev_gray, intrinsic, np.eye(4), jac, opt)
            # reject "teleport" blow-ups: odometry sometimes reports success but
            # returns a wildly wrong transform on blurry/low-texture frames, which
            # otherwise corrupts the whole path+map from that point on.
            if ok and np.linalg.norm(trans[:3, 3]) > max_step:
                ok = False
            if ok:
                pose = pose @ trans      # T_world_cur = T_world_prev @ T_prev_cur
            else:
                fails += 1               # keep last pose (brief dropout)
        prev_gray = gray

        # back-project this frame to the world and merge
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(colr, intrinsic)
        pcd.transform(pose)
        global_pcd += pcd
        if voxel > 0 and i % 5 == 4:
            global_pcd = global_pcd.voxel_down_sample(voxel)

        traj.append(pose.copy())
        cam_centers.append(pose[:3, 3].copy())

        # ---- log to Rerun ----
        rr.set_time("frame", sequence=i)
        R, t = pose[:3, :3], pose[:3, 3]
        rr.log("world/camera", rr.Transform3D(translation=t, mat3x3=R))
        rr.log("world/camera/image",
               rr.Pinhole(image_from_camera=intrinsic.intrinsic_matrix,
                          width=w, height=h))
        rr.log("world/camera/image/rgb", rr.Image(color_np).compress(jpeg_quality=70))
        rr.log("world/camera/image/depth", rr.DepthImage(depth_np, meter=1.0))
        if len(cam_centers) >= 2:
            rr.log("world/path", rr.LineStrips3D([np.asarray(cam_centers)],
                                                 colors=[255, 180, 0], radii=0.01))
        pts = np.asarray(pcd.points)
        cols = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        rr.log(f"world/points/{i:05d}", rr.Points3D(pts, colors=cols, radii=0.006))
        print(f"[recon] {i+1}/{len(files)}  pts={len(global_pcd.points):,}  fails={fails}")

    if voxel > 0:
        global_pcd = global_pcd.voxel_down_sample(voxel)
    o3d.io.write_point_cloud(os.path.join(out_dir, "map.ply"), global_pcd)
    np.save(os.path.join(out_dir, "trajectory.npy"), np.asarray(traj))
    print(f"[recon] DONE  map={len(global_pcd.points):,} pts  "
          f"path={len(traj)} poses  odometry-dropouts={fails}")
    print(f"[recon] saved -> {out_dir}\\map.ply , trajectory.npy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--out", default="output")
    ap.add_argument("--depth-dir", default=None, help="use precomputed .npy depth")
    ap.add_argument("--f-rel", type=float, default=0.85,
                    help="focal length as fraction of image width (~0.85 for phones)")
    ap.add_argument("--near", type=float, default=0.3)
    ap.add_argument("--far", type=float, default=8.0)
    ap.add_argument("--voxel", type=float, default=0.02, help="map downsample size")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--max-step", type=float, default=0.5,
                    help="reject odometry jumps larger than this (anti-teleport)")
    ap.add_argument("--no-view", action="store_true", help="save .rrd, don't open viewer")
    args = ap.parse_args()
    run(args.frames, args.out, args.depth_dir, args.f_rel, args.near, args.far,
        args.voxel, args.max_frames, args.max_step, view=not args.no_view)
