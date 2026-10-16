"""Fast, lightweight viewer: load a saved map.ply + trajectory.npy and show them
as a SINGLE static point cloud + path. No per-frame layers, no images, flat points
-> renders smoothly even on weak GPUs (the per-frame '.rrd' recordings are heavy).

Usage:
    python src/view_light.py --dir output/clip1 --max-points 800000
"""
import argparse
import os

import numpy as np
import open3d as o3d
import rerun as rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output/clip1", help="folder with map.ply + trajectory.npy")
    ap.add_argument("--max-points", type=int, default=800_000)
    ap.add_argument("--save", action="store_true", help="save recording_light.rrd instead of spawning")
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(os.path.join(args.dir, "map.ply"))
    pts = np.asarray(pcd.points)
    cols = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    # thin to max_points for smooth rendering
    if len(pts) > args.max_points:
        idx = np.random.choice(len(pts), args.max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    print(f"[view] showing {len(pts):,} points")

    rr.init("monomap_light", spawn=not args.save)
    if args.save:
        rr.save(os.path.join(args.dir, "recording_light.rrd"))
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    # flat points (no radii) = fast
    rr.log("world/map", rr.Points3D(pts, colors=cols), static=True)

    traj_path = os.path.join(args.dir, "trajectory.npy")
    if os.path.exists(traj_path):
        centers = np.load(traj_path)[:, :3, 3]
        rr.log("world/path",
               rr.LineStrips3D([centers], colors=[255, 180, 0], radii=0.01),
               static=True)
        print(f"[view] path with {len(centers)} poses")


if __name__ == "__main__":
    main()
