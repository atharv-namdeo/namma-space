"""Navigation output: 3D voxel occupancy + 2D ROS floorplan/costmap.

Detects the floor plane, builds a gravity-aligned frame, then produces:
  - occupancy_3d.npy (+ .json)   boolean voxel grid of occupied space
  - floormap.pgm + floormap.yaml  ROS map_server costmap (0=occupied,
                                   254=free, 205=unknown) + resolution/origin
  - floormap.png                  human-readable preview

This is the robot-facing map: feed floormap.pgm/.yaml straight into a ROS
nav stack, or occupancy_3d.npy into a 3D planner. Works on metric captures
(ARCore) and relative ones (the floor is detected either way).

Usage:
    python src/occupancy.py --cloud output/clip1_tsdf/map_metric.ply \
        --res 0.05 --obstacle-min 0.08 --obstacle-max 2.0
"""
import argparse
import json
import os

import cv2
import numpy as np
import open3d as o3d


def basis_from_up(up):
    up = up / (np.linalg.norm(up) + 1e-9)
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, up); u /= (np.linalg.norm(u) + 1e-9)
    v = np.cross(up, u)
    return u, v, up


def run(cloud_path, out_dir=None, res=0.05, obstacle_min=0.08, obstacle_max=2.0,
        plane_dist=0.03):
    out_dir = out_dir or os.path.dirname(cloud_path)
    os.makedirs(out_dir, exist_ok=True)
    pcd = o3d.io.read_point_cloud(cloud_path)
    pts = np.asarray(pcd.points)
    if len(pts) < 500:
        raise SystemExit("[occ] too few points")

    # floor plane -> up vector
    model, inliers = pcd.segment_plane(plane_dist, ransac_n=3, num_iterations=1000)
    up = np.array(model[:3], float)
    floor_c = pts[inliers].mean(axis=0)
    if np.dot(pts.mean(axis=0) - floor_c, up) < 0:
        up = -up
    u, v, w = basis_from_up(up)
    print(f"[occ] floor found ({len(inliers):,} pts); up ~ {np.round(w,2).tolist()}")

    # project into ground frame (x along u, y along v, h along up)
    x = pts @ u; y = pts @ v; h = pts @ w
    h -= np.median(h[inliers] if len(inliers) else h)       # 0 = floor level

    x0, y0 = x.min(), y.min()
    W = int(np.ceil((x.max() - x0) / res)) + 1
    H = int(np.ceil((y.max() - y0) / res)) + 1
    if W * H > 40_000_000:
        raise SystemExit(f"[occ] grid too big ({W}x{H}); increase --res")
    ix = np.clip(((x - x0) / res).astype(int), 0, W - 1)
    iy = np.clip(((y - y0) / res).astype(int), 0, H - 1)

    obstacle = (h > obstacle_min) & (h < obstacle_max)
    near_floor = (h <= obstacle_min)

    grid = np.full((H, W), 205, np.uint8)                   # unknown
    grid[iy[near_floor], ix[near_floor]] = 254              # free (floor seen)
    grid[iy[obstacle], ix[obstacle]] = 0                    # occupied

    # 3D voxel occupancy
    hmin = h.min()
    D = int(np.ceil((h.max() - hmin) / res)) + 1
    vox = np.zeros((W, H, D), bool)
    iz = np.clip(((h - hmin) / res).astype(int), 0, D - 1)
    vox[ix, iy, iz] = True
    np.save(os.path.join(out_dir, "occupancy_3d.npy"), vox)
    with open(os.path.join(out_dir, "occupancy_3d.json"), "w") as f:
        json.dump({"resolution": res, "dims": [W, H, D],
                   "origin_xyz": [float(x0), float(y0), float(hmin)],
                   "up": w.tolist(), "n_occupied": int(vox.sum())}, f, indent=2)

    # ROS costmap: flip so bottom row is min-y (ROS origin convention)
    pgm = np.flipud(grid)
    cv2.imwrite(os.path.join(out_dir, "floormap.pgm"), pgm)
    with open(os.path.join(out_dir, "floormap.yaml"), "w") as f:
        f.write(f"image: floormap.pgm\nresolution: {res}\n"
                f"origin: [{float(x0):.4f}, {float(y0):.4f}, 0.0]\n"
                "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")

    # pretty PNG preview (occupied=red, free=white, unknown=grey)
    prev = np.full((H, W, 3), 160, np.uint8)
    prev[grid == 254] = (255, 255, 255)
    prev[grid == 0] = (40, 40, 220)
    cv2.imwrite(os.path.join(out_dir, "floormap.png"), np.flipud(prev))

    occ = int((grid == 0).sum()); free = int((grid == 254).sum())
    print(f"[occ] grid {W}x{H} @ {res} m  occupied={occ:,} free={free:,} "
          f"voxels={int(vox.sum()):,}")
    print(f"[occ] saved -> {out_dir}  (floormap.pgm/.yaml/.png, occupancy_3d.npy)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True, help="map_metric.ply")
    ap.add_argument("--out", default=None)
    ap.add_argument("--res", type=float, default=0.05, help="cell size (m)")
    ap.add_argument("--obstacle-min", type=float, default=0.08,
                    help="height above floor where obstacles start (m)")
    ap.add_argument("--obstacle-max", type=float, default=2.0,
                    help="ignore points taller than this (m)")
    ap.add_argument("--plane-dist", type=float, default=0.03)
    args = ap.parse_args()
    run(args.cloud, args.out, args.res, args.obstacle_min, args.obstacle_max,
        args.plane_dist)
