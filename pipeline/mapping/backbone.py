"""Reconstruction engine: common capture/ -> clean mesh + metric point cloud.

Two fusion modes (the pose/depth quality decides which is right):

  fuse=tsdf   (metric path, e.g. PoseCam/ARCore): Open3D ScalableTSDFVolume fuses
              every frame's depth into a signed-distance volume -> one watertight,
              denoised mesh. Needs depth that is *consistent* frame-to-frame.

  fuse=points (relative path, plain-video odometry): monocular depth is normalised
              per frame, so TSDF would carve itself away. Instead we back-project
              every frame into one point cloud (like v1), then Poisson-mesh it.

Outputs (in --out):
    mesh.ply / mesh.obj / mesh.stl   fused surface  (PLY+OBJ keep colour; STL = geometry)
    map_metric.ply / .pcd            fused point cloud (metric if the source was)
    trajectory.ply                   camera path as a line set
    poses.tum                        trajectory copy in TUM format (robotics)

Usage:
    python src/backbone.py --cap capture/scan1 --out output/scan1_v2 --fuse tsdf
    python src/backbone.py --cap capture/clip1 --out output/clip1_v2 --fuse points
"""
import argparse
import glob
import os
import shutil

import cv2
import numpy as np
import open3d as o3d

import capture_format as cf


def _rgbd(cap_dir, i, far, depth_scale):
    rgb = os.path.join(cap_dir, "rgb", f"{i:05d}.jpg")
    dp = os.path.join(cap_dir, "depth", f"{i:05d}.png")
    draw = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
    if draw is None or not os.path.exists(rgb):
        return None
    color = o3d.io.read_image(rgb)
    depth = o3d.geometry.Image(np.ascontiguousarray(draw))
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=depth_scale, depth_trunc=far,
        convert_rgb_to_intensity=False)


def _poisson_mesh(pcd, voxel, depth=9, density_pct=3.0, max_pts=500_000):
    # orient_normals_consistent_tangent_plane builds a global graph -> very slow
    # on multi-million-point clouds; coarsen until it's tractable.
    v = voxel
    while len(pcd.points) > max_pts:
        v *= 1.5
        pcd = pcd.voxel_down_sample(v)
    print(f"[backbone] meshing {len(pcd.points):,} pts (voxel {v:.3f} m)")
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=v * 3, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(20)
    print(f"[backbone] Poisson (depth={depth}) ...")
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, density_pct / 100.0))
    mesh.remove_unreferenced_vertices()
    return mesh


def _write_outputs(mesh, pcd, centers, out_dir, cap_dir, target_tris):
    if target_tris and len(mesh.triangles) > target_tris:
        print(f"[backbone] decimating {len(mesh.triangles):,} -> ~{target_tris:,} tris")
        mesh = mesh.simplify_quadric_decimation(int(target_tris))
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(os.path.join(out_dir, "mesh.ply"), mesh)
    o3d.io.write_triangle_mesh(os.path.join(out_dir, "mesh.obj"), mesh)
    o3d.io.write_triangle_mesh(os.path.join(out_dir, "mesh.stl"), mesh)
    o3d.io.write_point_cloud(os.path.join(out_dir, "map_metric.ply"), pcd)
    o3d.io.write_point_cloud(os.path.join(out_dir, "map_metric.pcd"), pcd)

    if len(centers) >= 2:
        pts = np.asarray(centers)
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(
                np.column_stack([np.arange(len(pts) - 1), np.arange(1, len(pts))])))
        ls.paint_uniform_color([1.0, 0.7, 0.0])
        o3d.io.write_line_set(os.path.join(out_dir, "trajectory.ply"), ls)

    src = os.path.join(cap_dir, "poses.txt")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(out_dir, "poses.tum"))

    ext = pcd.get_axis_aligned_bounding_box().get_extent()
    print(f"[backbone] DONE  mesh: {len(mesh.vertices):,} v / {len(mesh.triangles):,} f"
          f"   cloud: {len(pcd.points):,} pts")
    print(f"[backbone] map extent (m if metric): "
          f"{ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f}")
    print(f"[backbone] saved -> {out_dir}")


def run(cap_dir, out_dir, voxel=0.02, sdf_trunc=0.08, far=8.0,
        target_tris=500_000, depth_scale=1000.0, fuse="tsdf"):
    os.makedirs(out_dir, exist_ok=True)
    intr = cf.load_intrinsics(cap_dir)
    poses = cf.load_poses(cap_dir)
    rgb_files = sorted(glob.glob(os.path.join(cap_dir, "rgb", "*.jpg")))
    if not rgb_files:
        raise SystemExit(f"[backbone] no rgb frames in {cap_dir}")
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intr["width"], intr["height"], intr["fx"], intr["fy"], intr["cx"], intr["cy"])
    n = min(len(rgb_files), len(poses))
    centers = [poses[i][:3, 3].copy() for i in range(n)]

    if fuse == "tsdf":
        vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel, sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        for i in range(n):
            rgbd = _rgbd(cap_dir, i, far, depth_scale)
            if rgbd is None:
                continue
            vol.integrate(rgbd, intrinsic, np.linalg.inv(poses[i]))
            if (i + 1) % 20 == 0 or i == n - 1:
                print(f"[backbone] integrated {i+1}/{n}")
        print("[backbone] extracting TSDF mesh + cloud ...")
        mesh = vol.extract_triangle_mesh()
        pcd = vol.extract_point_cloud()
    else:  # points fusion + Poisson (relative-depth path)
        acc = o3d.geometry.PointCloud()
        for i in range(n):
            rgbd = _rgbd(cap_dir, i, far, depth_scale)
            if rgbd is None:
                continue
            p = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
            p.transform(poses[i])
            acc += p
            if voxel > 0 and (i + 1) % 5 == 0:
                acc = acc.voxel_down_sample(voxel)
            if (i + 1) % 20 == 0 or i == n - 1:
                print(f"[backbone] fused {i+1}/{n}  pts={len(acc.points):,}")
        if voxel > 0:
            acc = acc.voxel_down_sample(voxel)
        pcd = acc
        mesh = _poisson_mesh(o3d.geometry.PointCloud(pcd), voxel)

    _write_outputs(mesh, pcd, centers, out_dir, cap_dir, target_tris)
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True, help="capture/ folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fuse", choices=["tsdf", "points"], default="tsdf",
                    help="tsdf = metric/consistent depth; points = relative mono depth")
    ap.add_argument("--voxel", type=float, default=0.02,
                    help="TSDF voxel / cloud downsample size in metres")
    ap.add_argument("--sdf-trunc", type=float, default=0.08, help="TSDF SDF band (~4x voxel)")
    ap.add_argument("--far", type=float, default=8.0, help="depth truncation (m)")
    ap.add_argument("--target-tris", type=int, default=500_000,
                    help="decimate mesh to ~this many triangles (0 = off)")
    args = ap.parse_args()
    run(args.cap, args.out, args.voxel, args.sdf_trunc, args.far,
        args.target_tris, fuse=args.fuse)
