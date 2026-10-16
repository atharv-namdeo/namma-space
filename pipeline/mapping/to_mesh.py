"""Convert a fused point cloud (map.ply) into a watertight mesh and export STL.

Point cloud -> normals -> Poisson surface reconstruction -> trim low-density
blobs -> STL (geometry only) + coloured PLY (for viewing).

Usage:
    python src/to_mesh.py --ply output/clip1/map.ply --out output/clip1/map.stl
    python src/to_mesh.py --ply output/map.ply --out output/map.stl --depth 9 --largest
"""
import argparse
import numpy as np
import open3d as o3d


def to_mesh(ply, out_stl, voxel=0.015, depth=10, density_pct=4.0, largest=False,
            target_tris=800_000):
    pcd = o3d.io.read_point_cloud(ply)
    print(f"[mesh] loaded {len(pcd.points):,} points")
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
        print(f"[mesh] downsampled to {len(pcd.points):,} @ {voxel} m")

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)
    print("[mesh] normals estimated; running Poisson (depth="
          f"{depth}) ... this can take a minute")

    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth)
    dens = np.asarray(dens)

    # Poisson balloons out thin low-confidence regions -> trim them
    thr = np.quantile(dens, density_pct / 100.0)
    mesh.remove_vertices_by_mask(dens < thr)
    mesh.remove_unreferenced_vertices()

    if largest:
        tri_clusters, n_tri, _ = mesh.cluster_connected_triangles()
        tri_clusters = np.asarray(tri_clusters)
        n_tri = np.asarray(n_tri)
        keep = tri_clusters == int(n_tri.argmax())
        mesh.remove_triangles_by_mask(~keep)
        mesh.remove_unreferenced_vertices()

    if target_tris and len(mesh.triangles) > target_tris:
        print(f"[mesh] decimating {len(mesh.triangles):,} -> ~{target_tris:,} tris")
        mesh = mesh.simplify_quadric_decimation(int(target_tris))
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    o3d.io.write_triangle_mesh(out_stl, mesh)
    ply_out = out_stl.rsplit(".", 1)[0] + "_mesh.ply"
    o3d.io.write_triangle_mesh(ply_out, mesh)  # keeps colour, for viewing
    print(f"[mesh] DONE  verts={len(mesh.vertices):,}  tris={len(mesh.triangles):,}")
    print(f"[mesh] saved -> {out_stl}  (+ {ply_out})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help="input point cloud (map.ply)")
    ap.add_argument("--out", required=True, help="output .stl path")
    ap.add_argument("--voxel", type=float, default=0.015,
                    help="pre-downsample size (smaller = finer, slower)")
    ap.add_argument("--depth", type=int, default=10,
                    help="Poisson octree depth (8-11; higher = more detail)")
    ap.add_argument("--density-pct", type=float, default=4.0,
                    help="trim this %% of lowest-confidence vertices")
    ap.add_argument("--largest", action="store_true",
                    help="keep only the largest connected surface")
    ap.add_argument("--target-tris", type=int, default=800_000,
                    help="decimate to about this many triangles (0 = no decimation)")
    args = ap.parse_args()
    to_mesh(args.ply, args.out, args.voxel, args.depth, args.density_pct,
            args.largest, args.target_tris)
