"""View the reconstructed SOLID mesh (not the point cloud) + camera path in Rerun.

Usage:
    python src/view_mesh.py --dir output/clip1_v2
    python src/view_mesh.py --mesh output/clip1_v2/mesh.ply --dir output/clip1_v2
"""
import argparse
import os

import numpy as np
import open3d as o3d
import rerun as rr


def load_centers(d):
    npy = os.path.join(d, "trajectory.npy")
    if os.path.exists(npy):
        return np.load(npy)[:, :3, 3]
    for name in ("poses.tum", "poses.txt"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            c = []
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    v = line.split()
                    if len(v) >= 4:
                        c.append([float(v[1]), float(v[2]), float(v[3])])
            if c:
                return np.asarray(c)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output/clip1_v2", help="folder with mesh.ply + poses")
    ap.add_argument("--mesh", default=None, help="mesh file (default: <dir>/mesh.ply)")
    args = ap.parse_args()

    mesh_path = args.mesh or os.path.join(args.dir, "mesh.ply")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.vertices) == 0:
        raise SystemExit(f"[view] empty/missing mesh: {mesh_path}")
    mesh.compute_vertex_normals()
    V = np.asarray(mesh.vertices)
    T = np.asarray(mesh.triangles)
    N = np.asarray(mesh.vertex_normals)
    cols = None
    if mesh.has_vertex_colors():
        cols = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    print(f"[view] mesh: {len(V):,} verts / {len(T):,} tris  colored={cols is not None}")

    rr.init("monomap_mesh", spawn=True)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    rr.log("world/mesh",
           rr.Mesh3D(vertex_positions=V, triangle_indices=T,
                     vertex_normals=N, vertex_colors=cols),
           static=True)

    centers = load_centers(args.dir)
    if centers is not None and len(centers) >= 2:
        rr.log("world/path",
               rr.LineStrips3D([centers], colors=[255, 180, 0], radii=0.01),
               static=True)
        print(f"[view] path with {len(centers)} poses")


if __name__ == "__main__":
    main()
