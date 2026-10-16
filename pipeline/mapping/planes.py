"""CAD primitives: extract floor / walls / ceiling planes + box primitives.

Iterative RANSAC pulls the dominant planes out of the fused cloud, classifies
each as floor / ceiling / wall relative to the 'up' axis, then clusters whatever
is left (furniture, objects) into oriented bounding boxes. The result is a
structured, lightweight description of the scene - the closest thing to
parametric CAD and directly useful for collision/planning in automation.

Outputs (in --out, default = the cloud's folder):
    planes.json      [{type, normal, d, centroid, size, n_points}, ...]
    boxes.json       [{center, extent, R, n_points}, ...]  (oriented boxes)
    primitives.ply   cloud coloured by plane + boxes drawn as line sets

Usage:
    python src/planes.py --cloud output/clip1_tsdf/map_metric.ply --up auto
"""
import argparse
import json
import os

import numpy as np
import open3d as o3d

_COLORS = np.array([
    [0.90, 0.30, 0.30], [0.30, 0.70, 0.90], [0.40, 0.85, 0.40],
    [0.95, 0.75, 0.25], [0.70, 0.45, 0.85], [0.95, 0.55, 0.75],
    [0.50, 0.80, 0.80], [0.85, 0.60, 0.40],
])


def classify(normal, centroid, up, cloud_mean):
    c = abs(float(np.dot(normal, up)))
    if c > 0.85:
        return "floor" if np.dot(centroid - cloud_mean, up) < 0 else "ceiling"
    if c < 0.35:
        return "wall"
    return "slope"


def run(cloud_path, out_dir=None, up="auto", max_planes=8, dist=0.03,
        min_plane_pts=800, voxel=0.0):
    out_dir = out_dir or os.path.dirname(cloud_path)
    os.makedirs(out_dir, exist_ok=True)
    pcd = o3d.io.read_point_cloud(cloud_path)
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    cloud_mean = np.asarray(pcd.points).mean(axis=0)
    print(f"[planes] {len(pcd.points):,} points loaded")

    rest = pcd
    planes, vis = [], []
    for k in range(max_planes):
        if len(rest.points) < min_plane_pts:
            break
        model, inliers = rest.segment_plane(dist, ransac_n=3, num_iterations=1000)
        if len(inliers) < min_plane_pts:
            break
        inlier = rest.select_by_index(inliers)
        pts = np.asarray(inlier.points)
        n = np.asarray(model[:3], float); n /= (np.linalg.norm(n) + 1e-9)
        planes.append({"model": [float(x) for x in model],
                       "normal": n.tolist(),
                       "centroid": pts.mean(axis=0).tolist(),
                       "size": (pts.max(0) - pts.min(0)).tolist(),
                       "n_points": int(len(pts))})
        inlier.paint_uniform_color(_COLORS[k % len(_COLORS)])
        vis.append(inlier)
        rest = rest.select_by_index(inliers, invert=True)

    if not planes:
        raise SystemExit("[planes] no planes found; try larger --dist / smaller --min")

    # decide 'up'
    if up == "y":
        up_vec = np.array([0, 1.0, 0])
    elif up == "z":
        up_vec = np.array([0, 0, 1.0])
    else:  # auto: normal of the plane with the most inliers
        up_vec = np.array(max(planes, key=lambda p: p["n_points"])["normal"])
    up_vec = up_vec / (np.linalg.norm(up_vec) + 1e-9)

    counts = {}
    for p in planes:
        p["type"] = classify(np.array(p["normal"]), np.array(p["centroid"]),
                             up_vec, cloud_mean)
        counts[p["type"]] = counts.get(p["type"], 0) + 1

    # cluster leftovers into oriented boxes (objects / furniture)
    boxes = []
    if len(rest.points) > 200:
        labels = np.array(rest.cluster_dbscan(eps=dist * 4, min_points=50))
        rest_pts = rest
        for lab in range(labels.max() + 1):
            idx = np.where(labels == lab)[0]
            if len(idx) < 120:
                continue
            sub = rest_pts.select_by_index(idx)
            try:
                obb = sub.get_oriented_bounding_box()
            except Exception:
                obb = sub.get_axis_aligned_bounding_box()
                obb = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(obb)
            boxes.append({"center": list(map(float, obb.center)),
                          "extent": list(map(float, obb.extent)),
                          "R": np.asarray(obb.R).tolist(),
                          "n_points": int(len(idx))})
            obb.color = (0.1, 0.1, 0.1)
            vis.append(obb)

    with open(os.path.join(out_dir, "planes.json"), "w") as f:
        json.dump({"up": up_vec.tolist(), "planes": planes}, f, indent=2)
    with open(os.path.join(out_dir, "boxes.json"), "w") as f:
        json.dump(boxes, f, indent=2)

    merged = o3d.geometry.PointCloud()
    for v in vis:
        if isinstance(v, o3d.geometry.PointCloud):
            merged += v
    o3d.io.write_point_cloud(os.path.join(out_dir, "primitives.ply"), merged)

    print(f"[planes] {len(planes)} planes {counts}  +  {len(boxes)} box primitives")
    print(f"[planes] up axis ~ {np.round(up_vec, 2).tolist()}")
    print(f"[planes] saved -> {out_dir}  (planes.json, boxes.json, primitives.ply)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True, help="map_metric.ply")
    ap.add_argument("--out", default=None)
    ap.add_argument("--up", default="auto", choices=["auto", "y", "z"])
    ap.add_argument("--max-planes", type=int, default=8)
    ap.add_argument("--dist", type=float, default=0.03, help="RANSAC inlier dist (m)")
    ap.add_argument("--min", dest="min_plane_pts", type=int, default=800)
    ap.add_argument("--voxel", type=float, default=0.0)
    args = ap.parse_args()
    run(args.cloud, args.out, args.up, args.max_planes, args.dist,
        args.min_plane_pts, args.voxel)
