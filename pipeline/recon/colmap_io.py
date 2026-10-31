"""Capture folder <-> COLMAP text model (cameras.txt / images.txt / points3D.txt).

COLMAP stores world->camera: qvec (w, x, y, z) and tvec with x_cam = R x_world + t,
OpenCV camera axes - the same camera axes as our capture, so the conversion is just
an inverse:  R_wc = R_cw^T,  t_wc = -R_cw^T t_cw.

The initial point cloud comes from back-projecting capture depth (when present):
a strided pixel grid per frame, voxel-deduplicated, coloured from the rgb frame.
"""
from __future__ import annotations

import os
import shutil

import cv2
import numpy as np

from ..capture import common as cm


def c2w_to_colmap(T):
    """camera->world 4x4 -> (qvec wxyz, tvec) world->camera."""
    Rcw = T[:3, :3].T
    t = -Rcw @ T[:3, 3]
    qx, qy, qz, qw = cm.mat_to_quat(Rcw)
    return np.array([qw, qx, qy, qz]), t


def colmap_to_c2w(qvec, tvec):
    qw, qx, qy, qz = qvec
    Rcw = cm.quat_to_mat(qx, qy, qz, qw)
    T = np.eye(4)
    T[:3, :3] = Rcw.T
    T[:3, 3] = -Rcw.T @ np.asarray(tvec, float)
    return T


def depth_points(cap, names, poses, intr, step=8, frame_stride=3, max_points=100000,
                 max_depth=None, seed=0):
    """Back-project capture depth -> (N,3) world points, (N,3) uint8 colours."""
    K = cm.K_of(intr)
    Kinv = np.linalg.inv(K)
    P, C = [], []
    for i in range(0, len(names), frame_stride):
        d = cm.read_depth_m(cap, names[i])
        if d is None:
            continue
        rgb = cv2.cvtColor(cv2.imread(os.path.join(cap, "rgb", names[i])), cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != d.shape:
            d = cv2.resize(d, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        vs, us = np.mgrid[step // 2:d.shape[0]:step, step // 2:d.shape[1]:step]
        z = d[vs, us]
        ok = z > 0
        if max_depth:
            ok &= z < max_depth
        # drop depth-clamp walls (monomap clips relative depth at `far`)
        ok &= z < 0.999 * d.max()
        uv1 = np.stack([us[ok], vs[ok], np.ones(ok.sum())], 0).astype(np.float64)
        Xc = (Kinv @ uv1) * z[ok]
        T = poses[i]
        P.append((T[:3, :3] @ Xc).T + T[:3, 3])
        C.append(rgb[vs[ok], us[ok]])
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    P, C = np.concatenate(P), np.concatenate(C)
    # voxel dedup at ~1/400 of the scene extent
    ext = np.percentile(P, 98, 0) - np.percentile(P, 2, 0)
    vox = max(float(ext.max()) / 400.0, 1e-4)
    key = np.floor(P / vox).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    P, C = P[first], C[first]
    if len(P) > max_points:
        sel = np.random.default_rng(seed).choice(len(P), max_points, replace=False)
        P, C = P[sel], C[sel]
    return P, C


def write_model(out_dir, intr, names, poses, points=None, colors=None):
    """Write sparse/0-style text model into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list\n# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        f.write(f"1 PINHOLE {int(intr['width'])} {int(intr['height'])} "
                f"{intr['fx']} {intr['fy']} {intr['cx']} {intr['cy']}\n")
    with open(os.path.join(out_dir, "images.txt"), "w") as f:
        f.write("# Image list\n# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
                "# (POINTS2D line left empty)\n")
        for i, (n, T) in enumerate(zip(names, poses)):
            q, t = c2w_to_colmap(T)
            f.write(f"{i + 1} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                    f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} 1 {n}\n\n")
    with open(os.path.join(out_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list\n# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        if points is not None:
            for i, (p, c) in enumerate(zip(points, colors)):
                f.write(f"{i + 1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])} 1.0\n")


def read_model(model_dir):
    """-> (intr dict, names list, poses (N,4,4) c2w, points (M,3), colors (M,3))."""
    with open(os.path.join(model_dir, "cameras.txt")) as f:
        cam = [l.split() for l in f if l.strip() and not l.startswith("#")][0]
    model, w, h = cam[1], int(cam[2]), int(cam[3])
    p = [float(x) for x in cam[4:]]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        intr = dict(fx=p[0], fy=p[0], cx=p[1], cy=p[2], width=w, height=h)
    else:
        intr = dict(fx=p[0], fy=p[1], cx=p[2], cy=p[3], width=w, height=h)
    names, poses, ids = [], [], []
    with open(os.path.join(model_dir, "images.txt")) as f:
        lines = [l.rstrip("\n") for l in f if not l.startswith("#")]
    k = 0
    while k < len(lines):
        t = lines[k].split()
        if len(t) >= 10:
            ids.append(int(t[0]))
            poses.append(colmap_to_c2w([float(x) for x in t[1:5]], [float(x) for x in t[5:8]]))
            names.append(t[9])
            k += 2  # skip POINTS2D line
        else:
            k += 1
    order = np.argsort(ids)
    names = [names[i] for i in order]
    poses = np.array([poses[i] for i in order])
    pts = np.loadtxt(os.path.join(model_dir, "points3D.txt"), comments="#", ndmin=2)
    if pts.size:
        P, C = pts[:, 1:4], pts[:, 4:7].astype(np.uint8)
    else:
        P, C = np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    return intr, names, poses, P, C


def capture_to_dataset(cap, dataset_dir, max_points=100000, copy_images=True):
    """Make a Brush/gsplat/nerfstudio-readable COLMAP dataset:
        <dataset>/images/*.jpg  <dataset>/sparse/0/{cameras,images,points3D}.txt"""
    intr = cm.load_intrinsics(cap)
    names = cm.frame_names(cap)
    poses = cm.load_poses(cap)
    if len(names) != len(poses):
        raise SystemExit(f"[colmap] {len(names)} frames but {len(poses)} poses in {cap}")
    pts, cols = depth_points(cap, names, poses, intr, max_points=max_points)
    sparse = os.path.join(dataset_dir, "sparse", "0")
    write_model(sparse, intr, names, poses, pts, cols)
    if copy_images:
        img = os.path.join(dataset_dir, "images")
        os.makedirs(img, exist_ok=True)
        for n in names:
            dst = os.path.join(img, n)
            if not os.path.exists(dst):
                shutil.copyfile(os.path.join(cap, "rgb", n), dst)
    print(f"[colmap] {len(names)} images, {len(pts):,} depth points -> {dataset_dir}")
    return dataset_dir
