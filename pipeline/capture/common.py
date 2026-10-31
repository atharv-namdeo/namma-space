"""Capture-folder IO for stage 1 (torch-free, open3d-free).

The folder layout is the contract in docs/ARCHITECTURE.md:

    <capture>/rgb/00000.jpg ...      frames
    <capture>/depth/00000.png ...    optional uint16 millimetres, same size as rgb
    <capture>/poses.npy              (N,4,4) camera->world, OpenCV convention
    <capture>/poses.txt              TUM copy (timestamp tx ty tz qx qy qz qw)
    <capture>/intrinsics.json        {fx, fy, cx, cy, width, height}
    <capture>/timestamps.txt         optional, one int ns per frame (sensor sync)
    <capture>/imu.csv                optional
    <capture>/sensors/               optional extra range sensors
    <capture>/meta.json              optional {source, metric, gravity, scale, ...}

pipeline/mapping/capture_format.py has the same maths but imports open3d at module
level, so it is mirrored here instead of imported.
"""
from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np

# OpenGL camera (ARCore/ARKit: -Z forward, +Y up) -> OpenCV camera (+Z fwd, +Y down).
# Right-multiply a camera->world matrix by this.
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


# ----------------------------------------------------------------------------- maths
def quat_to_mat(qx, qy, qz, qw):
    """Unit quaternion (x, y, z, w; Hamilton) -> 3x3 rotation."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = q / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def mat_to_quat(R):
    """3x3 rotation -> quaternion (x, y, z, w) with w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return -q if q[3] < 0 else q


# ----------------------------------------------------------------------------- IO
def ensure_dirs(cap, depth=True):
    os.makedirs(os.path.join(cap, "rgb"), exist_ok=True)
    if depth:
        os.makedirs(os.path.join(cap, "depth"), exist_ok=True)
    return cap


def save_poses(cap, timestamps, poses):
    poses = np.asarray(poses, dtype=np.float64)
    np.save(os.path.join(cap, "poses.npy"), poses)
    with open(os.path.join(cap, "poses.txt"), "w") as f:
        f.write("# TUM trajectory: timestamp tx ty tz qx qy qz qw (camera->world)\n")
        for t, T in zip(timestamps, poses):
            qx, qy, qz, qw = mat_to_quat(T[:3, :3])
            tx, ty, tz = T[:3, 3]
            f.write(f"{t} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")


def load_poses(cap):
    return np.load(os.path.join(cap, "poses.npy"))


def save_timestamps(cap, ts_ns):
    with open(os.path.join(cap, "timestamps.txt"), "w") as f:
        f.write("# frame timestamp, int nanoseconds (same clock as sensors/ logs)\n")
        for t in ts_ns:
            f.write(f"{int(t)}\n")


def load_timestamps(cap):
    """Frame timestamps in ns: timestamps.txt, else the TUM column of poses.txt."""
    p = os.path.join(cap, "timestamps.txt")
    if os.path.exists(p):
        return np.array([int(l) for l in open(p) if l.strip() and not l.startswith("#")],
                        dtype=np.int64)
    p = os.path.join(cap, "poses.txt")
    if os.path.exists(p):
        return np.array([int(float(l.split()[0])) for l in open(p)
                         if l.strip() and not l.startswith("#")], dtype=np.int64)
    return None


def save_intrinsics(cap, fx, fy, cx, cy, width, height):
    with open(os.path.join(cap, "intrinsics.json"), "w") as f:
        json.dump({"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
                   "width": int(width), "height": int(height)}, f, indent=2)


def load_intrinsics(cap):
    with open(os.path.join(cap, "intrinsics.json")) as f:
        return json.load(f)


def K_of(intr):
    return np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])


def load_meta(cap):
    p = os.path.join(cap, "meta.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def update_meta(cap, **kw):
    m = load_meta(cap)
    m.update(kw)
    with open(os.path.join(cap, "meta.json"), "w") as f:
        json.dump(m, f, indent=2)
    return m


def frame_names(cap):
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(cap, "rgb", "*.jpg")))


def depth_path(cap, name):
    return os.path.join(cap, "depth", os.path.splitext(name)[0] + ".png")


def read_depth_m(cap, name):
    """Depth png (uint16 mm) -> float32 metres (capture units); None if absent."""
    p = depth_path(cap, name)
    if not os.path.exists(p):
        return None
    d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    return None if d is None else d.astype(np.float32) / 1000.0


def write_depth_m(path, depth_m):
    mm = np.clip(np.nan_to_num(depth_m) * 1000.0, 0, 65535).round().astype(np.uint16)
    cv2.imwrite(path, mm)
