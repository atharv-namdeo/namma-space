"""Common capture format shared by every pose source (PoseCam / plain video /
later: rover IMU) and consumed by the reconstruction backbone.

A capture folder looks like:
    <cap>/rgb/000000.jpg ...        colour frames
    <cap>/depth/000000.png          uint16 depth in millimetres (metric)
    <cap>/poses.txt                 TUM:  t  tx ty tz  qx qy qz qw   (camera->world)
    <cap>/poses.npy                 (N,4,4) camera->world matrices
    <cap>/intrinsics.json           {fx, fy, cx, cy, width, height}

Poses are camera->world in the OpenCV camera convention (+X right, +Y down,
+Z forward) and a gravity-aligned world (so "up" is well defined). PoseCam /
ARCore give us that metrically; plain-video odometry gives it up to scale.

This module only does IO + the maths (quaternion<->matrix, GL<->CV) so every
other file imports from here instead of re-deriving conventions.
"""
import json
import os

import numpy as np
import open3d as o3d

# OpenGL camera (ARCore/ARKit/PoseCam: -Z forward, +Y up) -> OpenCV camera
# (+Z forward, +Y down). Right-multiply a camera->world matrix by this.
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


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


def quat_to_mat(qx, qy, qz, qw):
    """Unit quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def mat_to_quat(R):
    """3x3 rotation matrix -> quaternion (x,y,z,w)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=np.float64)


def pose_to_tum(t, T):
    """(timestamp, 4x4 camera->world) -> 'tx ty tz qx qy qz qw' TUM line body."""
    tx, ty, tz = T[:3, 3]
    qx, qy, qz, qw = mat_to_quat(T[:3, :3])
    return f"{t} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}"


def save_poses(cap_dir, timestamps, poses):
    """poses: (N,4,4) camera->world. Writes poses.txt (TUM) + poses.npy."""
    poses = np.asarray(poses)
    np.save(os.path.join(cap_dir, "poses.npy"), poses)
    with open(os.path.join(cap_dir, "poses.txt"), "w") as f:
        f.write("# TUM trajectory: timestamp tx ty tz qx qy qz qw (camera->world)\n")
        for t, T in zip(timestamps, poses):
            f.write(pose_to_tum(t, T) + "\n")


def load_poses(cap_dir):
    return np.load(os.path.join(cap_dir, "poses.npy"))


def save_intrinsics(cap_dir, fx, fy, cx, cy, width, height):
    with open(os.path.join(cap_dir, "intrinsics.json"), "w") as f:
        json.dump({"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                   "width": width, "height": height}, f, indent=2)


def load_intrinsics(cap_dir):
    with open(os.path.join(cap_dir, "intrinsics.json")) as f:
        return json.load(f)


def ensure_dirs(cap_dir):
    os.makedirs(os.path.join(cap_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(cap_dir, "depth"), exist_ok=True)
    return cap_dir
