import json
import os

import numpy as np

from conftest import REPO, SAMPLE, make_splat
from pipeline.capture import common as cm
from pipeline.recon import geometry as geo, plyio
from pipeline.recon.build_scene import compute_alignment


def test_sample_floor_alignment(sample_ply):
    poses = cm.load_poses(SAMPLE)
    al = compute_alignment(sample_ply, poses, cm.load_meta(SAMPLE))
    T = al["toWorld"]
    info = al["alignment"]
    # floor normal maps to +Y
    assert np.allclose(info["upInWorld"], [0, 1, 0], atol=1e-6)
    # floor inliers sit at y ~ 0 in world
    n, d = np.array(info["floorPlane_scene"][:3]), info["floorPlane_scene"][3]
    pts = geo.splat_points_for_alignment(plyio.xyz(sample_ply), plyio.opacity(sample_ply))
    floor = geo.apply(T, pts[np.abs(pts @ n + d) < info["thresh"]])
    assert abs(floor[:, 1].mean()) < 0.01 and floor[:, 1].std() < info["thresh"]
    assert info["inliers"] > 3000
    # the phone was held roughly upright: camera up vectors point to world +Y
    ups_w = (T[:3, :3] @ (-poses[:, :3, 1]).T).T
    assert np.median(ups_w[:, 1] / np.linalg.norm(ups_w, axis=1)) > 0.95
    # cameras are above the floor, first camera at the origin looking down -Z
    cams = geo.apply(T, poses[:, :3, 3])
    assert np.mean(cams[:, 1] > 0) > 0.9
    assert np.allclose(cams[0][[0, 2]], 0, atol=1e-9)
    assert abs(al["spawn"]["yaw"]) < 1e-9 and al["spawn"]["position"][1] > 0
    # proper rotation (no mirror)
    assert np.linalg.det(T[:3, :3]) > 0


def test_committed_scene_json_matches_recompute(sample_ply):
    sj = json.load(open(os.path.join(REPO, "web", "scenes", "cliproom", "scene.json")))
    al = compute_alignment(sample_ply, cm.load_poses(SAMPLE), cm.load_meta(SAMPLE))
    assert np.allclose(np.array(sj["toWorld"]).reshape(4, 4), al["toWorld"], atol=1e-6)
    assert sj["metric"] is False and sj["floorY"] == 0.0


def test_synthetic_tilted_room_metric():
    """Box room (floor y=0 .. ceiling 3 m, Y-up) viewed in an OpenCV-style scene frame that
    is rotated and translated; recover the room's floor exactly and keep metres."""
    rng = np.random.default_rng(0)
    n = 30000
    floor = np.c_[rng.uniform(-3, 3, n), np.zeros(n), rng.uniform(-4, 4, n)]
    ceil = np.c_[rng.uniform(-3, 3, n // 2), np.full(n // 2, 3.0), rng.uniform(-4, 4, n // 2)]
    wall = np.c_[np.full(n // 2, 3.0), rng.uniform(0, 3, n // 2), rng.uniform(-4, 4, n // 2)]
    table = np.c_[rng.uniform(-0.5, 0.5, 3000), np.full(3000, 0.75), rng.uniform(-0.5, 0.5, 3000)]
    room = np.concatenate([floor, ceil, wall, table])
    # world(Y-up) -> scene(OpenCV-ish, Y-down, tilted 20 deg, shifted)
    a = np.radians(20)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    flip = np.diag([1, -1, -1])
    M = Rx @ flip
    t = np.array([0.4, 1.3, -0.2])
    scene = room @ M.T + t
    splat = make_splat(len(scene))
    splat.data["x"], splat.data["y"], splat.data["z"] = scene.T.astype(np.float32)
    splat.data["opacity"] = 5.0
    # cameras at 1.5 m height walking along world -Z, looking along world -Z
    poses = []
    for k in range(10):
        c_w = np.array([0.0, 1.5, -0.2 * k])
        R_wc = np.stack([[1, 0, 0], [0, -1, 0], [0, 0, -1]], 1).astype(float)  # cv cam in Y-up world
        T = np.eye(4)
        T[:3, :3] = M @ R_wc
        T[:3, 3] = M @ c_w + t
        poses.append(T)
    al = compute_alignment(splat, np.array(poses), {"metric": True})
    W = geo.apply(al["toWorld"], scene)
    assert np.abs(W[:n, 1]).max() < 1e-3                           # floor at y = 0
    assert np.abs(W[n:n + n // 2, 1] - 3.0).max() < 1e-3          # ceiling at 3 m (metric kept)
    assert abs(al["spawn"]["position"][1] - 1.6) < 1e-9 and al["metric"] is True
    assert abs(al["alignment"]["cameraHeight"] - 1.5) < 1e-3
