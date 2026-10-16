import os

import numpy as np

from conftest import REPO, SAMPLE
from pipeline.capture import common as cm
from pipeline.recon import colmap_io


def _rand_pose(rng):
    q = rng.normal(size=4)
    T = np.eye(4)
    T[:3, :3] = cm.quat_to_mat(*q)
    T[:3, 3] = rng.normal(size=3)
    return T


def test_pose_conversion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        T = _rand_pose(rng)
        q, t = colmap_io.c2w_to_colmap(T)
        assert abs(np.linalg.norm(q) - 1) < 1e-9 and q[0] >= 0
        assert np.allclose(colmap_io.colmap_to_c2w(q, t), T, atol=1e-9)


def test_model_write_read_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    poses = np.array([_rand_pose(rng) for _ in range(20)])
    names = [f"{i:05d}.jpg" for i in range(20)]
    intr = dict(fx=500.0, fy=501.0, cx=320.0, cy=180.0, width=640, height=360)
    P = rng.normal(size=(100, 3))
    C = rng.integers(0, 255, (100, 3)).astype(np.uint8)
    colmap_io.write_model(str(tmp_path), intr, names, poses, P, C)
    intr2, names2, poses2, P2, C2 = colmap_io.read_model(str(tmp_path))
    assert intr2 == intr and names2 == names
    assert np.allclose(poses2, poses, atol=1e-7)
    assert np.allclose(P2, P, atol=1e-6) and np.array_equal(C2, C)


def test_sample_matches_committed_model():
    """The capture->COLMAP conversion reproduces the model the sample splat was trained on."""
    ref = os.path.join(REPO, "pipeline", "recon", "cliproom_colmap_sparse", "0")
    intr, names, poses, _, _ = colmap_io.read_model(ref)
    assert names == cm.frame_names(SAMPLE)
    assert np.allclose(poses, cm.load_poses(SAMPLE), atol=1e-6)
    assert intr == {k: (float(v) if k not in ("width", "height") else v)
                    for k, v in cm.load_intrinsics(SAMPLE).items()}


def test_depth_backprojection_is_consistent(tmp_path):
    """Back-projected points reproject to the pixel they came from with that pixel's depth."""
    import cv2
    cap = tmp_path / "cap"
    cm.ensure_dirs(str(cap))
    cv2.imwrite(str(cap / "rgb" / "00000.jpg"), np.full((60, 80, 3), 128, np.uint8))
    ramp = np.tile(np.linspace(1.0, 3.0, 80, dtype=np.float32), (60, 1))
    cm.write_depth_m(str(cap / "depth" / "00000.png"), ramp)
    cm.save_intrinsics(str(cap), 50, 50, 40, 30, 80, 60)
    T = np.eye(4)
    T[:3, 3] = [1, 2, 3]
    P, _ = colmap_io.depth_points(str(cap), ["00000.jpg"], [T], cm.load_intrinsics(str(cap)), step=4)
    assert len(P) > 50
    Xc = P - T[:3, 3]
    u = Xc[:, 0] / Xc[:, 2] * 50 + 40
    expect = 1.0 + 2.0 * np.round(u) / 79.0
    assert np.allclose(Xc[:, 2], expect, atol=2e-3)
