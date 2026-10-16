"""Sim3 registration of a re-capture onto the base capture (real SIFT on sample frames).

The 're-capture' is made from sample frames the registration does NOT use on the base
side (odd frames vs base stride 4 starting at 0), moved by a known Sim3."""
import shutil

import numpy as np

from conftest import SAMPLE
from pipeline.capture import common as cm
from pipeline.recon import geometry as geo, register as rg


def test_umeyama_exact():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    q = rng.normal(size=4)
    R = cm.quat_to_mat(*q)
    Y = 2.5 * X @ R.T + [1, 2, 3]
    s, R2, t = geo.umeyama(X, Y)
    assert abs(s - 2.5) < 1e-9 and np.allclose(R2, R) and np.allclose(t, [1, 2, 3])
    Y[:10] += rng.normal(0, 5, (10, 3))
    T, inl = geo.ransac_sim3(X, Y, 0.01, iters=300)
    assert inl[10:].all() and not inl[:10].any()


def test_feature_registration_recovers_sim3(tmp_path):
    names = cm.frame_names(SAMPLE)
    P = cm.load_poses(SAMPLE)
    a = np.radians(30)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    s = 0.6
    G = geo.sim3_matrix(s, R, np.array([0.5, -0.1, 2.0]))     # base <- new
    Gi = np.linalg.inv(G)
    new = str(tmp_path / "new")
    cm.ensure_dirs(new)
    Q = []
    for k, i in enumerate(range(81, 101, 2)):
        n = f"{k:05d}.jpg"
        shutil.copyfile(f"{SAMPLE}/rgb/{names[i]}", f"{new}/rgb/{n}")
        cm.write_depth_m(cm.depth_path(new, n), cm.read_depth_m(SAMPLE, names[i]) / s)
        T = np.eye(4)
        T[:3, :3] = R.T @ P[i][:3, :3]
        T[:3, 3] = geo.apply(Gi, P[i][:3, 3][None])[0]
        Q.append(T)
    cm.save_poses(new, range(len(Q)), Q)
    shutil.copyfile(f"{SAMPLE}/intrinsics.json", f"{new}/intrinsics.json")

    T, info = rg.register_features(new, SAMPLE, n_query=5, base_stride=4, topk=2)
    s2, R2, t2 = geo.sim3_decompose(T)
    assert abs(s2 - s) / s < 0.01
    ang = np.degrees(np.arccos(np.clip((np.trace(R2.T @ R) - 1) / 2, -1, 1)))
    assert ang < 0.5
    assert np.linalg.norm(t2 - G[:3, 3]) < 0.02
    assert info["inliers"] > 100

    # the registered copy lands the new cameras on the original ones
    reg = rg.write_registered_capture(new, str(tmp_path / "reg"), T, base_id="cliproom")
    Pr = cm.load_poses(reg)
    assert np.abs(Pr[:, :3, 3] - P[81:101:2, :3, 3]).max() < 0.02
    assert cm.load_meta(reg)["base_frame"] == "cliproom"
