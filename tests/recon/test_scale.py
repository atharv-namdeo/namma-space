"""Metric-scale solve from range sensors / ARCore depth on synthetic data."""
import cv2
import numpy as np
import pytest

from pipeline.capture import common as cm, fusion


def test_robust_scale_rejects_outliers():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0.5, 4.0, 300)
    true_s = 2.37
    met = pred * true_s * np.exp(rng.normal(0, 0.02, 300))   # 2 % sensor noise
    bad = rng.choice(300, 60, replace=False)                  # 20 % gross outliers
    met[bad] = rng.uniform(0.05, 8.0, 60)                     # multipath, wrong pixel, ...
    s, inl, spread = fusion.robust_scale(met, pred)
    assert abs(s - true_s) / true_s < 0.01
    assert inl.sum() >= 220 and inl[bad].mean() < 0.3 and spread < 0.05


def test_robust_scale_needs_samples():
    with pytest.raises(ValueError):
        fusion.robust_scale([1.0], [0.0])


def _synthetic_capture(tmp_path, s_true, n=12, W=64, H=48):
    """Relative capture whose depth/poses are 1/s_true of metric; returns (cap, metric depths)."""
    cap = str(tmp_path / "cap")
    cm.ensure_dirs(cap)
    rng = np.random.default_rng(1)
    ts = [1_000_000_000 + i * 100_000_000 for i in range(n)]  # 10 fps, ns
    metric = []
    poses = []
    for i in range(n):
        cv2.imwrite(f"{cap}/rgb/{i:05d}.jpg", np.zeros((H, W, 3), np.uint8))
        d = rng.uniform(1.0, 3.0) + 0.3 * np.sin(np.arange(W) / 7.0)[None].repeat(H, 0)
        metric.append(d.astype(np.float32))
        cm.write_depth_m(cm.depth_path(cap, f"{i:05d}.jpg"), d / s_true)
        T = np.eye(4)
        T[:3, 3] = [0.1 * i / s_true, 0, 0]
        poses.append(T)
    cm.save_poses(cap, ts, poses)
    cm.save_timestamps(cap, ts)
    cm.save_intrinsics(cap, 50, 50, W / 2, H / 2, W, H)
    return cap, metric, ts


def test_range_csv_fusion_end_to_end(tmp_path):
    s_true = 3.1
    cap, metric, ts = _synthetic_capture(tmp_path, s_true)
    K = cm.K_of(cm.load_intrinsics(cap))
    offset = 5_000_000_000            # sensor clock is 5 s behind the camera clock
    rows = ["timestamp_ns,range_m,u,v"]
    rng = np.random.default_rng(2)
    for i, t in enumerate(ts):
        for j in range(3):            # 3 readings per frame, +-10 ms jitter
            u, v = (int(rng.integers(10, 54)), int(rng.integers(10, 38))) if j else (None, None)
            uu, vv = (32, 24) if u is None else (u, v)
            z = metric[i][vv, uu]
            r = z * np.linalg.norm(np.linalg.inv(K) @ [uu, vv, 1])   # along the ray
            if i == 4 and j == 1:
                r = 0.3                                                # an outlier
            tt = t - offset + int(rng.integers(-10, 10)) * 1_000_000
            rows.append(f"{tt},{r:.4f}," + ("," if u is None else f"{u},{v}"))
    csv = tmp_path / "range.csv"
    csv.write_text("\n".join(rows) + "\n")
    meta = fusion.fuse(cap, range_csv=str(csv), offset_ns=offset)
    assert meta["metric"] is True
    assert abs(meta["scale"] - s_true) / s_true < 0.01
    assert meta["scale_inliers"] >= 30
    # depth and pose translations were rescaled
    d0 = cm.read_depth_m(cap, "00000.jpg")
    assert np.allclose(d0, metric[0], atol=0.01)
    assert abs(cm.load_poses(cap)[5][0, 3] - 0.5) < 0.01
    assert (tmp_path / "cap" / "sensors" / "range.csv").exists()


def test_tap_sync(tmp_path):
    cap, metric, ts = _synthetic_capture(tmp_path, 2.0)
    rng = {"t": np.array([100, 200, 300]), "r": np.array([1.0, 0.05, 1.0]),
           "u": np.full(3, np.nan), "v": np.full(3, np.nan)}
    off = fusion.sync_offset_from_tap(rng, cm.load_timestamps(cap), sync_frame=3)
    assert off == ts[3] - 200


def test_arcore_depth_scale(tmp_path):
    s_true = 1.8
    cap, metric, _ = _synthetic_capture(tmp_path, s_true)
    arc = {}
    for i, d in enumerate(metric):    # ARCore depth: quarter resolution, DEPTH16 conf bits set
        small = cv2.resize(d, (16, 12), interpolation=cv2.INTER_NEAREST)
        mm = (np.round(small * 1000).astype(np.uint16) & 0x1FFF) | (7 << 13)
        p = tmp_path / f"arc_{i}.png"
        cv2.imwrite(str(p), mm)
        arc[f"{i:05d}.jpg"] = str(p)
    meta = fusion.fuse(cap, arcore_depths=arc, scale_poses=False)
    assert abs(meta["scale"] - s_true) / s_true < 0.03
    assert meta["scale_applied_to_poses"] is False
