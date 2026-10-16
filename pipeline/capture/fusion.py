"""Sensor fusion: solve metric scale for monocular depth from range sensors.

Sources of metric truth a capture can carry:
  * an external range sensor (VL53L1X ToF / HC-SR04 ultrasonic) logged to a CSV:
        timestamp_ns,range_m[,u,v]
    (u,v) = the pixel the sensor boresight hits; default = image centre.
  * ARCore Depth API images (uint16 millimetres, any resolution).

Both reduce to pairs (metric z, predicted z) and one robust ratio:

    scale = median(metric / predicted) after iterative MAD rejection in log space.

Applying it: depth pngs are multiplied by scale; for captures whose poses came from
the same monocular depth (plain-video RGB-D odometry) the pose translations are
multiplied too, so poses and depth stay consistent. ARCore/PoseCam poses are already
metric, so only the depth is rescaled there (`scale_poses=False`).
"""
from __future__ import annotations

import csv
import os

import cv2
import numpy as np

from . import common as cm


# ----------------------------------------------------------------------------- robust solve
def robust_scale(metric, predicted, max_log_dev=0.25, k_mad=3.0, iters=5, min_inliers=3):
    """Robust ratio metric/predicted.

    Works in log space (ratios are multiplicative): start from the median, keep samples
    within max(k_mad * 1.4826 * MAD, small floor) and also within `max_log_dev` of the
    current estimate, re-estimate, repeat. Returns (scale, inlier_mask, log_spread).
    """
    metric = np.asarray(metric, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    ok = np.isfinite(metric) & np.isfinite(predicted) & (metric > 0) & (predicted > 0)
    if ok.sum() < min_inliers:
        raise ValueError(f"scale solve: only {int(ok.sum())} valid samples (< {min_inliers})")
    lr = np.full(metric.shape, np.nan)
    lr[ok] = np.log(metric[ok] / predicted[ok])
    inl = ok.copy()
    c = float(np.median(lr[inl]))
    for _ in range(iters):
        dev = np.abs(lr - c)
        mad = float(np.median(dev[inl]))
        thr = min(max(k_mad * 1.4826 * mad, 0.02), max_log_dev)
        new = ok & (dev <= thr)
        if new.sum() < min_inliers:
            break
        c_new = float(np.median(lr[new]))
        done = np.array_equal(new, inl) and abs(c_new - c) < 1e-9
        inl, c = new, c_new
        if done:
            break
    spread = float(np.std(lr[inl])) if inl.sum() > 1 else 0.0
    return float(np.exp(c)), inl, spread


# ----------------------------------------------------------------------------- range CSV
def load_range_csv(path):
    """-> dict of arrays: t (int64 ns), r (m), u, v (float, NaN if absent)."""
    t, r, u, v = [], [], [], []
    with open(path, newline="") as f:
        rows = [row for row in csv.reader(f) if row and not row[0].lstrip().startswith("#")]
    if rows and not _is_num(rows[0][0]):
        hdr = [h.strip().lower() for h in rows[0]]
        rows = rows[1:]
        ix = {k: hdr.index(k) for k in ("timestamp_ns", "range_m") if k in hdr}
        if len(ix) < 2:
            raise ValueError(f"{path}: need columns timestamp_ns,range_m[,u,v]; got {hdr}")
        iu = hdr.index("u") if "u" in hdr else None
        iv = hdr.index("v") if "v" in hdr else None
    else:
        ix = {"timestamp_ns": 0, "range_m": 1}
        iu, iv = 2, 3
    for row in rows:
        try:
            t.append(int(float(row[ix["timestamp_ns"]])))
            r.append(float(row[ix["range_m"]]))
        except (ValueError, IndexError):
            continue
        u.append(float(row[iu]) if iu is not None and iu < len(row) and row[iu].strip() else np.nan)
        v.append(float(row[iv]) if iv is not None and iv < len(row) and row[iv].strip() else np.nan)
    return {"t": np.array(t, dtype=np.int64), "r": np.array(r), "u": np.array(u), "v": np.array(v)}


def _is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def sync_offset_from_tap(rng, frame_ts, sync_frame, near_m=0.15):
    """'Hand tap' sync: the first reading closer than `near_m` happened at frame
    `sync_frame` -> offset (ns) to add to sensor timestamps."""
    idx = np.flatnonzero(rng["r"] < near_m)
    if not len(idx):
        raise ValueError(f"range sync: no reading < {near_m} m to align with frame {sync_frame}")
    return int(frame_ts[sync_frame] - rng["t"][idx[0]])


def range_samples(cap, rng, offset_ns=0, max_dt_ns=40_000_000, win=2,
                  min_range=0.05, max_range=6.0):
    """Pair every range reading with the nearest-in-time frame's predicted depth.

    Range sensors measure distance along their boresight ray; we convert to z-depth
    using the camera ray through (u, v):  z = range / ||K^-1 [u v 1]||.
    Returns (metric_z, predicted_z, frame_idx) arrays.
    """
    names = cm.frame_names(cap)
    fts = cm.load_timestamps(cap)
    if fts is None or len(fts) != len(names):
        raise ValueError("range fusion needs per-frame timestamps (timestamps.txt or poses.txt)")
    if len(fts) > 1 and (fts.max() - fts.min()) < 1_000_000 * (len(fts) - 1) * 0.5:
        raise ValueError("frame timestamps look like frame indices, not ns; pass --frame-period-ms")
    intr = cm.load_intrinsics(cap)
    Kinv = np.linalg.inv(cm.K_of(intr))
    W, H = intr["width"], intr["height"]
    order = np.argsort(fts)
    fs = fts[order]
    met, pred, fidx = [], [], []
    cache = {}
    for t, r, u, v in zip(rng["t"] + offset_ns, rng["r"], rng["u"], rng["v"]):
        if not (min_range <= r <= max_range):
            continue
        j = int(np.searchsorted(fs, t))
        cands = [k for k in (j - 1, j) if 0 <= k < len(fs)]
        if not cands:
            continue
        k = min(cands, key=lambda k: abs(int(fs[k]) - int(t)))
        if abs(int(fs[k]) - int(t)) > max_dt_ns:
            continue
        fi = int(order[k])
        u = intr["cx"] if not np.isfinite(u) else u
        v = intr["cy"] if not np.isfinite(v) else v
        if fi not in cache:
            cache[fi] = cm.read_depth_m(cap, names[fi])
        d = cache[fi]
        if d is None:
            continue
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < W and 0 <= vi < H):
            continue
        patch = d[max(0, vi - win):vi + win + 1, max(0, ui - win):ui + win + 1]
        patch = patch[patch > 0]
        if not len(patch):
            continue
        ray = Kinv @ np.array([u, v, 1.0])
        met.append(r / np.linalg.norm(ray))
        pred.append(float(np.median(patch)))
        fidx.append(fi)
    return np.array(met), np.array(pred), np.array(fidx, dtype=int)


# ----------------------------------------------------------------------------- ARCore depth
def arcore_depth_samples(cap, arcore_depths, per_frame=400, min_m=0.2, max_m=6.0, seed=0):
    """arcore_depths: {frame_name: path to uint16 mm png (any resolution)}.
    Samples pixel pairs (ARCore metric z, predicted z)."""
    rs = np.random.default_rng(seed)
    met, pred = [], []
    for name, p in arcore_depths.items():
        a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        d = cm.read_depth_m(cap, name)
        if a is None or d is None:
            continue
        # DEPTH16: low 13 bits = mm, high 3 bits = confidence. Plain mm pngs are < 8192 too.
        a = (a.astype(np.uint32) & 0x1FFF).astype(np.float32) / 1000.0
        a = cv2.resize(a, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST)
        ok = (a > min_m) & (a < max_m) & (d > 0)
        ys, xs = np.nonzero(ok)
        if not len(ys):
            continue
        s = rs.choice(len(ys), min(per_frame, len(ys)), replace=False)
        met.append(a[ys[s], xs[s]])
        pred.append(d[ys[s], xs[s]])
    if not met:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(met), np.concatenate(pred)


# ----------------------------------------------------------------------------- apply
def apply_scale(cap, scale, scale_poses=True):
    """Multiply depth pngs (and pose translations) by `scale` in place."""
    names = cm.frame_names(cap)
    for n in names:
        d = cm.read_depth_m(cap, n)
        if d is not None:
            cm.write_depth_m(cm.depth_path(cap, n), d * scale)
    if scale_poses and os.path.exists(os.path.join(cap, "poses.npy")):
        P = cm.load_poses(cap).copy()
        P[:, :3, 3] *= scale
        ts = cm.load_timestamps(cap)
        cm.save_poses(cap, ts if ts is not None else range(len(P)), P)


def fuse(cap, range_csv=None, arcore_depths=None, offset_ns=0, sync_frame=None,
         scale_poses=True, max_dt_ms=40.0):
    """Solve + apply metric scale; writes meta.json. Returns the meta dict."""
    met, pred, sources = [], [], []
    if range_csv:
        rng = load_range_csv(range_csv)
        if sync_frame is not None:
            offset_ns = sync_offset_from_tap(rng, cm.load_timestamps(cap), sync_frame)
            print(f"[fusion] tap sync: sensor clock offset = {offset_ns} ns")
        m, p, _ = range_samples(cap, rng, offset_ns, int(max_dt_ms * 1e6))
        print(f"[fusion] range csv: {len(rng['r'])} readings -> {len(m)} matched to frames")
        met.append(m); pred.append(p); sources.append("range-csv")
        os.makedirs(os.path.join(cap, "sensors"), exist_ok=True)
        import shutil
        dst = os.path.join(cap, "sensors", "range.csv")
        if os.path.abspath(range_csv) != os.path.abspath(dst):
            shutil.copy(range_csv, dst)
    if arcore_depths:
        m, p = arcore_depth_samples(cap, arcore_depths)
        print(f"[fusion] ARCore depth: {len(m)} pixel samples from {len(arcore_depths)} frames")
        met.append(m); pred.append(p); sources.append("arcore-depth")
    if not met:
        return cm.load_meta(cap)
    met, pred = np.concatenate(met), np.concatenate(pred)
    s, inl, spread = robust_scale(met, pred)
    print(f"[fusion] metric scale = {s:.4f}  inliers {int(inl.sum())}/{len(inl)}  "
          f"log-spread {spread:.3f}")
    apply_scale(cap, s, scale_poses=scale_poses)
    return cm.update_meta(cap, metric=True, scale=s, scale_inliers=int(inl.sum()),
                          scale_samples=int(len(inl)), scale_log_spread=spread,
                          scale_sources=sources, scale_applied_to_poses=bool(scale_poses))
