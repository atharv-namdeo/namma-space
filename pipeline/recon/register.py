"""Register a new (re-)capture to a base scene: Sim3  T (base/scene frame <- new frame).

Method `features` (default, robust):
  1. sample `n_query` frames of the new capture and every `base_stride`-th base frame;
     SIFT (fallback ORB) features with Lowe ratio test;
  2. for each query keep the `topk` base frames with most matches, geometrically verify
     each pair with a fundamental-matrix RANSAC;
  3. lift every verified match to 3D twice - with the new capture's depth + pose (new
     frame) and the base capture's depth + pose (base frame);
  4. RANSAC Umeyama Sim3 over all 3D-3D correspondences (scale included, so a relative
     re-capture registers onto a relative or metric base), refit on inliers.
Optional `icp=True`: point-to-point Sim3 ICP (scipy cKDTree) between the new capture's
depth points and the base's, started from the feature solution.

Method `skip`: the new capture already shares the base frame (e.g. ARCore session
re-localised to the same anchors, or the frames came from the base capture).
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from ..capture import common as cm
from . import colmap_io, geometry as geo


def _detector():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=1500), cv2.NORM_L2
    return cv2.ORB_create(2000), cv2.NORM_HAMMING


def _feats(det, cap, name):
    g = cv2.imread(os.path.join(cap, "rgb", name), cv2.IMREAD_GRAYSCALE)
    k, d = det.detectAndCompute(g, None)
    pts = np.float32([kp.pt for kp in k]) if k else np.zeros((0, 2), np.float32)
    return pts, d


def _match(bf, d0, d1, ratio=0.75):
    if d0 is None or d1 is None or len(d0) < 2 or len(d1) < 2:
        return []
    good = []
    for m in bf.knnMatch(d0, d1, k=2):
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            good.append((m[0].queryIdx, m[0].trainIdx))
    return good


def _lift(cap, name, pose, Kinv, uv, depth_cache):
    if name not in depth_cache:
        depth_cache[name] = cm.read_depth_m(cap, name)
    d = depth_cache[name]
    if d is None:
        raise SystemExit(f"[register] {cap} has no depth for {name}; feature registration needs depth")
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, d.shape[1] - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, d.shape[0] - 1)
    z = d[v, u].astype(np.float64)
    ok = (z > 0) & (z < 0.999 * d.max())        # drop invalid + clamp-at-far pixels
    X = (Kinv @ np.vstack([uv.T, np.ones(len(uv))])) * z
    return (pose[:3, :3] @ X).T + pose[:3, 3], ok


def register_features(new_cap, base_cap, n_query=12, base_stride=3, topk=3, min_pair=30,
                      rel_thresh=0.03, seed=0):
    det, norm = _detector()
    bf = cv2.BFMatcher(norm)
    nn, bn = cm.frame_names(new_cap), cm.frame_names(base_cap)
    nP, bP = cm.load_poses(new_cap), cm.load_poses(base_cap)
    nKi = np.linalg.inv(cm.K_of(cm.load_intrinsics(new_cap)))
    bKi = np.linalg.inv(cm.K_of(cm.load_intrinsics(base_cap)))
    q_idx = np.unique(np.linspace(0, len(nn) - 1, min(n_query, len(nn))).round().astype(int))
    b_idx = list(range(0, len(bn), base_stride))
    bfeat = {j: _feats(det, base_cap, bn[j]) for j in b_idx}
    src, dst, pairs = [], [], []
    ncache, bcache = {}, {}
    for i in q_idx:
        qp, qd = _feats(det, new_cap, nn[i])
        scored = sorted(((len(m), j, m) for j in b_idx
                         for m in [_match(bf, qd, bfeat[j][1])]), key=lambda x: -x[0])[:topk]
        for cnt, j, m in scored:
            if cnt < min_pair:
                continue
            a = qp[[x[0] for x in m]]
            b = bfeat[j][0][[x[1] for x in m]]
            F, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 2.0, 0.999)
            if F is None:
                continue
            mask = mask.ravel().astype(bool)
            if mask.sum() < min_pair // 2:
                continue
            Xa, oka = _lift(new_cap, nn[i], nP[i], nKi, a[mask], ncache)
            Xb, okb = _lift(base_cap, bn[j], bP[j], bKi, b[mask], bcache)
            ok = oka & okb
            src.append(Xa[ok]); dst.append(Xb[ok])
            pairs.append((int(i), int(j), int(ok.sum())))
    if not src:
        raise SystemExit("[register] no verified feature matches between the captures; "
                         "re-shoot with more overlap or pass --register skip")
    src, dst = np.concatenate(src), np.concatenate(dst)
    scale_ref = float(np.median(np.linalg.norm(dst - np.median(dst, 0), axis=1)))
    thresh = rel_thresh * max(scale_ref, 1e-6)
    T, inl = geo.ransac_sim3(src, dst, thresh, seed=seed)
    res = np.linalg.norm(geo.apply(T, src[inl]) - dst[inl], axis=1)
    info = dict(method="features+sim3-ransac", detector="sift" if norm == cv2.NORM_L2 else "orb",
                pairs=len(pairs), correspondences=int(len(src)), inliers=int(inl.sum()),
                thresh=thresh, rmse=float(np.sqrt(np.mean(res ** 2))))
    print(f"[register] {len(pairs)} frame pairs, {len(src)} 3D matches, {int(inl.sum())} inliers, "
          f"rmse {info['rmse']:.4f}")
    return T, info


def icp_sim3(T0, src_pts, dst_pts, iters=30, trim=0.8):
    """Trimmed point-to-point Sim3 ICP refinement (src -> dst)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(dst_pts)
    T = T0.copy()
    for _ in range(iters):
        p = geo.apply(T, src_pts)
        dd, j = tree.query(p)
        keep = dd <= np.quantile(dd, trim)
        s, R, t = geo.umeyama(src_pts[keep], dst_pts[j[keep]])
        T_new = geo.sim3_matrix(s, R, t)
        if np.abs(T_new - T).max() < 1e-7:
            T = T_new
            break
        T = T_new
    return T


def capture_cloud(cap, max_points=40000):
    names = cm.frame_names(cap)
    P, _ = colmap_io.depth_points(cap, names, cm.load_poses(cap), cm.load_intrinsics(cap),
                                  max_points=max_points)
    return P


def register(new_cap, base_cap, method="features", icp=False, **kw):
    if method == "skip":
        return np.eye(4), {"method": "skip (shared frame)"}
    T, info = register_features(new_cap, base_cap, **kw)
    if icp:
        T = icp_sim3(T, capture_cloud(new_cap), capture_cloud(base_cap))
        info["method"] += "+icp"
    return T, info


def write_registered_capture(new_cap, out, T, base_id=None):
    """Copy a capture into the base frame: poses c2w' = Sim3 * c2w (rotation part only on
    the camera axes), translations and depth scaled by the Sim3 scale."""
    import shutil
    s, R, t = geo.sim3_decompose(T)
    cm.ensure_dirs(out, depth=os.path.isdir(os.path.join(new_cap, "depth")))
    names = cm.frame_names(new_cap)
    for n in names:
        shutil.copyfile(os.path.join(new_cap, "rgb", n), os.path.join(out, "rgb", n))
        d = cm.read_depth_m(new_cap, n)
        if d is not None:
            cm.write_depth_m(cm.depth_path(out, n), d * s)
    P = cm.load_poses(new_cap)
    Q = np.repeat(np.eye(4)[None], len(P), 0)
    Q[:, :3, :3] = R @ P[:, :3, :3]
    Q[:, :3, 3] = (s * (R @ P[:, :3, 3].T)).T + t
    ts = cm.load_timestamps(new_cap)
    cm.save_poses(out, ts if ts is not None else range(len(Q)), Q)
    shutil.copyfile(os.path.join(new_cap, "intrinsics.json"), os.path.join(out, "intrinsics.json"))
    meta = cm.load_meta(new_cap)
    meta.update(base_frame=base_id, registered_from=new_cap, sim3=T.tolist())
    cm.update_meta(out, **meta)
    return out
