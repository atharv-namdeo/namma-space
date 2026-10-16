"""Geometry for stage 2: Sim3 / Umeyama, RANSAC, floor plane + scene->world transform.

Frames (see docs/ARCHITECTURE.md):
  scene frame  - whatever the splat was trained in (cliproom: first camera, OpenCV y-down)
  world frame  - three.js: right-handed, +Y up, metres; floor of this room at y = 0
  toWorld      - 4x4, world = toWorld @ scene (row-major when flattened into scene.json)

Spawn yaw convention (three.js): a camera with rotation.y = yaw looks along
(-sin yaw, 0, -cos yaw). toWorld is built so that the first capture camera looks
along world -Z, i.e. its yaw is 0.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------- Sim3
def umeyama(src, dst, with_scale=True):
    """Least squares s, R, t with dst ~= s * R @ src + t (Umeyama 1991)."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var) if with_scale and var > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def sim3_matrix(s, R, t):
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def sim3_decompose(T):
    A = T[:3, :3]
    s = float(np.cbrt(np.linalg.det(A)))
    return s, A / s, T[:3, 3].copy()


def ransac_sim3(src, dst, thresh, iters=2000, seed=0, with_scale=True, min_inliers=6):
    """Robust Sim3 from 3D-3D correspondences. thresh in dst units.
    Returns (T 4x4, inlier mask)."""
    rng = np.random.default_rng(seed)
    n = len(src)
    if n < 3:
        raise ValueError("need >= 3 correspondences")
    best = np.zeros(n, bool)
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            s, R, t = umeyama(src[idx], dst[idx], with_scale)
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(s) or s <= 0:
            continue
        r = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        inl = r < thresh
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < min_inliers:
        raise ValueError(f"Sim3 RANSAC: only {int(best.sum())} inliers")
    for _ in range(3):  # refit on inliers, re-score
        s, R, t = umeyama(src[best], dst[best], with_scale)
        r = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        new = r < thresh
        if new.sum() < min_inliers:
            break
        best = new
    s, R, t = umeyama(src[best], dst[best], with_scale)
    return sim3_matrix(s, R, t), best


def apply(T, pts):
    pts = np.asarray(pts, np.float64)
    return pts @ T[:3, :3].T + T[:3, 3]


# ----------------------------------------------------------------------------- planes
def fit_plane(pts):
    c = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
    n = Vt[2]
    return n, -float(n @ c)


def ransac_plane(pts, thresh, iters=3000, seed=0, prior=None, max_angle_deg=35.0,
                 accept=None):
    """Plane n.x + d = 0 maximising inliers. With `prior`, candidate normals are flipped
    to agree with it and rejected beyond max_angle_deg. `accept(n, d) -> bool` can veto
    candidates (e.g. 'must be below the cameras'). Returns (n, d, inlier mask) or None."""
    rng = np.random.default_rng(seed)
    cos_min = np.cos(np.radians(max_angle_deg))
    best_n, best_d, best_c = None, None, 0
    tri = rng.integers(0, len(pts), size=(iters, 3))
    for a, b, c in tri:
        n = np.cross(pts[b] - pts[a], pts[c] - pts[a])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        if prior is not None:
            if n @ prior < 0:
                n = -n
            if n @ prior < cos_min:
                continue
        d = -float(n @ pts[a])
        cnt = int((np.abs(pts @ n + d) < thresh).sum())
        if cnt > best_c and (accept is None or accept(n, d)):
            best_n, best_d, best_c = n, d, cnt
    if best_n is None:
        return None
    n, d = best_n, best_d
    for k in (1.0, 0.5, 0.33):  # least-squares refits with a shrinking band, so wall
        sel = np.abs(pts @ n + d) < thresh * k   # bottoms / clutter near the plane drop out
        if sel.sum() < 3:
            break
        n2, d2 = fit_plane(pts[sel])
        n, d = (n2, d2) if n2 @ n >= 0 else (-n2, -d2)
    inl = np.abs(pts @ n + d) < thresh
    return n, d, inl


# ----------------------------------------------------------------------------- floor / toWorld
def splat_points_for_alignment(xyz, opac, min_opacity=0.3, max_points=60000, seed=0):
    """Opacity-filtered, floater-trimmed, subsampled gaussian centres."""
    keep = opac >= min_opacity
    if keep.sum() < 1000:
        keep = opac >= np.quantile(opac, 0.5)
    p = xyz[keep]
    lo, hi = np.percentile(p, 1, axis=0), np.percentile(p, 99, axis=0)
    p = p[np.all((p >= lo) & (p <= hi), axis=1)]
    if len(p) > max_points:
        p = p[np.random.default_rng(seed).choice(len(p), max_points, replace=False)]
    return p


def detect_floor(pts, cam_centers=None, cam_ups=None, up_hint=None, rel_thresh=0.012,
                 seed=0):
    """Floor plane (n, d, inliers, info) with n = unit up normal pointing toward cameras."""
    med = np.median(pts, 0)
    extent = float(np.percentile(np.linalg.norm(pts - med, axis=1), 90))
    thresh = max(rel_thresh * extent, 1e-4)
    info = {"thresh": thresh, "extent": extent}
    prior = None
    if cam_ups is not None and len(cam_ups):
        prior = np.mean(cam_ups, 0)
        info["prior"] = "camera-up"
    elif up_hint is not None:
        prior = np.asarray(up_hint, float)
        info["prior"] = "up-hint"
    if prior is not None:
        prior = prior / np.linalg.norm(prior)
        accept = None
        if cam_centers is not None and len(cam_centers):
            # plane must lie below (almost) the whole trajectory
            accept = lambda n, d: np.mean(cam_centers @ n + d > 0) >= 0.9  # noqa: E731
        res = ransac_plane(pts, thresh, prior=prior, accept=accept, seed=seed)
        if res is None:
            res = ransac_plane(pts, thresh, prior=prior, seed=seed)
        if res is None:
            raise ValueError("no floor plane found near the up prior")
        n, d, inl = res
        if cam_centers is not None and len(cam_centers) and np.median(cam_centers @ n + d) < 0:
            n, d = -n, -d
    else:
        # no prior: largest plane that has (almost) all points on one side; up = toward bulk
        rest, best = pts, None
        for k in range(6):
            res = ransac_plane(rest, thresh, seed=seed + k)
            if res is None:
                break
            n, d, inl_r = res
            side = pts @ n + d
            frac = max(np.mean(side > -thresh), np.mean(side < thresh))
            if frac > 0.95:
                if np.mean(side > -thresh) < 0.5:
                    n, d = -n, -d
                best = (n, d)
                break
            rest = rest[~inl_r]
        if best is None:
            raise ValueError("no bounding plane found; pass an up hint")
        n, d = best
        inl = np.abs(pts @ n + d) < thresh
        info["prior"] = "none (largest bounding plane; floor/ceiling ambiguous)"
    info["inliers"] = int(inl.sum())
    return n, float(d), inl, info


def world_transform(n, d, origin, forward=None, scale=1.0, floor_pts=None):
    """Build toWorld: up n -> +Y, `forward` (scene) -> -Z, floor -> y=0,
    `origin` (scene point, e.g. first camera) -> x=z=0, then uniform `scale`."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    if forward is None or np.linalg.norm(forward - (forward @ n) * n) < 1e-6:
        if floor_pts is not None and len(floor_pts) > 3:
            c = floor_pts - floor_pts.mean(0)
            forward = np.linalg.svd(c - np.outer(c @ n, n), full_matrices=False)[2][0]
        else:
            forward = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 0, 1.0])
    f = forward - (forward @ n) * n
    f /= np.linalg.norm(f)
    Y, Z = n, -f
    X = np.cross(Y, Z)
    R = np.stack([X, Y, Z])          # rows: world axes expressed in scene coords
    T = np.eye(4)
    T[:3, :3] = scale * R
    o = np.asarray(origin, float)
    T[:3, 3] = [-scale * (X @ o), scale * d, -scale * (Z @ o)]
    return T


def yaw_of(forward_world):
    fx, _, fz = forward_world
    return float(np.arctan2(-fx, -fz))
