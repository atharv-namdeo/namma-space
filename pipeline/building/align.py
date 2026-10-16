"""Scene frame -> world (Y up, metres, floor at y=0) alignment.

Preferred source is ``web/scenes/<id>/scene.json`` ``toWorld`` (owned by the
reconstruction stage). When it is missing we estimate one ourselves:

1. gravity: RANSAC a plane whose normal is within ``max_tilt`` of the mean
   camera "down" axis and which lies below every camera; up = towards cameras.
   The floor height is refined as a robust mode of point heights near it.
2. scale: if the capture is not metric, scale so the median camera height
   above the floor equals ``camera_height`` metres (phone held at chest/eye
   height ~1.4 m).
3. yaw: Manhattan alignment - rotate about up so wall points project onto
   the sharpest x/z histograms (walls become axis-aligned); among the four
   90 degree solutions pick the one where the first camera looks along -Z.
4. origin: first camera position projected onto the floor.

The result is cached to ``pipeline/building/cache/<id>_alignment.json``; it is
never written to scene.json (another stage owns that file).
"""
from __future__ import annotations

import json
import os

import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def mat_from_rowmajor(a) -> np.ndarray:
    m = np.asarray(a, dtype=np.float64)
    return m.reshape(4, 4)


def apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def ransac_plane(pts, rng, iters=2000, thresh=0.02, normal_hint=None,
                 max_tilt_deg=35.0, below_of=None, below_penalty=3.0):
    """Supporting-plane RANSAC for the floor.

    Plane (n, d) with n.p + d = 0, n pointing "down" (along ``normal_hint``).
    Score = inliers - below_penalty * (points further down than the plane), so
    the winner is the *lowest dense* plane, not just the biggest one (monocular
    clouds are often fan-shaped and have dense layers above the real floor).
    ``below_of`` (camera centres): the median camera must lie above the plane.
    """
    best_cnt, best = -np.inf, None
    cos_tol = np.cos(np.radians(max_tilt_deg))
    n_pts = len(pts)
    cam_med = None if below_of is None else np.median(below_of, 0)
    for _ in range(iters):
        s = pts[rng.choice(n_pts, 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        if normal_hint is not None:
            c = float(n @ normal_hint)
            if abs(c) < cos_tol:
                continue
            if c < 0:
                n = -n                       # n points along hint ("down")
        d = -float(n @ s[0])
        if cam_med is not None and cam_med @ n + d > -thresh:
            continue                         # cameras must be above (n.c + d < 0)
        sd = pts @ n + d
        cnt = float((np.abs(sd) < thresh).sum() - below_penalty * (sd > thresh).sum())
        if cnt > best_cnt:
            best_cnt, best = cnt, (n, d)
    if best is None:
        raise RuntimeError("RANSAC found no admissible floor plane")
    n, d = best
    inl = np.abs(pts @ n + d) < thresh
    # least-squares refine on inliers
    q = pts[inl]
    c = q.mean(0)
    _, _, vt = np.linalg.svd(q - c, full_matrices=False)
    n2 = vt[2] if vt[2] @ n > 0 else -vt[2]
    return n2, -float(n2 @ c), int(inl.sum())


def manhattan_yaw(xz: np.ndarray, bin_size: float) -> float:
    """Yaw (radians) that makes the projected wall points most axis-aligned."""
    if len(xz) < 50:
        return 0.0
    best, best_score = 0.0, -1.0
    for deg in np.arange(0.0, 90.0, 0.5):
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        u = xz[:, 0] * c - xz[:, 1] * s
        v = xz[:, 0] * s + xz[:, 1] * c
        score = 0.0
        for w in (u, v):
            h = np.bincount(((w - w.min()) / bin_size).astype(np.int64))
            score += float((h.astype(np.float64) ** 2).sum())
        if score > best_score:
            best, best_score = a, score
    return best


def yaw_matrix(a: float) -> np.ndarray:
    """Rotation about +Y by angle a (right-handed, three.js)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def estimate_alignment(pts: np.ndarray, poses: np.ndarray, metric: bool = False,
                       camera_height: float = 1.4, room_height: float = 3.0,
                       scale_cue: str = "auto", seed: int = 0,
                       manhattan: bool = True) -> dict:
    """Estimate toWorld from splat centres + camera->world poses (scene frame)."""
    rng = np.random.default_rng(seed)
    cams = poses[:, :3, 3]
    down_hint = poses[:, :3, 1].mean(0)            # OpenCV y axis = image down
    down_hint /= np.linalg.norm(down_hint)
    extent = float(np.percentile(np.linalg.norm(pts - np.median(pts, 0), axis=1), 90))
    thresh = max(extent * 0.006, 1e-3)
    sub = pts if len(pts) <= 60000 else pts[rng.choice(len(pts), 60000, replace=False)]
    n_down, d, n_inl = ransac_plane(sub, rng, iters=3000, thresh=thresh,
                                    normal_hint=down_hint, below_of=cams)
    up = -n_down
    # heights above the plane (scene units); refine floor as a dense mode
    h = pts @ up - d                                # n_down.p + d = 0  <=>  up.p - d = 0
    band = np.abs(h) < 6 * thresh
    if band.sum() > 100:
        hist, edges = np.histogram(h[band], bins=60)
        k = int(np.argmax(hist))
        floor_off = 0.5 * (edges[k] + edges[k + 1])
    else:
        floor_off = 0.0
    cam_h = cams @ up - d - floor_off
    cam_h_med = float(np.median(cam_h))
    ceil_h = float(np.percentile(h - floor_off, 98))
    if metric:
        scale, cue = 1.0, "metric"
    elif scale_cue == "camera-height" or (scale_cue == "auto" and ceil_h < 2.5 * cam_h_med):
        scale, cue = camera_height / cam_h_med, "camera-height"
    else:
        # monocular depth compresses the near field, so the camera often looks
        # glued to the floor; the floor-to-ceiling span is the steadier cue
        scale, cue = room_height / ceil_h, "room-height"

    # basis: Y = up; X = first camera x-axis projected on floor
    x0 = poses[0, :3, 0] - (poses[0, :3, 0] @ up) * up
    x0 /= np.linalg.norm(x0)
    z0 = np.cross(x0, up)                           # X x Y = Z
    R = np.stack([x0, up, z0], 0)                   # rows: world axes in scene frame

    origin_scene = cams[0] - (cams[0] @ up - d - floor_off) * up
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = -scale * (R @ origin_scene)

    yaw = 0.0
    if manhattan:
        w = apply(T, pts)
        wall = (w[:, 1] > 0.3) & (w[:, 1] < 2.0)
        yaw = manhattan_yaw(w[wall][:, [0, 2]], bin_size=0.05)
        # pick the 90deg branch whose result keeps camera-0 forward closest to -Z
        fwd = R @ poses[0, :3, 2]
        best = None
        for k in range(4):
            a = yaw + k * np.pi / 2
            f = yaw_matrix(a) @ fwd
            score = -f[2]
            if best is None or score > best[0]:
                best = (score, a)
        yaw = best[1]
        Ry = np.eye(4)
        Ry[:3, :3] = yaw_matrix(yaw)
        T = Ry @ T

    T[np.abs(T) < 1e-12] = 0.0
    return {
        "toWorld": T.reshape(-1).tolist(),
        "source": "estimated",
        "method": f"ransac-floor + {cue} scale + manhattan yaw",
        "floorInliers": n_inl,
        "upScene": up.tolist(),
        "scale": scale,
        "metric": bool(metric),
        "scaleCue": cue,
        "assumedCameraHeight": camera_height if cue == "camera-height" else None,
        "assumedRoomHeight": room_height if cue == "room-height" else None,
        "medianCameraHeightScene": cam_h_med,
        "ceilingHeightScene": ceil_h,
        "medianCameraHeightWorld": cam_h_med * scale,
        "yawDeg": float(np.degrees(yaw)),
    }


def get_alignment(scene_dir: str, pts: np.ndarray | None = None,
                  poses_path: str | None = None, camera_height: float = 1.4,
                  room_height: float = 3.0, scale_cue: str = "auto",
                  force_estimate: bool = False, write_cache: bool = True) -> dict:
    """Return {'toWorld': 16 floats, 'source': ...}. Uses scene.json if present."""
    sid = os.path.basename(os.path.normpath(scene_dir))
    sj = os.path.join(scene_dir, "scene.json")
    meta = {}
    if os.path.exists(sj):
        with open(sj, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "toWorld" in meta and not force_estimate:
            return {"toWorld": [float(x) for x in meta["toWorld"]],
                    "source": "scene.json", "metric": bool(meta.get("metric", False))}
    if pts is None or poses_path is None:
        raise ValueError("no scene.json toWorld; need points + poses to estimate")
    poses = np.load(poses_path)
    al = estimate_alignment(pts, poses, metric=bool(meta.get("metric", False)),
                            camera_height=camera_height, room_height=room_height,
                            scale_cue=scale_cue)
    al["sceneId"] = sid
    al["poses"] = poses_path.replace("\\", "/")
    if write_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, f"{sid}_alignment.json"), "w",
                  encoding="utf-8") as f:
            json.dump(al, f, indent=2)
    return al
