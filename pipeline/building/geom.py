"""Small 2D polygon helpers (numpy + OpenCV only; no shapely)."""
from __future__ import annotations

import cv2
import numpy as np


def polygon_area(poly) -> float:
    p = np.asarray(poly, np.float64)
    if len(p) < 3:
        return 0.0
    x, z = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))))


def polygon_centroid(poly) -> list[float]:
    p = np.asarray(poly, np.float64)
    x, z = p[:, 0], p[:, 1]
    cross = x * np.roll(z, -1) - np.roll(x, -1) * z
    a = cross.sum() / 2.0
    if abs(a) < 1e-12:
        return p.mean(0).tolist()
    cx = ((x + np.roll(x, -1)) * cross).sum() / (6 * a)
    cz = ((z + np.roll(z, -1)) * cross).sum() / (6 * a)
    return [float(cx), float(cz)]


def point_in_polygon(pt, poly) -> bool:
    """Even-odd rule; points exactly on an edge may go either way."""
    x, z = float(pt[0]), float(pt[1])
    p = np.asarray(poly, np.float64)
    inside = False
    n = len(p)
    for k in range(n):
        x1, z1 = p[k]
        x2, z2 = p[(k + 1) % n]
        if (z1 > z) != (z2 > z):
            xi = x1 + (z - z1) * (x2 - x1) / (z2 - z1)
            if x < xi:
                inside = not inside
    return inside


def dist_point_segment(pt, a, b) -> float:
    p, a, b = (np.asarray(v, np.float64) for v in (pt, a, b))
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def dist_to_boundary(pt, poly) -> float:
    p = np.asarray(poly, np.float64)
    return min(dist_point_segment(pt, p[k], p[(k + 1) % len(p)]) for k in range(len(p)))


def bbox(poly):
    p = np.asarray(poly, np.float64)
    return p.min(0), p.max(0)


def rasterize(polys, res: float, lo=None, hi=None):
    """Rasterise polygons on a common grid; returns (masks list, lo)."""
    allp = np.concatenate([np.asarray(p, np.float64) for p in polys], 0)
    lo = allp.min(0) - res if lo is None else np.asarray(lo)
    hi = allp.max(0) + res if hi is None else np.asarray(hi)
    W = int(np.ceil((hi[0] - lo[0]) / res)) + 1
    H = int(np.ceil((hi[1] - lo[1]) / res)) + 1
    masks = []
    for poly in polys:
        m = np.zeros((H, W), np.uint8)
        q = np.round((np.asarray(poly, np.float64) - lo) / res - 0.5).astype(np.int32)
        cv2.fillPoly(m, [q], 1)
        masks.append(m.astype(bool))
    return masks, lo


def overlap_area(a, b, res: float = 0.05) -> float:
    """Approximate interior overlap of two polygons (m^2). Edges shared by
    touching polygons do not count: each mask is eroded by one pixel."""
    (ma, mb), _ = rasterize([a, b], res)
    k = np.ones((3, 3), np.uint8)
    ma = cv2.erode(ma.astype(np.uint8), k).astype(bool)
    mb = cv2.erode(mb.astype(np.uint8), k).astype(bool)
    return float((ma & mb).sum() * res * res)


def transform_2d(points, yaw_deg: float = 0.0, translate=(0.0, 0.0)):
    """Rotate about +Y by yaw (three.js convention, x'=x cos+z sin,
    z'=-x sin+z cos) then translate, for (x,z) points."""
    a = np.radians(yaw_deg)
    c, s = np.cos(a), np.sin(a)
    p = np.asarray(points, np.float64).reshape(-1, 2)
    x = p[:, 0] * c + p[:, 1] * s + translate[0]
    z = -p[:, 0] * s + p[:, 1] * c + translate[1]
    return np.stack([x, z], 1)


def placement_matrix(yaw_deg: float = 0.0, translate=(0.0, 0.0), elevation: float = 0.0):
    """Row-major 4x4 (as flat list) for the 2D placement, about +Y."""
    a = np.radians(yaw_deg)
    c, s = np.cos(a), np.sin(a)
    m = np.array([[c, 0, s, translate[0]],
                  [0, 1, 0, elevation],
                  [-s, 0, c, translate[1]],
                  [0, 0, 0, 1]], np.float64)
    m[np.abs(m) < 1e-12] = 0.0
    return m
