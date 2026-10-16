"""Top-down floor-plan rendering (PNG via OpenCV, SVG by hand).

Orientation: x to the right, z downwards (three.js -Z = "forward" points up
the page). SVG user units are metres in world x/z, so overlays from
building.json can be drawn with the same numbers.
"""
from __future__ import annotations

import cv2
import numpy as np

from .grid import BLOCKED, FREE, UNKNOWN, NavGrid

# BGR for OpenCV; hex for SVG
COL = {
    "free": ((246, 241, 236), "#ecf1f6"),
    "blocked": ((70, 58, 52), "#343a46"),
    "unknown": ((214, 210, 206), "#ced2d6"),
    "outline": ((48, 140, 232), "#e88c30"),
    "traj": ((60, 160, 60), "#3ca03c"),
}


def render_png(grid: NavGrid, path: str, px: int = 8, outline=None,
               draw_traj: bool = True) -> None:
    H, W = grid.cells.shape
    img = np.empty((H, W, 3), np.uint8)
    img[grid.cells == FREE] = COL["free"][0]
    img[grid.cells == BLOCKED] = COL["blocked"][0]
    img[grid.cells == UNKNOWN] = COL["unknown"][0]
    img = cv2.resize(img, (W * px, H * px), interpolation=cv2.INTER_NEAREST)
    # faint 1 m grid
    step = max(1, int(round(1.0 / grid.cell_size)))
    for i in range(0, W, step):
        img[:, i * px] = (img[:, i * px] * 0.9).astype(np.uint8)
    for j in range(0, H, step):
        img[j * px, :] = (img[j * px, :] * 0.9).astype(np.uint8)

    def to_px(x, z):
        return (int(round((x - grid.origin[0]) / grid.cell_size * px)),
                int(round((z - grid.origin[1]) / grid.cell_size * px)))

    if outline:
        pts = np.array([to_px(x, z) for x, z in outline], np.int32)
        cv2.polylines(img, [pts], True, COL["outline"][0], 2, cv2.LINE_AA)
    traj = grid.extra.get("trajectory") if draw_traj else None
    if traj:
        pts = np.array([to_px(x, z) for x, z in traj], np.int32)
        cv2.polylines(img, [pts], False, COL["traj"][0], 2, cv2.LINE_AA)
        cv2.circle(img, tuple(int(v) for v in pts[0]), 5, COL["traj"][0], -1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def _runs_svg(mask: np.ndarray, grid: NavGrid, fill: str) -> str:
    """One <path> of horizontal run rectangles (compact)."""
    cs, (x0, z0) = grid.cell_size, grid.origin
    parts = []
    for j in range(mask.shape[0]):
        row = mask[j]
        if not row.any():
            continue
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        z = z0 + j * cs
        for s, e in zip(starts, ends):
            x = x0 + s * cs
            parts.append(f"M{x:.2f} {z:.2f}h{(e - s) * cs:.2f}v{cs:.2f}h{-(e - s) * cs:.2f}z")
    if not parts:
        return ""
    return f'<path fill="{fill}" d="{"".join(parts)}"/>'


def render_svg(grid: NavGrid, path: str, outline=None, overlays: str = "",
               draw_traj: bool = True) -> None:
    cs, (x0, z0) = grid.cell_size, grid.origin
    w, h = grid.width * cs, grid.height * cs
    body = [
        f'<rect x="{x0:.2f}" y="{z0:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{COL["unknown"][1]}"/>',
        _runs_svg(grid.cells == FREE, grid, COL["free"][1]),
        _runs_svg(grid.cells == BLOCKED, grid, COL["blocked"][1]),
    ]
    if outline:
        pts = " ".join(f"{x:.2f},{z:.2f}" for x, z in outline)
        body.append(f'<polygon points="{pts}" fill="none" stroke="{COL["outline"][1]}" '
                    f'stroke-width="{max(cs * 0.8, 0.05):.2f}"/>')
    traj = grid.extra.get("trajectory") if draw_traj else None
    if traj:
        pts = " ".join(f"{x:.2f},{z:.2f}" for x, z in traj)
        body.append(f'<polyline points="{pts}" fill="none" stroke="{COL["traj"][1]}" '
                    f'stroke-width="{max(cs * 0.6, 0.04):.2f}" stroke-linecap="round"/>')
    body.append(overlays)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.2f} {z0:.2f} {w:.2f} {h:.2f}" '
           f'width="{grid.width * 6}" height="{grid.height * 6}">'
           f'<title>{grid.scene_id} floor plan (x right, z down, metres)</title>'
           + "".join(b for b in body if b) + "</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
