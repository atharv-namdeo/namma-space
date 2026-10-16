"""Build web/scenes/<id>/navgrid.json (+ floorplan.png/svg) from a splat scene.

    python -m pipeline.building.navgrid --scene web/scenes/cliproom [--cell 0.1]

Steps (all in world frame: Y up, metres, floor y=0):
 1. splat centres (opacity + scale filtered) -> world via scene.json toWorld,
    or our own estimate (pipeline/building/cache/<id>_alignment.json).
 2. per-cell counts: floor band (|y| < floor_tol), obstacle band
    (obstacle_lo..obstacle_hi, default 0.15-1.8 m) and everything.
 3. footprint = observed cells, closed + hole-filled, largest component.
 4. blocked = obstacle-band count >= min points (auto: adaptive to density)
    and obstacle-dominant column; footprint rim sealed as wall.
    free    = footprint, floor observed (or enclosed by free), not blocked.
    unknown = rest.
 5. camera trajectory (where the person walked) forced free.
 6. cleanup (drop tiny blocked specks, close free), inflate blocked by the
    walker radius (trajectory stays free), drop free pockets < 0.5 m^2 that
    do not touch the trajectory (--keep-connected: keep only the free space
    connected to the trajectory).

The stored grid is already inflated: a planner treats the walker as a point.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import align
from .floorplan import render_png, render_svg
from .grid import BLOCKED, FREE, UNKNOWN, NavGrid


def _disk(radius_cells: float) -> np.ndarray:
    r = int(np.ceil(radius_cells))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= radius_cells * radius_cells + 1e-9


def _bincount2(i, j, w, h):
    g = np.zeros((h, w), np.int32)
    np.add.at(g, (j, i), 1)
    return g


def _ground_field(i, j, y, W, H, cell, window, max_dev):
    order = np.lexsort((y, j * W + i))
    key = (j * W + i)[order]
    ys = y[order]
    uniq, start, counts = np.unique(key, return_index=True, return_counts=True)
    g = np.full(W * H, np.nan)
    keep = counts >= 3
    # 10th percentile of each cell's heights (sorted within cell)
    idx = start[keep] + (0.1 * (counts[keep] - 1)).astype(np.int64)
    g[uniq[keep]] = ys[idx]
    g = g.reshape(H, W)
    g = np.clip(g, -max_dev, max_dev)
    nan = np.isnan(g)
    if nan.all():
        return np.zeros((H, W))
    if nan.any():
        _, (nj, ni) = ndi.distance_transform_edt(nan, return_indices=True)
        g = g[nj, ni]
    k = max(3, int(round(window / cell)) | 1)
    return ndi.median_filter(g, size=k, mode="nearest")


def build_navgrid(points: np.ndarray, cam_xyz: np.ndarray | None = None,
                  cell: float = 0.1, floor_tol: float = 0.15,
                  obstacle_lo: float = 0.15, obstacle_hi: float = 1.8,
                  obstacle_min_points: int | None = None,
                  obstacle_ratio: float = 0.5, inflate: float = 0.2,
                  traj_radius: float = 0.3, seal_rim: bool = True,
                  margin: float = 0.3, min_blob_cells: int = 4,
                  local_ground: bool = True, ground_window: float = 1.5,
                  ground_max_dev: float = 0.6, keep_connected: bool = False,
                  min_free_area: float = 0.5,
                  obstacle_contrast: float = 1.5, contrast_window: float = 1.0,
                  debug: dict | None = None,
                  scene_id: str = "") -> NavGrid:
    """points: (N,3) world-frame points (Y up, floor y=0)."""
    pts = np.asarray(points, np.float64)
    pts = pts[(pts[:, 1] > -0.5) & (pts[:, 1] < obstacle_hi + 1.5)]
    xz = pts[:, [0, 2]]
    lo = np.percentile(xz, 0.5, 0) if len(xz) > 400 else xz.min(0)
    hi = np.percentile(xz, 99.5, 0) if len(xz) > 400 else xz.max(0)
    if cam_xyz is not None and len(cam_xyz):
        lo = np.minimum(lo, cam_xyz[:, [0, 2]].min(0))
        hi = np.maximum(hi, cam_xyz[:, [0, 2]].max(0))
    lo = np.floor((lo - margin) / cell) * cell
    hi = np.ceil((hi + margin) / cell) * cell
    W = int(round((hi[0] - lo[0]) / cell))
    H = int(round((hi[1] - lo[1]) / cell))
    if W * H > 25_000_000:
        raise ValueError(f"grid too large ({W}x{H}); increase --cell")
    i = np.floor((xz[:, 0] - lo[0]) / cell).astype(np.int64)
    j = np.floor((xz[:, 1] - lo[1]) / cell).astype(np.int64)
    ok = (i >= 0) & (i < W) & (j >= 0) & (j < H)
    i, j, y = i[ok], j[ok], pts[ok, 1]

    ground = None
    if local_ground:
        # Monocular / drifting scans have a floor that is not exactly y=0
        # (it bends with distance). Estimate a smooth ground height field:
        # low percentile of y per cell, median-filtered over ~ground_window m.
        ground = _ground_field(i, j, y, W, H, cell, ground_window, ground_max_dev)
        y = y - ground[j, i]

    floor_m = np.abs(y) < floor_tol
    obst_m = (y >= obstacle_lo) & (y <= obstacle_hi)
    n_floor = _bincount2(i[floor_m], j[floor_m], W, H)
    n_obst = _bincount2(i[obst_m], j[obst_m], W, H)
    n_all = _bincount2(i, j, W, H)

    # 3. footprint of the observed space
    seen = n_all > 0
    seen = ndi.binary_closing(seen, structure=_disk(2), iterations=1)
    seen = ndi.binary_fill_holes(seen)
    lab, nlab = ndi.label(seen)
    if nlab > 1:
        sizes = ndi.sum(seen, lab, index=np.arange(1, nlab + 1))
        keep = {int(np.argmax(sizes)) + 1}
        if cam_xyz is not None:
            for c in cam_xyz:
                ci = int((c[0] - lo[0]) / cell); cj = int((c[2] - lo[1]) / cell)
                if 0 <= ci < W and 0 <= cj < H and lab[cj, ci] > 0:
                    keep.add(int(lab[cj, ci]))
        seen = np.isin(lab, list(keep))

    # 4. obstacles: (a) enough obstacle-band evidence, (b) the column is
    #    dominated by it, (c) it stands out against the local obstacle
    #    density (walls/furniture are concentrated; monocular depth noise is
    #    a diffuse haze that is densest near the camera)
    if obstacle_min_points is None:
        nz = n_obst[seen & (n_obst > 0)]
        obstacle_min_points = int(max(2, np.ceil(np.percentile(nz, 50)))) if nz.size else 2
    # light smoothing so single noisy cells do not become obstacles
    s_obst = ndi.gaussian_filter(n_obst.astype(np.float64), sigma=0.7, mode="constant")
    s_floor = ndi.gaussian_filter(n_floor.astype(np.float64), sigma=0.7, mode="constant")
    blocked = (s_obst >= obstacle_min_points) & \
              (s_obst >= obstacle_ratio * (s_floor + s_obst)) & seen
    if obstacle_contrast > 0:
        local = ndi.gaussian_filter(n_obst.astype(np.float64), sigma=contrast_window / cell,
                                    mode="constant")
        blocked &= s_obst >= obstacle_contrast * local
    if min_blob_cells > 1:
        lab, nlab = ndi.label(blocked, structure=np.ones((3, 3)))
        if nlab:
            sizes = ndi.sum(blocked, lab, index=np.arange(1, nlab + 1))
            small = np.flatnonzero(sizes < min_blob_cells) + 1
            blocked &= ~np.isin(lab, small)
    if seal_rim:
        rim = seen & ~ndi.binary_erosion(seen, structure=np.ones((3, 3)))
        blocked |= rim

    if debug is not None:
        debug.update(n_floor=n_floor, n_obst=n_obst, n_all=n_all, ground=ground,
                     blocked0=blocked.copy(), seen=seen)
    free = seen & ~blocked & (n_floor > 0)
    free = ndi.binary_closing(free, structure=_disk(1.5)) & seen & ~blocked

    # 5. trajectory = walked = free
    traj = np.zeros((H, W), bool)
    if cam_xyz is not None and len(cam_xyz):
        ci = np.floor((cam_xyz[:, 0] - lo[0]) / cell).astype(int)
        cj = np.floor((cam_xyz[:, 2] - lo[1]) / cell).astype(int)
        okc = (ci >= 0) & (ci < W) & (cj >= 0) & (cj < H)
        traj[cj[okc], ci[okc]] = True
        # connect consecutive samples so the walked path is continuous
        pts_c = np.stack([ci, cj], 1)
        for a, b in zip(pts_c[:-1], pts_c[1:]):
            cv2.line(traj.view(np.uint8), tuple(int(v) for v in a), tuple(int(v) for v in b), 1, 1)
        traj = ndi.binary_dilation(traj, structure=_disk(traj_radius / cell))
    free |= traj
    blocked &= ~traj

    # 6. inflate obstacles by the walker radius
    raw_blocked = blocked.copy()
    if inflate > 0:
        infl = ndi.binary_dilation(blocked, structure=_disk(inflate / cell))
        blocked = (blocked | (infl & free)) & ~traj   # unknown stays unknown
        free &= ~blocked

    # free pockets: keep_connected -> only the component(s) touching the
    # trajectory (or the biggest); otherwise only drop specks < min_free_area
    lab, nlab = ndi.label(free)
    if nlab > 1:
        sizes = ndi.sum(free, lab, index=np.arange(1, nlab + 1))
        on_traj = set(np.unique(lab[traj & free]).tolist()) - {0}
        if keep_connected:
            keep_ids = on_traj or {int(np.argmax(sizes)) + 1}
        else:
            keep_ids = on_traj | set((np.flatnonzero(sizes * cell * cell >= min_free_area) + 1).tolist())
        free &= np.isin(lab, list(keep_ids))   # dropped pockets -> unknown
    lab, nlab = ndi.label(free)

    cells = np.full((H, W), UNKNOWN, np.uint8)
    cells[free] = FREE
    cells[blocked] = BLOCKED
    extra = {
        "inflateRadius": float(inflate),
        "obstacleBand": [float(obstacle_lo), float(obstacle_hi)],
        "obstacleMinPoints": int(obstacle_min_points),
        "rawBlockedCells": int(raw_blocked.sum()),
        "freeComponents": int(nlab),
    }
    if cam_xyz is not None and len(cam_xyz):
        step = max(1, len(cam_xyz) // 60)
        extra["trajectory"] = [[round(float(c[0]), 3), round(float(c[2]), 3)]
                               for c in cam_xyz[::step]]
    return NavGrid(cells=cells, cell_size=float(cell),
                   origin=(float(lo[0]), float(lo[1])), floor_y=0.0,
                   scene_id=scene_id, extra=extra)


def footprint_polygon(grid: NavGrid, simplify: float = 0.15) -> list[list[float]]:
    """Outline (world x,z) of the known (free|blocked) region, simplified
    with Douglas-Peucker tolerance ``simplify`` metres. Counter-clockwise in
    the x-z plane as drawn top-down (x right, z down)."""
    known = (grid.cells != UNKNOWN).astype(np.uint8)
    known = cv2.morphologyEx(known, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(known, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, simplify / grid.cell_size, True)[:, 0, :].astype(np.float64)
    # contour points are pixel centres of boundary cells; use cell corners-ish
    out = [[round(grid.origin[0] + (p[0] + 0.5) * grid.cell_size, 3),
            round(grid.origin[1] + (p[1] + 0.5) * grid.cell_size, 3)] for p in c]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True, help="web/scenes/<id> folder (has scene.ply)")
    ap.add_argument("--splat", default=None, help="override path to the splat PLY")
    ap.add_argument("--poses", default=None,
                    help="camera->world poses .npy in the scene frame "
                         "(default: data/sample/clip1/poses.npy for cliproom)")
    ap.add_argument("--cell", type=float, default=0.1)
    ap.add_argument("--min-opacity", type=float, default=0.25)
    ap.add_argument("--max-scale", type=float, default=None,
                    help="drop splats larger than this (scene units); default p98")
    ap.add_argument("--floor-tol", type=float, default=0.15)
    ap.add_argument("--obstacle-lo", type=float, default=0.15)
    ap.add_argument("--obstacle-hi", type=float, default=1.8)
    ap.add_argument("--obstacle-min-points", type=int, default=None)
    ap.add_argument("--obstacle-ratio", type=float, default=0.5)
    ap.add_argument("--obstacle-contrast", type=float, default=1.5,
                    help="obstacle density must exceed this x its 1 m neighbourhood mean (0=off)")
    ap.add_argument("--inflate", type=float, default=0.2, help="walker radius (m)")
    ap.add_argument("--keep-connected", action="store_true",
                    help="keep only free space connected to the camera trajectory")
    ap.add_argument("--flat-ground", action="store_true",
                    help="trust y=0 as the floor everywhere (no local ground field); "
                         "use for metric/ARCore scans")
    ap.add_argument("--camera-height", type=float, default=1.4)
    ap.add_argument("--room-height", type=float, default=3.0)
    ap.add_argument("--scale-cue", default="auto", choices=["auto", "camera-height", "room-height"])
    ap.add_argument("--force-estimate", action="store_true",
                    help="ignore scene.json toWorld and estimate alignment")
    ap.add_argument("--no-floorplan", action="store_true")
    args = ap.parse_args(argv)

    from .splat import load_splat_centres

    scene_dir = args.scene
    sid = os.path.basename(os.path.normpath(scene_dir))
    splat = args.splat or os.path.join(scene_dir, "scene.ply")
    poses_path = args.poses
    if poses_path is None:
        sj = os.path.join(scene_dir, "scene.json")
        if os.path.exists(sj):
            with open(sj, encoding="utf-8") as f:
                poses_path = json.load(f).get("poses")
        if poses_path is None and sid == "cliproom":
            poses_path = "data/sample/clip1/poses.npy"
    pts, st = load_splat_centres(splat, args.min_opacity, args.max_scale)
    print(f"[navgrid] {sid}: {st['kept']:,}/{st['total']:,} splats after opacity/scale filter")

    al = align.get_alignment(scene_dir, pts, poses_path, camera_height=args.camera_height,
                             room_height=args.room_height, scale_cue=args.scale_cue,
                             force_estimate=args.force_estimate)
    T = align.mat_from_rowmajor(al["toWorld"])
    print(f"[navgrid] alignment from {al['source']}"
          + (f" (cue={al.get('scaleCue')}, scale={al.get('scale'):.3f})" if al["source"] != "scene.json" else ""))
    print("[navgrid] toWorld =\n" + np.array2string(T, precision=4, suppress_small=True))
    wpts = align.apply(T, pts)
    cams = None
    if poses_path and os.path.exists(poses_path):
        cams = align.apply(T, np.load(poses_path)[:, :3, 3])

    grid = build_navgrid(wpts, cams, cell=args.cell, floor_tol=args.floor_tol,
                         obstacle_lo=args.obstacle_lo, obstacle_hi=args.obstacle_hi,
                         obstacle_min_points=args.obstacle_min_points,
                         obstacle_ratio=args.obstacle_ratio, inflate=args.inflate,
                         obstacle_contrast=args.obstacle_contrast,
                         keep_connected=args.keep_connected,
                         local_ground=not args.flat_ground, scene_id=sid)
    grid.extra["alignment"] = {"source": al["source"], "toWorld": [round(v, 6) for v in al["toWorld"]]}
    out = os.path.join(scene_dir, "navgrid.json")
    grid.save(out)
    c = grid.cells
    print(f"[navgrid] {grid.width}x{grid.height} @ {grid.cell_size} m, origin {grid.origin}: "
          f"free={int((c == FREE).sum())} blocked={int((c == BLOCKED).sum())} "
          f"unknown={int((c == UNKNOWN).sum())}  -> {out}")
    if not args.no_floorplan:
        poly = footprint_polygon(grid)
        render_png(grid, os.path.join(scene_dir, "floorplan.png"), outline=poly)
        render_svg(grid, os.path.join(scene_dir, "floorplan.svg"), outline=poly)
        print(f"[navgrid] floorplan.png/.svg written; outline has {len(poly)} vertices")


if __name__ == "__main__":
    main()
