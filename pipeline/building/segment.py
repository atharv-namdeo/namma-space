"""Segment a navgrid's free space into rooms / hallway + doors.

    python -m pipeline.building.segment --navgrid web/scenes/cliproom/navgrid.json
    python -m pipeline.building.segment --synthetic --out pipeline/building/cache/synthetic

Algorithm (all deterministic):
 1. free mask F; D = Euclidean distance (m) from each free cell to the
    nearest non-free cell.
 2. markers = connected components (8-conn) of {D >= door_width/2 + cell}:
    doorways are narrower than door_width, so the cores of rooms and
    corridors separate there.
 3. priority-flood watershed on -D restricted to F grows every marker
    over the remaining free cells (cells with higher D are claimed first).
 4. adjacency: two regions touching (4-neighbour) share a boundary; its
    length is the opening width. Openings wider than max_door_width are not
    doors - those regions are merged (repeat until stable). Regions smaller
    than min_area merge into the neighbour they share most boundary with.
 5. each remaining shared boundary is a door: position = boundary midpoint
    (cell closest to the mean), width = boundary length.
 6. kind: hallway if the region's min-area rectangle is long and thin
    (aspect >= hallway_aspect and short side <= hallway_max_width), else room.
 7. polygon = outer contour of the region, Douglas-Peucker simplified, in
    world (x, z).
"""
from __future__ import annotations

import argparse
import heapq
import json
import os

import cv2
import numpy as np
from scipy import ndimage as ndi

from .geom import polygon_area, polygon_centroid
from .grid import BLOCKED, FREE, UNKNOWN, NavGrid


def watershed_free(free: np.ndarray, dist: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """Priority-flood: labels from ``markers`` (0 = unlabelled) spread over
    ``free`` in order of decreasing ``dist`` (4-connected)."""
    H, W = free.shape
    lab = markers.copy()
    heap = []
    cnt = 0
    ys, xs = np.nonzero(lab)
    for y, x in zip(ys, xs):
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and free[ny, nx] and lab[ny, nx] == 0:
                heapq.heappush(heap, (-dist[ny, nx], cnt, ny, nx, lab[y, x]))
                cnt += 1
    while heap:
        _, _, y, x, l = heapq.heappop(heap)
        if lab[y, x]:
            continue
        lab[y, x] = l
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and free[ny, nx] and lab[ny, nx] == 0:
                heapq.heappush(heap, (-dist[ny, nx], cnt, ny, nx, l))
                cnt += 1
    return lab


def _boundaries(lab: np.ndarray) -> dict:
    """{(a,b): [(y,x,y2,x2), ...]} for 4-neighbour pairs with a<b, both >0."""
    out: dict = {}
    for dy, dx in ((0, 1), (1, 0)):
        a = lab[: lab.shape[0] - dy, : lab.shape[1] - dx]
        b = lab[dy:, dx:]
        m = (a > 0) & (b > 0) & (a != b)
        ys, xs = np.nonzero(m)
        for y, x in zip(ys, xs):
            la, lb = int(a[y, x]), int(b[y, x])
            key = (min(la, lb), max(la, lb))
            out.setdefault(key, []).append((y, x, y + dy, x + dx))
    return out


def _opening_width(pairs, cell: float) -> float:
    """Length of a shared boundary: number of distinct cells on the smaller
    side (a straight door of k cells has k pairs)."""
    s1 = {(p[0], p[1]) for p in pairs}
    s2 = {(p[2], p[3]) for p in pairs}
    return min(len(s1), len(s2)) * cell


def _relabel(lab, mapping):
    out = lab.copy()
    for src, dst in mapping.items():
        out[lab == src] = dst
    return out


def segment(grid: NavGrid, door_width: float = 1.0, max_door_width: float = 1.6,
            min_area: float = 1.5, hallway_aspect: float = 2.8,
            hallway_max_width: float = 3.0, simplify: float = 0.15) -> dict:
    cs = grid.cell_size
    free = grid.cells == FREE
    dist = ndi.distance_transform_edt(free) * cs
    core = dist >= door_width / 2.0 + cs
    markers, n = ndi.label(core, structure=np.ones((3, 3)))
    if n == 0:                                      # all narrow: one region per component
        markers, n = ndi.label(free)
    lab = watershed_free(free, dist, markers.astype(np.int32))
    # free cells not reachable from any marker -> own components
    rest = free & (lab == 0)
    if rest.any():
        extra, m = ndi.label(rest)
        lab[rest] = extra[rest] + lab.max()

    # merge loop: wide openings and tiny regions
    for _ in range(200):
        changed = False
        bnd = _boundaries(lab)
        for (a, b), pairs in sorted(bnd.items()):
            if _opening_width(pairs, cs) > max_door_width:
                lab[lab == b] = a
                changed = True
                break
        if changed:
            continue
        ids, areas = np.unique(lab[lab > 0], return_counts=True)
        for rid, ar in sorted(zip(ids.tolist(), areas.tolist()), key=lambda t: t[1]):
            if ar * cs * cs >= min_area:
                continue
            nb = [(len(p), k[0] if k[1] == rid else k[1]) for k, p in bnd.items() if rid in k]
            if nb:
                lab[lab == rid] = max(nb)[1]
                changed = True
                break
        if not changed:
            break

    # compact labels 1..K in reading order of their centroid (deterministic)
    ids = [int(v) for v in np.unique(lab[lab > 0])]
    cents = {i: ndi.center_of_mass(lab == i) for i in ids}
    order = sorted(ids, key=lambda i: (round(cents[i][0], 1), round(cents[i][1], 1)))
    lab = _relabel(lab, {old: -(k + 1) for k, old in enumerate(order)})
    lab = -lab
    lab[lab < 0] = 0

    spaces, rid_to_sid = [], {}
    for k in range(1, len(order) + 1):
        m = (lab == k).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        approx = cv2.approxPolyDP(c, max(simplify / cs, 0.5), True)[:, 0, :]
        # contour runs through boundary cell centres; push half a cell outward
        # by mapping each vertex to the corner of its cell away from the centroid
        cy, cx = ndi.center_of_mass(m)
        poly = []
        for x, y in approx:
            ox = 0.5 if x >= cx else -0.5
            oy = 0.5 if y >= cy else -0.5
            poly.append([round(grid.origin[0] + (x + 0.5 + ox) * cs, 3),
                         round(grid.origin[1] + (y + 0.5 + oy) * cs, 3)])
        rect = cv2.minAreaRect(np.argwhere(m)[:, ::-1].astype(np.float32))
        w_, h_ = sorted(rect[1])
        w_, h_ = (w_ + 1) * cs, (h_ + 1) * cs
        aspect = h_ / max(w_, 1e-6)
        kind = "hallway" if (aspect >= hallway_aspect and w_ <= hallway_max_width) else "room"
        area = float(m.sum() * cs * cs)
        sid = f"space-{k}"
        rid_to_sid[k] = sid
        spaces.append({"id": sid, "kind": kind, "polygon": poly,
                       "area": round(area, 2), "center": [round(v, 3) for v in polygon_centroid(poly)]
                       if polygon_area(poly) > 0 else None,
                       "size": [round(w_, 2), round(h_, 2)], "aspect": round(aspect, 2)})

    doors = []
    for n_d, ((a, b), pairs) in enumerate(sorted(_boundaries(lab).items())):
        pts = np.array([[(p[1] + p[3]) / 2.0, (p[0] + p[2]) / 2.0] for p in pairs])
        mean = pts.mean(0)
        mid = pts[np.argmin(((pts - mean) ** 2).sum(1))]
        x = grid.origin[0] + (mid[0] + 0.5) * cs
        z = grid.origin[1] + (mid[1] + 0.5) * cs
        doors.append({"id": f"door-{n_d + 1}", "between": [rid_to_sid[a], rid_to_sid[b]],
                      "position": [round(x, 3), round(grid.floor_y, 3), round(z, 3)],
                      "width": round(_opening_width(pairs, cs), 2)})
    return {"sceneId": grid.scene_id, "cellSize": cs, "spaces": spaces, "doors": doors,
            "labels": lab}


# ---------------------------------------------------------------- synthetic
def synthetic_floorplan(cell: float = 0.1, door: float = 0.9, wall: float = 0.2) -> NavGrid:
    """A 14 x 2 m hallway (z in [5, 7]) with three rooms north of it
    (4.5 x 5, 4 x 5, 4.5 x 5) each joined by one ``door`` wide opening.
    Outside the building is unknown, walls are blocked."""
    W, H = int(round(15.0 / cell)), int(round(8.0 / cell))
    origin = (-0.5, -0.5)
    c = np.full((H, W), UNKNOWN, np.uint8)

    def rect(x0, z0, x1, z1, v):
        i0, i1 = int(round((x0 - origin[0]) / cell)), int(round((x1 - origin[0]) / cell))
        j0, j1 = int(round((z0 - origin[1]) / cell)), int(round((z1 - origin[1]) / cell))
        c[j0:j1, i0:i1] = v

    rect(0, 0, 14, 7, BLOCKED)                       # building block of wall
    rect(wall, 5 + wall / 2, 14 - wall, 7 - wall, FREE)        # hallway
    xs = [0, 4.5, 8.5, 14]
    for k in range(3):
        x0, x1 = xs[k], xs[k + 1]
        rect(x0 + wall / 2 if k else wall, wall, x1 - (wall / 2 if k < 2 else wall),
             5 - wall / 2, FREE)                      # room k
        dc = (x0 + x1) / 2
        rect(dc - door / 2, 5 - wall, dc + door / 2, 5 + wall, FREE)   # doorway
    return NavGrid(cells=c, cell_size=cell, origin=origin, scene_id="synthetic")


def render_segments(grid: NavGrid, seg: dict, path: str, px: int = 8) -> None:
    lab = seg["labels"]
    H, W = lab.shape
    rng = np.random.default_rng(3)
    pal = rng.integers(90, 230, size=(lab.max() + 1, 3)).astype(np.uint8)
    img = np.full((H, W, 3), 205, np.uint8)
    img[grid.cells == BLOCKED] = (52, 58, 70)
    img[lab > 0] = pal[lab[lab > 0]]
    img = cv2.resize(img, (W * px, H * px), interpolation=cv2.INTER_NEAREST)
    for sp in seg["spaces"]:
        pts = np.array([[(x - grid.origin[0]) / grid.cell_size * px,
                         (z - grid.origin[1]) / grid.cell_size * px] for x, z in sp["polygon"]],
                       np.int32)
        cv2.polylines(img, [pts], True, (30, 30, 30), 1, cv2.LINE_AA)
        if sp["center"]:
            cx = int((sp["center"][0] - grid.origin[0]) / grid.cell_size * px)
            cz = int((sp["center"][1] - grid.origin[1]) / grid.cell_size * px)
            cv2.putText(img, f'{sp["id"]} {sp["kind"]}', (cx - 40, cz), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (20, 20, 20), 1, cv2.LINE_AA)
    for d in seg["doors"]:
        cx = int((d["position"][0] - grid.origin[0]) / grid.cell_size * px)
        cz = int((d["position"][2] - grid.origin[1]) / grid.cell_size * px)
        cv2.circle(img, (cx, cz), 6, (40, 140, 240), -1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Segment a navgrid into rooms/hallway + doors")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--navgrid", help="navgrid.json")
    src.add_argument("--synthetic", action="store_true",
                     help="use the built-in hallway + 3 rooms test floorplan")
    ap.add_argument("--out", default=None,
                    help="output prefix (default pipeline/building/cache/<sceneId>_segments)")
    ap.add_argument("--door-width", type=float, default=1.0)
    ap.add_argument("--max-door-width", type=float, default=1.6)
    ap.add_argument("--min-area", type=float, default=1.5)
    args = ap.parse_args(argv)
    grid = synthetic_floorplan() if args.synthetic else NavGrid.load(args.navgrid)
    seg = segment(grid, door_width=args.door_width, max_door_width=args.max_door_width,
                  min_area=args.min_area)
    out = args.out or os.path.join(os.path.dirname(__file__), "cache",
                                   f"{grid.scene_id or 'navgrid'}_segments")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    js = {k: v for k, v in seg.items() if k != "labels"}
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(js, f, indent=1)
    render_segments(grid, seg, out + ".png")
    for sp in seg["spaces"]:
        print(f"[segment] {sp['id']:9s} {sp['kind']:8s} area={sp['area']:6.2f} m2 "
              f"size={sp['size']} vertices={len(sp['polygon'])}")
    for d in seg["doors"]:
        print(f"[segment] {d['id']:9s} {d['between'][0]} <-> {d['between'][1]} "
              f"at x={d['position'][0]:.2f} z={d['position'][2]:.2f} width={d['width']:.2f} m")
    print(f"[segment] -> {out}.json / .png")


if __name__ == "__main__":
    main()
