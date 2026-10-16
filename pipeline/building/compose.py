"""Compose web/scenes/building.json from a small YAML spec, with validation.

    python -m pipeline.building.compose --spec pipeline/building/demo_building.yaml \
        --out web/scenes/building.json [--plan pipeline/building/demo_building]

Spec (YAML):

    name: Demo building
    floors: [{id: G, name: Ground, elevation: 0.0}]
    spaces:
      - id: room-a
        kind: room                     # room | hallway | other
        name: Room A
        floor: G
        sceneId: cliproom              # scanned space
        polygon:                       # explicit [[x,z],...] or derived from the scan:
          fromNavgrid: web/scenes/cliproom/navgrid.json
          mode: bbox                   # bbox | outline
          pad: 0.0
        placement: {yawDeg: 0, translate: [0, 0]}     # scan frame -> building
        # or: autoPlace: {door: d-a, inward: [0, -1], inset: 0.3}
      - id: hall
        kind: hallway
        name: Main hallway
        floor: G
        polygon: [[-11, 0.3], [9.3, 0.3], [9.3, 2.7], [-11, 2.7]]   # planned
    doors:
      - {id: d-a, between: [hall, room-a], position: [0.0, 0.3]}  # [x,z] | [x,y,z] | auto

Validation: unique ids, references resolve, polygons have >= 3 vertices and
positive area, no two spaces on a floor overlap (> 0.05 m^2), every door lies
on the boundary of both its spaces (<= door_tol), each scanned scene's known
navgrid cells fall inside its polygon (warn below 95 %), and the door graph is
connected (warn). Errors exit 1 unless --force.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import yaml

from . import geom
from .grid import BLOCKED, FREE, UNKNOWN, NavGrid


class SpecError(Exception):
    pass


def _poly_from_navgrid(spec: dict, base: str):
    from .navgrid import footprint_polygon
    path = spec["fromNavgrid"]
    p = path if os.path.isabs(path) else os.path.join(base, path)
    g = NavGrid.load(p)
    mode = spec.get("mode", "bbox")
    pad = float(spec.get("pad", 0.0))
    if mode == "bbox":
        k = np.argwhere(g.cells != UNKNOWN)
        if not len(k):
            raise SpecError(f"{path}: navgrid has no known cells")
        (j0, i0), (j1, i1) = k.min(0), k.max(0)
        x0 = g.origin[0] + i0 * g.cell_size - pad
        x1 = g.origin[0] + (i1 + 1) * g.cell_size + pad
        z0 = g.origin[1] + j0 * g.cell_size - pad
        z1 = g.origin[1] + (j1 + 1) * g.cell_size + pad
        poly = [[x0, z0], [x1, z0], [x1, z1], [x0, z1]]
    elif mode == "outline":
        poly = footprint_polygon(g, simplify=float(spec.get("simplify", 0.15)))
    else:
        raise SpecError(f"unknown polygon mode {mode!r}")
    return [[round(float(x), 3), round(float(z), 3)] for x, z in poly], g


def _grid_known_xz(g: NavGrid) -> np.ndarray:
    k = np.argwhere(g.cells != UNKNOWN)
    return np.stack([g.origin[0] + (k[:, 1] + 0.5) * g.cell_size,
                     g.origin[1] + (k[:, 0] + 0.5) * g.cell_size], 1)


def _shared_edge_midpoint(pa, pb, tol=0.05):
    """Midpoint of the longest collinear overlap between edges of pa and pb."""
    best = None
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    for k in range(len(pa)):
        a0, a1 = pa[k], pa[(k + 1) % len(pa)]
        da = a1 - a0
        La = np.linalg.norm(da)
        if La < 1e-9:
            continue
        u = da / La
        nrm = np.array([-u[1], u[0]])
        for m in range(len(pb)):
            b0, b1 = pb[m], pb[(m + 1) % len(pb)]
            if abs(np.dot(b0 - a0, nrm)) > tol or abs(np.dot(b1 - a0, nrm)) > tol:
                continue
            t = sorted([np.dot(b0 - a0, u), np.dot(b1 - a0, u)])
            lo, hi = max(0.0, t[0]), min(La, t[1])
            if hi - lo > 1e-6 and (best is None or hi - lo > best[0]):
                best = (hi - lo, a0 + u * (lo + hi) / 2)
    return None if best is None else best[1].tolist()


def compose(spec: dict, base: str = ".", door_tol: float = 0.15,
            fit_warn: float = 0.95, overlap_tol: float = 0.05):
    """Returns (building_dict, errors, warnings, grids_by_space)."""
    errors, warnings = [], []
    floors = spec.get("floors") or [{"id": "G", "name": "Ground", "elevation": 0.0}]
    floor_elev = {f["id"]: float(f.get("elevation", 0.0)) for f in floors}
    spaces_in = spec.get("spaces", [])
    doors_in = spec.get("doors", [])

    ids = [s.get("id") for s in spaces_in] + [d.get("id") for d in doors_in]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        errors.append(f"duplicate ids: {sorted(dup)}")
    door_by_id = {d["id"]: d for d in doors_in}

    spaces, grids, scenes = [], {}, {}
    pending_auto = []
    for s in spaces_in:
        sid = s["id"]
        fl = s.get("floor", floors[0]["id"])
        if fl not in floor_elev:
            errors.append(f"{sid}: unknown floor {fl!r}")
        scene_id = s.get("sceneId")
        status = s.get("status") or ("scanned" if scene_id else "planned")
        poly = s.get("polygon")
        g = None
        if isinstance(poly, dict):
            poly, g = _poly_from_navgrid(poly, base)
        elif scene_id:
            ng = os.path.join(base, "web", "scenes", scene_id, "navgrid.json")
            if os.path.exists(ng):
                g = NavGrid.load(ng)
        if not poly or len(poly) < 3:
            errors.append(f"{sid}: polygon needs >= 3 vertices")
            poly = poly or []
        out = {"id": sid, "kind": s.get("kind", "room"), "name": s.get("name", sid),
               "floor": fl, "sceneId": scene_id,
               "polygon": [[round(float(x), 3), round(float(z), 3)] for x, z in poly],
               "status": status}
        if s.get("note"):
            out["note"] = s["note"]
        if scene_id:
            scenes[scene_id] = f"scenes/{scene_id}/scene.json"
            out["navgrid"] = f"scenes/{scene_id}/navgrid.json"
            if g is None:
                warnings.append(f"{sid}: no navgrid for scene {scene_id!r}; run pipeline.building.navgrid")
            else:
                grids[sid] = g
            if not os.path.exists(os.path.join(base, "web", "scenes", scene_id, "scene.json")):
                warnings.append(f"{sid}: web/scenes/{scene_id}/scene.json not found yet "
                                "(owned by the reconstruction stage)")
        yaw, tr = 0.0, (0.0, 0.0)
        if "autoPlace" in s:
            pending_auto.append((out, s["autoPlace"]))
        else:
            pl = s.get("placement") or {}
            yaw = float(pl.get("yawDeg", 0.0))
            tr = tuple(float(v) for v in pl.get("translate", (0.0, 0.0)))
        out["_yaw"], out["_tr"] = yaw, tr
        spaces.append(out)

    sp_by_id = {s["id"]: s for s in spaces}

    # auto placement: line the scan's entrance (first trajectory point, camera
    # looking along local -Z) up with a door, facing ``inward``
    for out, ap in pending_auto:
        d = door_by_id.get(ap.get("door"))
        g = grids.get(out["id"])
        if d is None or g is None or not g.extra.get("trajectory"):
            errors.append(f"{out['id']}: autoPlace needs an existing door and a navgrid with a trajectory")
            continue
        p = d.get("position")
        if p in (None, "auto"):
            errors.append(f"{out['id']}: autoPlace door {d['id']} needs an explicit position")
            continue
        dxz = np.array([p[0], p[-1]], float)
        inward = np.asarray(ap.get("inward", [0, -1]), float)
        inward /= np.linalg.norm(inward)
        yaw = math.degrees(math.atan2(-inward[0], -inward[1]))
        ent = geom.transform_2d([g.extra["trajectory"][0]], yaw)[0]
        tr = dxz + float(ap.get("inset", 0.3)) * inward - ent
        out["_yaw"], out["_tr"] = yaw, (float(tr[0]), float(tr[1]))
        print(f"[compose] autoPlace {out['id']}: yaw={yaw:.1f} deg translate=({tr[0]:.3f}, {tr[1]:.3f})")

    for out in spaces:
        yaw, tr = out.pop("_yaw"), out.pop("_tr")
        ident = abs(yaw) < 1e-9 and abs(tr[0]) < 1e-9 and abs(tr[1]) < 1e-9
        if not ident and out["polygon"]:
            # polygons derived from the scan live in the scan frame -> move them
            src = next(s for s in spaces_in if s["id"] == out["id"])
            if isinstance(src.get("polygon"), dict):
                out["polygon"] = [[round(float(x), 3), round(float(z), 3)]
                                  for x, z in geom.transform_2d(out["polygon"], yaw, tr)]
        if out["sceneId"]:
            M = geom.placement_matrix(yaw, tr, 0.0)
            out["placement"] = [round(float(v), 6) for v in M.reshape(-1)]
            if not ident:
                warnings.append(f"{out['id']}: non-identity placement (yaw {yaw:.1f}, t {tr}); "
                                "scene.json toWorld must include it and the navgrid must be rebuilt, "
                                "or the viewer must apply `placement` after toWorld")
        out["_M"] = (yaw, tr)
        a = geom.polygon_area(out["polygon"]) if len(out["polygon"]) >= 3 else 0.0
        if len(out["polygon"]) >= 3 and a <= 1e-6:
            errors.append(f"{out['id']}: polygon has zero area")
        out["area"] = round(a, 2)
        if a > 0:
            out["center"] = [round(v, 3) for v in geom.polygon_centroid(out["polygon"])]

    # overlaps (same floor)
    for x in range(len(spaces)):
        for y in range(x + 1, len(spaces)):
            A, B = spaces[x], spaces[y]
            if A["floor"] != B["floor"] or len(A["polygon"]) < 3 or len(B["polygon"]) < 3:
                continue
            (a0, a1), (b0, b1) = geom.bbox(A["polygon"]), geom.bbox(B["polygon"])
            if (a1 < b0).any() or (b1 < a0).any():
                continue
            ov = geom.overlap_area(A["polygon"], B["polygon"])
            if ov > overlap_tol:
                errors.append(f"spaces {A['id']} and {B['id']} overlap by ~{ov:.2f} m^2")

    # doors
    doors = []
    for d in doors_in:
        bt = d.get("between", [])
        if len(bt) != 2 or any(b not in sp_by_id for b in bt):
            errors.append(f"door {d.get('id')}: between must name two existing spaces, got {bt}")
            continue
        A, B = sp_by_id[bt[0]], sp_by_id[bt[1]]
        p = d.get("position", "auto")
        if p == "auto" or p is None:
            mid = _shared_edge_midpoint(A["polygon"], B["polygon"])
            if mid is None:
                errors.append(f"door {d['id']}: {bt[0]} and {bt[1]} share no edge for an auto door")
                continue
            xz = mid
        else:
            xz = [float(p[0]), float(p[-1])]
        for S in (A, B):
            dd = geom.dist_to_boundary(xz, S["polygon"])
            if dd > door_tol:
                errors.append(f"door {d['id']} at ({xz[0]:.2f},{xz[1]:.2f}) is {dd:.2f} m off "
                              f"the boundary of {S['id']} (tol {door_tol})")
        y = floor_elev.get(A["floor"], 0.0)
        od = {"id": d["id"], "between": list(bt),
              "position": [round(xz[0], 3), round(y, 3), round(xz[1], 3)]}
        if "width" in d:
            od["width"] = float(d["width"])
        doors.append(od)

    # scanned footprint fits polygon
    for sid, g in grids.items():
        S = sp_by_id[sid]
        if len(S["polygon"]) < 3:
            continue
        yaw, tr = S["_M"]
        pts = geom.transform_2d(_grid_known_xz(g), yaw, tr)
        if not len(pts):
            continue
        mask, lo = geom.rasterize([S["polygon"]], g.cell_size / 2)
        res = g.cell_size / 2
        ii = np.floor((pts[:, 0] - lo[0]) / res).astype(int)
        jj = np.floor((pts[:, 1] - lo[1]) / res).astype(int)
        ok = (ii >= 0) & (jj >= 0) & (ii < mask[0].shape[1]) & (jj < mask[0].shape[0])
        inside = np.zeros(len(pts), bool)
        inside[ok] = mask[0][jj[ok], ii[ok]]
        frac = float(inside.mean())
        S["footprintFit"] = round(frac, 3)
        if frac < fit_warn:
            warnings.append(f"{sid}: only {frac:.0%} of scene {S['sceneId']!r} navgrid lies inside its polygon")

    # connectivity of the door graph
    if spaces:
        adj = {s["id"]: set() for s in spaces}
        for d in doors:
            a, b = d["between"]
            adj[a].add(b)
            adj[b].add(a)
        seen, stack = set(), [spaces[0]["id"]]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(adj[u] - seen)
        lonely = [s["id"] for s in spaces if s["id"] not in seen]
        if lonely:
            warnings.append(f"not reachable from {spaces[0]['id']} through doors: {lonely}")

    for s in spaces:
        s.pop("_M", None)
    building = {"name": spec.get("name", "Building"),
                "floors": [{"id": f["id"], "name": f.get("name", f["id"]),
                            "elevation": float(f.get("elevation", 0.0))} for f in floors],
                "spaces": spaces, "doors": doors, "scenes": scenes}
    for k in ("description", "notes"):
        if k in spec:
            building[k] = spec[k]
    building["generatedBy"] = "pipeline/building/compose.py"
    return building, errors, warnings, grids


# ---------------------------------------------------------------- rendering
def render_plan(building: dict, grids: dict, prefix: str, px_per_m: int = 40) -> None:
    allp = np.concatenate([np.asarray(s["polygon"], float) for s in building["spaces"]
                           if len(s["polygon"]) >= 3])
    lo, hi = allp.min(0) - 1.0, allp.max(0) + 1.0
    W, H = (np.ceil((hi - lo) * px_per_m)).astype(int)
    img = np.full((H, W, 3), 250, np.uint8)

    def P(xz):
        return (int(round((xz[0] - lo[0]) * px_per_m)), int(round((xz[1] - lo[1]) * px_per_m)))

    fills = {"hallway": (226, 232, 238), "room": (236, 236, 236)}
    for s in building["spaces"]:
        pts = np.array([P(p) for p in s["polygon"]], np.int32)
        col = (246, 226, 206) if s["status"] == "scanned" else fills.get(s["kind"], (236, 236, 236))
        cv2.fillPoly(img, [pts], col)
        g = grids.get(s["id"])
        if g is not None and s.get("placement"):
            M = np.array(s["placement"], float).reshape(4, 4)
            yaw = math.degrees(math.atan2(M[0, 2], M[0, 0]))
            k = np.argwhere(g.cells != UNKNOWN)
            xz = np.stack([g.origin[0] + (k[:, 1] + 0.5) * g.cell_size,
                           g.origin[1] + (k[:, 0] + 0.5) * g.cell_size], 1)
            xz = geom.transform_2d(xz, yaw, (M[0, 3], M[2, 3]))
            r = max(1, int(round(g.cell_size * px_per_m / 2)))
            for (x, z), v in zip(xz, g.cells[k[:, 0], k[:, 1]]):
                c = (250, 246, 242) if v == FREE else (90, 70, 60)
                cx, cz = P((x, z))
                cv2.rectangle(img, (cx - r, cz - r), (cx + r - 1, cz + r - 1), c, -1)
        if s["status"] == "planned":
            # dashed outline for planned spaces
            for k in range(len(pts)):
                a, b = pts[k].astype(float), pts[(k + 1) % len(pts)].astype(float)
                L = np.linalg.norm(b - a)
                n = max(1, int(L // 12))
                for t in range(0, n, 2):
                    p0 = a + (b - a) * t / n
                    p1 = a + (b - a) * min(t + 1, n) / n
                    cv2.line(img, tuple(p0.astype(int)), tuple(p1.astype(int)), (110, 110, 110), 2, cv2.LINE_AA)
        else:
            cv2.polylines(img, [pts], True, (40, 40, 40), 3, cv2.LINE_AA)
    for d in building["doors"]:
        cx, cz = P((d["position"][0], d["position"][2]))
        cv2.circle(img, (cx, cz), 9, (40, 140, 240), -1, cv2.LINE_AA)
        cv2.putText(img, d["id"], (cx + 10, cz - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 100, 200), 1, cv2.LINE_AA)
    for s in building["spaces"]:
        if "center" not in s:
            continue
        cx, cz = P(s["center"])
        label = s["name"]
        tag = "SCANNED" if s["status"] == "scanned" else "planned - not scanned"
        for k, (txt, sc, th) in enumerate(((label, 0.6, 2), (tag, 0.45, 1))):
            (tw, th_), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
            y0 = cz + k * 22
            cv2.rectangle(img, (cx - tw // 2 - 4, y0 - th_ - 4), (cx + tw // 2 + 4, y0 + 6),
                          (252, 252, 252), -1)
            cv2.putText(img, txt, (cx - tw // 2, cz + k * 22), cv2.FONT_HERSHEY_SIMPLEX, sc,
                        (30, 30, 30) if k == 0 else (90, 90, 90), th, cv2.LINE_AA)
    cv2.imwrite(prefix + ".png", img)

    # SVG (metres; x right, z down)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{lo[0]:.2f} {lo[1]:.2f} '
             f'{hi[0] - lo[0]:.2f} {hi[1] - lo[1]:.2f}" width="{W}" height="{H}">',
             f'<title>{building["name"]} - plan (x right, z down, metres)</title>',
             f'<rect x="{lo[0]:.2f}" y="{lo[1]:.2f}" width="{hi[0] - lo[0]:.2f}" '
             f'height="{hi[1] - lo[1]:.2f}" fill="#fafafa"/>']
    for s in building["spaces"]:
        pts = " ".join(f"{x:.2f},{z:.2f}" for x, z in s["polygon"])
        scanned = s["status"] == "scanned"
        fill = "#cee2f6" if scanned else ("#eee8e2" if s["kind"] == "hallway" else "#ececec")
        dash = "" if scanned else ' stroke-dasharray="0.3 0.2"'
        parts.append(f'<polygon id="{s["id"]}" points="{pts}" fill="{fill}" stroke="#282828" '
                     f'stroke-width="0.08"{dash}><title>{s["name"]} ({s["status"]})</title></polygon>')
        if "center" in s:
            parts.append(f'<text x="{s["center"][0]:.2f}" y="{s["center"][1]:.2f}" font-size="0.5" '
                         f'text-anchor="middle" font-family="sans-serif">{s["name"]}</text>')
            if not scanned:
                parts.append(f'<text x="{s["center"][0]:.2f}" y="{s["center"][1] + 0.6:.2f}" '
                             f'font-size="0.32" text-anchor="middle" fill="#666" '
                             f'font-family="sans-serif">planned - not scanned</text>')
    for d in building["doors"]:
        parts.append(f'<circle cx="{d["position"][0]:.2f}" cy="{d["position"][2]:.2f}" r="0.22" '
                     f'fill="#f08c28"><title>{d["id"]}: {d["between"][0]} - {d["between"][1]}</title></circle>')
    parts.append("</svg>")
    with open(prefix + ".svg", "w", encoding="utf-8") as f:
        f.write("".join(parts))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compose and validate building.json")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=".", help="repo root that spec paths are relative to")
    ap.add_argument("--plan", default=None, help="also render <prefix>.png/.svg plan")
    ap.add_argument("--door-tol", type=float, default=0.15)
    ap.add_argument("--force", action="store_true", help="write even if validation fails")
    args = ap.parse_args(argv)
    with open(args.spec, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    try:
        building, errors, warnings, grids = compose(spec, args.base, door_tol=args.door_tol)
    except SpecError as e:
        print(f"[compose] ERROR {e}")
        sys.exit(1)
    for w in warnings:
        print(f"[compose] warning: {w}")
    for e in errors:
        print(f"[compose] ERROR: {e}")
    if errors and not args.force:
        print("[compose] not written (fix the errors or pass --force)")
        sys.exit(1)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(building, f, indent=2)
    n_sc = sum(s["status"] == "scanned" for s in building["spaces"])
    print(f"[compose] {len(building['spaces'])} spaces ({n_sc} scanned), "
          f"{len(building['doors'])} doors -> {args.out}")
    if args.plan:
        render_plan(building, grids, args.plan)
        print(f"[compose] plan -> {args.plan}.png / .svg")


if __name__ == "__main__":
    main()
