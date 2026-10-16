"""Reference pathfinding: grid A* + cross-space routing over building doors.

The web app re-implements this in JS; README.md documents the exact
algorithm so both give the same answer.

    # inside one scan
    python -m pipeline.building.astar --navgrid web/scenes/cliproom/navgrid.json \
        --from 0,-0.5 --to -0.2,-1.6
    # across the building (room -> door -> hallway -> door -> room)
    python -m pipeline.building.astar --building web/scenes/building.json \
        --from 0,-1 --to 9,-3
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
from collections import deque

import numpy as np

from .geom import point_in_polygon
from .grid import FREE, UNKNOWN, NavGrid

SQRT2 = math.sqrt(2.0)
# neighbour order is part of the spec (affects tie-breaking): E, W, S, N, SE, SW, NE, NW
NEIGHBOURS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
              (1, 1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (-1, -1, SQRT2)]


def octile(i0, j0, i1, j1) -> float:
    dx, dy = abs(i0 - i1), abs(j0 - j1)
    return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)


def passable(grid: NavGrid, i: int, j: int, allow_unknown: bool = False) -> bool:
    if not grid.in_bounds(i, j):
        return False
    v = grid.cells[j, i]
    return v == FREE or (allow_unknown and v == UNKNOWN)


def astar(grid: NavGrid, start: tuple[int, int], goal: tuple[int, int],
          allow_unknown: bool = False):
    """8-connected A*, no corner cutting. start/goal = (i, j) cells.
    Returns (path [(i,j), ...], cost_in_cells) or (None, inf)."""
    if not (passable(grid, *start, allow_unknown) and passable(grid, *goal, allow_unknown)):
        return None, math.inf
    W = grid.width
    g = {start: 0.0}
    came: dict = {}
    closed = set()
    counter = 0
    heap = [(octile(*start, *goal), octile(*start, *goal), counter, start)]
    while heap:
        _, _, _, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1], g[goal]
        closed.add(cur)
        ci, cj = cur
        for di, dj, step in NEIGHBOURS:
            ni, nj = ci + di, cj + dj
            if not passable(grid, ni, nj, allow_unknown):
                continue
            if di and dj and not (passable(grid, ci + di, cj, allow_unknown)
                                  and passable(grid, ci, cj + dj, allow_unknown)):
                continue                                  # no corner cutting
            nxt = (ni, nj)
            if nxt in closed:
                continue
            ng = g[cur] + step
            if ng < g.get(nxt, math.inf) - 1e-12:
                g[nxt] = ng
                came[nxt] = cur
                h = octile(ni, nj, *goal)
                counter += 1
                heapq.heappush(heap, (ng + h, h, counter, nxt))
    return None, math.inf


def nearest_passable(grid: NavGrid, i: int, j: int, max_radius_cells: int,
                     allow_unknown: bool = False):
    """BFS (4-conn, through any cell, clamped into the grid) to the nearest
    passable cell; ties resolved by BFS order E, W, S, N."""
    i = min(max(i, 0), grid.width - 1)
    j = min(max(j, 0), grid.height - 1)
    if passable(grid, i, j, allow_unknown):
        return (i, j)
    seen = {(i, j)}
    q = deque([(i, j, 0)])
    while q:
        ci, cj, d = q.popleft()
        if d >= max_radius_cells:
            continue
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (ci + di, cj + dj)
            if n in seen or not grid.in_bounds(*n):
                continue
            if passable(grid, *n, allow_unknown):
                return n
            seen.add(n)
            q.append((n[0], n[1], d + 1))
    return None


def path_world(grid: NavGrid, start_xz, goal_xz, snap: float = 0.6,
               allow_unknown: bool = False):
    """World-space convenience wrapper. Returns (polyline [[x,z],...], metres)
    or (None, inf). The exact start/goal points are kept as the polyline ends."""
    r = max(1, int(round(snap / grid.cell_size)))
    s = nearest_passable(grid, *grid.world_to_cell(*start_xz), r, allow_unknown)
    t = nearest_passable(grid, *grid.world_to_cell(*goal_xz), r, allow_unknown)
    if s is None or t is None:
        return None, math.inf
    cells, cost = astar(grid, s, t, allow_unknown)
    if cells is None:
        return None, math.inf
    pts = [list(grid.cell_to_world(i, j)) for i, j in cells]
    pts = [list(map(float, start_xz))] + pts + [list(map(float, goal_xz))]
    length = sum(math.dist(a, b) for a, b in zip(pts[:-1], pts[1:]))
    return pts, length


# ------------------------------------------------------------ building graph
class Building:
    """Graph view of building.json: nodes = doors, plus temporary start/goal.
    Edge weights: door<->door inside one space = walking distance in that
    space (A* on the space's navgrid if it has one, else straight line)."""

    def __init__(self, data: dict, root: str = ".", allow_unknown: bool = False):
        self.data = data
        self.root = root
        self.allow_unknown = allow_unknown
        self.spaces = {s["id"]: s for s in data["spaces"]}
        self.doors = {d["id"]: d for d in data["doors"]}
        self.grids: dict = {}
        for s in data["spaces"]:
            ng = s.get("navgrid")
            if s.get("sceneId") and ng:
                p = ng if os.path.isabs(ng) else os.path.join(root, ng)
                if os.path.exists(p):
                    g = NavGrid.load(p)
                    pl = s.get("placement")
                    if pl and not np.allclose(np.array(pl, float).reshape(4, 4), np.eye(4)):
                        raise NotImplementedError(
                            f"{s['id']}: non-identity placement - fold it into scene.json "
                            "toWorld and rebuild the navgrid")
                    self.grids[s["id"]] = g
        self.space_doors: dict = {sid: [] for sid in self.spaces}
        for d in data["doors"]:
            for sid in d["between"]:
                self.space_doors[sid].append(d["id"])

    def space_at(self, xz) -> str | None:
        hits = [sid for sid, s in self.spaces.items() if point_in_polygon(xz, s["polygon"])]
        return hits[0] if hits else None

    @staticmethod
    def door_xz(d):
        p = d["position"]
        return [float(p[0]), float(p[2])] if len(p) == 3 else [float(p[0]), float(p[1])]

    def leg(self, sid: str, a_xz, b_xz):
        """Walk inside one space -> (polyline, metres)."""
        g = self.grids.get(sid)
        if g is None:
            return [list(a_xz), list(b_xz)], math.dist(a_xz, b_xz)
        return path_world(g, a_xz, b_xz, snap=1.0, allow_unknown=self.allow_unknown)

    def route(self, start_xz, goal_xz):
        """Dijkstra over {start, doors, goal}. Returns dict with 'spaces'
        (alternating space/door ids), 'polyline' and 'length', or None."""
        s_sp, t_sp = self.space_at(start_xz), self.space_at(goal_xz)
        if s_sp is None or t_sp is None:
            raise ValueError(f"point outside every space: start in {s_sp}, goal in {t_sp}")
        if s_sp == t_sp:
            pl, L = self.leg(s_sp, start_xz, goal_xz)
            return None if pl is None else {"spaces": [s_sp], "polyline": pl, "length": L}
        # nodes: 'S', 'T', door ids; edges carry (space, polyline)
        pos = {"S": list(start_xz), "T": list(goal_xz)}
        pos.update({did: self.door_xz(d) for did, d in self.doors.items()})
        members = {sid: list(ds) for sid, ds in self.space_doors.items()}
        members[s_sp] = members[s_sp] + ["S"]
        members[t_sp] = members[t_sp] + ["T"]
        adj: dict = {}
        for sid, nodes in members.items():
            for x in range(len(nodes)):
                for y in range(x + 1, len(nodes)):
                    a, b = nodes[x], nodes[y]
                    pl, L = self.leg(sid, pos[a], pos[b])
                    if pl is None:
                        continue
                    adj.setdefault(a, []).append((L, b, sid, pl))
                    adj.setdefault(b, []).append((L, a, sid, pl[::-1]))
        dist = {"S": 0.0}
        prev: dict = {}
        heap = [(0.0, "S")]
        done = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            if u == "T":
                break
            for L, v, sid, pl in sorted(adj.get(u, []), key=lambda e: (e[0], e[1])):
                nd = d + L
                if nd < dist.get(v, math.inf) - 1e-9:
                    dist[v] = nd
                    prev[v] = (u, sid, pl)
                    heapq.heappush(heap, (nd, v))
        if "T" not in dist:
            return None
        seq, poly, node = [], [], "T"
        while node != "S":
            u, sid, pl = prev[node]
            seq.append((sid, node))
            poly = pl[1:] + poly if poly else pl[1:]
            node = u
        seq.reverse()
        names = []
        for sid, node in seq:
            names.append(sid)
            if node != "T":
                names.append(node)
        return {"spaces": names, "polyline": [list(start_xz)] + poly, "length": dist["T"]}


def _xz(s: str):
    a = [float(v) for v in s.split(",")]
    return a[:2] if len(a) == 2 else [a[0], a[2]]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Print a path between two world points (x,z)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--navgrid")
    src.add_argument("--building")
    ap.add_argument("--from", dest="start", required=True, help="x,z (or x,y,z)")
    ap.add_argument("--to", dest="goal", required=True, help="x,z (or x,y,z)")
    ap.add_argument("--web-root", default=None,
                    help="folder building.json paths are relative to (default: parent of its folder)")
    ap.add_argument("--allow-unknown", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args(argv)
    a, b = _xz(args.start), _xz(args.goal)
    if args.navgrid:
        g = NavGrid.load(args.navgrid)
        pl, L = path_world(g, a, b, allow_unknown=args.allow_unknown)
        if pl is None:
            print("unreachable")
            raise SystemExit(1)
        res = {"polyline": pl, "length": L}
    else:
        with open(args.building, encoding="utf-8") as f:
            data = json.load(f)
        # paths inside building.json are relative to the web root (web/)
        root = args.web_root or os.path.dirname(os.path.dirname(os.path.abspath(args.building)))
        res = Building(data, root, args.allow_unknown).route(a, b)
        if res is None:
            print("unreachable")
            raise SystemExit(1)
        print("route: " + " -> ".join(res["spaces"]))
    if args.json:
        print(json.dumps(res, indent=1))
    else:
        print(f"length: {res['length']:.2f} m, {len(res['polyline'])} points")
        pl = res["polyline"]
        step = max(1, len(pl) // 12)
        print("  " + "  ".join(f"({x:.2f},{z:.2f})" for x, z in pl[::step] + [pl[-1]]))


if __name__ == "__main__":
    main()
