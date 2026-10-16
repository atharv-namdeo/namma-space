"""Fast tests for pipeline.building (numpy/scipy/opencv only, < a few seconds)."""
import json
import math

import numpy as np
import pytest

from pipeline.building import align, astar, compose, geom, segment
from pipeline.building.grid import BLOCKED, FREE, UNKNOWN, NavGrid, rle_decode, rle_encode
from pipeline.building.navgrid import build_navgrid, footprint_polygon


# ------------------------------------------------------------------ RLE
def test_rle_round_trip():
    rng = np.random.default_rng(0)
    a = rng.choice([0, 1, 2], size=(37, 53), p=[0.6, 0.3, 0.1]).astype(np.uint8)
    a[5:20, 10:40] = 0
    runs = rle_encode(a)
    assert sum(n for _, n in runs) == a.size
    assert all(runs[k][0] != runs[k + 1][0] for k in range(len(runs) - 1))
    assert np.array_equal(rle_decode(runs, a.size).reshape(a.shape), a)


def test_navgrid_json_round_trip(tmp_path):
    cells = np.full((4, 6), UNKNOWN, np.uint8)
    cells[1:3, 1:5] = FREE
    cells[0, :] = BLOCKED
    g = NavGrid(cells, 0.1, (-0.3, 1.2), scene_id="t", extra={"inflateRadius": 0.2})
    p = tmp_path / "navgrid.json"
    g.save(str(p))
    d = json.loads(p.read_text())
    assert d["encoding"] == "rle" and d["width"] == 6 and d["height"] == 4
    assert d["data"][0] == [BLOCKED, 6]           # row-major over z then x
    g2 = NavGrid.load(str(p))
    assert np.array_equal(g2.cells, cells) and g2.origin == (-0.3, 1.2)
    assert g2.extra["inflateRadius"] == 0.2
    # cell (i,j) covers [x0+i*cs, ...)
    assert g2.world_to_cell(-0.3 + 0.25, 1.2 + 0.15) == (2, 1)


# ------------------------------------------------------------- navgrid
def _box_room(w=5.0, d=4.0, h=2.5, step=0.05):
    xs = np.arange(0, w + 1e-9, step)
    zs = np.arange(0, d + 1e-9, step)
    X, Z = np.meshgrid(xs, zs)
    floor = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    ys = np.arange(0, h + 1e-9, step)
    walls = []
    for x in xs:
        for z in (0.0, d):
            walls += [[x, y, z] for y in ys]
    for z in zs:
        for x in (0.0, w):
            walls += [[x, y, z] for y in ys]
    return np.concatenate([floor, np.array(walls)], 0)


def test_navgrid_box_room():
    pts = _box_room()
    cams = np.array([[1.5, 1.4, 2.0], [2.5, 1.4, 2.0], [3.5, 1.4, 2.0]])
    g = build_navgrid(pts, cams, cell=0.1, inflate=0.2)

    def at(x, z):
        i, j = g.world_to_cell(x, z)
        return int(g.cells[j, i])

    for x, z in [(2.5, 2.0), (1.0, 1.0), (4.0, 3.0), (2.5, 0.8)]:
        assert at(x, z) == FREE, (x, z)
    for x, z in [(0.02, 2.0), (4.98, 2.0), (2.5, 0.02), (2.5, 3.98), (0.15, 2.0)]:
        assert at(x, z) == BLOCKED, (x, z)           # walls + 0.2 m inflation
    assert at(-0.2, 2.0) == UNKNOWN and at(2.5, 4.25) == UNKNOWN
    poly = footprint_polygon(g)
    assert 18 < geom.polygon_area(poly) < 22


def test_alignment_recovers_gravity_and_floor():
    # build a box room in a world frame, then express it in an OpenCV-like
    # scene frame (y down, tilted, scaled) and check we get the floor back
    world = _box_room(step=0.1)
    world = np.concatenate([world, world + [0, 2.5, 0] * np.array([0, 1, 0])[None]], 0)
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)    # Y up -> y down
    a = np.radians(12)
    tilt = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    s = 0.4
    to_scene = lambda p: s * (p - [2.5, 1.4, 2.0]) @ (tilt @ R).T
    pts = to_scene(world)
    poses = np.repeat(np.eye(4)[None], 5, 0)
    for k in range(5):
        poses[k, :3, :3] = tilt @ R @ np.diag([1, -1, -1])   # camera looks along world -Z
        poses[k, :3, 3] = to_scene(np.array([[2.5 - 0.2 * k, 1.4, 2.0]]))[0]
    al = align.estimate_alignment(pts, poses, metric=False, scale_cue="camera-height",
                                  camera_height=1.4)
    T = align.mat_from_rowmajor(al["toWorld"])
    w = align.apply(T, pts)
    floor = w[np.abs(world[:, 1]) < 1e-9]
    assert abs(np.median(floor[:, 1])) < 0.05
    assert np.percentile(np.abs(floor[:, 1]), 95) < 0.08
    assert abs(al["scale"] * s - 1.0) < 0.05
    cams = align.apply(T, poses[:, :3, 3])
    assert np.allclose(cams[:, 1], 1.4, atol=0.05)


# ------------------------------------------------------------ segmentation
def test_segment_synthetic_hallway_three_rooms():
    g = segment.synthetic_floorplan()
    seg = segment.segment(g)
    kinds = sorted(s["kind"] for s in seg["spaces"])
    assert kinds == ["hallway", "room", "room", "room"]
    assert len(seg["doors"]) == 3
    hall = next(s["id"] for s in seg["spaces"] if s["kind"] == "hallway")
    for d in seg["doors"]:
        assert hall in d["between"]
        assert abs(d["position"][2] - 5.0) < 0.21            # doors in the shared wall
        assert 0.6 <= d["width"] <= 1.1
    xs = sorted(d["position"][0] for d in seg["doors"])
    assert np.allclose(xs, [2.25, 6.5, 11.25], atol=0.25)
    for s in seg["spaces"]:
        assert geom.polygon_area(s["polygon"]) > 15


# ------------------------------------------------------------------ A*
def _open_grid(w=20, h=10):
    return NavGrid(np.zeros((h, w), np.uint8), 0.1, (0.0, 0.0))


def test_astar_straight_and_diagonal():
    g = _open_grid()
    path, cost = astar.astar(g, (0, 0), (9, 0))
    assert cost == pytest.approx(9.0) and len(path) == 10
    path, cost = astar.astar(g, (0, 0), (5, 5))
    assert cost == pytest.approx(5 * math.sqrt(2))


def test_astar_wall_detour_and_no_corner_cutting():
    g = _open_grid()
    g.cells[0:9, 10] = BLOCKED                         # wall with a gap at j=9
    path, cost = astar.astar(g, (5, 0), (15, 0))
    assert path is not None and all(g.cells[j, i] == FREE for i, j in path)
    assert any(j == 9 for i, j in path)
    # (5,0)->(9,9): 4 diag + 5 straight; through the gap (9,9)->(11,9): 2 straight
    # (no diagonal into/out of the gap: corner cutting); then mirror image
    assert cost == pytest.approx(2 * (5 + 4 * math.sqrt(2)) + 2)
    # corner cutting: two blocked cells touching diagonally must not be passed
    g2 = _open_grid(3, 3)
    g2.cells[0, 1] = BLOCKED
    g2.cells[1, 0] = BLOCKED
    p2, _ = astar.astar(g2, (0, 0), (1, 1))
    assert p2 is None


def test_astar_exact_optimum():
    g = _open_grid(7, 7)
    g.cells[1:6, 3] = BLOCKED
    _, cost = astar.astar(g, (1, 3), (5, 3))
    # round the wall end; the step past the wall tip must be straight
    # (2,6)->(3,6)->(4,6) because diagonals would cut the blocked corner:
    # (1,3)-(2,4) d, (2,5), (2,6), (3,6), (4,6), (4,5), (4,4), (5,3) d
    assert cost == pytest.approx(6 + 2 * math.sqrt(2))


def test_astar_unreachable():
    g = _open_grid()
    g.cells[:, 10] = BLOCKED
    path, cost = astar.astar(g, (2, 2), (15, 2))
    assert path is None and math.isinf(cost)
    pl, L = astar.path_world(g, (0.25, 0.25), (1.55, 0.25))
    assert pl is None


def test_building_route_through_doors():
    data = {
        "spaces": [
            {"id": "hall", "kind": "hallway", "polygon": [[0, 4], [12, 4], [12, 6], [0, 6]]},
            {"id": "r1", "kind": "room", "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]]},
            {"id": "r2", "kind": "room", "polygon": [[8, 0], [12, 0], [12, 4], [8, 4]]},
        ],
        "doors": [{"id": "d1", "between": ["hall", "r1"], "position": [2, 0, 4]},
                  {"id": "d2", "between": ["hall", "r2"], "position": [10, 0, 4]}],
    }
    res = astar.Building(data).route([2, 2], [10, 2])
    assert res["spaces"] == ["r1", "d1", "hall", "d2", "r2"]
    assert res["length"] == pytest.approx(2 + 8 + 2)


# --------------------------------------------------------------- compose
def _spec(**over):
    spec = {
        "name": "t",
        "floors": [{"id": "G", "elevation": 0.0}],
        "spaces": [
            {"id": "hall", "kind": "hallway", "polygon": [[0, 4], [12, 4], [12, 6], [0, 6]]},
            {"id": "r1", "kind": "room", "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]]},
            {"id": "r2", "kind": "room", "polygon": [[8, 0], [12, 0], [12, 4], [8, 4]]},
        ],
        "doors": [{"id": "d1", "between": ["hall", "r1"], "position": [2, 4]},
                  {"id": "d2", "between": ["hall", "r2"], "position": "auto"}],
    }
    spec.update(over)
    return spec


def test_compose_valid(tmp_path):
    b, errors, warnings, _ = compose.compose(_spec(), base=str(tmp_path))
    assert errors == []
    d2 = next(d for d in b["doors"] if d["id"] == "d2")
    assert d2["position"] == [10.0, 0.0, 4.0]
    assert all(s["status"] == "planned" and s["sceneId"] is None for s in b["spaces"])


def test_compose_catches_overlap_and_bad_door(tmp_path):
    spec = _spec()
    spec["spaces"][2]["polygon"] = [[3, 0], [12, 0], [12, 4], [3, 4]]      # overlaps r1
    spec["doors"][0]["position"] = [2, 3]                                   # inside r1
    _, errors, _, _ = compose.compose(spec, base=str(tmp_path))
    assert any("overlap" in e for e in errors)
    assert any("d1" in e and "boundary" in e for e in errors)


def test_compose_scanned_fit_and_autoplace(tmp_path):
    # tiny scan: 2x3 m known area, trajectory starting at its south edge
    cells = np.zeros((30, 20), np.uint8)
    g = NavGrid(cells, 0.1, (-1.0, -2.8), scene_id="scan",
                extra={"trajectory": [[0.0, 0.0], [0.0, -1.0]]})
    d = tmp_path / "web" / "scenes" / "scan"
    d.mkdir(parents=True)
    g.save(str(d / "navgrid.json"))
    spec = _spec()
    spec["spaces"][1] = {"id": "r1", "kind": "room", "sceneId": "scan",
                         "polygon": {"fromNavgrid": "web/scenes/scan/navgrid.json", "mode": "bbox"},
                         "autoPlace": {"door": "d1", "inward": [0, -1], "inset": 0.2}}
    b, errors, warnings, _ = compose.compose(spec, base=str(tmp_path))
    r1 = next(s for s in b["spaces"] if s["id"] == "r1")
    assert errors == [], errors
    assert r1["status"] == "scanned" and r1["footprintFit"] == 1.0
    # entrance (0,0) lands 0.2 m inside door d1 at (2,4) -> translate (2, 3.8)
    M = np.array(r1["placement"]).reshape(4, 4)
    assert np.allclose(M[[0, 2], 3], [2.0, 3.8])
    # an explicit polygon that is too small must warn about footprint fit
    spec["spaces"][1]["polygon"] = [[1.5, 3.0], [2.5, 3.0], [2.5, 4.0], [1.5, 4.0]]
    _, _, warnings, _ = compose.compose(spec, base=str(tmp_path))
    assert any("inside its polygon" in w for w in warnings)
