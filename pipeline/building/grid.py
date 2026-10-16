"""navgrid.json model + RLE codec (see docs/ARCHITECTURE.md).

Array layout: ``cells[j, i]`` with j = z index (row), i = x index (column);
cell (i, j) covers x in [x0 + i*cs, x0 + (i+1)*cs), z in [z0 + j*cs, ...).
Flattened row-major over z then x, i.e. ``cells.reshape(-1)``.
Values: 0 free, 1 blocked, 2 unknown.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

FREE, BLOCKED, UNKNOWN = 0, 1, 2


def rle_encode(flat) -> list[list[int]]:
    a = np.asarray(flat).reshape(-1)
    if a.size == 0:
        return []
    change = np.flatnonzero(a[1:] != a[:-1]) + 1
    starts = np.concatenate([[0], change])
    lengths = np.diff(np.concatenate([starts, [a.size]]))
    return [[int(a[s]), int(n)] for s, n in zip(starts, lengths)]


def rle_decode(runs, size: int | None = None) -> np.ndarray:
    if not runs:
        out = np.zeros(0, np.uint8)
    else:
        vals = np.array([r[0] for r in runs], np.uint8)
        lens = np.array([r[1] for r in runs], np.int64)
        out = np.repeat(vals, lens)
    if size is not None and out.size != size:
        raise ValueError(f"RLE decodes to {out.size} cells, expected {size}")
    return out


@dataclass
class NavGrid:
    cells: np.ndarray                    # (height, width) uint8
    cell_size: float
    origin: tuple[float, float]          # (x0, z0)
    floor_y: float = 0.0
    scene_id: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.cells.shape[1])

    @property
    def height(self) -> int:
        return int(self.cells.shape[0])

    # --- coordinates -----------------------------------------------------
    def world_to_cell(self, x: float, z: float) -> tuple[int, int]:
        """-> (i, j) = (column, row); may be out of range."""
        return (int(np.floor((x - self.origin[0]) / self.cell_size)),
                int(np.floor((z - self.origin[1]) / self.cell_size)))

    def cell_to_world(self, i: int, j: int) -> tuple[float, float]:
        """Centre of cell (i, j) -> (x, z)."""
        return (self.origin[0] + (i + 0.5) * self.cell_size,
                self.origin[1] + (j + 0.5) * self.cell_size)

    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.width and 0 <= j < self.height

    def is_free(self, i: int, j: int) -> bool:
        return self.in_bounds(i, j) and self.cells[j, i] == FREE

    # --- JSON ------------------------------------------------------------
    def to_json(self) -> dict:
        d = {
            "sceneId": self.scene_id,
            "cellSize": float(self.cell_size),
            "origin": [round(float(self.origin[0]), 4), round(float(self.origin[1]), 4)],
            "width": self.width,
            "height": self.height,
            "floorY": float(self.floor_y),
            "encoding": "rle",
            "data": rle_encode(self.cells),
        }
        d.update(self.extra)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "NavGrid":
        w, h = int(d["width"]), int(d["height"])
        if d.get("encoding", "rle") != "rle":
            raise ValueError("only rle navgrids supported")
        cells = rle_decode(d["data"], w * h).reshape(h, w)
        known = {"sceneId", "cellSize", "origin", "width", "height", "floorY",
                 "encoding", "data"}
        return cls(cells=cells, cell_size=float(d["cellSize"]),
                   origin=(float(d["origin"][0]), float(d["origin"][1])),
                   floor_y=float(d.get("floorY", 0.0)), scene_id=d.get("sceneId", ""),
                   extra={k: v for k, v in d.items() if k not in known})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, separators=(",", ":"))

    @classmethod
    def load(cls, path: str) -> "NavGrid":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(json.load(f))
