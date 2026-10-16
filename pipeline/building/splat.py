"""Read Gaussian-splat centres from a 3DGS PLY (Brush / INRIA layout).

Only numpy. Property order is taken from the header, so exports that shuffle
columns (Brush writes f_rest_* in lexical order) still parse.
"""
from __future__ import annotations

import numpy as np

_PLY_TYPES = {
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
    "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
}


def read_ply_vertices(path: str) -> dict[str, np.ndarray]:
    """Return {property_name: array} for the vertex element of a binary LE PLY."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        fields, n_vert, in_vertex = [], 0, False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: truncated header")
            tok = line.decode("ascii", "replace").split()
            if not tok:
                continue
            if tok[0] == "format" and tok[1] != "binary_little_endian":
                raise ValueError(f"{path}: only binary_little_endian PLY supported")
            if tok[0] == "element":
                in_vertex = tok[1] == "vertex"
                if in_vertex:
                    n_vert = int(tok[2])
            elif tok[0] == "property" and in_vertex:
                if tok[1] == "list":
                    raise ValueError(f"{path}: list properties in vertex element unsupported")
                fields.append((tok[2], _PLY_TYPES[tok[1]]))
            elif tok[0] == "end_header":
                break
        data = np.fromfile(f, dtype=np.dtype(fields), count=n_vert)
    return {name: data[name] for name, _ in fields}


def load_splat_centres(path: str, min_opacity: float = 0.25,
                       max_scale: float | None = None) -> tuple[np.ndarray, dict]:
    """Load gaussian centres, filtered by opacity and size.

    opacity is stored as a logit, scale_* as log. ``max_scale`` (scene units)
    drops big blurry splats; default = the 98th percentile of max-axis scale.
    Returns (N,3) float64 points in the scene frame and a stats dict.
    """
    v = read_ply_vertices(path)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    keep = np.isfinite(xyz).all(1)
    stats = {"total": int(len(xyz))}
    if "opacity" in v:
        op = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
        keep &= op >= min_opacity
    scales = [k for k in ("scale_0", "scale_1", "scale_2") if k in v]
    if scales:
        smax = np.exp(np.max(np.stack([v[k] for k in scales], 1), 1).astype(np.float64))
        if max_scale is None:
            max_scale = float(np.percentile(smax, 98))
        keep &= smax <= max_scale
    stats.update(kept=int(keep.sum()), min_opacity=min_opacity,
                 max_scale=None if max_scale is None else float(max_scale))
    return xyz[keep], stats
