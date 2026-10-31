"""Exact numpy PLY reader/writer for Gaussian splats (binary or ascii in, binary out).

`read_ply` returns a `Ply` whose `.data` is a numpy *structured* array with the file's
own property names, order and dtypes, so nothing is converted or reordered. Writing a
Ply back reproduces the input bytes exactly for binary files produced by Brush,
gsplat, nerfstudio, INRIA 3DGS, Scaniverse/Polycam exports (header lines are kept
verbatim, data is the raw record bytes).

Gaussian-splat helpers (`xyz`, `opacity_logit` ...) assume the INRIA 3DGS property
names: x y z, opacity (logit), scale_0..2 (log), rot_0..3 (wxyz), f_dc_*, f_rest_*.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}
_NAMES = {"i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort", "i4": "int",
          "u4": "uint", "f4": "float", "f8": "double"}


@dataclass
class Ply:
    data: np.ndarray                      # structured array, one record per vertex
    comments: list = field(default_factory=list)   # raw header lines between format and element
    fmt: str = "binary_little_endian"
    header_lines: list | None = None      # verbatim header (for exact round trip)
    tail: bytes = b""                     # bytes of any elements after 'vertex' (kept raw)

    def __len__(self):
        return len(self.data)

    @property
    def names(self):
        return list(self.data.dtype.names)

    def subset(self, idx):
        return Ply(self.data[idx], list(self.comments), self.fmt, None, self.tail)


def read_ply(path):
    with open(path, "rb") as f:
        buf = f.read()
    end = buf.find(b"end_header")
    if not buf.startswith(b"ply") or end < 0:
        raise ValueError(f"{path}: not a PLY file")
    nl = buf.find(b"\n", end)
    body = buf[nl + 1:]
    lines = buf[:nl + 1].decode("ascii", errors="strict").splitlines()
    fmt, comments, props, n, elem = None, [], [], 0, None
    others = []  # elements after vertex: (name, count, props)
    for ln in lines:
        t = ln.strip().split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] in ("comment", "obj_info"):
            comments.append(ln)
        elif t[0] == "element":
            elem = t[1]
            if elem == "vertex":
                n = int(t[2])
            else:
                others.append((elem, int(t[2]), []))
        elif t[0] == "property":
            if t[1] == "list":
                if elem == "vertex":
                    raise ValueError(f"{path}: list properties on vertex not supported")
                others[-1][2].append(("list",) + tuple(t[2:]))
            elif elem == "vertex":
                props.append((t[2], _TYPES[t[1]]))
            else:
                others[-1][2].append((t[2], _TYPES[t[1]]))
    if fmt is None:
        raise ValueError(f"{path}: missing format line")
    if fmt == "ascii":
        dt = np.dtype([(nm, ty) for nm, ty in props])
        rows = body.decode("ascii").split("\n")[:n]
        data = np.zeros(n, dtype=dt)
        vals = np.array([r.split()[:len(props)] for r in rows], dtype=np.float64)
        for k, (nm, _) in enumerate(props):
            data[nm] = vals[:, k]
        return Ply(data, comments, "binary_little_endian", None, b"")
    end_ch = "<" if fmt == "binary_little_endian" else ">"
    dt = np.dtype([(nm, end_ch + ty) for nm, ty in props])
    nbytes = n * dt.itemsize
    if len(body) < nbytes:
        raise ValueError(f"{path}: truncated ({len(body)} < {nbytes} bytes)")
    data = np.frombuffer(body[:nbytes], dtype=dt).copy()
    return Ply(data, comments, fmt, lines, body[nbytes:])


def _header(ply: Ply):
    if ply.header_lines is not None:
        # verbatim, but with the vertex count updated
        out = []
        for ln in ply.header_lines:
            t = ln.split()
            if len(t) == 3 and t[0] == "element" and t[1] == "vertex":
                ln = f"element vertex {len(ply.data)}"
            out.append(ln)
        return ("\n".join(out) + "\n").encode("ascii")
    out = ["ply", f"format {ply.fmt} 1.0", *ply.comments, f"element vertex {len(ply.data)}"]
    for nm in ply.data.dtype.names:
        out.append(f"property {_NAMES[ply.data.dtype[nm].str[1:]]} {nm}")
    out.append("end_header")
    return ("\n".join(out) + "\n").encode("ascii")


def write_ply(path, ply: Ply):
    if ply.tail and ply.header_lines is None:
        raise ValueError("cannot write extra PLY elements without the original header")
    data = ply.data
    want = "<" if ply.fmt == "binary_little_endian" else ">"
    if data.dtype != data.dtype.newbyteorder(want):
        data = data.astype(data.dtype.newbyteorder(want))
    with open(path, "wb") as f:
        f.write(_header(ply))
        f.write(np.ascontiguousarray(data).tobytes())
        f.write(ply.tail)


# ----------------------------------------------------------------------------- splat helpers
def xyz(ply: Ply):
    d = ply.data
    return np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def opacity(ply: Ply):
    return sigmoid(ply.data["opacity"].astype(np.float64)) if "opacity" in ply.names else \
        np.ones(len(ply))


def _sh_rest_count(names):
    return sum(1 for n in names if n.startswith("f_rest_"))


def conform(src: Ply, ref: Ply):
    """Re-express `src` records with `ref`'s property list (order + dtypes).

    f_rest_* are channel-major (R coeffs, then G, then B), so when SH degrees differ the
    coefficients are remapped per channel; bands the source lacks are zero-filled.
    Other missing properties are zero-filled, extra ones dropped.
    """
    if src.data.dtype == ref.data.dtype:
        return src
    out = np.zeros(len(src.data), dtype=ref.data.dtype)
    have = set(src.names)
    for nm in ref.names:
        if nm in have and not nm.startswith("f_rest_"):
            out[nm] = src.data[nm]
    ks, kr = _sh_rest_count(src.names) // 3, _sh_rest_count(ref.names) // 3
    for c in range(3):
        for k in range(min(ks, kr)):
            a, b = f"f_rest_{c * ks + k}", f"f_rest_{c * kr + k}"
            if a in have and b in out.dtype.names:
                out[b] = src.data[a]
    return Ply(out, list(ref.comments), ref.fmt, None, b"")
