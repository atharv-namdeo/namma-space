import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SAMPLE = os.path.join(REPO, "data", "sample", "clip1")
SCENE_PLY = os.path.join(REPO, "web", "scenes", "cliproom", "scene.ply")


@pytest.fixture(scope="session")
def sample_ply():
    from pipeline.recon import plyio
    return plyio.read_ply(SCENE_PLY)


def make_splat(n, seed=0, sh_rest=45, box=((-2, -1, -2), (2, 1, 2))):
    """Random little 3DGS splat (Brush property order)."""
    from pipeline.recon import plyio
    rng = np.random.default_rng(seed)
    names = ["f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(sh_rest)] + \
            ["opacity", "rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2", "x", "y", "z"]
    d = np.zeros(n, dtype=[(k, "<f4") for k in names])
    for k in names:
        d[k] = rng.normal(0, 1, n)
    lo, hi = np.array(box[0]), np.array(box[1])
    p = rng.uniform(lo, hi, (n, 3))
    d["x"], d["y"], d["z"] = p.T
    return plyio.Ply(d, ["comment synthetic"], "binary_little_endian", None, b"")
