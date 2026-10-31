"""Stand-in patch splat for testing refine/merge without a GPU.

NOT a reconstruction: copies the base gaussians in region+margin, jitters their centres
and recolours them (tint), so a merged scene visibly shows the refined box.

    python -m pipeline.recon.standin_patch --scene web/scenes/cliproom \
        --region -0.5,0,-3,0.5,1,-2 --out build/patch.ply
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from . import merge as mg, plyio
from .build_scene import from_world_list, load_scene_json
from .refine import parse_region


def standin(base, to_box, bmin, bmax, margin, jitter=0.01, tint=(0.6, -0.3, -0.3), seed=0):
    rng = np.random.default_rng(seed)
    inside, band, _ = mg.classify(base, to_box, np.asarray(bmin), np.asarray(bmax), margin)
    d = base.data[inside | band].copy()
    for k in "xyz":
        d[k] = d[k] + rng.normal(0, jitter, len(d)).astype(d[k].dtype)
    for k, dt in enumerate(tint):
        nm = f"f_dc_{k}"
        if nm in d.dtype.names:
            d[nm] = d[nm] + np.float32(dt)
    return plyio.Ply(d, list(base.comments) + ["comment stand-in patch (jittered/recoloured base subset)"],
                     base.fmt, None, b"")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pipeline.recon.standin_patch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--region", required=True, type=parse_region)
    ap.add_argument("--frame", choices=["world", "scene"], default="world")
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--jitter", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sj = load_scene_json(os.path.join(a.scene, "scene.json"))
    base = plyio.read_ply(os.path.join(a.scene, sj.get("splat", "scene.ply")))
    to_box = from_world_list(sj["toWorld"]) if a.frame == "world" else np.eye(4)
    p = standin(base, to_box, *a.region, a.margin, a.jitter)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    plyio.write_ply(a.out, p)
    print(f"[standin] {len(p):,} gaussians -> {a.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
