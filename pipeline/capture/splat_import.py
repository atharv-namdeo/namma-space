"""Import an already-trained splat (.ply from Scaniverse, Polycam, Luma, Brush ...).

Scene-only capture: no frames, no poses. Writes

    <capture>/scene/scene.ply     the splat, byte-for-byte (validated as a 3DGS PLY)
    <capture>/meta.json           {source, metric, scene_only: true, n_gaussians, up_hint}

`python -m pipeline.recon.build_scene <capture> --id X --trainer none` then aligns it
(floor RANSAC; no camera trajectory, so `--up-hint` / meta.up_hint disambiguate floor
vs ceiling) and publishes it to web/scenes/X/.
"""
from __future__ import annotations

import os
import shutil

from . import common as cm
from ..recon import plyio

REQUIRED = ("x", "y", "z")
SPLAT = ("opacity", "scale_0", "rot_0", "f_dc_0")


def import_splat(ply_path, out, metric=None, up_hint=None):
    p = plyio.read_ply(ply_path)
    missing = [k for k in REQUIRED if k not in p.names]
    if missing:
        raise SystemExit(f"[splat] {ply_path}: missing {missing}; not a point/splat PLY")
    is_splat = all(k in p.names for k in SPLAT)
    os.makedirs(os.path.join(out, "scene"), exist_ok=True)
    shutil.copyfile(ply_path, os.path.join(out, "scene", "scene.ply"))
    # Scaniverse/Polycam exports come from LiDAR/ARKit sessions and are in metres.
    meta = dict(source=f"ply:{os.path.basename(ply_path)}", scene_only=True,
                metric=bool(metric) if metric is not None else False,
                n_gaussians=len(p), gaussian_splat=is_splat)
    if up_hint is not None:
        meta["up_hint"] = [float(v) for v in up_hint]
    cm.update_meta(out, **meta)
    print(f"[splat] {len(p):,} {'gaussians' if is_splat else 'points'} -> {out}/scene/scene.ply")
    return out
