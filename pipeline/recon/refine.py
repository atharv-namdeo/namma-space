"""Iterative refinement: re-shoot one region, rebuild only that region, merge.

    python -m pipeline.recon.refine --scene web/scenes/<id> --capture <new_capture> \
        --region minx,miny,minz,maxx,maxy,maxz [--frame world|scene] [--margin 0.2] \
        [--trainer brush|gsplat|none] [--patch patch.ply] [--register features|skip] [--icp]

  a) register the new capture to the base scene (Sim3 from SIFT matches lifted to 3D
     with both captures' depth, RANSAC Umeyama; or `skip` when it already shares the
     base frame - automatic if its meta.json has base_frame == <id>), and write the
     registered copy to <work>/registered;
  b) build a COLMAP dataset from the registered frames only, initialised with the base
     gaussians inside region+margin (centre + DC colour) plus the new depth points
     there; train a patch splat with the pluggable trainer (or take --patch);
  c) merge (pipeline/recon/merge.py): drop base gaussians inside the region, feather
     the margin band, insert the patch cropped to region+margin. Everything outside
     region+margin stays byte-identical. The previous scene.ply is kept as
     scene.v<N>.ply (gitignored) and a record is appended to scene.json.refinements.

--region is in the world frame (metres, as shown in the web viewer; default) or the
scene/splat frame (--frame scene). --margin is in the same frame's units.
Torch-free.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import os
import re
import shutil
import sys

import numpy as np

from ..capture import common as cm
from . import colmap_io, geometry as geo, merge as mg, plyio, register as rg, trainers
from .build_scene import REPO, from_world_list, load_scene_json, relpath, save_scene_json

SH_C0 = 0.28209479177387814


def parse_region(s):
    v = [float(x) for x in s.split(",")]
    if len(v) != 6:
        raise argparse.ArgumentTypeError("region = minx,miny,minz,maxx,maxy,maxz")
    lo, hi = np.minimum(v[:3], v[3:]), np.maximum(v[:3], v[3:])
    return lo, hi


def next_version(scene_dir):
    vs = [int(m.group(1)) for p in glob.glob(os.path.join(scene_dir, "scene.v*.ply"))
          if (m := re.search(r"scene\.v(\d+)\.ply$", p))]
    return max(vs, default=0) + 1


def patch_init_points(base, to_box, bmin, bmax, margin, reg_cap, max_new=60000):
    """Base gaussians in region+margin (centre, DC colour) + registered depth points there."""
    inside, band, _ = mg.classify(base, to_box, bmin, bmax, margin)
    sel = inside | band
    P = plyio.xyz(base)[sel]
    if all(f"f_dc_{k}" in base.names for k in range(3)):
        dc = np.stack([base.data[f"f_dc_{k}"][sel] for k in range(3)], 1)
        C = np.clip((0.5 + SH_C0 * dc) * 255, 0, 255).astype(np.uint8)
    else:
        C = np.full((len(P), 3), 128, np.uint8)
    if os.path.isdir(os.path.join(reg_cap, "depth")):
        Q, QC = colmap_io.depth_points(reg_cap, cm.frame_names(reg_cap), cm.load_poses(reg_cap),
                                       cm.load_intrinsics(reg_cap), max_points=max_new)
        if len(Q):
            qb = geo.apply(to_box, Q)
            m = mg.outside_distance(qb, bmin, bmax) < margin
            P, C = np.concatenate([P, Q[m]]), np.concatenate([C, QC[m]])
    return P, C


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pipeline.recon.refine", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="web/scenes/<id>")
    ap.add_argument("--capture", required=True, help="the re-capture of the region")
    ap.add_argument("--region", required=True, type=parse_region)
    ap.add_argument("--frame", choices=["world", "scene"], default="world")
    ap.add_argument("--margin", type=float, default=0.2, help="feather band width (frame units)")
    ap.add_argument("--register", choices=["auto", "features", "skip"], default="auto")
    ap.add_argument("--icp", action="store_true", help="refine the Sim3 with ICP")
    ap.add_argument("--base-capture", help="default: scene.json.capture")
    ap.add_argument("--trainer", choices=["brush", "gsplat", "none"], default="brush")
    ap.add_argument("--patch", help="patch splat (scene frame) for --trainer none")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--brush-exe")
    ap.add_argument("--trainer-args", default="")
    ap.add_argument("--work", help="default build/refine/<id>/r<N>")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)

    scene_dir = a.scene
    sj_path = os.path.join(scene_dir, "scene.json")
    sj = load_scene_json(sj_path)
    if not sj:
        raise SystemExit(f"[refine] {sj_path} missing; run build_scene first")
    sid = sj.get("id", os.path.basename(os.path.normpath(scene_dir)))
    ply_path = os.path.join(scene_dir, sj.get("splat", "scene.ply"))
    base = plyio.read_ply(ply_path)
    to_world = from_world_list(sj["toWorld"])
    to_box = to_world if a.frame == "world" else np.eye(4)
    bmin, bmax = a.region
    rid = f"r{len(sj.get('refinements', [])) + 1}"
    work = a.work or os.path.join(REPO, "build", "refine", sid, rid)
    os.makedirs(work, exist_ok=True)

    # (a) registration
    new_meta = cm.load_meta(a.capture)
    method = a.register
    if method == "auto":
        method = "skip" if new_meta.get("base_frame") == sid else "features"
    if method == "skip":
        T, reg_info = np.eye(4), {"method": "skip (shared frame)"}
    else:
        base_cap = a.base_capture or (os.path.join(REPO, sj["capture"]) if sj.get("capture") else None)
        if not base_cap or not os.path.isdir(base_cap):
            raise SystemExit("[refine] feature registration needs the base capture "
                             "(scene.json.capture or --base-capture), or use --register skip")
        T, reg_info = rg.register(a.capture, base_cap, "features", icp=a.icp)
    reg_cap = os.path.join(work, "registered")
    rg.write_registered_capture(a.capture, reg_cap, T, base_id=sid)
    cm.update_meta(reg_cap, metric=bool(sj.get("metric", False)))
    s, _, _ = geo.sim3_decompose(T)
    print(f"[refine] registration: {reg_info.get('method')}  scale {s:.4f}")

    # (b) patch training from the new frames only
    if a.trainer == "none":
        if not a.patch:
            raise SystemExit("[refine] --trainer none needs --patch <ply in the scene frame>")
        patch_path = a.patch
    else:
        dataset = os.path.join(work, "dataset")
        colmap_io.capture_to_dataset(reg_cap, dataset)
        P, C = patch_init_points(base, to_box, bmin, bmax, a.margin, reg_cap)
        intr = cm.load_intrinsics(reg_cap)
        colmap_io.write_model(os.path.join(dataset, "sparse", "0"), intr, cm.frame_names(reg_cap),
                              cm.load_poses(reg_cap), P, C)
        print(f"[refine] patch init: {len(P):,} points (base gaussians + new depth in region)")
        patch_path = trainers.train(a.trainer, dataset, work, a.steps, None, a.brush_exe,
                                    a.trainer_args)
        if patch_path is None:
            print("[refine] train the patch elsewhere, then rerun with --trainer none --patch <ply>")
            return None
    patch = plyio.read_ply(patch_path)

    # (c) merge
    merged, stats = mg.merge(base, patch, to_box, bmin, bmax, a.margin)
    v = next_version(scene_dir)
    backup = os.path.join(scene_dir, f"scene.v{v}.ply")
    shutil.copyfile(ply_path, backup)
    tmp = ply_path + ".tmp"
    plyio.write_ply(tmp, merged)
    os.replace(tmp, ply_path)
    print(f"[refine] base {stats['base']:,}: removed {stats['removed']:,}, feathered "
          f"{stats['feathered']:,}, untouched {stats['untouched']:,}; patch +{stats['added']:,} "
          f"-> {stats['total']:,} gaussians. backup {os.path.basename(backup)}")

    rec = {"id": rid, "region": {"min": [float(x) for x in bmin], "max": [float(x) for x in bmax]},
           "frame": a.frame, "margin": a.margin, "date": _dt.date.today().isoformat(),
           "capture": relpath(a.capture), "registration": {**reg_info, "sim3": np.round(T, 9).tolist()},
           "trainer": a.trainer if a.trainer != "none" else f"none ({relpath(patch_path)})",
           "backup": os.path.basename(backup), "stats": stats}
    if a.note:
        rec["note"] = a.note
    sj.setdefault("refinements", []).append(rec)
    sj["gaussians"] = stats["total"]
    save_scene_json(sj_path, sj)
    print(f"[refine] appended {rid} to {sj_path}")
    return rec


if __name__ == "__main__":
    main(sys.argv[1:])
