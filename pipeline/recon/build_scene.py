"""Stage 2 CLI: capture folder -> web/scenes/<id>/scene.ply + scene.json.

    python -m pipeline.recon.build_scene <capture> --id <sceneId> [--trainer brush|gsplat|none]

Steps
  1. capture -> COLMAP text model + images in --work (default build/recon/<id>/dataset):
     c2w poses inverted to w2c (qvec wxyz), points3D from depth back-projection.
  2. trainer: brush (local, Vulkan) | gsplat (writes a Colab/CUDA script) | none (--ply).
  3. publish scene.ply, then compute scene.json.toWorld:
       - floor = RANSAC plane on opacity-filtered gaussian centres, constrained to be
         near the cameras' up direction (or meta.gravity / --up-hint) and below the
         camera trajectory; up = normal pointing toward the cameras
       - rotate up -> +Y and the first camera's forward -> -Z, floor -> y = 0,
         first camera -> x = z = 0
       - scale: 1 if meta.json says metric (capture already in metres); --scale S to
         force; --assume-eye-height H rescales a relative capture so the median camera
         height is H (heuristic, metric stays false); otherwise unscaled, metric:false
     bounds (0.5..99.5 percentile of gaussian centres, world), spawn (first camera at
     --eye-height above the floor when metric, else at the first camera's own height;
     yaw = its forward in three.js convention).
Torch-free.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import sys

import numpy as np

from ..capture import common as cm
from . import colmap_io, geometry as geo, plyio, trainers

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SCENES = os.path.join(REPO, "web", "scenes")


def relpath(p):
    try:
        return os.path.relpath(os.path.abspath(p), REPO).replace("\\", "/")
    except ValueError:
        return os.path.abspath(p).replace("\\", "/")


def load_scene_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_scene_json(path, sj):
    with open(path, "w") as f:
        json.dump(sj, f, indent=2)
        f.write("\n")


def to_world_list(T):
    return [round(float(v), 9) + 0.0 for v in np.asarray(T).reshape(-1)]


def from_world_list(lst):
    return np.array(lst, dtype=np.float64).reshape(4, 4)


def compute_alignment(ply, poses=None, meta=None, up_hint=None, eye_height=1.6,
                      assume_eye_height=None, scale=None, seed=0):
    """-> dict with toWorld (4x4), bounds, spawn, metric, alignment info."""
    meta = meta or {}
    xyz, op = plyio.xyz(ply), plyio.opacity(ply)
    pts = geo.splat_points_for_alignment(xyz, op, seed=seed)
    centers = ups = fwd0 = c0 = None
    if poses is not None and len(poses):
        centers = poses[:, :3, 3]
        ups = -poses[:, :3, 1]          # OpenCV camera: -y is up in the image
        fwd0, c0 = poses[0][:3, 2], poses[0][:3, 3]
    gravity = meta.get("gravity")
    if gravity is not None:              # measured gravity beats the camera-up heuristic
        up_hint, ups = -np.asarray(gravity, float), None
    elif up_hint is None and meta.get("up_hint") is not None:
        up_hint = meta["up_hint"]
    n, d, inl, info = geo.detect_floor(pts, centers, ups, up_hint, seed=seed)

    metric = bool(meta.get("metric", False))
    scale_src = "capture metric" if metric else "none (relative units)"
    s = 1.0
    cam_h = float(np.median(centers @ n + d)) if centers is not None else None
    if scale is not None:
        s, scale_src = float(scale), "--scale"
        metric = True
    elif not metric and assume_eye_height and cam_h:
        s, scale_src = float(assume_eye_height) / cam_h, f"eye-height heuristic ({assume_eye_height} m)"
    if c0 is None:
        c0 = pts[inl].mean(0)
    T = geo.world_transform(n, d, c0, fwd0, s, floor_pts=pts[inl])

    w = geo.apply(T, pts)
    lo, hi = np.percentile(w, 0.5, 0), np.percentile(w, 99.5, 0)
    if metric or assume_eye_height:
        eye = float(eye_height)
    elif fwd0 is not None:  # relative units: put the viewer where the phone actually was
        eye = float(n @ c0 + d) * s
    else:
        eye = float(hi[1] - lo[1]) * 0.45
    spawn_pos = [0.0, eye, 0.0]
    yaw = 0.0
    if fwd0 is not None:
        yaw = geo.yaw_of(T[:3, :3] @ fwd0)
    floor_w = geo.apply(T, pts[inl])
    up_w = T[:3, :3] @ n / s
    info.update(floorPlane_scene=[*map(float, n), float(d)],
                floorResidualStd=float(floor_w[:, 1].std()),
                upInWorld=[float(v) for v in up_w],
                cameraHeight=None if cam_h is None else cam_h * s,
                scale=s, scaleSource=scale_src)
    return dict(toWorld=T, bounds={"min": [float(v) for v in lo], "max": [float(v) for v in hi]},
                spawn={"position": [round(v, 4) for v in spawn_pos], "yaw": round(yaw, 6)},
                metric=metric, alignment=info)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pipeline.recon.build_scene", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--id", required=True, help="scene id -> web/scenes/<id>/")
    ap.add_argument("--name", help="display name (default: keep existing / id)")
    ap.add_argument("--trainer", choices=["brush", "gsplat", "none"], default="brush")
    ap.add_argument("--ply", help="existing splat for --trainer none")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--brush-exe")
    ap.add_argument("--trainer-args", default="", help="extra flags passed to the trainer")
    ap.add_argument("--work", help="scratch dir (default build/recon/<id>)")
    ap.add_argument("--no-dataset", action="store_true",
                    help="skip writing the COLMAP dataset (only valid with --trainer none)")
    ap.add_argument("--scenes-dir", default=DEFAULT_SCENES)
    ap.add_argument("--source", help="scene.json source string")
    ap.add_argument("--up-hint", help="x,y,z approximate up in the scene frame")
    ap.add_argument("--eye-height", type=float, default=1.6)
    ap.add_argument("--assume-eye-height", type=float,
                    help="relative captures: scale so the median camera height = this (m)")
    ap.add_argument("--scale", type=float, help="force scene->metres scale")
    a = ap.parse_args(argv)

    cap = a.capture
    meta = cm.load_meta(cap)
    work = a.work or os.path.join(REPO, "build", "recon", a.id)
    scene_dir = os.path.join(a.scenes_dir, a.id)
    os.makedirs(scene_dir, exist_ok=True)
    poses = None

    if meta.get("scene_only"):
        ply_path = os.path.join(cap, "scene", "scene.ply")
        trainer_desc = "imported splat"
    else:
        poses = cm.load_poses(cap)
        dataset = os.path.join(work, "dataset")
        if not (a.no_dataset and a.trainer == "none"):
            colmap_io.capture_to_dataset(cap, dataset)
        ply_path = trainers.train(a.trainer, dataset, work, a.steps, a.ply, a.brush_exe,
                                  a.trainer_args)
        if ply_path is None:
            print("[build] training must run elsewhere (see script above); stopping.")
            return None
        trainer_desc = f"{a.trainer} {a.steps} steps" if a.trainer != "none" else "existing splat"

    dst = os.path.join(scene_dir, "scene.ply")
    if os.path.abspath(ply_path) != os.path.abspath(dst):
        shutil.copyfile(ply_path, dst)
    size_mb = os.path.getsize(dst) / 1e6
    if size_mb > 50:
        print(f"[build] WARNING: scene.ply is {size_mb:.1f} MB (> 50 MB repo limit); prune it")

    ply = plyio.read_ply(dst)
    up_hint = [float(v) for v in a.up_hint.split(",")] if a.up_hint else None
    al = compute_alignment(ply, poses, meta, up_hint, a.eye_height, a.assume_eye_height, a.scale)

    sj_path = os.path.join(scene_dir, "scene.json")
    sj = load_scene_json(sj_path)
    sj.update({
        "id": a.id,
        "name": a.name or sj.get("name") or a.id,
        "splat": "scene.ply",
        "toWorld": to_world_list(al["toWorld"]),
        "bounds": al["bounds"],
        "floorY": 0.0,
        "spawn": al["spawn"],
        "metric": al["metric"],
        "source": a.source or sj.get("source") or f"{meta.get('source', relpath(cap))} + {trainer_desc}",
        "refinements": sj.get("refinements", []),
        "capture": relpath(cap),
        "gaussians": len(ply),
        "alignment": al["alignment"],
        "built": _dt.date.today().isoformat(),
    })
    save_scene_json(sj_path, sj)
    T = al["toWorld"]
    print(f"[build] {len(ply):,} gaussians -> {dst}")
    print(f"[build] floor inliers {al['alignment']['inliers']}, up(world) "
          f"{np.round(al['alignment']['upInWorld'], 4).tolist()}, scale {al['alignment']['scale']:.4f}")
    print("[build] toWorld =\n" + np.array2string(T, precision=6, suppress_small=True))
    print(f"[build] wrote {sj_path}")
    return sj_path


if __name__ == "__main__":
    main(sys.argv[1:])
