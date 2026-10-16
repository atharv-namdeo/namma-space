"""Stage 1 CLI: anything a phone produces -> contract-format capture folder.

    python -m pipeline.capture.ingest <input> --out <capture> [options]

<input> kind is auto-detected (override with --kind):
    *.mp4/*.mov/*.mkv/...     plain phone video  -> frames (+ blur rejection) + poses
    *.zip or folder w/ poses.csv  PoseCam RAW session (ARCore metric poses, IMU, depth)
    *.ply                     trained splat export (Scaniverse/Polycam/Brush): scene only
    folder w/ rgb/ + poses.npy  an existing capture: only run sensor fusion on it

Sensor fusion (any kind with depth):
    --range-csv sensors.csv   timestamp_ns,range_m[,u,v]  (ToF / ultrasonic)
    --arcore-depth DIR        ARCore depth pngs named like the rgb frames (or auto from PoseCam)
  -> robust median scale -> depth (and relative poses) rescaled, meta.json {metric:true, scale,
     scale_inliers}.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

from . import common as cm

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".3gp")


def detect_kind(path):
    low = path.lower()
    if low.endswith(VIDEO_EXT):
        return "video"
    if low.endswith(".ply"):
        return "ply"
    if low.endswith(".zip"):
        return "posecam"
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "poses.npy")) and os.path.isdir(os.path.join(path, "rgb")):
            return "capture"
        if glob.glob(os.path.join(path, "**", "poses.csv"), recursive=True):
            return "posecam"
    raise SystemExit(f"[ingest] cannot tell what {path} is; pass --kind")


def _vec3(s):
    v = [float(x) for x in s.split(",")]
    if len(v) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pipeline.capture.ingest",
                                 description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--out", help="capture folder to write (default: next to input)")
    ap.add_argument("--kind", choices=["video", "posecam", "ply", "capture"])
    g = ap.add_argument_group("video")
    g.add_argument("--fps", type=float, default=4.0, help="kept frames per second")
    g.add_argument("--oversample", type=int, default=3, help="pick sharpest of N candidates")
    g.add_argument("--blur-rel", type=float, default=0.5,
                   help="drop frames sharper than < this x rolling median")
    g.add_argument("--size", type=int, default=640, help="longest side, px")
    g.add_argument("--max-frames", type=int, default=0)
    g.add_argument("--poses", choices=["monomap", "none"], default="monomap",
                   help="monomap = Depth Anything + RGB-D odometry (needs torch+open3d)")
    g.add_argument("--f-rel", type=float, default=0.85, help="fx = f_rel * width if unknown")
    g.add_argument("--depth-model", default=None,
                   help="video: relative|indoor (default relative); posecam: none|indoor|relative "
                        "(default none = use ARCore depth if recorded)")
    g.add_argument("--video-start-ns", type=int, default=0,
                   help="sensor-clock time of video frame 0 (for --range-csv)")
    g.add_argument("--no-ffmpeg", action="store_true", help="force OpenCV decoding")
    g = ap.add_argument_group("posecam")
    g.add_argument("--stride", type=int, default=2)
    g = ap.add_argument_group("ply")
    g.add_argument("--metric", action="store_true", help="the imported splat is in metres")
    g.add_argument("--up-hint", type=_vec3, help="approx. up vector in the PLY frame, x,y,z")
    g = ap.add_argument_group("sensor fusion")
    g.add_argument("--range-csv", help="timestamp_ns,range_m[,u,v] from a ToF/ultrasonic logger")
    g.add_argument("--range-offset-ns", type=int, default=0, help="add to sensor timestamps")
    g.add_argument("--range-sync-frame", type=int,
                   help="tap sync: first reading < 0.15 m happened at this frame index")
    g.add_argument("--range-max-dt-ms", type=float, default=40.0)
    g.add_argument("--frame-period-ms", type=float,
                   help="captures without real frame timestamps (e.g. the sample, whose poses.txt "
                        "stamps are frame indices): frame i is at i * period")
    g.add_argument("--arcore-depth", help="folder of ARCore depth pngs named like rgb/ frames")
    a = ap.parse_args(argv)

    kind = a.kind or detect_kind(a.input)
    base = os.path.splitext(a.input.rstrip("/\\"))[0]
    out = a.out or base + ("_fused" if kind == "capture" else "_capture")
    print(f"[ingest] {kind}: {a.input} -> {out}")
    if kind == "capture" and os.path.abspath(out) != os.path.abspath(a.input):
        import shutil  # never rescale the input capture in place
        shutil.copytree(a.input, out, dirs_exist_ok=True)

    from . import fusion
    if kind == "ply":
        from .splat_import import import_splat
        import_splat(a.input, out, metric=a.metric or None, up_hint=a.up_hint)
        return out
    if kind == "posecam":
        from .posecam import ingest_posecam
        ingest_posecam(a.input, out, stride=a.stride, size=a.size, max_frames=a.max_frames,
                       depth_model=a.depth_model or "none", range_csv=a.range_csv,
                       range_offset_ns=a.range_offset_ns)
        return out  # posecam path runs its own fusion (depth only; poses already metric)
    if kind == "video":
        from .video import ingest_video
        ingest_video(a.input, out, fps=a.fps, oversample=a.oversample, size=a.size,
                     blur_rel=a.blur_rel, max_frames=a.max_frames, poses=a.poses,
                     f_rel=a.f_rel, model=a.depth_model or "relative",
                     start_ns=a.video_start_ns, use_ffmpeg=False if a.no_ffmpeg else None)

    # fusion for video / existing captures
    if a.frame_period_ms:
        n = len(cm.frame_names(out))
        cm.save_timestamps(out, (np.arange(n) * a.frame_period_ms * 1e6).astype(np.int64))
    if a.range_csv or a.arcore_depth:
        if not os.path.isdir(os.path.join(out, "depth")):
            raise SystemExit("[ingest] sensor fusion needs depth/ in the capture")
        arc = None
        if a.arcore_depth:
            arc = {n: os.path.join(a.arcore_depth, os.path.splitext(n)[0] + ".png")
                   for n in cm.frame_names(out)}
            arc = {k: v for k, v in arc.items() if os.path.exists(v)}
        meta = cm.load_meta(out)
        # relative poses came from the same depth -> scale them too; ARCore poses are metric
        scale_poses = meta.get("poses") not in ("arcore",)
        fusion.fuse(out, range_csv=a.range_csv, arcore_depths=arc, offset_ns=a.range_offset_ns,
                    sync_frame=a.range_sync_frame, scale_poses=scale_poses,
                    max_dt_ms=a.range_max_dt_ms)
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
