# Stage 1: capture (`pipeline/capture`)

One CLI turns anything a phone produces into a **capture folder** in the format of
`docs/ARCHITECTURE.md`: `rgb/`, optional `depth/` (uint16 mm), `poses.npy`
(camera→world, OpenCV), `poses.txt` (TUM), `intrinsics.json`, plus `timestamps.txt`,
`imu.csv`, `sensors/` and `meta.json` when available.

```
python -m pipeline.capture.ingest <input> --out <capture> [options]
```

| Input (auto-detected) | Path | Torch? |
|---|---|---|
| `*.mp4/mov/mkv/...` | ffmpeg (or OpenCV) frames → sharpest of N + blur rejection → poses via `--poses monomap` (Depth Anything + Open3D RGB-D odometry, ported from `pipeline/mapping/make_capture.py`) or `--poses none` (frames only, for COLMAP) | monomap only |
| PoseCam RAW `*.zip` / folder with `poses.csv` | ARCore metric gravity-aligned poses (GL→CV), longest jump-free run, IMU copied, ARCore depth used if recorded (`--depth-model none`, default) | only with `--depth-model indoor/relative` |
| `*.ply` (Scaniverse/Polycam/Brush splat) | scene-only import → `scene/scene.ply` + meta (`--metric`, `--up-hint x,y,z`) | no |
| an existing capture folder | copy to `--out` (default `<input>_fused`) and run sensor fusion only | no |

**Sensor fusion** (`fusion.py`): `--range-csv` (`timestamp_ns,range_m[,u,v]`, VL53L1X ToF or
HC-SR04, with (u,v) defaulting to the image centre) and/or `--arcore-depth DIR`. Each reading is
paired with the nearest frame in time (±40 ms), converted from along-ray range to z-depth, and
compared to the capture's depth at that pixel. The scale is the robust median of ratios
(iterative MAD rejection in log space). Depth is rescaled, and so are pose translations for
relative-pose captures (not for ARCore poses, which are already metric). `meta.json` gets
`{metric: true, scale, scale_inliers, scale_samples, scale_log_spread, scale_sources}`.
Clock sync uses `--range-offset-ns` or `--range-sync-frame K` (the hand-tap method, see
`CAPTURE_SOP.md`).

## Examples on the sample (`data/sample/clip1`)

```bash
# the sample already is a capture: add metric scale from your range log (copies to --out; sample untouched)
python -m pipeline.capture.ingest data/sample/clip1 --out build/clip1_metric --frame-period-ms 66.7 \
    --range-csv my_range.csv --range-sync-frame 0

# a phone video -> frames only (no torch), poses later by COLMAP
python -m pipeline.capture.ingest walk.mp4 --out data/captures/walk --poses none --fps 4

# a phone video -> full capture with relative poses + depth (needs torch + open3d: Colab)
python -m pipeline.capture.ingest walk.mp4 --out data/captures/walk

# PoseCam raw session (metric), no torch
python -m pipeline.capture.ingest capture-2026-10-20.zip --out data/captures/roomB

# a Scaniverse splat
python -m pipeline.capture.ingest scan.ply --out data/captures/scaniverse_lab --metric --up-hint 0,1,0
```
Then run stage 2: `python -m pipeline.recon.build_scene <capture> --id <id>`.

## Files
- `ingest.py`: the CLI and input-kind detection
- `video.py`: frame extraction, sharpness selection, and the monomap pose backend (lazy torch/open3d import)
- `posecam.py`: PoseCam session/zip ingest (torch-free port of `pipeline/mapping/ingest_posecam.py`)
- `splat_import.py`: `.ply` scene-only import
- `fusion.py`: range CSV / ARCore depth → robust metric scale
- `common.py`: capture IO and quaternion maths (open3d-free mirror of `pipeline/mapping/capture_format.py`)
- `CAPTURE_SOP.md`: how to film (Nothing Phone 3a, rooms vs hallways, range sensor wiring and sync, re-capture for refinement)
- `BUILD_AND_CAPTURE.md`: building and using the PoseCam app

Tests: `python -m pytest tests/recon` (CPU, about 15 s).
