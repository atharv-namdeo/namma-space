# monomap — single-camera 3D mapping + path tracking

Record a video with **one camera**, and this rebuilds the **3D map of the area**
plus the **path the camera travelled**, shown together in an interactive 3D viewer.

Built to run on a **4 GB GPU (RTX 3050 Laptop) / 32 GB RAM / i5-12th** — offline,
record-then-reconstruct, relative scale.

## Why this stack (and not TAPNet)

TAPNet = *Tracking Any Point* — it follows individual pixels across a video. It does
**not** estimate depth, build maps, or track the camera path. This task is **monocular
dense SLAM**, so the pipeline is:

| Job | Tool | Where it runs |
|---|---|---|
| per-frame depth | **Depth Anything V2 (Small)** | GPU (tiny, fits 4 GB) |
| camera path (odometry) | **Open3D RGB-D odometry** | CPU |
| fuse the 3D map | **Open3D** point-cloud fusion | CPU / 32 GB RAM |
| show it in 3D | **Rerun** (map + animated path + timeline) | — |

The heavy lifting is on the CPU/RAM, leaving the 4 GB GPU free for just the depth net.
Pure-pip, no ORB-SLAM3 C++ build. (TAPNet/TAPIP3D is a future add-on if you later want
to track *moving objects* inside the map.)

## Install (one time)

Already set up in `venv/` with torch cu128 + open3d + rerun + transformers.
To recreate:
```
python -m venv venv
venv\Scripts\python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
venv\Scripts\python -m pip install open3d rerun-sdk transformers huggingface_hub opencv-python pillow tqdm
```

## Capture tips (matters a lot for quality)

- Walk **slowly**, keep the camera roughly level, avoid pure rotation-in-place.
- Lots of **texture** (furniture, posters) tracks well; blank walls drift.
- Good, even lighting. 10–30 seconds is plenty to start.
- Record on your phone, or stream via the **IP Webcam** app and save the clip.

## Run

End-to-end (video in, 3D viewer out):
```
venv\Scripts\python src\run.py --video data\clip.mp4
```

Useful knobs:
```
--stride 3        keep every Nth frame (lower = smoother but slower)
--max-frames 120  cap frame count (start small while tuning)
--far 6           far plane in relative metres (shrink for small rooms)
--f-rel 0.85      focal length as fraction of width (phone ~0.8–0.9)
--voxel 0.02      map resolution (smaller = denser, heavier)
--no-view         save output/recording.rrd instead of opening the viewer
```

Run stages individually (for debugging):
```
venv\Scripts\python src\extract_frames.py --video data\clip.mp4 --out data\frames
venv\Scripts\python src\depth.py --frames data\frames --out output\depth
venv\Scripts\python src\reconstruct.py --frames data\frames --depth-dir output\depth
```

## Outputs

- `output/map.ply` — the fused coloured 3D point cloud (open in MeshLab / CloudCompare)
- `output/trajectory.npy` — `(N,4,4)` camera-to-world poses (the path)
- `output/recording.rrd` — Rerun recording; reopen with `rerun output\recording.rrd`

## If results drift / look wrong

1. **Shrink `--far`** to match the real scene depth (e.g. 4 for a small room).
2. **Lower `--stride`** so consecutive frames overlap more (better odometry).
3. Tune **`--f-rel`** (0.7–1.0) to your phone's field of view.
4. Persistent drift on relative depth → upgrade the depth stage to *Video Depth
   Anything* (temporally consistent) — the biggest quality lever here.

---

# v2 — metric, CAD-grade, automation-ready (IMU / ARCore)

v1 gives a relative-scale point cloud. v2 adds **metric scale + gravity alignment**
(so the floor is level and distances are real metres) and four
automation-ready outputs. The key design: the **pose source is decoupled from the
reconstruction**, so an Android/ARCore phone now and a hardware IMU on the rover
later both feed the *same* engine.

### The common capture format (the contract)

Every source produces one folder:
```
capture/<name>/
  rgb/000000.jpg …        frames
  depth/000000.png        uint16 depth in millimetres (metric)
  poses.txt / poses.npy   camera→world, TUM + (N,4,4); gravity-aligned
  intrinsics.json         fx fy cx cy w h
```

### Pose sources (front-ends)

- **Android / ARCore (metric) — recommended.** Record with **[PoseCam](https://github.com/Krish-Kheni/PoseCam)**
  (any ARCore phone): it logs synced RGB + 6-DoF metric poses + raw IMU at 30 fps,
  on-device. Then:
  ```
  venv\Scripts\python src\run_pipeline.py --posecam data\posecam_scan1 --name scan1
  ```
  Poses are metric and gravity-aligned, so scale + level floor come for free — no
  custom VIO filter needed. (We convert ARCore's OpenGL camera frame to OpenCV,
  and scale the estimated depth to metres via triangulation from the known poses.)
- **Plain video (relative scale) — fallback / testing.** Uses the v1 odometry so
  you can exercise the whole output stack on any clip:
  ```
  venv\Scripts\python src\run_pipeline.py --video data\clip1.mp4 --name clip1 --stride 2
  ```
- **Hardware IMU on the rover (later).** Add a front-end that fuses the rover's
  IMU + camera into the same `capture/` format; the rest of the pipeline is unchanged.
  PoseCam already logs IMU, so the fusion code can be prototyped on phone data first.

### Engine + four outputs (identical for every source)

```
src/backbone.py    TSDF fusion (Open3D ScalableTSDFVolume)  → mesh + metric cloud
src/planes.py      RANSAC floor/walls/ceiling + box prims    → planes.json, boxes.json
src/occupancy.py   floor-aligned voxel grid + ROS costmap    → floormap.pgm/.yaml, occupancy_3d.npy
src/run_pipeline.py  orchestrates: source → backbone → planes → occupancy
```

Outputs land in `output/<name>_v2/`:
| File | What | Use |
|---|---|---|
| `mesh.obj` / `mesh.stl` / `mesh.ply` | watertight fused surface | CAD / Blender / slicer |
| `map_metric.ply` / `.pcd` + `poses.tum` | cloud + trajectory | robotics / SLAM tooling |
| `planes.json` / `boxes.json` / `primitives.ply` | fitted floor/walls + object boxes | parametric CAD, collision |
| `floormap.pgm`+`.yaml`, `occupancy_3d.npy` | 2D costmap + 3D voxel grid | ROS nav / path planning |

### Why TSDF instead of point splatting
TSDF fuses every frame's depth into a signed-distance volume, so overlapping views
*average out* noise and you get one clean watertight mesh — exactly what a splatted
point cloud can't give. It needs good poses + depth, which is why the ARCore path
matters.

### Depth models (`--depth-model`)
PoseCam phones have no depth sensor, so monomap estimates depth. Three choices:
| value | model | depth | fits 4 GB? |
|---|---|---|---|
| `relative` | Depth-Anything-V2-Small | affine-invariant (needs near/far) | yes |
| `indoor` | Depth-Anything-V2-**Metric**-Indoor-Small | **real metres**, best for rooms | yes |
| `outdoor` | Depth-Anything-V2-Metric-Outdoor-Small | real metres, outdoor scenes | yes |

`run_pipeline.py` defaults to `auto` → **indoor (metric)** for `--posecam`, `relative`
for `--video`. Metric depth is scale-consistent, so the pipeline then uses **TSDF**
automatically (clean mesh); relative depth falls back to point-fusion.

**Video Depth Anything** (temporal consistency across frames) needs ~6.8 GB VRAM, so
it does **not** fit the 4 GB RTX 3050 — it's wired into the **Colab** path instead,
where the T4 has the VRAM. The local metric model is the equivalent quality win that
fits here.

## Roadmap

- v1: offline, relative scale, point-cloud map + path. ✅
- **v2 (this): metric + gravity-aligned (ARCore/PoseCam), TSDF mesh (OBJ/STL),
  CAD planes/primitives, ROS occupancy/floorplan.** ✅ engine; ARCore path pending a scan.
- v2.1: swap depth → **Video Depth Anything** (temporally consistent) = the biggest
  remaining quality lever; pose-graph loop closure for long walks.
- v3 (rover): hardware-IMU front-end → tight visual-inertial fusion into `capture/`.
- optional: **MASt3R-SLAM** (8 GB+ GPU/cloud) front-end; **TAPIP3D** to track moving
  objects inside the map.
