# Namma Space — architecture and data contract

Product: phone video of an indoor space becomes a photorealistic, navigable, labelled 3D
digital twin of a building, explorable on desktop, phone and VR. Techfest IIT Bombay 2026-27.

This file is the contract every module builds against. If you need to change a schema,
change it here in the same commit and keep old fields readable.

## The five stages

| Stage | What | Code | Output |
|---|---|---|---|
| 1 Capture | Phone video/photos, optional sensor data (ARCore poses + depth, IMU, external ToF/ultrasonic) | `pipeline/capture/` | a **capture folder** |
| 2 Reconstruct + navigate | Capture folder to Gaussian splat; **iterative refinement**: re-shoot one region, rebuild only that region, merge without touching the rest | `pipeline/recon/` | `web/scenes/<id>/scene.ply` + `scene.json` |
| 3 Label | Open-vocabulary detection on capture frames, lifted to 3D with poses + depth, deduplicated | `pipeline/labeling/` | `web/scenes/<id>/labels.json` |
| 4 Building map | Per-scene floor/occupancy grid, room segmentation, hallway + rooms graph, pathfinding | `pipeline/building/` (reuses `pipeline/mapping/` = monomap) | `web/scenes/<id>/navgrid.json`, `web/scenes/building.json` |
| 5 Explore / VR twin | Web app: walk, search POIs, building overview, click a room to enter, route to a POI, WebXR VR with interactable objects | `web/` | static site (GitHub Pages) |

## Coordinate conventions

- **Capture/camera frame**: OpenCV (x right, y down, z forward). `poses.npy` is `(N,4,4)` camera→world.
- **Scene frame**: whatever frame the splat was trained in (for `cliproom`, the monomap world frame, which equals the first camera's OpenCV frame).
- **World/building frame**: three.js convention: right-handed, **Y up**, metres. `scene.json.toWorld` (row-major 4x4) maps scene-frame points into this frame; it includes the gravity/floor alignment, scale-to-metres, and the room's placement in the building. Floor of the ground storey is y = 0.
- All JSON files store positions in the **world/building frame** unless the field name ends in `_scene`.

## Capture folder (stage 1 output, stage 2/3/4 input)

```
<capture>/
  rgb/00000.jpg ...            # frames, any resolution
  depth/00000.png ...          # optional, uint16 millimetres, same size as rgb
  poses.npy                    # (N,4,4) float, camera->world, OpenCV convention
  poses.txt                    # TUM format copy (timestamp tx ty tz qx qy qz qw)
  intrinsics.json              # {fx, fy, cx, cy, width, height}
  imu.csv                      # optional
  sensors/                     # optional extra range sensors, see pipeline/capture
  meta.json                    # optional {source, metric: bool, gravity: [x,y,z] in world}
```
Sample: `data/sample/clip1/` (215 frames 640x360, relative-scale monocular depth, poses from monomap).
`pipeline/recon/cliproom_colmap_sparse/` is the same poses as a COLMAP text model (used to train the sample splat with Brush).

## web/scenes/<id>/scene.json

```json
{
  "id": "cliproom",
  "name": "Room A",
  "splat": "scene.ply",
  "toWorld": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
  "bounds": {"min": [x,y,z], "max": [x,y,z]},
  "floorY": 0.0,
  "spawn": {"position": [x,y,z], "yaw": 0.0},
  "metric": false,
  "source": "monomap clip1 + Brush 6000 steps",
  "refinements": [{"id": "r1", "region": {"min": [...], "max": [...]}, "date": "2026-10-20"}]
}
```

## web/scenes/<id>/labels.json

```json
{
  "sceneId": "cliproom",
  "model": "owlv2-base-patch16-ensemble",
  "objects": [{
    "id": "obj-001",
    "label": "chair",
    "category": "furniture",
    "score": 0.62,
    "center": [x,y,z],
    "bbox": {"min": [x,y,z], "max": [x,y,z]},
    "views": [12, 13, 40],
    "thumbnail": "thumbs/obj-001.jpg",
    "interact": {"type": "info|toggle|grab", "description": "..."}
  }]
}
```

## web/scenes/<id>/navgrid.json

```json
{
  "sceneId": "cliproom",
  "cellSize": 0.1,
  "origin": [x0, z0],
  "width": 120, "height": 80,
  "floorY": 0.0,
  "encoding": "rle",
  "data": [[value, runLength], ...]
}
```
Row-major over z then x: cell `(i,j)` covers x in `[x0+i*cs, x0+(i+1)*cs)`, z in `[z0+j*cs, ...)`. Values: 0 free, 1 blocked, 2 unknown.

## web/scenes/building.json

```json
{
  "name": "Demo building",
  "floors": [{"id": "G", "name": "Ground", "elevation": 0.0}],
  "spaces": [
    {"id": "hall", "kind": "hallway", "name": "Main hallway", "floor": "G",
     "sceneId": null, "polygon": [[x,z], ...], "status": "scanned|planned"},
    {"id": "room-a", "kind": "room", "name": "Room A", "floor": "G",
     "sceneId": "cliproom", "polygon": [[x,z], ...], "status": "scanned"}
  ],
  "doors": [{"id": "d1", "between": ["hall", "room-a"], "position": [x,y,z]}],
  "scenes": {"cliproom": "scenes/cliproom/scene.json"}
}
```
A space whose `sceneId` is null has no scan yet: the web app draws it as a floor-plan extrusion.

## Rules for contributors

- Python 3.11+, CPU-runnable where at all possible (the owner's laptop blocks torch, so local runs need onnxruntime/numpy/opencv only; torch is fine for cloud/Colab scripts).
- Every pipeline module has a CLI (`python -m pipeline.<stage>.<tool> --help`) and a README with a one-command example on `data/sample/clip1`.
- The web app is a static site: no build step, ES modules from jsdelivr via an import map, works from GitHub Pages and from `python -m http.server`.
- Keep files under 50 MB each.
