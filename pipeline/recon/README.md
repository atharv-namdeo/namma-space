# Stage 2: reconstruction and iterative refinement (`pipeline/recon`)

Torch-free on the laptop. Training is delegated to Brush (local, Vulkan, no CUDA) or to
gsplat on Colab.

## build_scene: capture → `web/scenes/<id>/scene.ply` + `scene.json`

```bash
# the committed sample: splat already trained (Brush, 6000 steps) -> just align + publish
python -m pipeline.recon.build_scene data/sample/clip1 --id cliproom --name "Room A" \
    --trainer none --ply web/scenes/cliproom/scene.ply --source "monomap clip1 + Brush 6000 steps"

# a new capture, trained locally with Brush
python -m pipeline.recon.build_scene data/captures/roomB --id roomB --trainer brush --steps 6000

# no local GPU: write the COLMAP dataset + a gsplat script, train on Colab, then:
python -m pipeline.recon.build_scene data/captures/roomB --id roomB --trainer gsplat
python -m pipeline.recon.build_scene data/captures/roomB --id roomB --trainer none --ply roomB_gsplat.ply
```

Steps:
1. **COLMAP text model** in `build/recon/<id>/dataset/{images,sparse/0}`. Camera→world poses
   are inverted to world→camera with qvec in wxyz order, and `points3D.txt` comes from
   back-projected depth (voxel-deduplicated, at most 100k points). On the sample, the generated
   `images.txt`/`cameras.txt` are **byte-identical** to `cliproom_colmap_sparse/0`, which is
   what the committed splat was trained from.
2. **Trainer** (`trainers.py`): `brush` runs `brush_app <dataset> --total-steps N
   --export-path ... --export-name ...` (`--brush-exe`/`$BRUSH_EXE`, extra flags via
   `--trainer-args`); `gsplat` writes `train_gsplat.sh` for a CUDA box; `none` uses `--ply`.
3. **toWorld** (`geometry.py`):
   - Floor: RANSAC on opacity-filtered (≥0.3), floater-trimmed gaussian centres. The plane
     normal must be within 35° of the cameras' mean up vector (or `meta.gravity` /
     `--up-hint`), and at least 90% of the camera centres must lie above it. Then
     least-squares refits with a shrinking band.
   - Rotation: floor normal → +Y, first camera's forward → −Z (so the spawn yaw is 0), right-handed.
   - Translation: floor → y = 0, first camera → x = z = 0.
   - Scale: 1 when `meta.json` says metric, or `--scale S`. `--assume-eye-height H` rescales a
     relative capture so the median phone height is H; it is a heuristic and leaves
     `metric:false`. Otherwise the scene stays in relative units with `metric:false`.
   - `bounds` covers the 0.5-99.5 percentile of gaussian centres in world. `spawn` is at the
     first camera, at `--eye-height` (1.6) above the floor when metric, otherwise at the first
     camera's own height. Its `yaw` uses the three.js convention, where `rotation.y = yaw`
     looks along (−sin yaw, 0, −cos yaw).

`scene.json` follows the contract and adds `capture` (base capture path, used by refine),
`gaussians`, `alignment` (floor plane in the scene frame, inliers, residual, cameraHeight,
scale source) and `built`.

### cliproom result (committed)
`toWorld` (row-major):
```
[ 0.996586 -0.082565  0.000000  0.000000]
[-0.078620 -0.948975 -0.305394  0.312850]
[ 0.025215  0.304351 -0.952226  0.000000]
[ 0         0         0         1       ]
```
That is about 18° of pitch plus a Y flip (the scene frame is OpenCV y-down), with the floor
0.313 units below the first camera. The sample is a **relative-scale** monocular capture
(`metric:false`), and its depth comes from a non-linear disparity mapping. The splat is
therefore projectively stretched: the floor is planar within about 3 units of the trajectory,
and far geometry fans out in y (see `bounds`).

## refine: re-shoot a region and merge it in

```bash
python -m pipeline.recon.refine --scene web/scenes/cliproom --capture data/captures/cliproom_fix1 \
    --region=-0.6,0,-3,0.6,0.6,-1.5 --frame world --margin 0.2 --trainer brush
```
(Write `--region=...` with `=` when the first number is negative.)

1. **Register** (`register.py`). With `--register auto`, registration is skipped if the new
   capture's `meta.base_frame` equals the scene id; otherwise it runs **features**: SIFT on 12
   new frames against every 3rd base frame, ratio test, top-3 base frames per query,
   fundamental-matrix RANSAC, both sides lifted to 3D with their own depth and pose, then a
   **RANSAC Umeyama Sim3** that includes scale. `--icp` adds trimmed point-to-point Sim3 ICP.
   The registered copy goes to `build/refine/<id>/r<N>/registered`.
2. **Patch training** from the new frames only. The dataset's `points3D` holds the base
   gaussians in region+margin (centre and DC colour) plus the new depth points there. Use
   `--trainer brush|gsplat`, or `--trainer none --patch <ply in the scene frame>`.
3. **Merge** (`merge.py`):
   - base gaussians inside the box are removed;
   - those within `--margin` outside it are faded (opacity × d/m);
   - patch gaussians inside are inserted, and those in the band are faded (opacity × (1 − d/m));
   - the rest of the patch is dropped;
   - **base records outside region+margin are byte-identical and keep their order**;
   - the patch is conformed to the base's property layout, with SH degree remapped per channel.

   The old file is saved as `scene.v<N>.ply` (gitignored), and a record with the region,
   registration Sim3, stats and backup name is appended to `scene.json.refinements`.

Without a GPU, use a **stand-in patch** (the base subset in the box, jittered and tinted red;
this is not a reconstruction) to exercise the whole flow:
```bash
python -m pipeline.recon.standin_patch --scene web/scenes/cliproom --region=-0.6,0,-3,0.6,0.6,-1.5 --out build/patch.ply
python -m pipeline.recon.refine --scene web/scenes/cliproom --capture data/sample/clip1 --register skip \
    --region=-0.6,0,-3,0.6,0.6,-1.5 --trainer none --patch build/patch.ply
git checkout web/scenes/cliproom   # undo the demo
```

## Files
`build_scene.py` · `refine.py` · `merge.py` · `register.py` · `geometry.py` (Sim3, RANSAC,
floor, toWorld) · `colmap_io.py` · `plyio.py` (exact PLY IO) · `trainers.py` ·
`standin_patch.py`. The older RoomSplat material (`ROOMSPLAT_README.md`, `run_pipeline.ps1`,
`train_colab.ipynb`, `INSTALL_WINDOWS.md`) still describes the plain COLMAP + Brush route.

Tests: `python -m pytest tests/recon` covers PLY round-trip exactness, merge invariance,
COLMAP round-trip, floor alignment on the sample, scale solve, PoseCam/video/ply ingest and
Sim3 registration. It runs on CPU in about 15 s.
