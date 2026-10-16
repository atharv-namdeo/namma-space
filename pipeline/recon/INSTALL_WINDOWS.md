# Install (Windows)

You need three tools. **ffmpeg** you already have; this covers **Rust + Brush** and **COLMAP**.

> ⚠️ **Smart App Control (SAC):** if Windows blocks an unsigned `.exe` (COLMAP, or your freshly
> built `brush.exe`) with *"An app control policy has blocked this file"*, you'll need to either
> allow that file, turn SAC off (Settings → Privacy & security → **Smart App Control**), or use the
> **Colab notebook** instead. This is the one gotcha on locked-down laptops.

---

## 1. ffmpeg  ✅ (already installed)

Check:
```powershell
ffmpeg -version
```
If missing: `winget install Gyan.FFmpeg`

---

## 2. Rust + Brush (the Gaussian-splat trainer)

Brush is Rust + wgpu — **no CUDA, no PyTorch**. You build it once.

**a) Install Rust** (signed installer, SAC-safe):
```powershell
winget install Rustlang.Rustup
# then open a NEW terminal so cargo is on PATH
rustc --version
```

**b) Build Brush:**
```powershell
cd $HOME\Desktop
git clone https://github.com/ArthurBrussee/brush
cd brush
cargo build --release
```
The first build downloads crates and takes ~10–20 min. The binary lands at:
```
brush\target\release\brush.exe
```
Add it to PATH, or pass it to the pipeline with `-BrushExe "C:\...\brush\target\release\brush.exe"`.

**c) Test it** (also a great standalone viewer):
```powershell
brush            # opens the Brush window; you can drag a .ply/.zip dataset in
```
Brush uses Vulkan/DX12 via wgpu — your RTX 3050 is fine. For a **4 GB** card, train **smaller
rooms** and keep frame counts modest; if it runs out of memory, use the Colab notebook.

---

## 3. COLMAP (camera poses / structure-from-motion)

Download the prebuilt Windows build from the releases page:
<https://github.com/colmap/colmap/releases>  → grab `colmap-x.x-windows-*.zip`
(the **-cuda** build is faster on NVIDIA; the plain build works everywhere).

```powershell
# extract, then make colmap visible to the pipeline, e.g.:
$env:Path += ";C:\Users\Atharv\Desktop\COLMAP-x.x-windows-cuda"
colmap --help
```
Or pass it explicitly: `-Colmap "C:\...\COLMAP\colmap.exe"`.

> If COLMAP's GPU SIFT errors, re-run the pipeline with `-UseGpuSift 0` (CPU — slower but reliable).

---

## 4. Run it

```powershell
cd $HOME\Desktop\roomsplat
python scripts\extract_frames.py data\myroom.mp4 --fps 3 --max 300
powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1 -Images frames -Out output `
    -Colmap "C:\path\to\colmap.exe" -BrushExe "C:\path\to\brush.exe"
powershell -ExecutionPolicy Bypass -File scripts\serve_viewer.ps1
```

Stuck on installs or SAC? Open `notebooks/train_colab.ipynb` — it runs the whole pipeline on a free
cloud GPU and gives you a `scene.ply` to drop into `viewer/index.html`.
