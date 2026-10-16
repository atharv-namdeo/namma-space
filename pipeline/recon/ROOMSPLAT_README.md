# RoomSplat 🏠✨

**Turn a phone video of a room into a photorealistic, walk-through 3D model — on your own machine.**

RoomSplat is an open-source, local alternative to apps like Scaniverse/Polycam. It does the same job
(video → **3D Gaussian Splat**) using only free, open tools, and ships a browser viewer so anyone can
walk through the result with no install.

> **Why not just reverse-engineer Scaniverse?** No need. Scaniverse's core is **3D Gaussian Splatting**
> — published research (INRIA, 2023), not a company secret. RoomSplat uses open implementations of the
> exact same technique, so it's legal to share and modify.

Built for the **Techfest "Namma Space"** competition (Round 1: 3D reconstruction + web roaming).

---

## The pipeline

```
 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │  Phone video │──▶│   Frames     │──▶│   COLMAP     │──▶│    Brush     │──▶│  Web viewer  │
 │  of the room │   │  (ffmpeg)    │   │  camera poses│   │  trains 3DGS │   │  walk through│
 └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
      input           extract_frames      run_pipeline        run_pipeline       viewer/
      data/*.mp4        → frames/           → sparse/          → output/          index.html
                                                                scene.ply
   [ your phone ]      [ CPU, fast ]      [ CPU, minutes ]    [ GPU, Vulkan ]    [ any browser ]
```

| Stage | Tool | Runs on | Output |
|-------|------|---------|--------|
| 1. Extract frames | **ffmpeg** | CPU | `frames/*.jpg` |
| 2. Camera poses (SfM) | **COLMAP** | CPU/GPU | `sparse/0/` |
| 3. Train the splat | **Brush** (Rust + wgpu, no CUDA) | GPU (Vulkan/DX12) | `output/scene.ply` |
| 4. View & roam | **RoomSplat viewer** (three.js) | Any browser | — |

**Brush** is the key choice: it trains Gaussian splats on the GPU through **wgpu** (Vulkan/DX12/Metal),
so it needs **no CUDA and no PyTorch** — the usual pain on Windows and low-VRAM laptops.

---

## Quickstart (Windows)

1. **Install the tools** once — see [`INSTALL_WINDOWS.md`](INSTALL_WINDOWS.md) (Rust + Brush, COLMAP; ffmpeg you already have).
2. **Film a good room** — see [`CAPTURE_GUIDE.md`](CAPTURE_GUIDE.md). Put the video in `data/`.
3. **Run the whole pipeline:**
   ```powershell
   # extract frames from your video (every ~0.3s)
   python scripts\extract_frames.py data\myroom.mp4 --fps 3

   # COLMAP poses + Brush training -> output\scene.ply
   powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1 -Images frames -Out output
   ```
4. **Walk through it:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\serve_viewer.ps1
   ```
   Opens `http://localhost:8000` and auto-loads `output/scene.ply`. WASD + mouse to move.
   (Or just open `viewer/index.html` and drag the `.ply` onto it.)

**No GPU / want it reliable?** Use [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) — same
pipeline on a free Google Colab GPU. Hand it to a friend; it works on any laptop.

---

## What's in here

```
roomsplat/
├── README.md               ← you are here
├── INSTALL_WINDOWS.md       install Rust + Brush + COLMAP
├── CAPTURE_GUIDE.md         how to film a room that reconstructs well
├── scripts/
│   ├── extract_frames.py    video → frames (ffmpeg)
│   ├── run_pipeline.ps1     COLMAP poses → Brush training → scene.ply
│   └── serve_viewer.ps1     local web server for the viewer
├── viewer/
│   └── index.html           first-person splat walkthrough (three.js)
├── notebooks/
│   └── train_colab.ipynb    cloud-GPU fallback (any laptop)
├── data/                    put your room video here
└── output/                  scene.ply lands here
```

## Sharing it

Zip the whole `roomsplat/` folder and send it. Your friend follows `INSTALL_WINDOWS.md`
(or just opens the Colab notebook). The `viewer/` works standalone — drag any `.ply`/`.ksplat` splat onto it.

## Hardware notes

- Brush runs on **any Vulkan/DX12 GPU**. A **4 GB** card (e.g. RTX 3050) works for **small/medium rooms**;
  cap the splat count if you hit out-of-memory, or use the Colab notebook for big scenes.
- COLMAP's feature matching is CPU-heavy but fine on a laptop for a few hundred frames.

## License

MIT — see [`LICENSE`](LICENSE). Uses COLMAP (BSD), Brush (Apache-2.0), three.js (MIT). 3D Gaussian
Splatting method © INRIA/MPII (research). RoomSplat contains no Scaniverse code.
