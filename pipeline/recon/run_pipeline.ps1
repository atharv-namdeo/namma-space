<#
  RoomSplat pipeline: frames -> COLMAP camera poses -> Brush Gaussian-splat training.

  Usage:
    powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1 -Images frames -Out output

  Params:
    -Images     folder of extracted frames (default: frames)
    -Out        output folder (default: output); COLMAP work goes in <Out>\colmap
    -Colmap     path to colmap.exe (default: "colmap" on PATH)
    -BrushExe   path to brush.exe (default: "brush" on PATH)
    -Steps      Brush training steps (default: 30000)
    -UseGpuSift 1 to let COLMAP use the GPU for SIFT (default 1; set 0 if it errors)
    -SkipColmap / -SkipBrush   skip a stage (e.g. re-train only)

  Needs COLMAP + Brush installed — see INSTALL_WINDOWS.md.
#>
param(
  [string]$Images = "frames",
  [string]$Out = "output",
  [string]$Colmap = "colmap",
  [string]$BrushExe = "brush",
  [int]$Steps = 30000,
  [int]$UseGpuSift = 1,
  [switch]$SkipColmap,
  [switch]$SkipBrush
)
$ErrorActionPreference = "Stop"
function Say($m){ Write-Host "[roomsplat] $m" -ForegroundColor Cyan }
function Die($m){ Write-Host "[roomsplat] ERROR: $m" -ForegroundColor Red; exit 1 }

$Images = (Resolve-Path $Images).Path
$work = Join-Path $Out "colmap"
$imgDir = Join-Path $work "images"
$db = Join-Path $work "database.db"
$sparse = Join-Path $work "sparse"
New-Item -ItemType Directory -Force -Path $Out, $work, $sparse | Out-Null

$nFrames = (Get-ChildItem $Images -Filter *.jpg -ErrorAction SilentlyContinue).Count
if ($nFrames -lt 10) { Die "only $nFrames frames in $Images — run extract_frames.py first." }
Say "$nFrames frames in $Images"

# ---------------- 1. COLMAP: structure-from-motion (camera poses) ----------------
if (-not $SkipColmap) {
  if (-not (Get-Command $Colmap -ErrorAction SilentlyContinue) -and -not (Test-Path $Colmap)) {
    Die "COLMAP not found. Install it (see INSTALL_WINDOWS.md) or pass -Colmap <path\to\colmap.exe>."
  }
  # COLMAP wants its own images/ folder
  New-Item -ItemType Directory -Force -Path $imgDir | Out-Null
  Copy-Item (Join-Path $Images "*.jpg") $imgDir -Force

  # GPU SIFT flags only exist in CUDA builds; the nocuda build rejects them.
  $gpuFeat = if ($UseGpuSift -eq 1) { @("--SiftExtraction.use_gpu","1") } else { @() }
  $gpuMatch = if ($UseGpuSift -eq 1) { @("--SiftMatching.use_gpu","1") } else { @() }

  Say "COLMAP 1/3  feature extraction"
  & $Colmap feature_extractor --database_path $db --image_path $imgDir `
      --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV @gpuFeat
  if ($LASTEXITCODE -ne 0) { Die "feature_extractor failed. If you have the nocuda COLMAP build, run with -UseGpuSift 0." }

  Say "COLMAP 2/3  feature matching (sequential — good for video frames)"
  & $Colmap sequential_matcher --database_path $db @gpuMatch
  if ($LASTEXITCODE -ne 0) { Die "matcher failed. With the nocuda build, run with -UseGpuSift 0." }

  Say "COLMAP 3/3  mapping (sparse reconstruction)"
  & $Colmap mapper --database_path $db --image_path $imgDir --output_path $sparse
  if ($LASTEXITCODE -ne 0) { Die "mapper failed." }

  if (-not (Test-Path (Join-Path $sparse "0\cameras.bin"))) {
    Die "COLMAP produced no reconstruction (sparse\0 missing). Usually means too few features — see CAPTURE_GUIDE.md."
  }
  Say "COLMAP done -> $sparse\0"
}

# ---------------- 2. Brush: train the Gaussian splat ----------------
# Brush reads a COLMAP dataset directory that contains images\ and sparse\0\.
$dataset = $work
$plyOut = Join-Path $Out "scene.ply"

if (-not $SkipBrush) {
  if (-not (Get-Command $BrushExe -ErrorAction SilentlyContinue) -and -not (Test-Path $BrushExe)) {
    Write-Host ""
    Say "Brush not found on PATH. COLMAP dataset is ready at:"
    Write-Host "    $dataset" -ForegroundColor Yellow
    Say "Install Brush (INSTALL_WINDOWS.md), then either:"
    Write-Host "    brush `"$dataset`" --total-steps $Steps --export-path `"$plyOut`"" -ForegroundColor Yellow
    Write-Host "    (or open that folder in the Brush GUI and Export .ply when training looks good)"
    exit 0
  }
  Say "Brush training ($Steps steps) — a Vulkan window will open; let it converge"
  # NOTE: flags follow Brush's CLI; if they differ in your build, run `brush --help`.
  & $BrushExe "$dataset" --total-steps $Steps --export-path "$plyOut"
  if ($LASTEXITCODE -ne 0) {
    Say "CLI train returned non-zero. Fallback: open `"$dataset`" in the Brush GUI and Export .ply."
    exit 1
  }
  if (Test-Path $plyOut) { Say "DONE -> $plyOut  (open the viewer to walk through it)" }
}
