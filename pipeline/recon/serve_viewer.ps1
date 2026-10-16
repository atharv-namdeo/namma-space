<#
  Serve the RoomSplat viewer locally so it can auto-load output\scene.ply.
  (Opening index.html directly via file:// can't fetch the .ply — a tiny web server fixes that.)

  Usage:
    powershell -ExecutionPolicy Bypass -File scripts\serve_viewer.ps1 [-Port 8000]
#>
param([int]$Port = 8000)

$root = Split-Path $PSScriptRoot -Parent          # the roomsplat/ folder
$ply  = Join-Path $root "output\scene.ply"

# expose output\scene.ply to the viewer as viewer\scene.ply (auto-load target)
if (Test-Path $ply) {
  Copy-Item $ply (Join-Path $root "viewer\scene.ply") -Force
  Write-Host "[roomsplat] linked output\scene.ply -> viewer\scene.ply (auto-loads)" -ForegroundColor Cyan
} else {
  Write-Host "[roomsplat] no output\scene.ply yet — the viewer will show its load panel (drag a .ply in)." -ForegroundColor Yellow
}

$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { Write-Host "[roomsplat] Python not found; just open viewer\index.html and drag your .ply onto it." -ForegroundColor Yellow; exit 0 }

Set-Location (Join-Path $root "viewer")
Start-Process "http://localhost:$Port/index.html"
Write-Host "[roomsplat] serving viewer at http://localhost:$Port/  (Ctrl+C to stop)" -ForegroundColor Green
& $py -m http.server $Port
