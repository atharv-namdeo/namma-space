"""Extract frames from a video for COLMAP / Brush.

Usage:
    python scripts/extract_frames.py data/myroom.mp4 --fps 3
    python scripts/extract_frames.py data/myroom.mp4 --fps 2 --max 300 --out frames

Samples the video at a fixed rate, writes JPGs into <out>/ (default: ./frames).
Uses ffmpeg (on PATH, or the winget install location on Windows).
"""
import argparse
import os
import shutil
import subprocess
import sys


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    # common Windows winget location
    base = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    if os.path.isdir(base):
        for root, _dirs, files in os.walk(base):
            if "ffmpeg.exe" in files:
                return os.path.join(root, "ffmpeg.exe")
    return None


def main():
    ap = argparse.ArgumentParser(description="Extract frames from a video with ffmpeg.")
    ap.add_argument("video", help="input video file (e.g. data/myroom.mp4)")
    ap.add_argument("--fps", type=float, default=3.0, help="frames per second to sample (default 3)")
    ap.add_argument("--out", default="frames", help="output folder (default: frames)")
    ap.add_argument("--max", type=int, default=0, help="optional cap on number of frames (0 = no cap)")
    ap.add_argument("--width", type=int, default=1600, help="downscale longest side to this (0 = keep)")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        sys.exit(f"[extract] video not found: {args.video}")
    ff = find_ffmpeg()
    if not ff:
        sys.exit("[extract] ffmpeg not found. Install it:  winget install Gyan.FFmpeg")

    os.makedirs(args.out, exist_ok=True)
    # clear old frames
    for f in os.listdir(args.out):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            os.remove(os.path.join(args.out, f))

    vf = [f"fps={args.fps}"]
    if args.width and args.width > 0:
        # scale longest side to width, keep aspect, even dims
        vf.append(f"scale='if(gt(iw,ih),{args.width},-2)':'if(gt(iw,ih),-2,{args.width})'")
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-i", args.video,
           "-vf", ",".join(vf), "-q:v", "2",
           os.path.join(args.out, "frame_%05d.jpg")]
    print("[extract] running ffmpeg ...")
    subprocess.run(cmd, check=True)

    frames = sorted(f for f in os.listdir(args.out) if f.lower().endswith(".jpg"))
    if args.max and len(frames) > args.max:
        # evenly thin to <= max
        keep = set(frames[i] for i in
                   [round(i * (len(frames) - 1) / (args.max - 1)) for i in range(args.max)])
        for f in frames:
            if f not in keep:
                os.remove(os.path.join(args.out, f))
        frames = sorted(keep)

    print(f"[extract] wrote {len(frames)} frames to {args.out}/")
    if len(frames) < 20:
        print("[extract] WARNING: very few frames — raise --fps or film a longer clip.")
    elif len(frames) > 400:
        print("[extract] NOTE: >400 frames; COLMAP matching will be slow. Consider --max 300.")


if __name__ == "__main__":
    main()
