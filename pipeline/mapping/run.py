"""End-to-end: video -> frames -> depth+odometry+fusion -> 3D viewer.

Usage:
    python src/run.py --video data/clip.mp4
    python src/run.py --video data/clip.mp4 --stride 2 --max-frames 120 --far 6
"""
import argparse
import os

from extract_frames import extract
from reconstruct import run as reconstruct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--out", default="output")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--f-rel", type=float, default=0.85)
    ap.add_argument("--near", type=float, default=0.3)
    ap.add_argument("--far", type=float, default=8.0)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--max-step", type=float, default=0.5,
                    help="reject odometry jumps larger than this (anti-teleport)")
    ap.add_argument("--no-view", action="store_true")
    args = ap.parse_args()

    extract(args.video, args.frames, args.stride, args.size, args.max_frames)
    reconstruct(args.frames, args.out, None, args.f_rel, args.near, args.far,
                args.voxel, 0, args.max_step, view=not args.no_view)


if __name__ == "__main__":
    main()
