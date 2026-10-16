"""Stage 1 - Extract frames from a recorded video.

Pulls frames at a fixed stride, resizes so the longest side == --size,
and writes them as zero-padded JPEGs into an output folder.

Usage:
    python src/extract_frames.py --video data/clip.mp4 --out data/frames --stride 3 --size 640
"""
import argparse
import os
import cv2


def extract(video, out_dir, stride=3, size=640, max_frames=0):
    os.makedirs(out_dir, exist_ok=True)
    # wipe old frames so stale runs don't leak in
    for f in os.listdir(out_dir):
        if f.lower().endswith((".jpg", ".png")):
            os.remove(os.path.join(out_dir, f))

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[extract] {video}  frames={total}  fps={fps:.1f}  stride={stride}")

    kept, idx = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            h, w = frame.shape[:2]
            scale = size / max(h, w)
            if scale < 1.0:
                frame = cv2.resize(frame, (round(w * scale), round(h * scale)),
                                   interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_dir, f"{kept:05d}.jpg"), frame)
            kept += 1
            if max_frames and kept >= max_frames:
                break
        idx += 1

    cap.release()
    print(f"[extract] wrote {kept} frames -> {out_dir}")
    return kept


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="data/frames")
    ap.add_argument("--stride", type=int, default=3,
                    help="keep every Nth frame (3 ~= 10fps from 30fps video)")
    ap.add_argument("--size", type=int, default=640, help="longest side in pixels")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()
    extract(args.video, args.out, args.stride, args.size, args.max_frames)
