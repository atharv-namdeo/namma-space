"""Stage 2 - Monocular depth with Depth Anything V2.

Supports two kinds of model:
  * "relative"  - affine-invariant inverse depth (Depth-Anything-V2-Small). Needs a
                  near/far mapping to become metres; scale is arbitrary.
  * "indoor" / "outdoor" - METRIC variants (Depth-Anything-V2-Metric-*-Small) that
                  output depth in real metres directly. Small enough for a 4 GB GPU.

`infer_depth()` returns depth in METRES for every kind, so callers don't care which
model is loaded. (`infer()` still returns relative disparity for back-compat.)

Note: Video Depth Anything (temporal consistency) needs ~6.8 GB VRAM -> it lives in
the Colab path, not the 4 GB local GPU. The metric model is the local upgrade.

Usage:
    python src/depth.py --frames data/frames --out output/depth --model indoor
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODELS = {
    "relative": "depth-anything/Depth-Anything-V2-Small-hf",
    "indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
}


class DepthEstimator:
    def __init__(self, model="relative", near=0.3, far=8.0, device=None, half=True):
        self.kind = model
        self.metric = model != "relative"
        self.near, self.far = near, far
        model_id = MODELS.get(model, model)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = half and self.device == "cuda"
        print(f"[depth] loading {model_id} on {self.device} "
              f"(half={self.half}, metric={self.metric})")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_id, torch_dtype=torch.float16 if self.half else torch.float32
        ).to(self.device).eval()

    @torch.inference_mode()
    def _raw(self, pil_img):
        """Model's predicted_depth resized to the image size (float32, not normalised)."""
        w, h = pil_img.size
        inputs = self.processor(images=pil_img, return_tensors="pt")
        pv = inputs["pixel_values"].to(self.device)
        if self.half:
            pv = pv.half()
        pred = self.model(pixel_values=pv).predicted_depth
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1).float(), size=(h, w), mode="bicubic", align_corners=False
        ).squeeze()
        return pred.cpu().numpy().astype(np.float32)

    def infer(self, pil_img):
        """Relative models: normalised disparity [0,1] (1=near). (back-compat)"""
        d = self._raw(pil_img)
        lo, hi = np.percentile(d, 1), np.percentile(d, 99)
        return np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)

    def infer_depth(self, pil_img):
        """Depth in METRES. Metric models: direct. Relative: via near/far mapping."""
        if self.metric:
            # Metric model already outputs metres; clamp absurd values.
            return np.clip(self._raw(pil_img), 0.0, self.far * 4).astype(np.float32)
        disp = self.infer(pil_img)
        inv = disp * (1.0 / self.near - 1.0 / self.far) + 1.0 / self.far
        return (1.0 / inv).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--out", default="output/depth")
    ap.add_argument("--model", default="relative", choices=list(MODELS))
    ap.add_argument("--near", type=float, default=0.3)
    ap.add_argument("--far", type=float, default=8.0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    est = DepthEstimator(model=args.model, near=args.near, far=args.far,
                         device="cpu" if args.cpu else None)
    files = sorted(glob.glob(os.path.join(args.frames, "*.jpg")) +
                   glob.glob(os.path.join(args.frames, "*.png")))
    if not files:
        raise SystemExit(f"No frames in {args.frames}")
    for i, f in enumerate(files):
        d = est.infer_depth(Image.open(f).convert("RGB"))   # metres
        np.save(os.path.join(args.out, f"{i:05d}.npy"), d)
        print(f"[depth] {i+1}/{len(files)} {os.path.basename(f)}")
    print(f"[depth] done ({args.model}, metres) -> {args.out}")


if __name__ == "__main__":
    main()
