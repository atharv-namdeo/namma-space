"""Pluggable splat trainers. All take a COLMAP dataset dir and return a .ply path.

  brush  - shells out to Brush (Rust + wgpu; Vulkan/DX12, no CUDA, no torch):
           brush_app <dataset> --total-steps N --export-path <dir> --export-name <file>
           (flag names follow Brush's CLI; override the executable with --brush-exe /
           $BRUSH_EXE and add flags with --trainer-args).
  gsplat - writes a shell script for a CUDA box / Colab running gsplat's simple_trainer
           and returns None (run it there, then rerun with --trainer none --ply <result>).
  none   - no training; use an existing ply (--ply), e.g. trained earlier or on Colab.

Nothing here imports torch.
"""
from __future__ import annotations

import glob
import os
import shlex
import shutil
import subprocess


def train(trainer, dataset, work, steps=6000, ply=None, brush_exe=None, extra_args=""):
    if trainer == "none":
        if not ply or not os.path.exists(ply):
            raise SystemExit("[train] --trainer none needs --ply <existing splat .ply>")
        return ply
    if trainer == "brush":
        return train_brush(dataset, work, steps, brush_exe, extra_args)
    if trainer == "gsplat":
        write_gsplat_script(dataset, work, steps, extra_args)
        return None
    raise SystemExit(f"[train] unknown trainer {trainer}")


def _find_brush(brush_exe=None):
    for c in (brush_exe, os.environ.get("BRUSH_EXE"), "brush_app", "brush"):
        if c and (shutil.which(c) or os.path.exists(c)):
            return shutil.which(c) or c
    return None


def train_brush(dataset, work, steps, brush_exe=None, extra_args=""):
    exe = _find_brush(brush_exe)
    out_dir = os.path.join(work, "brush_out")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [exe or "brush_app", os.path.abspath(dataset), "--total-steps", str(steps),
           "--export-every", str(steps), "--export-path", os.path.abspath(out_dir),
           "--export-name", "export_{iter}.ply", *shlex.split(extra_args)]
    if exe is None:
        raise SystemExit("[train] Brush not found (PATH, $BRUSH_EXE or --brush-exe). Dataset is "
                         f"ready; run:\n    {subprocess.list2cmdline(cmd)}\n"
                         "then rerun build_scene with --trainer none --ply <exported ply>.")
    print("[train] " + subprocess.list2cmdline(cmd))
    subprocess.run(cmd, check=True)
    plys = sorted(glob.glob(os.path.join(out_dir, "*.ply")), key=os.path.getmtime)
    if not plys:
        raise SystemExit(f"[train] Brush finished but wrote no .ply in {out_dir}")
    return plys[-1]


def write_gsplat_script(dataset, work, steps, extra_args=""):
    os.makedirs(work, exist_ok=True)
    sh = os.path.join(work, "train_gsplat.sh")
    with open(sh, "w", newline="\n") as f:
        f.write(f"""#!/usr/bin/env bash
# Run on a CUDA machine / Colab (needs torch + gsplat). Upload {os.path.basename(dataset)}/ next to this.
set -e
pip -q install gsplat
[ -d gsplat ] || git clone --depth 1 https://github.com/nerfstudio-project/gsplat.git
pip -q install -r gsplat/examples/requirements.txt
python gsplat/examples/simple_trainer.py default \\
    --data_dir {os.path.basename(dataset)} --data_factor 1 \\
    --result_dir gsplat_out --max_steps {steps} --save_ply --ply_steps {steps} {extra_args}
ls gsplat_out/ply/   # download the .ply, then on the laptop:
# python -m pipeline.recon.build_scene <capture> --id <id> --trainer none --ply <downloaded.ply>
""")
    print(f"[train] gsplat is a GPU/Colab path. Wrote {sh}; dataset: {dataset}")
    return sh
