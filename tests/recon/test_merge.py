"""Merge invariance: gaussians outside region+margin are byte-identical after a refine.

The 'patch' here is a STAND-IN (jittered/recoloured subset of the base), not a trained
splat - no GPU in CI."""
import json
import os

import numpy as np

from conftest import make_splat
from pipeline.recon import geometry as geo, merge as mg, plyio, standin_patch


def _check_invariance(base, merged, to_box, lo, hi, margin):
    inside, band, _ = mg.classify(base, to_box, lo, hi, margin)
    far = ~inside & ~band
    kept = ~inside
    n_kept = int(kept.sum())
    # kept base records come first, in original order; far ones are untouched bytes
    assert merged.data[:n_kept][far[kept]].tobytes() == base.data[far].tobytes()
    # band records changed only in opacity, and only downwards
    bb = merged.data[:n_kept][band[kept]]
    ob = base.data[band]
    for nm in base.names:
        if nm != "opacity":
            assert np.array_equal(bb[nm], ob[nm])
    assert np.all(bb["opacity"] <= ob["opacity"] + 1e-6)
    # nothing from the patch lies outside region+margin
    pm = merged.subset(np.arange(n_kept, len(merged)))
    p = geo.apply(to_box, plyio.xyz(pm))
    assert np.all(mg.outside_distance(p, lo, hi) < margin + 1e-5)
    return far


def test_merge_invariance_synthetic():
    base = make_splat(20000, seed=1)
    lo, hi, margin = np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5]), 0.25
    patch = make_splat(5000, seed=2, box=((-1, -1, -1), (1, 1, 1)))
    merged, st = mg.merge(base, patch, np.eye(4), lo, hi, margin)
    far = _check_invariance(base, merged, np.eye(4), lo, hi, margin)
    assert st["untouched"] == far.sum() and st["removed"] > 0 and st["added"] > 0
    assert st["total"] == st["base"] - st["removed"] + st["added"]


def test_merge_invariance_sample_world_region(sample_ply):
    sj = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "web", "scenes",
                                     "cliproom", "scene.json")))
    T = np.array(sj["toWorld"]).reshape(4, 4)
    lo, hi, margin = np.array([-0.6, 0.0, -3.0]), np.array([0.6, 0.6, -1.5]), 0.2
    patch = standin_patch.standin(sample_ply, T, lo, hi, margin)
    merged, st = mg.merge(sample_ply, patch, T, lo, hi, margin)
    _check_invariance(sample_ply, merged, T, lo, hi, margin)
    assert st["removed"] > 1000


def test_refine_cli_file_level_invariance(tmp_path):
    """Full refine CLI on a tiny scene (register skip, stand-in patch): the bytes of every
    far-away record in the new scene.ply equal those in the backup scene.v1.ply."""
    from pipeline.capture import common as cm
    from pipeline.recon import refine
    scene = tmp_path / "scenes" / "tiny"
    scene.mkdir(parents=True)
    base = make_splat(5000, seed=3)
    plyio.write_ply(str(scene / "scene.ply"), base)
    T = np.eye(4)
    (scene / "scene.json").write_text(json.dumps({"id": "tiny", "splat": "scene.ply",
                                                  "toWorld": T.reshape(-1).tolist(), "refinements": []}))
    cap = tmp_path / "recap"
    cm.ensure_dirs(str(cap), depth=False)
    import cv2
    for i in range(3):
        cv2.imwrite(str(cap / "rgb" / f"{i:05d}.jpg"), np.zeros((8, 8, 3), np.uint8))
    cm.save_poses(str(cap), range(3), np.repeat(np.eye(4)[None], 3, 0))
    cm.save_intrinsics(str(cap), 8, 8, 4, 4, 8, 8)
    cm.update_meta(str(cap), base_frame="tiny")
    lo, hi, margin = np.array([-0.4, -0.3, -0.4]), np.array([0.4, 0.3, 0.4]), 0.2
    patch = standin_patch.standin(base, T, lo, hi, margin)
    plyio.write_ply(str(tmp_path / "patch.ply"), patch)
    rec = refine.main(["--scene", str(scene), "--capture", str(cap), "--region=-0.4,-0.3,-0.4,0.4,0.3,0.4",
                       "--margin", "0.2", "--trainer", "none", "--patch", str(tmp_path / "patch.ply"),
                       "--work", str(tmp_path / "work")])
    assert rec["registration"]["method"].startswith("skip")
    old = plyio.read_ply(str(scene / "scene.v1.ply"))
    assert (scene / "scene.v1.ply").read_bytes() == plyio_bytes(base, tmp_path)
    new = plyio.read_ply(str(scene / "scene.ply"))
    _check_invariance(old, new, T, lo, hi, margin)
    sj = json.loads((scene / "scene.json").read_text())
    assert sj["refinements"][-1]["id"] == "r1" and sj["gaussians"] == len(new)
    # header preserved verbatim apart from the count
    assert (scene / "scene.ply").read_bytes()[:30] == (scene / "scene.v1.ply").read_bytes()[:30]


def plyio_bytes(p, tmp_path):
    f = tmp_path / "_tmp.ply"
    plyio.write_ply(str(f), p)
    return f.read_bytes()
