"""Region merge of a patch splat into a base splat.

Given an axis-aligned box [bmin, bmax] in some *box frame* (world or scene; `to_box`
maps scene coords into it) and a feather margin m (box-frame units):

  base gaussians   centre inside the box        -> removed
                   centre within m outside it    -> kept, opacity * (dist / m)
                   further away                  -> untouched, byte-identical, same order
  patch gaussians  centre inside the box        -> inserted as-is
                   centre within m outside it    -> inserted, opacity * (1 - dist / m)
                   further away                  -> dropped

So base and patch cross-fade across the margin band and nothing outside region+margin
changes. Output = [kept base records in original order] + [patch records]; the base
PLY header (comments, property order, dtypes) is reused, and the patch is conformed to
it (SH degree remapped per channel, missing properties zero-filled).
"""
from __future__ import annotations

import numpy as np

from . import geometry as geo, plyio

EPS = 1e-6


def outside_distance(p, bmin, bmax):
    """Euclidean distance from points to the box (0 inside)."""
    q = np.maximum(np.maximum(bmin - p, 0.0), p - bmax)
    return np.linalg.norm(q, axis=1)


def _scale_opacity(logit, w):
    a = plyio.sigmoid(logit.astype(np.float64)) * np.clip(w, EPS, 1.0)
    a = np.clip(a, EPS, 1 - EPS)
    return np.log(a / (1 - a)).astype(logit.dtype)


def classify(ply, to_box, bmin, bmax, margin):
    p = geo.apply(to_box, plyio.xyz(ply))
    dist = outside_distance(p, np.asarray(bmin, float), np.asarray(bmax, float))
    inside = np.all((p >= bmin) & (p <= bmax), axis=1)
    band = ~inside & (dist < margin)
    return inside, band, dist


def merge(base, patch, to_box, bmin, bmax, margin=0.2):
    """-> (merged Ply, stats dict). `to_box`: 4x4 scene -> box frame."""
    bmin, bmax = np.asarray(bmin, float), np.asarray(bmax, float)
    b_in, b_band, b_dist = classify(base, to_box, bmin, bmax, margin)
    keep_b = ~b_in
    out_b = base.data[keep_b].copy()
    if margin > 0 and "opacity" in base.names:
        bi = np.flatnonzero(b_band[keep_b])
        out_b["opacity"][bi] = _scale_opacity(out_b["opacity"][bi], b_dist[keep_b][bi] / margin)

    patch = plyio.conform(patch, base)
    p_in, p_band, p_dist = classify(patch, to_box, bmin, bmax, margin)
    keep_p = p_in | p_band
    out_p = patch.data[keep_p].copy()
    if margin > 0 and "opacity" in base.names:
        pi = np.flatnonzero(p_band[keep_p])
        out_p["opacity"][pi] = _scale_opacity(out_p["opacity"][pi], 1.0 - p_dist[keep_p][pi] / margin)

    merged = plyio.Ply(np.concatenate([out_b, out_p]), list(base.comments), base.fmt,
                       base.header_lines, base.tail)
    stats = dict(base=len(base), removed=int(b_in.sum()), feathered=int(b_band.sum()),
                 untouched=int((~b_in & ~b_band).sum()), patch=len(patch),
                 added=int(keep_p.sum()), added_feathered=int(p_band.sum()),
                 total=len(merged))
    return merged, stats


def untouched_mask(base, to_box, bmin, bmax, margin):
    """Base records that must survive byte-identical."""
    b_in, b_band, _ = classify(base, to_box, np.asarray(bmin, float), np.asarray(bmax, float), margin)
    return ~b_in & ~b_band
