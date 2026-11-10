#!/usr/bin/env python3
"""
error_analysis.py - why is lesion AP low? Lesion-level error analysis of saved maps (postproc_search.py maps).

Uses the inner validation folds 1 and 2 only (models trained on the other three of folds 1-4), never fold 0.
Maps go through the current detection step (postproc_search.DEFAULT_PP). For every expert lesion
(3D connected component of the lesion mask) and every extracted candidate:
  lesions     hit (IoU >= 0.1 with some candidate, picai_eval's rule), missed (no candidate overlaps),
              low-IoU (overlap but IoU < 0.1), split into several candidates, merged / oversized candidate
  candidates  size vs lesion size for hits; false positives (no lesion with IoU >= 0.1) by zone at the peak
              (PZ / TZ / unlabelled) and slice level; in csPCa-negative vs positive cases
  ranking     AUROC of candidate confidence, true vs false positives (AP depends on this ranking)

  uv run python error_analysis.py --models t119,intensity
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import ndimage
from scipy.special import expit
from sklearn.metrics import roc_auc_score

import postproc_search as pp

ZN = {0: "unlabelled", 1: "PZ", 2: "TZ"}


def candidates(m, off, temp, p):
    from report_guided_annotation import extract_lesion_candidates
    prob = expit((np.nan_to_num(m.astype(np.float32), nan=-1e3) - off) / temp).astype(np.float64)
    det, conf, idx = extract_lesion_candidates(prob, threshold=p["mode"], min_voxels_detection=p["minvox"],
                                               num_lesions_to_extract=p["nles"], dynamic_threshold_factor=p["factor"],
                                               remove_adjacent_lesion_candidates=p["adj"])
    return det, dict(conf), idx, prob


def analyse(maps_val: dict, p: dict):
    ref = np.concatenate([c["map"][~np.isnan(c["map"])].astype(np.float32).ravel()
                          for c in maps_val.values() if c["cspca"] == 0])
    off, temp = float(np.percentile(ref, p["q"])), float(p["tf"] * ref.std() + 1e-6)
    les_rows, cand_rows = [], []
    for case, c in maps_val.items():
        det, conf, idx, prob = candidates(c["map"], off, temp, p)
        gt, ngt = ndimage.label(c["lesion"] > 0)
        cands = [k for k in conf]
        best_iou_of_cand = {k: 0.0 for k in cands}
        for g in range(1, ngt + 1):
            G = gt == g
            ov = [k for k in cands if (G & (idx == k)).any()]
            ious = {k: float((G & (idx == k)).sum() / (G | (idx == k)).sum()) for k in ov}
            for k, v in ious.items():
                best_iou_of_cand[k] = max(best_iou_of_cand[k], v)
            best = max(ious.values()) if ious else 0.0
            size_l = int(G.sum())
            if not ov:
                kind = "missed"
            elif best >= 0.1:
                kind = "hit"
            elif len(ov) > 1:
                kind = "split"
            else:
                k = ov[0]
                kind = "oversized/merged" if (idx == k).sum() > 2 * size_l else "low-IoU (offset or undersized)"
            kb = max(ious, key=ious.get) if ious else None
            les_rows.append({"case": int(case), "kind": kind, "iou": best, "lesion_vox": size_l,
                             "cand_vox": int((idx == kb).sum()) if kb is not None else 0,
                             "zone": ZN[int(np.bincount(c["zone"][G].ravel(), minlength=3).argmax())],
                             "peak_prob_in_lesion": float(prob[G].max())})
        for k in cands:
            K = idx == k
            pk = np.unravel_index(np.argmax(np.where(K, prob, -1)), K.shape)
            cand_rows.append({"case": int(case), "cspca": int(c["cspca"]), "conf": float(conf[k]), "vox": int(K.sum()),
                              "tp": bool(best_iou_of_cand[k] >= 0.1), "zone": ZN[int(c["zone"][pk])],
                              "level": int(min(c["rel_z"][pk[0]] * 3, 2))})
    return les_rows, cand_rows


def summarise(les, cand):
    from collections import Counter
    out = {"lesions": len(les), "lesion_outcome": dict(Counter(r["kind"] for r in les))}
    hits = [r for r in les if r["kind"] == "hit"]
    if hits:
        out["hit_size_ratio_median"] = float(np.median([r["cand_vox"] / r["lesion_vox"] for r in hits]))
    lo = [r for r in les if r["kind"] not in ("hit", "missed")]
    if lo:
        out["lowiou_size_ratio_median"] = float(np.median([r["cand_vox"] / max(r["lesion_vox"], 1) for r in lo]))
    out["lesion_vox_median"] = float(np.median([r["lesion_vox"] for r in les]))
    out["missed_by_zone"] = dict(Counter(r["zone"] for r in les if r["kind"] == "missed"))
    out["lesions_by_zone"] = dict(Counter(r["zone"] for r in les))
    fp = [r for r in cand if not r["tp"]]
    tp = [r for r in cand if r["tp"]]
    out["candidates"] = len(cand)
    out["tp_candidates"] = len(tp)
    out["fp_candidates"] = len(fp)
    out["fp_in_negative_cases"] = sum(1 for r in fp if r["cspca"] == 0)
    out["fp_by_zone"] = dict(Counter(r["zone"] for r in fp))
    out["fp_by_level(0=apex..2=base)"] = dict(Counter(r["level"] for r in fp))
    out["cand_vox_median_tp_fp"] = [float(np.median([r["vox"] for r in tp])) if tp else None,
                                    float(np.median([r["vox"] for r in fp])) if fp else None]
    if tp and fp:
        out["confidence_auroc_tp_vs_fp"] = float(roc_auc_score([1] * len(tp) + [0] * len(fp),
                                                               [r["conf"] for r in tp] + [r["conf"] for r in fp]))
    return out


def overlays(model, maps_val, p, path, n=6):
    """A few cases with lesions: map, candidates (outline) and expert lesion (green), on the largest-lesion slice."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ref = np.concatenate([c["map"][~np.isnan(c["map"])].astype(np.float32).ravel()
                          for c in maps_val.values() if c["cspca"] == 0])
    off, temp = float(np.percentile(ref, p["q"])), float(p["tf"] * ref.std() + 1e-6)
    pos = sorted([k for k, c in maps_val.items() if c["lesion"].any()])[:n]
    fig, axes = plt.subplots(1, len(pos), figsize=(2.6 * len(pos), 2.9))
    for ax, k in zip(np.atleast_1d(axes), pos):
        c = maps_val[k]
        det, conf, idx, prob = candidates(c["map"], off, temp, p)
        j = int(c["lesion"].reshape(len(c["lesion"]), -1).sum(1).argmax())
        ax.imshow(np.nan_to_num(prob[j], nan=0), cmap="magma", vmin=0, vmax=1)
        ax.contour(c["lesion"][j], levels=[0.5], colors="lime", linewidths=1)
        if idx[j].any():
            ax.contour(idx[j] > 0, levels=[0.5], colors="cyan", linewidths=0.8, linestyles="--")
        ax.set_title(f"{model} {k} z{j}", fontsize=7)
        ax.axis("off")
    fig.suptitle("probability map, expert lesion (green), extracted candidates (cyan dashed)", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="t119,intensity")
    args = ap.parse_args()
    out_dir = os.path.join(pp.ROOT, "error_analysis")
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for model in args.models.split(","):
        les, cand = [], []
        for r in ("inner1", "inner2"):
            mv = pp.load(r, model)["val"]
            a, b = analyse(mv, pp.DEFAULT_PP)
            les += a; cand += b
            if r == "inner1":
                overlays(model, mv, pp.DEFAULT_PP, os.path.join(out_dir, f"overlays_{model}.png"))
        summary[model] = summarise(les, cand)
        json.dump({"lesions": les, "candidates": cand}, open(os.path.join(out_dir, f"rows_{model}.json"), "w"))
        print(model, json.dumps(summary[model], indent=1))
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
