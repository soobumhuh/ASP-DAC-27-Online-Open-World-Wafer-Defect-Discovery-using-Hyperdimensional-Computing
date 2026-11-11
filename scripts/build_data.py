#!/usr/bin/env python3
"""Build the slice file and the ResNet-18 layer-2 feature cache named in the config.

  uv run python scripts/build_data.py --config configs/best.yaml            # slices (5 folds + merge) + features
  uv run python scripts/build_data.py --config configs/best.yaml --features_only

The five public PI-CAI folds are built from their Zenodo file lists (paths.fold_lists, '{f}' = fold) into
paths.fold_slices, two at a time, then merged into paths.slices with a 'fold' key. Existing files are kept.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import _path  # noqa: F401
from mri_hdc import data as D, features as Fe
from mri_hdc.pipeline import load_config


def build_fold(cfg, f):
    p = cfg["paths"]
    out = p["fold_slices"].format(f=f)
    if not os.path.exists(out):
        D.build_slices(p["images_dir"], p["labels_dir"], out, size=cfg["data"]["size"],
                       patient_list=p["fold_lists"].format(f=f), spacing_mm=cfg["data"].get("spacing_mm"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--features_only", action="store_true")
    ap.add_argument("--device", default="cpu", help="ResNet forward pass: cpu (as published) or cuda")
    args = ap.parse_args()
    cfg = load_config(args.config)
    p = cfg["paths"]
    if not args.features_only and not os.path.exists(p["slices"]):
        with ProcessPoolExecutor(cfg["run"]["parallel_jobs"]) as ex:
            parts = list(ex.map(build_fold, [cfg] * 5, range(5)))
        D.merge_folds(parts, p["slices"])
    if not os.path.exists(p["feature_cache"]):
        d = D.load_slices(p["slices"])
        Fe.resnet_l2_maps(d["images"], p["feature_cache"], device=args.device)
        print("saved", p["feature_cache"])


if __name__ == "__main__":
    main()
