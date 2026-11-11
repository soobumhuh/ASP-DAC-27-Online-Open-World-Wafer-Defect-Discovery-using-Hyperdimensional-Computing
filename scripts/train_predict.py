#!/usr/bin/env python3
"""Train the HDC patch memories of one split with one projection seed and save the val / test maps.

  uv run python scripts/train_predict.py --config configs/best.yaml --split fold0 --seed 0
"""
import argparse
import os

import _path  # noqa: F401
from mri_hdc import data as D, features as Fe, maps as M
from mri_hdc.pipeline import load_config, map_path, splits


def feature_maps(cfg, d):
    """ResNet layer-2 cache: 'rgb' = T2W/ADC/HBV as R/G/B (default); 'slices3' = per sequence, slices z-1/z/z+1."""
    if cfg["model"].get("features", "rgb") == "slices3":
        import torch
        prev, nxt = Fe.neighbour_slices(d)
        return Fe.resnet_l2_slices_maps(d["images"], prev, nxt, cfg["paths"]["feature_cache"].replace(".npy", "_slices3.npy"),
                                        device="cuda" if torch.cuda.is_available() else "cpu")
    return Fe.resnet_l2_maps(d["images"], cfg["paths"]["feature_cache"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True, help="fold0 .. fold4, inner1, inner2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = map_path(cfg, args.seed, args.split)
    if os.path.exists(out):
        print("exists:", out)
        return
    d = M.prepare_slices(D.load_slices(cfg["paths"]["slices"]), cfg["model"])
    fmaps = feature_maps(cfg, d)
    maps = M.train_and_map(d, fmaps, splits(cfg)[args.split], cfg["model"], args.seed,
                           log=lambda s: print(s, flush=True))
    M.save(maps, out)
    print("saved", out)


if __name__ == "__main__":
    main()
