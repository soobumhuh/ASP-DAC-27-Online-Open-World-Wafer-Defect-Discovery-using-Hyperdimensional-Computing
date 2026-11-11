#!/usr/bin/env python3
"""Train and save the HDC patch memories of one split and seed (for inference on new exams).

  uv run python scripts/fit_model.py --config configs/best.yaml --split fold0 --seed 0

Saves <paths.out>/models/seed<s>_<split>.pkl. If the split's saved maps exist (maps/seed<s>/<split>.pkl), the
test maps of the saved model are recomputed and compared with them (they must be identical).
"""
import argparse
import os
import pickle

import numpy as np

import _path  # noqa: F401
from mri_hdc import data as D, features as Fe, maps as M
from mri_hdc.pipeline import load_config, map_path, out_path, splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="fold0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = M.prepare_slices(D.load_slices(cfg["paths"]["slices"]), cfg["model"])
    fmaps = Fe.resnet_l2_maps(d["images"], cfg["paths"]["feature_cache"])
    sp = splits(cfg)[args.split]
    model = M.PatchModel().fit(d, fmaps, sp, cfg["model"], args.seed, log=lambda s: print(s, flush=True))
    out = out_path(cfg, "models", f"seed{args.seed}_{args.split}.pkl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pickle.dump(model, open(out, "wb"), protocol=4)
    print("saved", out, f"{os.path.getsize(out) / 2**20:.0f} MiB", flush=True)
    ref = map_path(cfg, args.seed, args.split)
    if os.path.exists(ref):
        old = M.load(ref)["test"]
        new = model.maps(d, fmaps, M.split_indices(d, sp)["test"])
        same = sum(np.array_equal(np.nan_to_num(old[k]["map"]), np.nan_to_num(new[k]["map"])) for k in old)
        print(f"test maps identical to {ref}: {same} of {len(old)} cases")


if __name__ == "__main__":
    main()
