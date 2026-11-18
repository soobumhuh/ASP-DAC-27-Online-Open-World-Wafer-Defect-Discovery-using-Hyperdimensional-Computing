#!/usr/bin/env python3
"""Build the maps an AUROC-only run needs (fold0-4 and the 20 nested splits) for one config, then the seed
ensembles. python scripts/auroc_maps.py --config configs/auroc/triad_s3.yaml"""
import argparse

import _path  # noqa: F401
from mri_hdc.pipeline import build_maps, ensemble_maps, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
a = ap.parse_args()
cfg = load_config(a.config)
names = [f"fold{f}" for f in range(5)] + [f"nest{f}{g}" for f in range(5) for g in range(5) if g != f]
build_maps(cfg, names)
for n in names:
    ensemble_maps(cfg, n)
print("maps done", flush=True)
