#!/usr/bin/env python3
"""Inner-split screen (folds 1-4 only): seed-ensemble maps of inner1 / inner2 (train on folds 1-4 minus v, validate
on fold v), one extraction setting (the config's, or --extraction <tune json>), mean validation PI-CAI / AP / AUROC.

  uv run python scripts/screen_inner.py configs/sota/base.yaml configs/sota/ens_rp.yaml [--extraction f.json]
"""
import argparse
import json

import numpy as np

import _path  # noqa: F401
from mri_hdc import evaluate as E, extraction as X
from mri_hdc.pipeline import ensemble_maps, load_config

ap = argparse.ArgumentParser()
ap.add_argument("configs", nargs="+")
ap.add_argument("--extraction", default="")
ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
for c in a.configs:
    cfg = load_config(c)
    pp = {**cfg["extraction"], **(json.load(open(a.extraction))["best"] if a.extraction else {})}
    r = []
    for n in ("inner1", "inner2"):
        m = ensemble_maps(cfg, n)["val"]
        r.append(E.picai(E.detection_maps(X.extract(m, m, pp, None, a.workers)), m, workers=a.workers))
    out = {k: round(float(np.mean([x[k] for x in r])), 4) for k in ("picai_score", "lesion_ap", "patient_auroc")}
    print(c, out, "per inner:", [round(x["picai_score"], 4) for x in r], flush=True)
