#!/usr/bin/env python3
"""Nested ("clean") 5-fold CV of the full pipeline (mri_hdc/clean.py).

  uv run python scripts/run_cv_clean.py --config configs/best_clean.yaml
  uv run python scripts/run_cv_clean.py --config configs/sota/x.yaml --tag _tuned --extraction results/sota/x/tune_extraction.json \
      --set rescorer.kind=knn rescorer.k_grid=[5,10,20]

Builds the fold maps and the 20 nested maps per seed (nest<f><g>), then per test fold chooses the re-scorer
settings and the clinical merge weight on the nested folds only; writes <paths.out>/cv_clean<tag>.json / .md / _cases.pkl.
--extraction: a tune.py result (its 'best' setting replaces cfg.extraction); --set: dotted config overrides (YAML values).
"""
import argparse
import json

import yaml

import _path  # noqa: F401
from mri_hdc.clean import run_cv_clean
from mri_hdc.pipeline import load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--extraction", default="")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config)
    cfg["_tag"] = a.tag
    if a.extraction:
        cfg["extraction"] = {**cfg["extraction"], **json.load(open(a.extraction))["best"]}
    for kv in a.set:
        k, v = kv.split("=", 1)
        node = cfg
        *path, last = k.split(".")
        for p in path:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(v)
    print("extraction:", cfg["extraction"], "| overrides:", a.set, flush=True)
    run_cv_clean(cfg)
