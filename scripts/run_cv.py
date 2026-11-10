#!/usr/bin/env python3
"""Full pipeline, 5-fold CV (fold 0 = the held-out test fold, reported in the first row).

  uv run python scripts/run_cv.py --config configs/best.yaml

Builds any missing maps (ensemble_seeds x fold0..fold4, run.parallel_jobs at a time), then extraction, re-scorer and
patient merge with the settings in the config; writes <paths.out>/cv.json and cv.md.
"""
import argparse

import _path  # noqa: F401
from mri_hdc.pipeline import load_config, run_cv

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    run_cv(load_config(ap.parse_args().config))
