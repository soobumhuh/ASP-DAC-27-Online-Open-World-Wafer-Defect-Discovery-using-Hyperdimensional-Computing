#!/usr/bin/env python3
"""Nested ("clean") 5-fold CV of the full pipeline (mri_hdc/clean.py).

  uv run python scripts/run_cv_clean.py --config configs/best_clean.yaml

Builds the fold maps and the 20 nested maps per seed (nest<f><g>), then per test fold chooses the re-scorer
settings and the clinical merge weight on the nested folds only; writes <paths.out>/cv_clean.json / .md.
"""
import argparse

import _path  # noqa: F401
from mri_hdc.clean import run_cv_clean
from mri_hdc.pipeline import load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    run_cv_clean(load_config(ap.parse_args().config))
