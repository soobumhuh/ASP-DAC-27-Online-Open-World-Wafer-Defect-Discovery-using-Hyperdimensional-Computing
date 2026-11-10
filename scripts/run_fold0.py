#!/usr/bin/env python3
"""The held-out fold-0 test of the full pipeline (models trained on folds 1-4).

  uv run python scripts/run_fold0.py --config configs/best.yaml

The re-scorer of fold 0 is trained on the out-of-sample candidates of folds 1-4 (their CV test candidates), so the
maps of all five folds are built (or reused) as in run_cv.py; only fold 0 is scored. Writes <paths.out>/fold0.json.
"""
import argparse

import _path  # noqa: F401
from mri_hdc.pipeline import load_config, run_cv

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    run_cv(load_config(ap.parse_args().config), folds=(0,))
