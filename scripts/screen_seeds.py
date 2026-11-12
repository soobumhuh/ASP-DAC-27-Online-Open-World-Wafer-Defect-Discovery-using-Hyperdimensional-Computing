#!/usr/bin/env python3
"""3-seed screen of a model variant on the inner splits (folds 1-4 only).

Builds the inner1 / inner2 maps of seeds 0, 1, 2 for the given config (run.parallel_jobs at a time; existing maps are
reused) and scores, with one fixed extraction setting, each seed and the seed ensemble (mean of the 3 maps):
mean PI-CAI, lesion AP and patient AUROC over the two inner validation folds.

  uv run python scripts/screen_seeds.py --config results/d2sweep/ssl3d/config.yaml --pp results/round6/pp_t110.json
"""
import argparse
import json

import numpy as np

import _path  # noqa: F401
from mri_hdc import evaluate as E, extraction as X, maps as M
from mri_hdc.pipeline import build_maps, ensemble_maps, load_config, map_path, out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pp", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = [int(s) for s in args.seeds.split(",")]
    cfg["ensemble_seeds"] = seeds
    build_maps(cfg, ["inner1", "inner2"], seeds)
    pp = {k: v for k, v in {**X.DEFAULT, **json.load(open(args.pp))}.items() if k in X.DEFAULT}
    w = cfg["run"]["workers"]

    def score(get):
        r = []
        for n in ("inner1", "inner2"):
            m = get(n)
            r.append(E.picai(E.detection_maps(X.extract(m, m, pp, None, w)), m, workers=w))
        return {k: float(np.mean([x[k] for x in r])) for k in r[0]}

    res = {f"seed{s}": score(lambda n, s=s: M.load(map_path(cfg, s, n))["val"]) for s in seeds}
    res["ensemble"] = score(lambda n: ensemble_maps(cfg, n)["val"])
    json.dump(res, open(out_path(cfg, "screen_seeds.json"), "w"), indent=2)
    for k, v in res.items():
        print(k, {a: round(b, 4) for a, b in v.items()}, flush=True)


if __name__ == "__main__":
    main()
