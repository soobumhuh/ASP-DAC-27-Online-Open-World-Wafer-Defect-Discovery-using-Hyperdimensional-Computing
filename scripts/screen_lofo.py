#!/usr/bin/env python3
"""Fairer screen of a model variant on all four tuning folds (folds 1-4; fold 0 never touched).

  1. maps of inner splits 1-4 (train on three of folds 1-4, validate on the fourth) for each seed; seed ensemble
  2. the variant's own extraction setting: Optuna (--trials) on the mean validation PI-CAI over the 4 inner splits
  3. fixed re-scorer (epochs 50, alpha 0.25): for inner split v trained on the candidates of the other three inner
     splits (each out-of-sample for its own model)
  4. per inner split: lesion AP, patient AUROC (max re-scored candidate), PI-CAI, and lesion recall (share of expert
     lesions hit by any candidate); mean over the 4 splits -> <out>/screen_lofo.json

  uv run python scripts/screen_lofo.py --config results/screen/foc/config.yaml --trials 50
"""
import argparse
import json

import numpy as np

import _path  # noqa: F401
from mri_hdc import data as D, evaluate as E, extraction as X
from mri_hdc.pipeline import build_maps, ensemble_maps, load_config, out_path
from mri_hdc.rescorer import Rescorer

INNER = ["inner1", "inner2", "inner3", "inner4"]


def suggest(trial):
    c = trial.suggest_categorical
    return dict(sxy=c("sxy", [0.0, 1.0, 2.0]), sz=c("sz", [0.0, 0.5, 1.0, 2.0]), q=c("q", [95.0, 97.0, 99.0, 99.5]),
                tf=c("tf", [0.25, 0.5, 1.0, 2.0]), mode=c("mode", ["dynamic", "dynamic-fast"]), mapping="sigmoid",
                conf=c("conf", ["max", "top10pct"]), factor=c("factor", [1.5, 2.0, 2.5, 3.0]),
                minvox=c("minvox", [10, 25, 50]), nles=c("nles", [3, 5, 10]), adj=c("adj", [True, False]), gamma=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["ensemble_seeds"] = [int(s) for s in args.seeds.split(",")]
    build_maps(cfg, INNER)
    w = cfg["run"]["workers"]
    mp = {n: ensemble_maps(cfg, n)["val"] for n in INNER}

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        pp = suggest(trial)
        return float(np.mean([E.picai(E.detection_maps(X.extract(mp[n], mp[n], pp, None, w)), mp[n], workers=w)["picai_score"]
                              for n in INNER]))

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0, multivariate=True))
    st.enqueue_trial({k: v for k, v in cfg["extraction"].items() if k not in ("mapping", "gamma")})
    st.optimize(objective, n_trials=args.trials, catch=(Exception,))
    pp = {**cfg["extraction"], **st.best_params}
    d = D.load_slices(cfg["paths"]["slices"])
    ext = {n: X.extract(mp[n], mp[n], pp, d, w) for n in INNER}
    rows = {}
    for v in INNER:
        rs = Rescorer(seed=0, epochs=50).fit([c for g in INNER if g != v for c in X.flat(ext[g])])
        conf = rs.confidences(ext[v], 0.25)
        keep = {}
        r = E.picai(E.detection_maps(ext[v], conf), mp[v], workers=w, keep=keep)
        lr = keep["lesion_results"]
        hits = [t == 1 and c > 0 for v_ in lr.values() for t, c, o in v_ if t == 1]
        r["recall"] = float(np.mean(hits)) if hits else float("nan")
        r["n_lesions"] = len(hits)
        rows[v] = r
        print(v, {k: round(x, 4) if isinstance(x, float) else x for k, x in r.items()}, flush=True)
    mean = {k: float(np.mean([r[k] for r in rows.values()])) for k in ("patient_auroc", "lesion_ap", "picai_score", "recall")}
    res = {"extraction": pp, "extraction_search_best": st.best_value, "per_split": rows, "mean": mean}
    json.dump(res, open(out_path(cfg, "screen_lofo.json"), "w"), indent=2)
    print("MEAN", {k: round(v, 4) for k, v in mean.items()}, flush=True)


if __name__ == "__main__":
    main()
