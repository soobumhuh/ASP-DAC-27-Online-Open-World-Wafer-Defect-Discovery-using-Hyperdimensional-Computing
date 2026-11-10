#!/usr/bin/env python3
"""Tuning searches; only folds 1-4 are used (fold 0 stays untouched).

  uv run python scripts/tune.py --config configs/best.yaml extraction --trials 200
      Optuna (TPE) over the map -> candidate step on the inner splits (train on folds 1-4 minus v, validate on fold
      v = 1, 2): objective = mean validation PI-CAI of the seed-ensemble maps.
  uv run python scripts/tune.py --config configs/best.yaml rescorer
      re-scorer epochs x alpha by leave-one-fold-out over folds 1-4 (train on three folds' CV test candidates,
      validate on the fourth); needs the fold maps (run_cv.py builds them).
  uv run python scripts/tune.py --config configs/best.yaml merge
      clinical merge weight w on the fold-0 split's validation patients (all from folds 1-4), un-re-scored image score.
Each prints the best setting and writes <paths.out>/tune_<stage>.json; copy it into the config by hand.
"""
import argparse
import json
import os

import numpy as np

import _path  # noqa: F401
from mri_hdc import data as D, evaluate as E, extraction as X, patient as PT
from mri_hdc.pipeline import build_maps, ensemble_maps, load_config, out_path, splits
from mri_hdc.rescorer import Rescorer


def suggest(trial):
    c = trial.suggest_categorical
    return dict(sxy=c("sxy", [0.0, 1.0, 2.0, 3.0]), sz=c("sz", [0.0, 0.5, 1.0, 2.0]),
                q=c("q", [90.0, 95.0, 97.0, 99.0, 99.5, 99.9]), tf=c("tf", [0.1, 0.25, 0.5, 1.0, 2.0]),
                mode=c("mode", ["dynamic", "dynamic-fast", "0.3", "0.5", "0.7", "0.9"]),
                mapping=c("mapping", ["sigmoid", "rank"]), conf=c("conf", ["max", "mean", "top10pct"]),
                factor=c("factor", [1.5, 2.0, 2.5, 3.0, 4.0, 6.0]), minvox=c("minvox", [10, 25, 50, 100, 200]),
                nles=c("nles", [1, 2, 3, 5, 10]), adj=c("adj", [True, False]), gamma=c("gamma", [0.0, 0.5, 1.0, 2.0]))


def tune_extraction(cfg, trials, seed):
    import optuna
    build_maps(cfg, ["inner1", "inner2"])
    inner = {n: ensemble_maps(cfg, n)["val"] for n in ("inner1", "inner2")}
    w = cfg["run"]["workers"]

    def score(pp):
        r = [E.picai(E.detection_maps(X.extract(inner[n], inner[n], pp, None, w)), inner[n], workers=w) for n in inner]
        return {k: float(np.mean([x[k] for x in r])) for k in r[0]}

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True))
    study.enqueue_trial(dict(cfg["extraction"]))  # the current setting first

    def objective(trial):
        r = score(suggest(trial))
        trial.set_user_attr("lesion_ap", r["lesion_ap"]); trial.set_user_attr("patient_auroc", r["patient_auroc"])
        return r["picai_score"]

    study.optimize(objective, n_trials=trials, catch=(Exception,))
    best = study.best_trial
    return {"best": best.params, "inner_val_picai": best.value, **best.user_attrs,
            "current_inner_val_picai": study.trials[0].value}


def tune_rescorer(cfg):
    build_maps(cfg, [f"fold{f}" for f in range(1, 5)])
    d = D.load_slices(cfg["paths"]["slices"])
    w = cfg["run"]["workers"]
    mp = {f: ensemble_maps(cfg, f"fold{f}") for f in range(1, 5)}
    ext = {f: X.extract(mp[f]["test"], mp[f]["val"], cfg["extraction"], d, w) for f in mp}
    res = {}
    for ep in (5, 20, 50):
        for a in (0.0, 0.1, 0.25, 0.5, 0.75):
            r = []
            for v in mp:
                rs = Rescorer(epochs=ep).fit([c for g in mp if g != v for c in X.flat(ext[g])])
                r.append(E.picai(E.detection_maps(ext[v], rs.confidences(ext[v], a)), mp[v]["test"], workers=w))
            res[f"{ep}/{a}"] = {k: float(np.mean([x[k] for x in r])) for k in r[0]}
            print(ep, a, res[f"{ep}/{a}"], flush=True)
    best = max(res, key=lambda k: (res[k]["picai_score"], -float(k.split("/")[1])))
    return {"best": {"epochs": int(best.split("/")[0]), "alpha": float(best.split("/")[1])}, "all": res}


def tune_merge(cfg):
    from sklearn.metrics import roc_auc_score
    sp = splits(cfg)["fold0"]
    build_maps(cfg, ["fold0"])
    val = ensemble_maps(cfg, "fold0")["val"]
    d = np.load(cfg["paths"]["slices"], mmap_mode="r")
    pid = {int(s): int(p) for s, p in zip(d["study_id"], d["patient_id"])}
    yv = {int(s): int(c) for s, c in zip(d["study_id"], d["case_cspca"])}
    groups = {k: set(int(p) for p in sp[k]) for k in ("train", "val", "test")}
    cases = {k: sorted(s for s in pid if pid[s] in groups[k]) for k in groups}
    y = {k: [yv[c] for c in cases[k]] for k in cases}
    rows = PT.read_marksheet(os.path.join(cfg["paths"]["labels_dir"], "clinical_information", "marksheet.csv"))
    clin = PT.clinical_scores(rows, D.scanner_info(cfg["paths"]["images_dir"]), cases, y, seed=0)
    ev = X.extract(val, val, cfg["extraction"], None, cfg["run"]["workers"])
    img = [max([c["conf"] for c in ev[k][0]], default=0.0) for k in cases["val"]]
    grid = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
    auc = {g: float(roc_auc_score(y["val"], PT.merge(img, img, clin["val"], clin["val"], g))) for g in grid}
    return {"best": {"merge_weight": max(grid, key=lambda g: (auc[g], -g))}, "val_auroc": auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("stage", choices=["extraction", "rescorer", "merge"])
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    r = {"extraction": lambda: tune_extraction(cfg, args.trials, args.seed), "rescorer": lambda: tune_rescorer(cfg),
         "merge": lambda: tune_merge(cfg)}[args.stage]()
    json.dump(r, open(out_path(cfg, f"tune_{args.stage}.json"), "w"), indent=2)
    print(json.dumps(r["best"]))


if __name__ == "__main__":
    main()
