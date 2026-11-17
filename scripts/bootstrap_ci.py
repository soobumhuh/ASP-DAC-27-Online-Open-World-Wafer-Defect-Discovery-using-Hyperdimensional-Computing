#!/usr/bin/env python3
"""Paired bootstrap confidence intervals over exams for PI-CAI score, lesion AP and patient AUROC.

Reads the per-exam results that run_cv.py saves (<out>/cv_cases.pkl: picai_eval lesion lists and case scores per
stage and fold). Each bootstrap draw resamples exams with replacement (within each fold, the same draw for every
model, so differences are paired); metrics are recomputed with picai_eval's own Metrics class.

  uv run python scripts/bootstrap_ci.py --runs pick=results/review/pick:merged t110=results/review/t110:base \
      --ref pick --n 1000
"""
import argparse
import json
import pickle

import numpy as np
from picai_eval.metrics import Metrics

import _path  # noqa: F401


def metrics(K, folds, draws):
    """K[(stage, fold)] -> one pooled Metrics over the drawn exams of the given folds."""
    lr, ct, cp = {}, {}, {}
    for f, idx in zip(folds, draws):
        keys = sorted(K[f]["case_target"])
        for j, i in enumerate(idx):
            k = keys[i]
            n = f"{f}_{j}"
            lr[n], ct[n], cp[n] = K[f]["lesion_results"][k], K[f]["case_target"][k], K[f]["case_pred"][k]
    m = Metrics(lesion_results=lr, case_target=ct, case_pred=cp)
    return np.array([m.auroc, m.AP, (m.auroc + m.AP) / 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=out_dir:stage or name=file.pkl:stage")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="results/review/bootstrap.json")
    args = ap.parse_args()
    runs = {}
    for r in args.runs:
        name, rest = r.split("=", 1)
        path, stage = rest.rsplit(":", 1)
        K = pickle.load(open(path if path.endswith(".pkl") else f"{path}/cv_cases.pkl", "rb"))  # or a native_eval --save-cases file
        runs[name] = {f: K[(stage, f)] for st, f in K if st == stage}  # folds 0-4, or e.g. centres
    rng = np.random.default_rng(0)
    F = sorted(runs[args.ref], key=str)
    sizes = {f: len(runs[args.ref][f]["case_target"]) for f in F}
    for name in runs:  # the same exams in every run
        assert all(sorted(runs[name][f]["case_target"]) == sorted(runs[args.ref][f]["case_target"]) for f in F)
    parts = {"cv_pooled": F, **({"fold0": [0]} if 0 in F else {})}
    point = {n: {p: metrics(runs[n], fs, [np.arange(sizes[f]) for f in fs]) for p, fs in parts.items()} for n in runs}
    boot = {n: {p: [] for p in parts} for n in runs}
    for _ in range(args.n):
        dr = {f: rng.integers(0, sizes[f], sizes[f]) for f in F}
        for n in runs:
            for p, fs in parts.items():
                boot[n][p].append(metrics(runs[n], fs, [dr[f] for f in fs]))
    out = {}
    names = ["patient_auroc", "lesion_ap", "picai_score"]
    for n in runs:
        out[n] = {}
        for part in parts:
            b = np.array(boot[n][part])
            diff = b - np.array(boot[args.ref][part])
            out[n][part] = {m: {"value": float(point[n][part][i]),
                                "ci95": [float(np.percentile(b[:, i], 2.5)), float(np.percentile(b[:, i], 97.5))],
                                f"diff_vs_{args.ref}": float(point[n][part][i] - point[args.ref][part][i]),
                                "diff_ci95": [float(np.percentile(diff[:, i], 2.5)), float(np.percentile(diff[:, i], 97.5))],
                                "p_better": float((diff[:, i] > 0).mean())} for i, m in enumerate(names)}
        print(n, json.dumps({p: {m: [round(out[n][p][m]["value"], 3), [round(x, 3) for x in out[n][p][m]["ci95"]],
                                     round(out[n][p][m][f"diff_vs_{args.ref}"], 3),
                                     [round(x, 3) for x in out[n][p][m]["diff_ci95"]]] for m in names}
                                 for p in out[n]}), flush=True)
    json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
