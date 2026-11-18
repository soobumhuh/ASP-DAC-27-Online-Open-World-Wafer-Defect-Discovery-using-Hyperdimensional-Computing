#!/usr/bin/env python3
"""Paired bootstrap on an external set (apply_final.py outputs on the same exams): differences of patient AUROC,
lesion AP, PI-CAI score vs a reference run, exams resampled with replacement (same draws for every run).

  uv run python scripts/ext_boot.py --ref best_clean_adc=/data/sota/external/prostate158/eval_best_clean_adc \
      --runs best_sota=/data/sota/external/prostate158/eval_best_sota --out /data/sota/external/prostate158/boot.json
"""
import argparse
import json
import pickle

import numpy as np
from picai_eval.metrics import Metrics


def load(path, reader="reader1"):
    K = pickle.load(open(f"{path}/cases.pkl", "rb"))
    keys = sorted(k for k in K["native"] if reader in K["native"][k])
    return keys, {k: K["native"][k][reader][0] for k in keys}, {k: K["native"][k][reader][1] for k in keys}, K["case_pred"]


def score(keys, lr, ct, cp, idx):
    m = Metrics(lesion_results={str(j): lr[keys[i]] for j, i in enumerate(idx)},
                case_target={str(j): ct[keys[i]] for j, i in enumerate(idx)},
                case_pred={str(j): cp[keys[i]] for j, i in enumerate(idx)})
    return np.array([m.auroc, m.AP, (m.auroc + m.AP) / 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = dict(r.split("=", 1) for r in [a.ref] + a.runs)
    ref = a.ref.split("=")[0]
    data = {n: load(p) for n, p in runs.items()}
    keys = data[ref][0]
    assert all(d[0] == keys for d in data.values())
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(keys), len(keys)) for _ in range(a.n)]
    full = np.arange(len(keys))
    B = {n: np.array([score(keys, d[1], d[2], d[3], i) for i in draws]) for n, d in data.items()}
    out = {}
    for n, d in data.items():
        pt, rp = score(keys, d[1], d[2], d[3], full), score(keys, *data[ref][1:], full)
        diff = B[n] - B[ref]
        out[n] = {m: {"value": float(pt[i]), "ci95": [float(x) for x in np.percentile(B[n][:, i], [2.5, 97.5])],
                      f"diff_vs_{ref}": float(pt[i] - rp[i]),
                      "diff_ci95": [float(x) for x in np.percentile(diff[:, i], [2.5, 97.5])]}
                  for i, m in enumerate(("patient_auroc", "lesion_ap", "picai_score"))}
        print(n, json.dumps({m: [round(v["value"], 3), round(v[f"diff_vs_{ref}"], 3), [round(x, 3) for x in v["diff_ci95"]]]
                             for m, v in out[n].items()}), flush=True)
    json.dump(out, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
