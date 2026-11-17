#!/usr/bin/env python3
"""Tier-1 evaluation extras on the nested 5-fold CV (native-grid per-exam results from native_eval.py --save-cases).

Per run, pooled over the five test folds, with 95% bootstrap CIs (exams resampled within each fold; the same draws
for every run, so differences are paired):
  - patient AUROC, lesion AP, PI-CAI score
  - FROC: lesion sensitivity at 0.5 and 1 false positive per patient (picai_eval lesion TPR vs FP per case)
  - patient sensitivity / specificity at operating points fixed WITHOUT the test fold: for test fold f the threshold
    is taken from the pooled out-of-fold scores and labels of the other four folds (90% sensitivity; Youden index)
  - comparator: PSA density alone (marksheet psad, else psa / prostate_volume; missing -> median of the other folds),
    patient AUROC, paired difference vs each run
Labels: expert lesion masks (human_expert/Pooch25 where present, else human_expert/resampled).

  uv run python scripts/eval_extras.py --runs best=results/sota/base/native_cases_hknnw.pkl --out results/sota/extras.json
"""
import argparse
import csv
import json
import pickle

import numpy as np
from picai_eval.metrics import Metrics
from sklearn.metrics import roc_auc_score


def pooled(K, draws):
    lr, ct, cp = {}, {}, {}
    for f, idx in draws.items():
        keys = sorted(K[f]["case_target"])
        for j, i in enumerate(idx):
            k = keys[i]
            n = f"{f}_{j}"
            lr[n], ct[n], cp[n] = K[f]["lesion_results"][k], K[f]["case_target"][k], K[f]["case_pred"][k]
    return Metrics(lesion_results=lr, case_target=ct, case_pred=cp)


def froc(m, fps=(0.5, 1.0)):
    fpr, tpr = np.asarray(m.lesion_FPR), np.asarray(m.lesion_TPR)  # FP per case, lesion sensitivity
    o = np.argsort(fpr, kind="stable")
    return [float(np.interp(x, fpr[o], tpr[o])) for x in fps]


def thresholds(scores, y):
    s, y = np.asarray(scores), np.asarray(y)
    t90 = np.sort(s[y == 1])[int(np.floor(0.1 * (y == 1).sum()))]  # >= t90 catches 90% of positives
    cand = np.unique(s)
    j = [((s[y == 1] >= t).mean() + (s[y == 0] < t).mean()) for t in cand]
    return {"sens90": float(t90), "youden": float(cand[int(np.argmax(j))])}


def operating_points(K):
    out = {}
    for name in ("sens90", "youden"):
        tp = fn = tn = fp = 0
        for f in K:
            other = [g for g in K if g != f]
            s = [K[g]["case_pred"][k] for g in other for k in K[g]["case_pred"]]
            y = [K[g]["case_target"][k] for g in other for k in K[g]["case_pred"]]
            t = thresholds(s, y)[name]
            for k, v in K[f]["case_pred"].items():
                pos, yk = v >= t, K[f]["case_target"][k]
                tp += pos and yk; fn += (not pos) and yk; fp += pos and not yk; tn += (not pos) and not yk
        out[name] = {"sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp), "tp": tp, "fn": fn, "tn": tn, "fp": fp}
    return out


def psad_scores(K, marksheet):
    rows = {f"{r['patient_id']}_{r['study_id']}": r for r in csv.DictReader(open(marksheet))}

    def val(k):
        r = rows[k]
        try:
            return float(r["psad"])
        except ValueError:
            try:
                return float(r["psa"]) / float(r["prostate_volume"])
            except (ValueError, ZeroDivisionError):
                return np.nan
    out = {}
    for f in K:
        other = [val(k) for g in K if g != f for k in K[g]["case_target"]]
        med = float(np.nanmedian(other))
        out[f] = {k: (v if not np.isnan(v) else med) for k in K[f]["case_target"] for v in [val(k)]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=native_cases.pkl[:stage]")
    ap.add_argument("--marksheet", default="/data/picai_labels/clinical_information/marksheet.csv")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {}
    for r in a.runs:
        name, rest = r.split("=", 1)
        path, stage = rest.rsplit(":", 1) if ":" in rest else (rest, "merged")
        K = pickle.load(open(path, "rb"))
        runs[name] = {f: K[(stage, f)] for f in range(5)}
    ref = next(iter(runs.values()))
    sizes = {f: len(ref[f]["case_target"]) for f in range(5)}
    ps = psad_scores(ref, a.marksheet)
    keys = {f: sorted(ref[f]["case_target"]) for f in range(5)}

    def stats(draws):
        out = {}
        for n, K in runs.items():
            m = pooled(K, draws)
            out[n] = [m.auroc, m.AP, (m.auroc + m.AP) / 2] + froc(m)
        y = [ref[f]["case_target"][keys[f][i]] for f, idx in draws.items() for i in idx]
        s = [ps[f][keys[f][i]] for f, idx in draws.items() for i in idx]
        out["psad"] = [roc_auc_score(y, s)]
        return out

    point = stats({f: np.arange(sizes[f]) for f in range(5)})
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(a.n):
        boot.append(stats({f: rng.integers(0, sizes[f], sizes[f]) for f in range(5)}))
    names = ["patient_auroc", "lesion_ap", "picai_score", "froc_sens_at_0.5fp", "froc_sens_at_1fp"]
    res = {"labels": "expert lesion masks: human_expert/Pooch25 (205 exams, 2025) else human_expert/resampled",
           "psad_auroc": {"value": point["psad"][0],
                          "ci95": list(np.percentile([b["psad"][0] for b in boot], [2.5, 97.5]))}}
    for n in runs:
        b = np.array([x[n] for x in boot])
        res[n] = {m: {"value": float(point[n][i]), "ci95": [float(x) for x in np.percentile(b[:, i], [2.5, 97.5])]}
                  for i, m in enumerate(names)}
        d = b[:, 0] - np.array([x["psad"][0] for x in boot])
        res[n]["auroc_minus_psad"] = {"value": float(point[n][0] - point["psad"][0]),
                                      "ci95": [float(x) for x in np.percentile(d, [2.5, 97.5])]}
        res[n]["operating_points"] = operating_points(runs[n])
        print(n, json.dumps({k: (round(v["value"], 3), [round(x, 3) for x in v["ci95"]]) for k, v in res[n].items()
                             if isinstance(v, dict) and "value" in v}), json.dumps(res[n]["operating_points"]), flush=True)
    print("psad", res["psad_auroc"])
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
