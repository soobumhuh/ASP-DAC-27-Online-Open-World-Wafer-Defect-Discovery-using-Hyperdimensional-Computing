#!/usr/bin/env python3
"""Patient-level AUROC (csPCa = Grade Group >= 2) on the UCSD RSI export, from a de-identified label CSV
(pds, in_export, pirads, gg; built on the server from the patient sheet, never committed) + scores.csv of
apply_final.py + compare-stage CSV of external_liss.py. 95% bootstrap CIs over exams, paired differences."""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def pirads_max(v):
    if not isinstance(v, str):
        return np.nan
    xs = [float(x) for x in v.replace("?", "").split(",") if x.strip()]
    return max(xs) if xs else np.nan


ap = argparse.ArgumentParser()
ap.add_argument("--labels", default="/data/liss_work/labels_deid.csv")
ap.add_argument("--scores", default="/data/liss_work/eval_best_sota/scores.csv")
ap.add_argument("--compare", default="/data/liss_work/seg_compare.csv")
ap.add_argument("--n", type=int, default=3000)
a = ap.parse_args()
L = pd.read_csv(a.labels, dtype={"pirads": str})
L = L[L.in_export & L.gg.notna()].copy()
L["key"] = "liss_" + L.pds.str[3:]
d = L.merge(pd.read_csv(a.scores), on="key").merge(pd.read_csv(a.compare), on=["key", "study_id"])
d["y"] = (d.gg >= 2).astype(int)
d["pirads_max"] = d.pirads.apply(pirads_max)
d["neg_volume"] = -d.vol_ml_seg
print("n", len(d), "csPCa", int(d.y.sum()))
cols = {"HDC best_sota": "score", "max RSIrs (export contour)": "max_rsirs_export_contour",
        "max RSIrs (our gland)": "max_rsirs_seg_gland", "PI-RADS max": "pirads_max", "smaller gland": "neg_volume"}
rng, y = np.random.default_rng(0), d.y.values
idx = [i for i in (rng.integers(0, len(d), len(d)) for _ in range(a.n)) if 0 < y[i].sum() < len(i)]
res = {}
for name, c in cols.items():
    x = d[c].values.astype(float)
    m = ~np.isnan(x)
    res[name] = np.array([roc_auc_score(y[i][m[i]], x[i][m[i]]) for i in idx])
    print(f"{name:28s} n={m.sum()} AUROC {roc_auc_score(y[m], x[m]):.3f} "
          f"[{np.percentile(res[name], 2.5):.3f}, {np.percentile(res[name], 97.5):.3f}]")
for name in ["max RSIrs (export contour)", "PI-RADS max"]:
    df = res["HDC best_sota"] - res[name]
    print(f"HDC - {name}: {df.mean():+.3f} [{np.percentile(df, 2.5):+.3f}, {np.percentile(df, 97.5):+.3f}]")
