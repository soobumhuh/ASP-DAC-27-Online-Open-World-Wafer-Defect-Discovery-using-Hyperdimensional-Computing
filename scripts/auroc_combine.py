#!/usr/bin/env python3
"""Fixed equal-weight rank mean of several out-of-fold patient predictions (no weights tuned), pooled AUROC and
paired patient-cluster bootstrap against best_sota.

  python scripts/auroc_combine.py /data/auroc/runs/e2_full_pred.pkl /data/auroc/fe_runs/triad_mae_s3_pred.pkl --with-ref
"""
import argparse
import pickle

import numpy as np
from scipy.stats import rankdata

import _path  # noqa: F401
from run_auroc import auc, bootstrap, marksheet

ap = argparse.ArgumentParser()
ap.add_argument("preds", nargs="+")
ap.add_argument("--with-ref", action="store_true", help="add best_sota's merged scores as one more member")
ap.add_argument("--reference", default="/data/sota/base/cv_clean_hknnw_cases.pkl")
a = ap.parse_args()
_, Y, _, PID = marksheet("/data/picai_labels/clinical_information/marksheet.csv")
K = pickle.load(open(a.reference, "rb"))
ref = {}
for f in range(5):
    ref.update(K[("merged", f)]["case_pred"])
ids = sorted(ref)
y = np.array([Y[i] for i in ids])
pid = np.array([PID[i] for i in ids])
b = np.array([ref[i] for i in ids])
s = sum(rankdata([pickle.load(open(p, "rb"))[i] for i in ids]) for p in a.preds)
if a.with_ref:
    s = s + rankdata(b)
d = bootstrap(y, s, b, pid)
print(f"pooled AUROC {auc(y, s):.4f} vs best_sota {auc(y, b):.4f}: delta {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")
