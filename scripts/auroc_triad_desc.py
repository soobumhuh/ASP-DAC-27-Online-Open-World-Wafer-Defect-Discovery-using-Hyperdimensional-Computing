#!/usr/bin/env python3
"""Triad patient-probe score as a descriptor family for run_auroc ('triadp'): for every split (fold<f>: train = the
other four folds, test = fold f; nest<f><g>: train = folds outside {f, g}, test = fold g) the zone-pooled Triad stage-3
features are whitened (PCA 64 on the split's training cases), random-projected to 10,000-D bipolar HVs and scored by
a fixed prototype (5 retraining epochs) + kNN (k = 50) mix; no tuning, so no fold is read twice.
Writes /data/auroc/desc/triadp/<split>.pkl = {'val': {}, 'test': {study_id: {'triad_score': s}}}."""
import json
import os
import pickle

import numpy as np
from sklearn.decomposition import PCA

import _path  # noqa: F401
from fe_probe import rp
from mri_hdc.patient_hdc import KNNScorer, ProtoScorer
from run_auroc import ecdf, marksheet

pool = pickle.load(open("/data/auroc/pool/triad_mae_s3.pkl", "rb"))
_, Y, _, PID = marksheet("/data/picai_labels/clinical_information/marksheet.csv")
sp = json.load(open("/data/sota/base/splits.json"))
fold_of = {}
for f in range(5):
    tp = set(sp[f"fold{f}"]["test"])
    fold_of.update({s: f for s in Y if PID[s] in tp})
ids = sorted(s for s in pool if s in fold_of)
X = np.stack([pool[s] for s in ids]).astype(np.float64)
y = np.array([Y[s] for s in ids])
fo = np.array([fold_of[s] for s in ids])
out = "/data/auroc/desc/triadp"
os.makedirs(out, exist_ok=True)
for f in range(5):
    for name, excl, test in [(f"fold{f}", {f}, f)] + [(f"nest{f}{g}", {f, g}, g) for g in range(5) if g != f]:
        tr, te = ~np.isin(fo, list(excl)), fo == test
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z = (X - mu) / sd
        pca = PCA(64, random_state=f).fit(Z[tr])
        H = rp(pca.transform(Z) / np.sqrt(pca.explained_variance_[None]), 10000, seed=f)
        p = ProtoScorer(epochs=5, seed=0).fit(H[tr], y[tr])
        k = KNNScorer(k=50).fit(H[tr], y[tr])
        ptr, ktr = p.score(H[tr]), k.score(H[tr])
        s = 0.5 * ecdf(ptr, p.score(H[te])) + 0.5 * ecdf(ktr, k.score(H[te]))
        pickle.dump({"val": {}, "test": {ids[i]: {"triad_score": float(v)} for i, v in zip(np.where(te)[0], s)}},
                    open(f"{out}/{name}.pkl", "wb"))
        print(name, int(te.sum()), flush=True)
