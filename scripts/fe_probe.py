#!/usr/bin/env python3
"""Patient-level front-end screen: zone-pooled frozen features (scripts/auroc_pool.py) -> HDC patient scorer.

Per outer fold f (official folds; features are label-free, so the outer training cases are the other four folds):
  scaling     z-score with the training cases' mean / std, then PCA to `--pca` dims fitted on the training cases
  encoding    random projection to D = 10,000 and sign -> bipolar hypervector (classic HDC encoding of a real vector)
  scorers     prototype retraining and cosine kNN (mri_hdc.patient_hdc), epochs / k / mix chosen by leave-one-fold-out
              over the four training folds (pooled out-of-fold AUROC); fold f never read before scoring
Readouts: pooled 5-fold AUROC of the front end alone, of a fixed equal rank mean with a reference prediction
(the E2 patient record, so the gain over the map evidence shows), and a diagnostic logistic regression (not HDC).

  python scripts/fe_probe.py --pool /data/auroc/pool/dinov2_l.pkl --tag dinov2_l
"""
import argparse
import itertools
import json
import os
import pickle

import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import _path  # noqa: F401
from mri_hdc.patient_hdc import KNNScorer, ProtoScorer
from run_auroc import bootstrap, ecdf, marksheet


def rp(X, D, seed):
    R = np.random.default_rng(seed).standard_normal((X.shape[1], D)).astype(np.float32)
    return np.sign(X.astype(np.float32) @ R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--D", type=int, default=10000)
    ap.add_argument("--epochs", nargs="+", type=int, default=[0, 5, 20])
    ap.add_argument("--k", nargs="+", type=int, default=[25, 50, 100, 200])
    ap.add_argument("--mix", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--splits", default="/data/sota/base/splits.json")
    ap.add_argument("--labels", default="/data/picai_labels/clinical_information/marksheet.csv")
    ap.add_argument("--combine", default="/data/auroc/runs/e2_full_pred.pkl")
    ap.add_argument("--reference", default="/data/sota/base/cv_clean_hknnw_cases.pkl")
    ap.add_argument("--out", default="/data/auroc/fe_runs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pool = pickle.load(open(a.pool, "rb"))
    _, Y, _, PID = marksheet(a.labels)
    sp = json.load(open(a.splits))
    fold_of = {}
    for f in range(5):
        tp = set(sp[f"fold{f}"]["test"])
        for s in Y:
            if PID[s] in tp:
                fold_of[s] = f
    ids = sorted(s for s in pool if s in fold_of)
    X = np.stack([pool[s] for s in ids]).astype(np.float64)
    y = np.array([Y[s] for s in ids])
    fo = np.array([fold_of[s] for s in ids])
    pred, diag, rows = np.zeros(len(ids)), np.zeros(len(ids)), []
    for f in range(5):
        tr, te = fo != f, fo == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z = (X - mu) / sd
        pca = PCA(min(a.pca, Z.shape[1]), random_state=f).fit(Z[tr])
        P = pca.transform(Z) / np.sqrt(pca.explained_variance_[None])  # whitened
        H = rp(P, a.D, seed=f)
        diag[te] = LogisticRegression(C=0.05, max_iter=5000).fit(P[tr], y[tr]).predict_proba(P[te])[:, 1]
        inner = [g for g in range(5) if g != f]
        cache, best = {}, None
        for ep, k, mix in itertools.product(a.epochs, a.k, a.mix):
            if (mix == 0 and ep != a.epochs[0]) or (mix == 1 and k != a.k[0]):
                continue
            oof = {"proto": np.zeros(len(ids)), "knn": np.zeros(len(ids))}
            for v in inner:
                m, val = tr & (fo != v), fo == v
                if mix > 0:
                    key = ("p", v, ep)
                    if key not in cache:
                        cache[key] = ProtoScorer(epochs=ep, seed=0).fit(H[m], y[m]).score(H[val])
                    oof["proto"][val] = cache[key]
                if mix < 1:
                    key = ("k", v, k)
                    if key not in cache:
                        cache[key] = KNNScorer(k=k).fit(H[m], y[m]).score(H[val])
                    oof["knn"][val] = cache[key]
            o = {s: oof[s][tr] for s in oof}
            s = (mix * ecdf(o["proto"], o["proto"]) if mix > 0 else 0) + \
                ((1 - mix) * ecdf(o["knn"], o["knn"]) if mix < 1 else 0)
            r = roc_auc_score(y[tr], s)
            if best is None or r > best[0]:
                best = (r, dict(epochs=ep, k=k, mix=mix), o)
        r_in, cfg, o = best
        s = 0.0
        if cfg["mix"] > 0:
            s = s + cfg["mix"] * ecdf(o["proto"], ProtoScorer(epochs=cfg["epochs"], seed=0).fit(H[tr], y[tr]).score(H[te]))
        if cfg["mix"] < 1:
            s = s + (1 - cfg["mix"]) * ecdf(o["knn"], KNNScorer(k=cfg["k"]).fit(H[tr], y[tr]).score(H[te]))
        pred[te] = s
        rows.append({"fold": f, "cfg": cfg, "inner": r_in, "test": roc_auc_score(y[te], s)})
        print(f"fold {f}: {cfg} inner {r_in:.3f} test {rows[-1]['test']:.3f}", flush=True)
    pid = np.array([PID[s] for s in ids])
    res = {"tag": a.tag, "pool": a.pool, "per_fold": rows, "pooled_auroc": roc_auc_score(y, pred),
           "diag_lr_auroc": roc_auc_score(y, diag)}
    e2 = pickle.load(open(a.combine, "rb"))
    e = np.array([e2[s] for s in ids])
    comb = rankdata(pred) + rankdata(e)
    res["e2_auroc"] = roc_auc_score(y, e)
    res["e2_plus_fe_auroc"] = roc_auc_score(y, comb)
    res["e2_plus_fe_delta_vs_e2"] = bootstrap(y, comb, e, pid)
    pickle.dump(dict(zip(ids, map(float, pred))), open(os.path.join(a.out, f"{a.tag}_pred.pkl"), "wb"))
    json.dump(res, open(os.path.join(a.out, f"{a.tag}.json"), "w"), indent=1)
    d = res["e2_plus_fe_delta_vs_e2"]
    print(f"{a.tag}: front end alone {res['pooled_auroc']:.4f} (diag LR {res['diag_lr_auroc']:.4f}); "
          f"E2 {res['e2_auroc']:.4f} -> E2 + front end {res['e2_plus_fe_auroc']:.4f} "
          f"delta {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]", flush=True)


if __name__ == "__main__":
    main()
