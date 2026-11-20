#!/usr/bin/env python3
"""AUROC-only patient-level evaluation: nested 5-fold CV of the patient HDC scorer (mri_hdc.patient_hdc) on the
per-case descriptors of saved maps (scripts/auroc_desc.py) plus pre-biopsy clinical fields.

Per outer test fold f:
  training records  the cases of the four other folds g, each described from the nested map nest<f><g> (a model
                    trained on the three folds that are neither f nor g, so it never saw the case or fold f)
  test records      fold f's cases described from the fold-f map (model trained on the other four folds)
  scaling           every map descriptor -> empirical CDF of the same split's validation cases (so the 3-fold nested
                    models and the 4-fold outer model sit on one scale); clinical fields raw
  tuning            levels L, retraining epochs, kNN k and the proto / kNN mix by leave-one-fold-out over the four g
                    (pooled AUROC of the out-of-fold scores); nothing reads fold f
  scoring           the scorers retrained on all four g; each score mapped through the empirical CDF of its own
                    out-of-fold training scores, mixed
Main number: pooled AUROC over the 1,500 test cases; paired patient-cluster bootstrap against a reference.

  python scripts/run_auroc.py --tag e2 --families base --groups map zone tissue clin
"""
import argparse
import csv
import itertools
import json
import os
import pickle
import time

import numpy as np
from sklearn.metrics import roc_auc_score

import _path  # noqa: F401
from mri_hdc.patient_hdc import KNNScorer, ProtoScorer, RecordEncoder

MAP = ["q50", "q90", "q95", "q99", "q99.9", "max", "top10", "top50", "top200", "top1000", "vol1", "vol2", "vol3",
       "vol4", "frac1", "frac2", "frac3", "frac4", "smax", "cc2_max", "cc3_n", "slices3"]
ZONE = [f"z{z}_{s}" for z in (1, 2) for s in ("max", "q99", "top50", "vol2")]
TISSUE_MAP = ["adc_top", "adc_rel", "hbv_rel"]       # under the map's most suspicious voxels (per family)
TISSUE_IMG = ["adc_p5", "adc_p10", "hbv_p99"]        # image only (same for every family)
CLIN = ["psad", "psa", "prostate_volume", "patient_age"]
NN = ["nn_max", "nn_top10", "nn_top100", "nn_top1000", "nn_vol0.1", "nn_vol0.3", "nn_vol0.5", "nn_vol0.7", "nn_cc_max",
      "nn_cc_n", "nn_smax", "nn_gland_mean", "nn_z1_max", "nn_z1_top100", "nn_z2_max", "nn_z2_top100"]  # nnU-Net detector
GROUPS = {"triadp": ["triad_score"], "nn": NN, "map": MAP, "zone": ZONE, "tissue": TISSUE_MAP, "maxonly": ["max"], "top": ["top50", "smax", "vol2"]}


def ecdf(ref, x):
    ref = np.sort(ref[~np.isnan(ref)])
    if len(ref) == 0:
        return np.full(len(x), np.nan)
    out = np.searchsorted(ref, x, side="right") / len(ref)
    out[np.isnan(x)] = np.nan
    return out


def marksheet(path, derived_psad=True):
    rows = {int(r["study_id"]): r for r in csv.DictReader(open(path))}

    def v(r, k):
        s = (r.get(k) or "").strip()
        return float(s) if s else np.nan

    clin, y, isup, pid = {}, {}, {}, {}
    for s, r in rows.items():
        c = {k: v(r, k) for k in CLIN}
        if derived_psad and np.isnan(c["psad"]) and c["psa"] > 0 and c["prostate_volume"] > 0:
            c["psad"] = c["psa"] / c["prostate_volume"]
        clin[s], y[s] = c, int(r["case_csPCa"] == "YES")
        isup[s] = int(v(r, "case_ISUP")) if not np.isnan(v(r, "case_ISUP")) else 0
        pid[s] = int(r["patient_id"])
    return clin, y, isup, pid


def load_desc(root, fam, split):
    return pickle.load(open(os.path.join(root, fam, f"{split}.pkl"), "rb"))


def features(desc_by_fam, split, grp, cases, fams, groups, clin):
    """(n, F) matrix: per family the chosen map descriptors scaled by the split's val ECDF, then image-only tissue
    fields (first family) and clinical fields."""
    cols, names = [], []
    for i, fam in enumerate(fams):
        dd = desc_by_fam[fam][split]
        fl = [n for g in groups if g in GROUPS for n in GROUPS[g]
              if n in next(iter(dd["test"].values()))]  # only the fields this family has
        if "tissue" in groups and i == 0 and fam != "nnunet":
            fl = fl + TISSUE_IMG
        for n in fl:
            ref = np.array([dd["val"][c].get(n, np.nan) for c in dd["val"]], float)
            x = np.array([dd[grp][c].get(n, np.nan) for c in cases], float)
            cols.append(ecdf(ref, x) if len(ref) else x)  # no validation group (nnU-Net): raw probabilities
            names.append(f"{fam}:{n}")
    if "clin" in groups:
        for n in CLIN:
            cols.append(np.array([clin[c][n] for c in cases], float))
            names.append(f"clin:{n}")
    return np.stack(cols, 1), names


def auc(y, s):
    return float(roc_auc_score(y, s))


def fit_score(cfg, Htr, ytr, wtr, Hte):
    out = {}
    if cfg["mix"] > 0:
        out["proto"] = ProtoScorer(epochs=cfg["epochs"], seed=0).fit(Htr, ytr, wtr).score(Hte)
    if cfg["mix"] < 1:
        out["knn"] = KNNScorer(k=cfg["k"]).fit(Htr, ytr, wtr).score(Hte)
    return out


def combine(cfg, sc, oof):
    s = 0.0
    if cfg["mix"] > 0:
        s = s + cfg["mix"] * ecdf(oof["proto"], sc["proto"])
    if cfg["mix"] < 1:
        s = s + (1 - cfg["mix"]) * ecdf(oof["knn"], sc["knn"])
    # tie-break inside equal ECDF steps with the raw scores (tiny weight)
    for k in sc:
        s = s + 1e-6 * (sc[k] - np.mean(oof[k])) / (np.std(oof[k]) + 1e-9)
    return s


def bootstrap(y, a, b, pid, n=2000, seed=0):
    """Paired patient-cluster bootstrap of pooled AUROC(a) - AUROC(b)."""
    rng = np.random.default_rng(seed)
    up = np.unique(pid)
    rows = {p: np.where(pid == p)[0] for p in up}
    d = []
    for _ in range(n):
        idx = np.concatenate([rows[p] for p in rng.choice(up, len(up))])
        if y[idx].min() == y[idx].max():
            continue
        d.append(auc(y[idx], a[idx]) - auc(y[idx], b[idx]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--families", nargs="+", default=["base"])
    ap.add_argument("--groups", nargs="+", default=["map", "zone", "tissue", "clin"])
    ap.add_argument("--marksheet_psad", action="store_true", help="marksheet PSAD only (no PSA / volume fill-in)")
    ap.add_argument("--levels", nargs="+", type=int, default=[8, 16])
    ap.add_argument("--epochs", nargs="+", type=int, default=[0, 5, 20])
    ap.add_argument("--k", nargs="+", type=int, default=[25, 50, 100, 200])
    ap.add_argument("--mix", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--hardneg", nargs="+", type=float, default=[1.0], help="training weight of ISUP 1 negatives")
    ap.add_argument("--D", type=int, default=10000)
    ap.add_argument("--desc", default="/data/auroc/desc")
    ap.add_argument("--splits", default="/data/sota/base/splits.json")
    ap.add_argument("--labels", default="/data/picai_labels/clinical_information/marksheet.csv")
    ap.add_argument("--reference", default="/data/sota/base/cv_clean_hknnw_cases.pkl")
    ap.add_argument("--out", default="/data/auroc/runs")
    a = ap.parse_args()
    t0 = time.time()
    clin, Y, ISUP, PID = marksheet(a.labels, not a.marksheet_psad)
    sp = json.load(open(a.splits))
    os.makedirs(a.out, exist_ok=True)
    needs_maps = any(g in GROUPS for g in a.groups)
    fams = a.families if needs_maps else []
    pred, rows = {}, []
    for f in range(5):
        desc = {fam: {s: load_desc(a.desc, fam, s) for s in [f"fold{f}"] + [f"nest{f}{g}" for g in range(5) if g != f]}
                for fam in fams}
        inner = [g for g in range(5) if g != f]
        test_cases = sorted(desc[fams[0]][f"fold{f}"]["test"]) if fams else \
            sorted(s for s in Y if PID[s] in set(sp[f"fold{f}"]["test"]))
        tr_cases, tr_fold = [], []
        Xs = []
        for g in inner:
            cs = sorted(desc[fams[0]][f"nest{f}{g}"]["test"]) if fams else \
                sorted(s for s in Y if PID[s] in set(sp[f"nest{f}{g}"]["test"]))
            Xg, names = features(desc, f"nest{f}{g}", "test", cs, fams, a.groups, clin)
            Xs.append(Xg)
            tr_cases += cs
            tr_fold += [g] * len(cs)
        Xtr, tr_fold = np.concatenate(Xs), np.array(tr_fold)
        Xte, _ = features(desc, f"fold{f}", "test", test_cases, fams, a.groups, clin)
        ytr = np.array([Y[c] for c in tr_cases])
        yte = np.array([Y[c] for c in test_cases])
        isup_tr = np.array([ISUP[c] for c in tr_cases])
        # leave-one-fold-out tuning over the four inner folds
        best, tune = None, {}
        for L, hn in itertools.product(a.levels, a.hardneg):
            wtr = np.where((ytr == 0) & (isup_tr == 1), hn, 1.0)
            enc = {}
            for v in inner:  # encoder quantiles from the LOFO training part only
                m = tr_fold != v
                e = RecordEncoder(names, a.D, L, seed=f).fit(Xtr[m])
                enc[v] = (e.encode(Xtr[m]), e.encode(Xtr[~m]))
            cache = {}
            for ep, k, mix in itertools.product(a.epochs, a.k, a.mix):
                if mix == 0 and ep != a.epochs[0]:
                    continue
                if mix == 1 and k != a.k[0]:
                    continue
                cfg = {"L": L, "hardneg": hn, "epochs": ep, "k": k, "mix": mix}
                oof = {s: np.zeros(len(ytr)) for s in ("proto", "knn")}
                for v in inner:
                    m = tr_fold != v
                    for s in ("proto", "knn"):
                        if (s == "proto" and mix == 0) or (s == "knn" and mix == 1):
                            continue
                        key = (s, v, ep if s == "proto" else k)
                        if key not in cache:
                            c1 = {**cfg, "mix": 1.0 if s == "proto" else 0.0}
                            cache[key] = fit_score(c1, enc[v][0], ytr[m], wtr[m], enc[v][1])[s]
                        oof[s][~m] = cache[key]
                # mix on per-fold ECDFs of the out-of-fold scores (pooled over the four folds)
                comb = combine(cfg, {s: oof[s] for s in oof if oof[s].any()}, {s: oof[s] for s in oof if oof[s].any()})
                r = auc(ytr, comb)
                tune[json.dumps(cfg)] = r
                if best is None or r > best[0]:
                    best = (r, cfg, {s: oof[s].copy() for s in oof if oof[s].any()})
        r_in, cfg, oof = best
        wtr = np.where((ytr == 0) & (isup_tr == 1), cfg["hardneg"], 1.0)
        e = RecordEncoder(names, a.D, cfg["L"], seed=f).fit(Xtr)
        sc = fit_score(cfg, e.encode(Xtr), ytr, wtr, e.encode(Xte))
        s = combine(cfg, sc, oof)
        pred.update(dict(zip(test_cases, map(float, s))))
        rows.append({"fold": f, "cfg": cfg, "inner_auroc": r_in, "test_auroc": auc(yte, s),
                     **{f"test_auroc_{k}": auc(yte, v) for k, v in sc.items()}})
        print(f"fold {f}: {cfg} inner {r_in:.3f} test {auc(yte, s):.3f} "
              + " ".join(f"{k} {auc(yte, v):.3f}" for k, v in sc.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    ids = sorted(pred)
    y = np.array([Y[i] for i in ids])
    p = np.array([pred[i] for i in ids])
    pid = np.array([PID[i] for i in ids])
    res = {"tag": a.tag, "args": vars(a), "per_fold": rows, "pooled_auroc": auc(y, p),
           "fold_mean_auroc": float(np.mean([r["test_auroc"] for r in rows])),
           "fold_sd_auroc": float(np.std([r["test_auroc"] for r in rows]))}
    if a.reference and os.path.exists(a.reference):
        K = pickle.load(open(a.reference, "rb"))
        ref = {}
        for f in range(5):
            ref.update(K[("merged", f)]["case_pred"])
        rr = np.array([ref[i] for i in ids])
        res["reference_pooled_auroc"] = auc(y, rr)
        res["delta_vs_reference"] = bootstrap(y, p, rr, pid)
    json.dump(res, open(os.path.join(a.out, f"{a.tag}.json"), "w"), indent=2)
    pickle.dump(pred, open(os.path.join(a.out, f"{a.tag}_pred.pkl"), "wb"))
    msg = f"{a.tag}: pooled AUROC {res['pooled_auroc']:.4f} (fold mean {res['fold_mean_auroc']:.3f} ± {res['fold_sd_auroc']:.3f})"
    if "delta_vs_reference" in res:
        d = res["delta_vs_reference"]
        msg += f"; ref {res['reference_pooled_auroc']:.4f}; delta {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]"
    print(msg, flush=True)


if __name__ == "__main__":
    main()
