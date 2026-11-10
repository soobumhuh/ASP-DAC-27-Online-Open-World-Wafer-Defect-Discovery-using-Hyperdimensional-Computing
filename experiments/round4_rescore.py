#!/usr/bin/env python3
"""
round4_rescore.py - idea 4: a better HDC candidate re-scorer.

The first re-scorer (rescore_hdc.py) learned from ~65 true candidates per fold and used only size / peak / zone fields.
This one (a) trains on many more out-of-sample candidates: the CV test candidates of the other folds (each scored by a
model that never saw them), and (b) adds the contrast a radiologist reads: mean T2W / ADC / HBV inside the candidate
minus the same in a 3 px ring around it (in-plane, inside the gland), plus compactness (voxels / bounding-box volume).

Candidate record hypervector (2,048 bits): every numeric field as a 16-level code (quantiles of the training
candidates), zone as a symbol, each bound (XOR) to a field key, bundled by majority. Classifier: true-positive vs
false-positive class hypervectors + retraining (hdc_classifier.HDCBinaryClassifier).
New confidence = conf ** (1 - a) * rank ** a, rank = the HDC score's rank among the training candidates.
Protocol: a is chosen by leave-one-fold-out over folds 1-4 (train on three folds' candidates, validate on the fourth);
fold 0's test is then re-scored by a model trained on folds 1-4 candidates; the CV re-scores fold f with folds != f.

  uv run python round4_rescore.py --model t110 --pp results/round4/pp_zone_t110.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import ndimage

import postproc_search as pps
import rescore_hdc as rh
from hdc_classifier import HDCBinaryClassifier, LevelEncoder, SymbolEncoder, bundle

D = 2048
NUM = ["logsize", "peak", "mean", "nslices", "level", "c_t2w", "c_adc", "c_hbv", "compact"]


def case_images(d, study, n_slices):
    j = np.where(d["study_id"] == study)[0]
    j = j[np.argsort(d["slice_z"][j])]
    assert len(j) == n_slices
    return np.asarray(d["images"][j], dtype=np.float32), np.asarray(d["gland_mask"][j]) > 0


def add_contrast(ext, d):
    ring_st = np.zeros((1, 7, 7), bool)
    ring_st[0, :, :] = True  # 3 px in-plane dilation
    for k, (cands, shape) in ext.items():
        if not cands:
            continue
        img, gland = case_images(d, k, shape[0])
        for c in cands:
            K = np.zeros(int(np.prod(shape)), bool)
            K[c["mask"]] = True
            K = K.reshape(shape)
            R = ndimage.binary_dilation(K, ring_st) & ~K & gland
            for ch, name in enumerate(("c_t2w", "c_adc", "c_hbv")):
                a, b = img[:, ch][K], img[:, ch][R]
                c["feats"][name] = float(a.mean() - (b.mean() if len(b) else 0.0))
            zz, yy, xx = np.nonzero(K)
            box = (np.ptp(zz) + 1) * (np.ptp(yy) + 1) * (np.ptp(xx) + 1)
            c["feats"]["compact"] = float(K.sum() / box)


class Rescorer2:
    def __init__(self, seed=0, levels=16, epochs=20):
        rng = np.random.default_rng(seed)
        self.keys = {f: rng.integers(0, 2, D, dtype=np.uint8) for f in NUM + ["zone"]}
        self.tie = rng.integers(0, 2, D, dtype=np.uint8)
        self.lv = {f: LevelEncoder(D, levels, rng) for f in NUM}
        self.zone = SymbolEncoder(D, rng).fit([0, 1, 2])
        self.epochs, self.seed = epochs, seed

    def encode(self, cands):
        parts = [self.lv[f].encode(np.array([c["feats"][f] for c in cands])) ^ self.keys[f] for f in NUM]
        parts.append(self.zone.encode([c["feats"]["zone"] for c in cands]) ^ self.keys["zone"])
        return bundle(parts, self.tie)

    def fit(self, cands):
        for f in NUM:
            self.lv[f].fit(np.array([c["feats"][f] for c in cands]))
        y = np.array([int(c["tp"]) for c in cands])
        H = self.encode(cands)
        self.clf = HDCBinaryClassifier(epochs=self.epochs, seed=self.seed).fit(H, y)
        self.train_scores = np.sort(self.clf.score(H))
        return self

    def rank(self, cands):
        s = self.clf.score(self.encode(cands))
        return (np.searchsorted(self.train_scores, s) + 0.5) / (len(self.train_scores) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="t110")
    ap.add_argument("--pp", default="results/round4/pp_zone_t110.json")
    ap.add_argument("--out", default="results/round4/rescore")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pp = {**pps.DEFAULT_PP, **json.load(open(args.pp))}
    d = np.load(pps.MODEL_NPZ.get(args.model, pps.NPZ), mmap_mode="r")
    maps = {f: pps.load(f"fold{f}", args.model) for f in range(5)}
    ext = {}
    for f in range(5):
        ext[f] = rh.extract_all(maps[f]["test"], maps[f]["val"], pp, args.workers)
        add_contrast(ext[f], d)
        c = rh.flat(ext[f])
        print(f"fold{f}: {len(c)} candidates, {sum(x['tp'] for x in c)} TP", flush=True)

    # how well does each new field separate TP from FP on folds 1-4 (diagnostic only)
    from sklearn.metrics import roc_auc_score
    c14 = [c for f in (1, 2, 3, 4) for c in rh.flat(ext[f])]
    y14 = np.array([int(c["tp"]) for c in c14])
    diag = {f: float(roc_auc_score(y14, [c["feats"][f] for c in c14])) for f in NUM}
    diag["conf"] = float(roc_auc_score(y14, [c["conf"] for c in c14]))
    print("TP-vs-FP AUROC per field (folds 1-4):", {k: round(v, 3) for k, v in diag.items()}, flush=True)

    # choose alpha and epochs by leave-one-fold-out over folds 1-4
    tune = {}
    for ep in (5, 20, 50):
        for a in (0.0, 0.1, 0.25, 0.5, 0.75):
            r = []
            for v in (1, 2, 3, 4):
                rs = Rescorer2(epochs=ep).fit([c for g in (1, 2, 3, 4) if g != v for c in rh.flat(ext[g])])
                r.append(rh.evaluate(ext[v], maps[v]["test"], rs, a, args.workers))
            tune[(ep, a)] = {k: float(np.mean([x[k] for x in r])) for k in r[0]}
            print("tune", ep, a, tune[(ep, a)], flush=True)
    ep, a = max(tune, key=lambda k: (tune[k]["picai_score"], -k[1]))
    print("chosen epochs", ep, "alpha", a, flush=True)

    rows = []
    for f in range(5):
        rs = Rescorer2(epochs=ep).fit([c for g in range(5) if g != f for c in rh.flat(ext[g])]) if f else \
            Rescorer2(epochs=ep).fit(c14)
        rows.append({"fold": f, "no_rescore": rh.evaluate(ext[f], maps[f]["test"], None, 0, args.workers),
                     "rescored": rh.evaluate(ext[f], maps[f]["test"], rs, a, args.workers)})
        print(rows[-1], flush=True)
    cv = {lab: {m: [float(np.mean([r[lab][m] for r in rows])), float(np.std([r[lab][m] for r in rows]))]
                for m in ("patient_auroc", "lesion_ap", "picai_score")} for lab in ("no_rescore", "rescored")}
    json.dump({"pp": pp, "field_auroc": diag, "tune": {f"{k[0]}/{k[1]}": v for k, v in tune.items()},
               "chosen": [ep, a], "per_fold": rows, "cv": cv},
              open(os.path.join(args.out, f"rescore2_{args.model}.json"), "w"), indent=2)
    print(json.dumps(cv, indent=1))


if __name__ == "__main__":
    main()
