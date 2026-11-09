#!/usr/bin/env python3
"""
rescore_hdc.py - HDC candidate re-scoring (HDnn / HDDB style) to raise lesion AP.

Lesion AP depends on how candidate confidences rank true vs false positives across all cases. Here every
extracted candidate gets a candidate hypervector (2,048 bits, the patch-HV dimension):
  patches   bundle (bitwise majority) of the hypervectors of the patches whose centre lies inside the candidate
  fields    level codes for log size, peak probability, mean probability, number of slices, slice level;
            symbol codes for the zone at the peak (PZ / TZ / unlabelled); each bound (XOR) to a field key
  record    bundle of all of the above
Two class hypervectors (true positive = IoU >= 0.1 with an expert lesion, false positive otherwise) are trained
by bundling + retraining epochs (hdc_classifier.HDCBinaryClassifier). New confidence = blend of the extraction
confidence and the HDC score's rank among training candidates: conf' = conf^(1 - a) * rank^a.

Training candidates always come from maps of models that did not train on those cases (out-of-sample):
  tune   train on inner fold 1 val candidates, validate on inner fold 2 (and the reverse)
  fold0  train on inner folds 1 + 2 val candidates, test on fold-0 test candidates (fold-0 model)
  cv     test fold f: train on the CV test candidates of the other four folds

  uv run python rescore_hdc.py --model t119 --pp results/lesion_ap/best_pp_t119.json
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import get_context

import numpy as np
from scipy import ndimage
from scipy.special import expit

import postproc_search as pps
from hdc_classifier import HDCBinaryClassifier, LevelEncoder, SymbolEncoder, bundle, unpack_codes

D = 2048
FIELDS = ["logsize", "peak", "mean", "nslices", "level"]
_PP, _REF = {}, {}


def _extract(item):
    """One case -> candidate list [(det-map value, voxel mask indices, features, patch-HV bundle)] + label info."""
    from report_guided_annotation import extract_lesion_candidates
    key, c, off, temp, qgrid = item[:5]
    offs = item[5] if len(item) > 5 else None  # zone-specific calibration offsets {zone: offset}
    p = {**pps.DEFAULT_PP, **_PP}
    m = c["map"].astype(np.float32)
    if p["sxy"] > 0 or p["sz"] > 0:
        w = (~np.isnan(m)).astype(np.float32)
        num = ndimage.gaussian_filter(np.nan_to_num(m), sigma=(p["sz"], p["sxy"], p["sxy"]))
        den = ndimage.gaussian_filter(w, sigma=(p["sz"], p["sxy"], p["sxy"]))
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(w > 0, num / np.maximum(den, 1e-6), np.nan)
    if p["mapping"] == "rank":
        prob = np.interp(np.nan_to_num(m, nan=-1e3), qgrid, np.linspace(0, 1, len(qgrid))) ** (1.0 / max(p["tf"], 1e-3))
        prob[np.isnan(m)] = 0
    else:
        o = off
        if offs:
            o = np.full(m.shape, off, np.float32)
            for zz, oz in offs.items():
                o[c["zone"] == zz] = oz
        prob = expit((np.nan_to_num(m, nan=-1e3) - o) / temp)
    thr = p["mode"] if p["mode"] in ("dynamic", "dynamic-fast") else float(p["mode"])
    _, conf, idx = extract_lesion_candidates(prob.astype(np.float64), threshold=thr,
                                             min_voxels_detection=int(p["minvox"]), num_lesions_to_extract=int(p["nles"]),
                                             dynamic_threshold_factor=float(p["factor"]),
                                             remove_adjacent_lesion_candidates=bool(p["adj"]))
    gt, ngt = ndimage.label(c["lesion"] > 0)
    cands = []
    pos, codes = c.get("pos"), c.get("codes")
    for k, cf in conf:
        K = idx == k
        v = prob[K]
        if p["conf"] == "mean":
            cf = float(v.mean())
        elif p["conf"] == "top10pct":
            cf = float(np.sort(v)[-max(1, len(v) // 10):].mean())
        iou = 0.0
        for g in np.unique(gt[K]):
            if g > 0:
                G = gt == g
                iou = max(iou, float((G & K).sum() / (G | K).sum()))
        pk = np.unravel_index(np.argmax(np.where(K, prob, -1)), K.shape)
        zs = np.where(K.any((1, 2)))[0]
        feats = {"logsize": float(np.log(K.sum())), "peak": float(v.max()), "mean": float(v.mean()),
                 "nslices": float(len(zs)), "level": float(c["rel_z"][pk[0]]), "zone": int(c["zone"][pk])}
        hv = None
        if codes is not None and len(codes):
            inside = K[pos[:, 0], pos[:, 1], pos[:, 2]]
            if not inside.any():  # no patch centre inside a small blob: take the patch nearest to the peak
                inside = np.zeros(len(pos), bool)
                inside[np.argmin(np.abs(pos - np.array(pk)).sum(1))] = True
            bits = unpack_codes(codes[inside], D)
            s = bits.sum(0).astype(np.int32) * 2 - len(bits)
            hv = np.where(s > 0, 1, np.where(s < 0, 0, _REF["tie"])).astype(np.uint8)
        cands.append({"conf": float(cf), "tp": iou >= 0.1, "feats": feats, "hv": hv, "mask": np.flatnonzero(K)})
    return key, cands, c["lesion"].shape


def extract_all(maps, ref_maps, pp, workers=24):
    ref = np.concatenate([c["map"][~np.isnan(c["map"])].astype(np.float32).ravel()
                          for c in ref_maps.values() if c["cspca"] == 0])
    off, temp = float(np.percentile(ref, pp["q"])), float(pp["tf"] * ref.std() + 1e-6)
    qgrid = np.percentile(ref, np.linspace(0, 100, 1001))
    offs = {}
    for zz, qk in ((1, "q_pz"), (2, "q_tz")):
        if pp.get(qk) is not None:
            v = np.concatenate([c["map"][(c["zone"] == zz) & ~np.isnan(c["map"])].astype(np.float32)
                                for c in ref_maps.values() if c["cspca"] == 0])
            offs[zz] = float(np.percentile(v, pp[qk]))
    with get_context("fork").Pool(workers, initializer=_init, initargs=(pp,)) as pool:
        out = pool.map(_extract, [(k, maps[k], off, temp, qgrid, offs) for k in sorted(maps)])
    return {k: (cands, shape) for k, cands, shape in out}


def _init(pp):
    global _PP, _REF
    _PP = pp
    _REF = {"tie": np.random.default_rng(7).integers(0, 2, D, dtype=np.uint8)}


class Rescorer:
    def __init__(self, seed=0, levels=16, epochs=20, use_patches=True):
        rng = np.random.default_rng(seed)
        self.keys = {f: rng.integers(0, 2, D, dtype=np.uint8) for f in FIELDS + ["zone", "patches"]}
        self.tie = rng.integers(0, 2, D, dtype=np.uint8)
        self.lv = {f: LevelEncoder(D, levels, rng) for f in FIELDS}
        self.zone = SymbolEncoder(D, rng).fit([0, 1, 2])
        self.epochs, self.use_patches, self.seed = epochs, use_patches, seed

    def encode(self, cands):
        parts = [self.lv[f].encode(np.array([c["feats"][f] for c in cands])) ^ self.keys[f] for f in FIELDS]
        parts.append(self.zone.encode([c["feats"]["zone"] for c in cands]) ^ self.keys["zone"])
        if self.use_patches and cands[0]["hv"] is not None:
            parts.append(np.stack([c["hv"] for c in cands]) ^ self.keys["patches"])
        return bundle(parts, self.tie)

    def fit(self, cands):
        for f in FIELDS:
            self.lv[f].fit(np.array([c["feats"][f] for c in cands]))
        y = np.array([int(c["tp"]) for c in cands])
        H = self.encode(cands)
        self.clf = HDCBinaryClassifier(epochs=self.epochs, seed=self.seed).fit(H, y)
        self.train_scores = np.sort(self.clf.score(H))
        return self

    def rank(self, cands):
        s = self.clf.score(self.encode(cands))
        return (np.searchsorted(self.train_scores, s) + 0.5) / (len(self.train_scores) + 1)


def evaluate(ext, maps, rescorer=None, alpha=0.0, workers=24):
    """picai_eval on candidates; with a rescorer, conf' = conf^(1 - alpha) * hdc_rank^alpha."""
    from picai_eval import evaluate as pe
    keys = sorted(ext)
    y_det = []
    for k in keys:
        cands, shape = ext[k]
        det = np.zeros(int(np.prod(shape)), np.float32)
        if cands:
            conf = np.array([c["conf"] for c in cands])
            if rescorer is not None and alpha > 0:
                conf = conf ** (1 - alpha) * rescorer.rank(cands) ** alpha
            for c, v in zip(cands, conf):
                det[c["mask"]] = v
        y_det.append(det.reshape(shape))
    m = pe(y_det=y_det, y_true=[maps[k]["lesion"].astype(np.int32) for k in keys], subject_list=[str(k) for k in keys],
           num_parallel_calls=workers)
    return {"patient_auroc": float(m.auroc), "lesion_ap": float(m.AP), "picai_score": float(m.score)}


def flat(ext):
    return [c for cands, _ in ext.values() for c in cands]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="t119")
    ap.add_argument("--pp", default=None, help="json of post-processing settings (default: postproc DEFAULT_PP)")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--use_patches", type=int, default=1)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    pp = {**pps.DEFAULT_PP, **(json.load(open(args.pp)) if args.pp else {})}
    alphas = [float(a) for a in args.alphas.split(",")]
    out = {"pp": pp, "use_patches": args.use_patches}

    # tune alpha on the inner folds (train on one, validate on the other)
    inner = {n: pps.load(n, args.model)["val"] for n in ("inner1", "inner2")}
    ext = {n: extract_all(inner[n], inner[n], pp, args.workers) for n in inner}
    for n in ext:
        c = flat(ext[n])
        print(f"{n}: {len(c)} candidates, {sum(x['tp'] for x in c)} true positives", flush=True)
    tune = {}
    for a in alphas:
        r = []
        for tr, va in (("inner1", "inner2"), ("inner2", "inner1")):
            rs = Rescorer(use_patches=bool(args.use_patches)).fit(flat(ext[tr]))
            r.append(evaluate(ext[va], inner[va], rs, a, args.workers))
        tune[a] = {k: float(np.mean([x[k] for x in r])) for k in r[0]}
        print("alpha", a, tune[a], flush=True)
    best_a = max(tune, key=lambda a: tune[a]["picai_score"])
    out["tune"], out["alpha"] = tune, best_a

    # fold 0 test: train on inner folds 1 + 2 candidates
    rs = Rescorer(use_patches=bool(args.use_patches)).fit(flat(ext["inner1"]) + flat(ext["inner2"]))
    f0 = pps.load("fold0", args.model)
    e0 = extract_all(f0["test"], f0["val"], pp, args.workers)
    out["fold0"] = {"no_rescore": evaluate(e0, f0["test"], None, 0, args.workers),
                    "rescored": evaluate(e0, f0["test"], rs, best_a, args.workers)}
    print("fold0", out["fold0"], flush=True)

    # 5-fold CV: test fold f re-scored by a model trained on the other folds' (out-of-sample) test candidates
    folds = {f: pps.load(f"fold{f}", args.model) for f in range(5)}
    ef = {f: (e0 if f == 0 else extract_all(folds[f]["test"], folds[f]["val"], pp, args.workers)) for f in range(5)}
    cv = []
    for f in range(5):
        rs = Rescorer(use_patches=bool(args.use_patches)).fit([c for g in range(5) if g != f for c in flat(ef[g])])
        cv.append({"fold": f, "no_rescore": evaluate(ef[f], folds[f]["test"], None, 0, args.workers),
                   "rescored": evaluate(ef[f], folds[f]["test"], rs, best_a, args.workers)})
        print("cv", cv[-1], flush=True)
    out["cv"] = cv
    for kind in ("no_rescore", "rescored"):
        out[f"cv_mean_{kind}"] = {k: [float(np.mean([x[kind][k] for x in cv])), float(np.std([x[kind][k] for x in cv]))]
                                  for k in cv[0][kind]}
    tag = os.path.splitext(os.path.basename(args.pp))[0] if args.pp else "defaultpp"
    json.dump(out, open(os.path.join(pps.ROOT, f"rescore_{args.model}_{tag}_p{args.use_patches}.json"), "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("alpha", "fold0", "cv_mean_no_rescore", "cv_mean_rescored")}, indent=1))


if __name__ == "__main__":
    main()
