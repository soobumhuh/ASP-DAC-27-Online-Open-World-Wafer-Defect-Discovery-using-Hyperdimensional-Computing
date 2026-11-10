#!/usr/bin/env python3
"""
round4_refine.py - idea 5: coarse-to-fine candidate shapes with a small-patch HDC model.

The detector's candidates come from 20 px patch maps and are about 1.6x the lesion size; a loose shape can miss the
IoU >= 0.1 rule. A second HDC model with the same recipe but 8 px patches (model t110p8) gives a sharper map.
Two ways to use it:
  trim   inside each candidate, keep only the voxels whose small-patch z is in the top (1 - tau) of the candidate's
         voxels, then keep the connected part that holds the candidate's peak; confidence is unchanged
  blend  z = (1 - b) * z_base + b * z_small before extraction (then the usual extraction setting)
tau / b are chosen on inner validation folds 1-2; the best is applied once to fold 0 and to the CV test folds.

  uv run python round4_refine.py --base t110 --small t110p8 --pp results/round4/pp_zone_t110.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import ndimage

import postproc_search as pps
import rescore_hdc as rh


def trim(ext, small_maps, tau):
    if tau <= 0:
        return ext
    out = {}
    for k, (cands, shape) in ext.items():
        zs = small_maps[k]["map"].astype(np.float32).ravel()
        new = []
        for c in cands:
            v = zs[c["mask"]]
            ok = ~np.isnan(v)
            if ok.sum() < 10:
                new.append(c)
                continue
            keep = c["mask"][ok & (v >= np.quantile(v[ok], tau))]
            K = np.zeros(int(np.prod(shape)), bool)
            K[keep] = True
            lab, n = ndimage.label(K.reshape(shape))
            if n > 1:  # the connected part with the most voxels of the original candidate's top
                sizes = np.bincount(lab.ravel())[1:]
                K = (lab == (1 + int(np.argmax(sizes)))).ravel()
            new.append({**c, "mask": np.flatnonzero(K)})
        out[k] = (new, shape)
    return out


def blend(base, small, b):
    out = {}
    for k, c in base.items():
        m = (1 - b) * c["map"].astype(np.float32) + b * small[k]["map"].astype(np.float32)
        out[k] = {**c, "map": m.astype(np.float16)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="t110")
    ap.add_argument("--small", default="t110p8")
    ap.add_argument("--pp", default="results/round4/pp_zone_t110.json")
    ap.add_argument("--out", default="results/round4/refine")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--tune_only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pp = {**pps.DEFAULT_PP, **json.load(open(args.pp))}

    tune = {}
    for n in ("inner1", "inner2"):
        b, s = pps.load(n, args.base)["val"], pps.load(n, args.small)["val"]
        ext = rh.extract_all(b, b, pp, args.workers)
        for tau in (0.0, 0.2, 0.4, 0.6):
            tune.setdefault(("trim", tau), []).append(rh.evaluate(trim(ext, s, tau), b, None, 0, args.workers))
        for w in (0.25, 0.5):
            bm = blend(b, s, w)
            tune.setdefault(("blend", w), []).append(
                rh.evaluate(rh.extract_all(bm, bm, pp, args.workers), b, None, 0, args.workers))
    tune = {k: {m: float(np.mean([x[m] for x in v])) for m in v[0]} for k, v in tune.items()}
    for k, v in tune.items():
        print("tune", k, v, flush=True)
    best = max(tune, key=lambda k: (tune[k]["picai_score"], k == ("trim", 0.0)))
    print("chosen", best, flush=True)
    res = {"tune": {f"{k[0]}/{k[1]}": v for k, v in tune.items()}, "chosen": list(best), "pp": pp}
    if not args.tune_only:
        rows = []
        for f in range(5):
            b, s = pps.load(f"fold{f}", args.base), pps.load(f"fold{f}", args.small)
            ext = rh.extract_all(b["test"], b["val"], pp, args.workers)
            r = {"fold": f, "base": rh.evaluate(ext, b["test"], None, 0, args.workers)}
            if best[0] == "trim":
                r["refined"] = rh.evaluate(trim(ext, s["test"], best[1]), b["test"], None, 0, args.workers)
            else:
                bt, bv = blend(b["test"], s["test"], best[1]), blend(b["val"], s["val"], best[1])
                r["refined"] = rh.evaluate(rh.extract_all(bt, bv, pp, args.workers), b["test"], None, 0, args.workers)
            rows.append(r)
            print(r, flush=True)
        res["per_fold"] = rows
        res["cv"] = {lab: {m: [float(np.mean([r[lab][m] for r in rows])), float(np.std([r[lab][m] for r in rows]))]
                           for m in ("patient_auroc", "lesion_ap", "picai_score")} for lab in ("base", "refined")}
        print(json.dumps(res["cv"], indent=1))
    json.dump(res, open(os.path.join(args.out, f"refine_{args.base}_{args.small}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
