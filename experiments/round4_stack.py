#!/usr/bin/env python3
"""round4_stack.py - the round-4 stack: base maps -> tuned extraction -> HDC candidate re-scoring (idea 4) ->
small-patch shape trim (idea 5). Every setting was chosen earlier on folds 1-4 only (inner folds or leave-one-fold-out);
this script only applies them: fold 0 once, plus the 5-fold CV (re-scorer for test fold f trained on the other folds'
out-of-sample candidates).

  uv run python round4_stack.py --base t110 --pp results/round4/pp_zone_t110.json --epochs 5 --alpha 0.25 --tau 0.2
"""
import argparse
import json
import os

import numpy as np

import postproc_search as pps
import rescore_hdc as rh
from round4_refine import trim
from round4_rescore import Rescorer2, add_contrast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="t110")
    ap.add_argument("--pp", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--small", default="t110p8")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    pp = {**pps.DEFAULT_PP, **json.load(open(args.pp))}
    d = np.load(pps.MODEL_NPZ.get(args.base, pps.NPZ), mmap_mode="r")
    maps = {f: pps.load(f"fold{f}", args.base) for f in range(5)}
    small = {f: pps.load(f"fold{f}", args.small)["test"] for f in range(5)} if args.tau > 0 else None
    ext = {}
    for f in range(5):
        ext[f] = rh.extract_all(maps[f]["test"], maps[f]["val"], pp, args.workers)
        add_contrast(ext[f], d)
    rows = []
    for f in range(5):
        rs = Rescorer2(epochs=args.epochs).fit([c for g in range(5) if g != f for c in rh.flat(ext[g])])
        e = ext[f]
        r = {"fold": f, "voxel_auroc": pps.voxel_auroc(maps[f]["test"]),
             "base": rh.evaluate(e, maps[f]["test"], None, 0, args.workers),
             "rescore": rh.evaluate(e, maps[f]["test"], rs, args.alpha, args.workers)}
        if small is not None:
            et = trim(e, small[f], args.tau)
            r["trim"] = rh.evaluate(et, maps[f]["test"], None, 0, args.workers)
            r["rescore+trim"] = rh.evaluate(et, maps[f]["test"], rs, args.alpha, args.workers)
        rows.append(r)
        print(f, {k: (round(v["picai_score"], 3) if isinstance(v, dict) else round(v, 3)) for k, v in r.items() if k != "fold"},
              flush=True)
    labs = [k for k in rows[0] if isinstance(rows[0][k], dict)]
    cv = {l: {m: [float(np.mean([r[l][m] for r in rows])), float(np.std([r[l][m] for r in rows]))]
              for m in ("patient_auroc", "lesion_ap", "picai_score")} for l in labs}
    out = {"args": vars(args), "pp": pp, "per_fold": rows, "cv": cv}
    os.makedirs("results/round4/stack", exist_ok=True)
    json.dump(out, open(f"results/round4/stack/stack_{args.base}.json", "w"), indent=2)
    for l in labs:
        f0 = rows[0][l]
        print(f"{l:14s} fold0 {f0['patient_auroc']:.3f}/{f0['lesion_ap']:.3f}/{f0['picai_score']:.3f} | CV "
              f"{cv[l]['patient_auroc'][0]:.3f}/{cv[l]['lesion_ap'][0]:.3f}/{cv[l]['picai_score'][0]:.3f} ± {cv[l]['picai_score'][1]:.3f}")


if __name__ == "__main__":
    main()
