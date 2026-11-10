#!/usr/bin/env python3
"""round6_cal.py - ideas A1 / A2: scanner-group calibration and per-case background normalisation of the maps.
Grid on inner validation folds 1-2 (model maps from postproc_search.py), on top of the model's tuned extraction pp:
  scal       None | vendor | bvalue | center | vendor_b : the sigmoid offset is the q-th percentile of the val-normal
             voxels of the case's scanner group (groups with < 5 normal val cases use the global offset)
  case_norm  0 | 0.25 | 0.5 | 1 : subtract that share of the case's own median map value (its background level)
Writes results/round6/cal_<model>.json and the best setting as results/round6/pp_cal_<model>.json."""
import itertools, json, sys
import numpy as np
import postproc_search as pps

model, base_pp = sys.argv[1], sys.argv[2]
base = {**pps.DEFAULT_PP, **json.load(open(base_pp))}
inner = {n: pps.load(n, model)["val"] for n in ("inner1", "inner2")}
rows = []
for scal, cn in itertools.product([None, "vendor", "bvalue", "center", "vendor_b"], [0.0, 0.25, 0.5, 1.0]):
    pp = {**base, "scal": scal, "case_norm": cn}
    r = [pps.evaluate(inner[n], inner[n], pp, 32) for n in inner]
    m = {k: float(np.mean([x[k] for x in r])) for k in r[0]}
    rows.append({"scal": scal, "case_norm": cn, **m})
    print(scal, cn, {k: round(v, 4) for k, v in m.items()}, flush=True)
best = max(rows, key=lambda r: (r["picai_score"], r["scal"] is None, -r["case_norm"]))
json.dump(rows, open(f"results/round6/cal_{model}.json", "w"), indent=1)
json.dump({**base, "scal": best["scal"], "case_norm": best["case_norm"]}, open(f"results/round6/pp_cal_{model}.json", "w"))
print("baseline", rows[0], "\nbest", best)
