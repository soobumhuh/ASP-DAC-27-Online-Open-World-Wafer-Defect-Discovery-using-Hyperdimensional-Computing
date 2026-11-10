#!/usr/bin/env python3
"""round4_zone.py - idea 7: zone-specific calibration of the detection step (PZ vs TZ offsets).

The sigmoid offset is set per zone: the q_pz-th / q_tz-th percentile of the validation csPCa-negative voxels of that
zone, instead of one global percentile. The rest of the extraction setting stays the model's tuned one. The grid is
scored on inner validation folds 1-2; the best pair is then written as a pp json for postproc_search.py final."""
import itertools, json, os, sys
import numpy as np
import postproc_search as pps

model = sys.argv[1] if len(sys.argv) > 1 else "t110"
base = {**pps.DEFAULT_PP, **json.load(open(sys.argv[2] if len(sys.argv) > 2 else "results/lesion_ap2/final_t110.json"))["tuned"]["pp"]}
inner = {n: pps.load(n, model)["val"] for n in ("inner1", "inner2")}
grid = [95.0, 97.0, 98.0, 99.0, 99.5, 99.9]
rows = []
for qp, qt in [(None, None)] + list(itertools.product(grid, grid)):
    pp = {**base, "q_pz": qp, "q_tz": qt}
    r = [pps.evaluate(inner[n], inner[n], pp, 32) for n in inner]
    m = {k: float(np.mean([x[k] for x in r])) for k in r[0]}
    rows.append({"q_pz": qp, "q_tz": qt, **m})
    print(qp, qt, m, flush=True)
best = max(rows[1:], key=lambda r: r["picai_score"])
os.makedirs("results/round4", exist_ok=True)
json.dump(rows, open(f"results/round4/zone_grid_{model}.json", "w"), indent=1)
json.dump({**{k: v for k, v in base.items()}, "q_pz": best["q_pz"], "q_tz": best["q_tz"]},
          open(f"results/round4/pp_zone_{model}.json", "w"))
print("baseline", rows[0], "\nbest", best)
