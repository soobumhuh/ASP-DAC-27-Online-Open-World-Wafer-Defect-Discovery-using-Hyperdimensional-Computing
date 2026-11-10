"""Inner-val (folds 1-2) PI-CAI of saved maps with a fixed extraction setting: used to pick round-4 variants."""
import json, sys, numpy as np, postproc_search as pps
pp = {**pps.DEFAULT_PP, **json.load(open(sys.argv[2] if len(sys.argv) > 2 else "results/lesion_ap2/final_t110.json"))["tuned"]["pp"]} \
    if (len(sys.argv) <= 2 or sys.argv[2].endswith("final_t110.json")) else {**pps.DEFAULT_PP, **json.load(open(sys.argv[2]))}
for model in sys.argv[1].split(","):
    r = [pps.evaluate(pps.load(n, model)["val"], pps.load(n, model)["val"], pp, 32) for n in ("inner1", "inner2")]
    v = [pps.voxel_auroc(pps.load(n, model)["val"]) for n in ("inner1", "inner2")]
    print(model, "inner val:", {k: round(float(np.mean([x[k] for x in r])), 4) for k in r[0]}, "voxel", round(float(np.mean(v)), 4), flush=True)
