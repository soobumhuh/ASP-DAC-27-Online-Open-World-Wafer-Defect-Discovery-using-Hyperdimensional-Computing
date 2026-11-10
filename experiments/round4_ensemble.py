"""round4_ensemble.py - idea 3: average the z maps of #110 copies that differ only in their random projection seed.
Creates derived map dumps results/lesion_ap/maps/<name>/<split>/maps.pkl (no patch HVs)."""
import os, pickle, sys
import numpy as np
models, name = sys.argv[1].split(","), sys.argv[2]
R = "results/lesion_ap/maps"
for n in ["inner1", "inner2"] + [f"fold{f}" for f in range(5)]:
    ms = [pickle.load(open(f"{R}/{m}/{n}/maps.pkl", "rb")) for m in models]
    out = {}
    for s in ("val", "test"):
        out[s] = {}
        for k, c in ms[0][s].items():
            m = np.mean([x[s][k]["map"].astype(np.float32) for x in ms], axis=0)
            out[s][k] = {x: v for x, v in c.items() if x not in ("codes", "pos", "map")}
            out[s][k]["map"] = m.astype(np.float16)
    os.makedirs(f"{R}/{name}/{n}", exist_ok=True)
    pickle.dump(out, open(f"{R}/{name}/{n}/maps.pkl", "wb"), protocol=4)
    print(name, n, "ok", flush=True)
