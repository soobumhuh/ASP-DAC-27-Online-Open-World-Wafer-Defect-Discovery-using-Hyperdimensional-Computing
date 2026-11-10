#!/usr/bin/env python3
"""round6_hw.py - round 6 B: hardware-friendliness of the HDC detector (single #110 model, fold-0 split).

Every variant is trained on folds 1-4 and scored once on fold 0 with #110's tuned extraction setting (no re-tuning),
so the numbers isolate the effect of the hardware parameter:
  hwD<D>      hypervector size D = 512 ... 8192 bits (4 thermometer levels x D/4 projections); D = 2048 is #110
  hwP<r>      share r of the bits of every stored prototype flipped at random (memory errors, e.g. FeNAND cells)
  hwQ<r>      share r of the bits of every query patch hypervector flipped at random (encoder / transfer errors)
Memory and compute per exam are counted analytically for the deployed model (135 position bins x (20 normal +
5 lesion) prototypes per model; patches per exam counted from the gland masks; Hamming similarity = XOR + popcount).
"""
import json
import os

import numpy as np

import postproc_search as pps

pp = {**pps.DEFAULT_PP, **json.load(open("results/lesion_ap2/final_t110.json"))["tuned"]["pp"]}
models = {"hwD512": 512, "hwD1024": 1024, "t110": 2048, "hwD4096": 4096, "hwD8192": 8192}
noise = {f"hw{k}{r}": (k, float(r)) for k in ("P", "Q") for r in ("0.01", "0.05", "0.1", "0.2")}
d = np.load(pps.NPZ, mmap_mode="r")
slices_per_exam = len(d["study_id"]) / len(np.unique(d["study_id"]))
g = np.asarray(d["gland_mask"])[:, 10:87:2, 10:87:2] > 0  # patch centres (stride 2, 20 px patches) inside the gland
patches_per_slice = float(g.reshape(len(g), -1).sum(1).mean())
rows = []
for m in list(models) + list(noise):
    p = os.path.join(pps.ROOT, "maps", m, "fold0", "maps.pkl")
    if not os.path.exists(p):
        print("missing", m)
        continue
    mp = pps.load("fold0", m)
    r = pps.evaluate(mp["test"], mp["val"], pp, 32)
    r["voxel_auroc"] = pps.voxel_auroc(mp["test"])
    D = models.get(m, 2048)
    patches = patches_per_slice * slices_per_exam  # patch hypervectors scored per exam
    r.update(model=m, D=D, noise=noise.get(m, ("-", 0.0)),
             proto_kib=135 * 25 * D / 8 / 1024, query_kib_per_exam=patches * D / 8 / 1024,
             popcount64_per_exam=patches * 25 * D / 64)
    rows.append(r)
    print(m, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
print(f"slices per exam {slices_per_exam:.1f}, patches per slice {patches_per_slice:.0f}")
os.makedirs("results/round6", exist_ok=True)
json.dump(rows, open("results/round6/hardware.json", "w"), indent=1)
print("| variant | D | bit errors | voxel AUROC | patient AUROC | lesion AP | PI-CAI | prototype memory | query HVs / exam | 64-bit popcounts / exam |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    n = r["noise"]
    print(f"| {r['model']} | {r['D']} | {'none' if n[0] == '-' else ('prototypes ' if n[0] == 'P' else 'queries ') + f'{n[1]:.0%}'} | "
          f"{r['voxel_auroc']:.3f} | {r['patient_auroc']:.3f} | {r['lesion_ap']:.3f} | {r['picai_score']:.3f} | "
          f"{r['proto_kib']:.0f} KiB | {r['query_kib_per_exam'] / 1024:.1f} MiB | {r['popcount64_per_exam'] / 1e6:.0f} M |")
