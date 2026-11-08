#!/usr/bin/env python3
"""
patch_sweep.py - run patch_anomaly.py over seeds x patch sizes x strides, choose settings on VAL
lesion AP (mean over seeds), and report TEST metrics as mean +- std over seeds.

  results/<root>/seed<s>/P<p>_s<st>/patch_<variant>/results.json   one run each
  results/<root>/summary.json, summary.md                           chosen settings + table

Raw per-pixel baselines (raw_*) do not depend on patch / stride, so they run once per seed.

Example:
  uv run python patch_sweep.py --slices_npz /data/picai_fold0_masks.npz --root patch_sweep
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

METRICS = ["voxel_auroc_in_gland", "patient_auroc", "lesion_ap", "picai_score"]


def run(cmd, log):
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"failed ({r.returncode}): {' '.join(cmd)} -- see {log}")


def load(root, seed, cfg, variant):
    p = os.path.join(root, f"seed{seed}", cfg, f"patch_{variant}", "results.json")
    return json.load(open(p)) if os.path.exists(p) else None


def flat(res, split, setting):
    r = res[split]["by_offset"][setting]
    return {"voxel_auroc_in_gland": res[split]["voxel_auroc_in_gland"], **r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices_npz", required=True)
    ap.add_argument("--root", default="patch_sweep", help="folder under results/")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--configs", default="6x2,8x2,8x4,12x2,12x4", help="PATCHxSTRIDE list (stride <= patch/2)")
    ap.add_argument("--knn_variants", default="hdc256,float,smooth_adc,smooth_hbv,smooth_mix")
    ap.add_argument("--raw_variants", default="raw_adc,raw_hbv,raw_mix")
    ap.add_argument("--descriptor", default="stats")
    ap.add_argument("--parallel", type=int, default=3, help="patch_anomaly.py processes at once")
    ap.add_argument("--extra", default="", help="extra arguments passed to every patch_anomaly.py run")
    ap.add_argument("--aggregate_only", action="store_true")
    args = ap.parse_args()
    root = os.path.join("results", args.root)
    seeds = [int(s) for s in args.seeds.split(",")]
    cfgs = [tuple(int(v) for v in c.split("x")) for c in args.configs.split(",")]

    if not args.aggregate_only:
        jobs = []
        for s in seeds:
            for i, (P, st) in enumerate(cfgs):
                variants = args.knn_variants + ("," + args.raw_variants if i == 0 and args.raw_variants else "")
                out = os.path.join(root, f"seed{s}", f"P{P}_s{st}")
                os.makedirs(out, exist_ok=True)
                cmd = [sys.executable, "patch_anomaly.py", "--slices_npz", args.slices_npz, "--seed", str(s),
                       "--patch", str(P), "--stride", str(st), "--descriptor", args.descriptor,
                       "--variants", variants, "--save_root", out, "--workers", str(max(8, 96 // args.parallel))] + args.extra.split()
                jobs.append((cmd, os.path.join(out, "log.txt")))
        with ThreadPoolExecutor(args.parallel) as ex:
            for f in [ex.submit(run, c, l) for c, l in jobs]:
                f.result()

    # ---- aggregate: choose (config, setting) per variant by mean val lesion AP over seeds
    variants = [v for v in (args.knn_variants + "," + args.raw_variants).split(",") if v]
    summary = {}
    for v in variants:
        cands = []  # (mean val AP, cfg, setting)
        for (P, st) in (cfgs if not v.startswith("raw_") else cfgs[:1]):
            cfg = f"P{P}_s{st}"
            runs = [load(root, s, cfg, v) for s in seeds]
            if any(r is None for r in runs):
                continue
            for setting in runs[0]["val"]["by_offset"]:
                cands.append((float(np.mean([r["val"]["by_offset"][setting]["lesion_ap"] for r in runs])), cfg, setting))
        if not cands:
            continue
        val_ap, cfg, setting = max(cands)
        runs = [load(root, s, cfg, v) for s in seeds]
        test = {m: [flat(r, "test", setting)[m] for r in runs] for m in METRICS}
        summary[v] = {
            "config": "pixel" if v.startswith("raw_") else cfg, "setting": setting, "val_lesion_ap_mean": val_ap,
            "test_mean": {m: float(np.mean(x)) for m, x in test.items()},
            "test_std": {m: float(np.std(x)) for m, x in test.items()},
            "test_per_seed": test,
            "runtime_s_per_seed": [r["runtime_s"] for r in runs],
            "split_counts_per_seed": [r["split_counts"] for r in runs],
        }
    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    lines = ["| variant | patch/stride | setting | val AP | voxel AUROC | patient AUROC | lesion AP | PI-CAI |",
             "|---|---|---|---|---|---|---|---|"]
    for v, r in summary.items():
        cells = [f"{r['test_mean'][m]:.3f} ± {r['test_std'][m]:.3f}" for m in METRICS]
        lines.append(f"| {v} | {r['config']} | {r['setting']} | {r['val_lesion_ap_mean']:.3f} | " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    open(os.path.join(root, "summary.md"), "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
