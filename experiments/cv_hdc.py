#!/usr/bin/env python3
"""
cv_hdc.py - 5-fold cross-validation over the official PI-CAI public folds (1,500 cases).

For each fold f: test = all patients of fold f; the other 4 folds are the training pool, from which 15% of the
patients (stratified by csPCa) form an internal validation split for z-scoring and for choosing the detection
setting. Nothing about fold f is used before its test metrics are computed.

  A  design-A patch maps (patch_anomaly.py: HDC prototypes + retraining, zone bins, mean/std descriptor) and the
     smoothed (HBV-ADC)/2 comparison; per fold the detection setting is chosen on the internal val lesion AP.
  B  design-B patient record HVs (patient_hdc.py --splits_json), image fields cross-fitted inside the training pool;
     PI-CAI score = (B patient AUROC + A lesion AP of the same fold) / 2.

Fold membership: fold 0 = the patients in --fold0_npz; folds 1-4 = the file lists saved from the Zenodo zips
(/data/picai_fold<f>_files.txt).

Example:
  uv run python cv_hdc.py --slices_npz /data/picai_all_masks.npz --root cv
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


def fold_of_patients(fold0_npz: str, list_tmpl: str) -> dict:
    out = {int(p): 0 for p in np.unique(np.load(fold0_npz)["patient_id"])}
    for f in (1, 2, 3, 4):
        for line in open(list_tmpl.format(f)):
            name = os.path.basename(line.strip())
            if name.endswith(".mha") and name.count("_") == 2:
                out.setdefault(int(name.split("_")[0]), f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices_npz", required=True)
    ap.add_argument("--fold0_npz", default="/data/picai_fold0_masks.npz")
    ap.add_argument("--fold_lists", default="/data/picai_fold{}_files.txt")
    ap.add_argument("--root", default="cv")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--a_variants", default="proto256_negplus,proto256,smooth_mix")
    ap.add_argument("--a_args", default="--descriptor stats --bin_mode zone --patch 12 --stride 4")
    ap.add_argument("--b_variant", default="proto256_negplus", help="design-A variant feeding design B")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--skip_b", action="store_true")
    args = ap.parse_args()
    root = os.path.join("results", args.root)
    os.makedirs(root, exist_ok=True)

    d = np.load(args.slices_npz)
    pids, cspca = d["patient_id"], d["case_cspca"]
    fold = ({int(p): int(f) for p, f in zip(d["patient_id"], d["fold"])} if "fold" in d.files
            else fold_of_patients(args.fold0_npz, args.fold_lists))
    pats = np.unique(pids)
    missing = [int(p) for p in pats if int(p) not in fold]
    pos = {int(p): int(cspca[pids == p].max()) for p in pats}
    counts = {f: {"patients": sum(1 for p in pats if fold.get(int(p)) == f),
                  "patients_pos": sum(1 for p in pats if fold.get(int(p)) == f and pos[int(p)])} for f in range(5)}
    print("patients per fold:", counts, "| not in any fold list:", len(missing))

    splits = []
    for f in range(5):
        test = [int(p) for p in pats if fold.get(int(p)) == f]
        pool = np.array([int(p) for p in pats if fold.get(int(p)) not in (f, None)])
        rng = np.random.default_rng(100 + f)
        val = []
        for flag in (0, 1):
            g = pool[np.array([pos[p] == flag for p in pool])]
            val += list(rng.permutation(g)[: int(round(args.val_frac * len(g)))])
        val = sorted(int(p) for p in val)
        train = sorted(int(p) for p in pool if int(p) not in set(val))
        splits.append({"name": f"fold{f}", "train": train, "val": val, "test": test})
    json.dump(splits, open(os.path.join(root, "splits.json"), "w"))

    # ---- A: patch maps per fold
    def run_a(sp):
        out = os.path.join(root, "A", sp["name"])
        os.makedirs(out, exist_ok=True)
        sf = os.path.join(out, "split.json")
        json.dump({k: sp[k] for k in ("train", "val", "test")}, open(sf, "w"))
        if all(os.path.exists(os.path.join(out, f"patch_{v}", "results.json")) for v in args.a_variants.split(",")):
            return
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "patch_anomaly.py"), "--slices_npz", args.slices_npz, "--split_file", sf,
               "--variants", args.a_variants, "--save_root", out, "--workers", "32"] + args.a_args.split()
        with open(os.path.join(out, "log.txt"), "w") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True)

    with ThreadPoolExecutor(args.parallel) as ex:
        for fu in [ex.submit(run_a, sp) for sp in splits]:
            fu.result()

    table, ap_b = {}, {}
    for v in args.a_variants.split(","):
        per = []
        for sp in splits:
            r = json.load(open(os.path.join(root, "A", sp["name"], f"patch_{v}", "results.json")))
            setting = max(r["val"]["by_offset"], key=lambda k: r["val"]["by_offset"][k]["lesion_ap"])
            t = r["test"]["by_offset"][setting]
            per.append({"fold": sp["name"], "setting": setting, "voxel_auroc_in_gland": r["test"]["voxel_auroc_in_gland"],
                        "patient_auroc": t["patient_auroc"], "lesion_ap": t["lesion_ap"], "picai_score": t["picai_score"],
                        "runtime_s": r["runtime_s"]})
            if v == args.b_variant:
                ap_b[sp["name"]] = t["lesion_ap"]
        table[f"A:{v}"] = per
    json.dump(ap_b, open(os.path.join(root, "ap_for_b.json"), "w"))

    # ---- B: patient record HVs per fold
    if not args.skip_b:
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "patient_hdc.py"), "--slices_npz", args.slices_npz, "--splits_json",
               os.path.join(root, "splits.json"), "--ap_json", os.path.join(root, "ap_for_b.json"),
               "--root", os.path.join(args.root, "B"),
               "--patch_args", f"{' '.join(a for a in args.a_args.split() if a not in ('--patch', '--stride', '12', '4'))} --variants {args.b_variant}"]
        with open(os.path.join(root, "B_log.txt"), "w") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True)
        B = json.load(open(os.path.join(root, "B", "summary.json")))
        for name in B["per_seed"][splits[0]["name"]]:
            if name == "image_score_raw_auroc":
                continue
            table[f"B:{name}"] = [{"fold": sp["name"], **B["per_seed"][sp["name"]][name]} for sp in splits]

    # ---- summary
    lines = ["| method | " + " | ".join(f"fold {f}" for f in range(5)) + " | mean ± std (PI-CAI) | voxel AUROC | patient AUROC | lesion AP |",
             "|---|" + "---|" * 9]
    summ = {}
    for name, per in table.items():
        ms = {m: (float(np.mean([p[m] for p in per if m in p])), float(np.std([p[m] for p in per if m in p])))
              for m in METRICS if any(m in p for p in per)}
        summ[name] = {"per_fold": per, "mean_std": ms}
        cell = lambda m: f"{ms[m][0]:.3f} ± {ms[m][1]:.3f}" if m in ms else ""
        lines.append(f"| {name} | " + " | ".join(f"{p.get('picai_score', float('nan')):.3f}" for p in per) +
                     f" | {cell('picai_score')} | {cell('voxel_auroc_in_gland')} | {cell('patient_auroc')} | {cell('lesion_ap')} |")
    md = (f"# 5-fold CV on PI-CAI public folds\n\npatients per fold: {json.dumps(counts)}\n\n" + "\n".join(lines) +
          "\n\nPer-fold columns = PI-CAI score. B rows: PI-CAI = (B patient AUROC + lesion AP of A "
          f"{args.b_variant} on the same fold) / 2.\n")
    open(os.path.join(root, "summary.md"), "w").write(md)
    json.dump({"counts": counts, "summary": summ}, open(os.path.join(root, "summary.json"), "w"), indent=2)
    print(md)


if __name__ == "__main__":
    main()
