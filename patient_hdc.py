#!/usr/bin/env python3
"""
patient_hdc.py - design B: HDDB-style patient record hypervectors for case-level csPCa classification.

Each case (study) becomes one binary record hypervector, built as in HDDB (Zhao et al., arXiv 2511.18234):
every field value is encoded, bound (XOR) to a random field-key hypervector, and all fields are bundled
(bitwise majority).
  numeric fields  psad, psa, prostate_volume, patient_age   -> order-preserving level codes (16 quantile levels)
  symbol fields   center, scanner model, HBV b-value        -> random hypervectors (+ 'unknown')
  missing value   a dedicated random 'missing' hypervector per field
  image fields    img_max : the case's max patch score from a design-A patch model (level code)
                  img_top : bundle of the hypervectors of the case's 5 highest-scoring patches
Never used as inputs: case_ISUP, lesion_GS, lesion_ISUP, histopath_type, case_csPCa, lesion_PIRADS.

Image scores for TRAIN cases must not come from a model trained on them, so they are cross-fitted: train
patients are split into 3 folds, the patch model is retrained on the other two folds and scores the held-out
one (patch_anomaly.py --split_file ... --dump_cases ...). Val and test cases are scored by the model trained
on all train patients.

Classifier: two class hypervectors (csPCa / not) by bundling bit counts + 20 retraining epochs on train cases;
score = sim(csPCa) - sim(not). Reported: test patient AUROC per field set, and the PI-CAI score with the lesion
AP of the design-A detection maps (picai_eval's lesion AP does not depend on the case-level score, so
(AUROC_B + AP_A) / 2 is what picai_eval gives when B's score is used as the case-level prediction).

Example:
  uv run python patient_hdc.py --slices_npz /data/picai_fold0_zones.npz --seeds 0,1,2 \\
      --patch_args "--descriptor stats --bin_mode zone --variants proto256" --root patient_hdc
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from hdc_classifier import HDCBinaryClassifier, LevelEncoder, SymbolEncoder, bundle, unpack_codes
from picai_bundle import split_patients

NUMERIC = ["psad", "psa", "prostate_volume", "patient_age"]
SYMBOL = ["center", "scanner", "bvalue"]
IMG_NUM = ["img_max", "img_max_fine", "pz_max", "tz_max", "pz_cnt", "tz_cnt"]  # numeric image fields (level codes)
IMG_TOP = {"img_top": "top", "pz_top": "top_pz", "tz_top": "top_tz"}           # bundled top-5 patch HVs
ZONES = ["pz_max", "tz_max", "pz_cnt", "tz_cnt"]
FIELD_SETS = {
    "clinical": NUMERIC + SYMBOL,
    "image_max": ["img_max"],
    "image_max+top5": ["img_max", "img_top"],
    "clinical+image_max": NUMERIC + SYMBOL + ["img_max"],
    "clinical+image_max+top5": NUMERIC + SYMBOL + ["img_max", "img_top"],
    # finer image level code (--img_levels, quantile = rank within train) and per-zone image fields
    "image_fine": ["img_max_fine"],
    "clinical+image_fine": NUMERIC + SYMBOL + ["img_max_fine"],
    "zones": ZONES,
    "clinical+zones": NUMERIC + SYMBOL + ZONES,
    "clinical+image_fine+zones": NUMERIC + SYMBOL + ["img_max_fine"] + ZONES,
    "clinical+image_fine+zones+zone_top5": NUMERIC + SYMBOL + ["img_max_fine"] + ZONES + ["pz_top", "tz_top"],
}


def scanner_info(images_dir: str) -> dict:
    """study_id -> (scanner model, HBV b-value) from the .mha headers."""
    import SimpleITK as sitk
    out = {}
    for f in glob.glob(os.path.join(images_dir, "**", "*_hbv.mha"), recursive=True):
        r = sitk.ImageFileReader(); r.SetFileName(f); r.ReadImageInformation()
        g = lambda k: r.GetMetaData(k) if r.HasMetaDataKey(k) else "NA"
        out[int(os.path.basename(f).split("_")[1])] = (f"{g('0008|0070')} {g('0008|1090')}", g("0018|9087"))
    return out


def run_patch(args, name, split, tag, seed: int):
    """Run the design-A patch model with an explicit split; return its per-case dump."""
    out_dir = os.path.join("results", args.root, name)
    os.makedirs(out_dir, exist_ok=True)
    sf, dump = os.path.join(out_dir, f"split_{tag}.json"), os.path.join(out_dir, f"cases_{tag}.npz")
    if not os.path.exists(dump):
        json.dump({k: [int(p) for p in v] for k, v in split.items()}, open(sf, "w"))
        cmd = [sys.executable, "patch_anomaly.py", "--slices_npz", args.slices_npz, "--seed", str(seed),
               "--split_file", sf, "--dump_cases", dump, "--skip_eval", "--save_root", out_dir,
               "--patch", str(args.patch), "--stride", str(args.stride)] + args.patch_args.split()
        with open(os.path.join(out_dir, f"log_{tag}.txt"), "w") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True)
    return dict(np.load(dump))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices_npz", required=True)
    ap.add_argument("--images_dir", default="/data/picai_images")
    ap.add_argument("--marksheet", default="/data/picai_labels/clinical_information/marksheet.csv")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--root", default="patient_hdc")
    ap.add_argument("--patch_args", default="--descriptor stats --bin_mode zone --variants proto256")
    ap.add_argument("--patch", type=int, default=12)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--ap_summary", default=None,
                    help="patch_sweep summary.json + variant (path:variant) whose test lesion AP per seed is "
                         "paired with B's AUROC for the PI-CAI score")
    ap.add_argument("--splits_json", default=None,
                    help="json list of {name, train, val, test} patient-id splits used instead of --seeds "
                         "(cv_hdc.py: the 5 official PI-CAI folds)")
    ap.add_argument("--ap_json", default=None, help="json {split name: design-A test lesion AP} for the PI-CAI score")
    ap.add_argument("--D", type=int, default=2048, help="record dimension (= patch HV dimension for img_top)")
    ap.add_argument("--levels", type=int, default=16, help="levels for clinical numeric fields and img_max")
    ap.add_argument("--img_levels", type=int, default=64, help="levels for img_max_fine and the per-zone fields")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    d = np.load(args.slices_npz)
    pids, cspca, studies = d["patient_id"], d["case_cspca"], d["study_id"]
    split_keep = (d["lesion_pixels"] >= 10) | (cspca == 0)
    rows = {int(r["study_id"]): r for r in csv.DictReader(open(args.marksheet))}
    scan = scanner_info(args.images_dir)
    case_ids = np.unique(studies)
    case_pid = {int(s): int(pids[studies == s][0]) for s in case_ids}
    case_y = {int(s): int(cspca[studies == s][0]) for s in case_ids}
    ap_by_seed = None
    if args.ap_summary:
        path, var = args.ap_summary.rsplit(":", 1)
        ap_by_seed = json.load(open(path))[var]["test_per_seed"]["lesion_ap"]

    if args.splits_json:
        plan = [(i, sp["name"], {k: sorted(int(p) for p in sp[k]) for k in ("train", "val", "test")})
                for i, sp in enumerate(json.load(open(args.splits_json)))]
    else:
        plan = []
        for seed in (int(s) for s in args.seeds.split(",")):
            so = split_patients(pids[split_keep], cspca[split_keep], 0.15, 0.25, np.random.default_rng(seed))
            plan.append((seed, f"seed{seed}", {k: sorted(p for p, v in so.items() if v == k) for k in ("train", "val", "test")}))
    ap_by_name = json.load(open(args.ap_json)) if args.ap_json else None

    results = {}
    for si, (seed, sname, groups) in enumerate(plan):
        split_of = {p: k for k, v in groups.items() for p in v}
        # image scores: val/test from the full-train model, train from 3-fold cross-fitting
        img = {}
        base = run_patch(args, sname, groups, "base", seed)
        def take(dmp, s):
            for i, c in enumerate(dmp[f"{s}_study"]):
                r = {"img_max": float(dmp[f"{s}_max"][i]), "top": dmp[f"{s}_top"][i]}
                r["img_max_fine"] = r["img_max"]
                for zn in ("pz", "tz"):
                    if f"{s}_max_{zn}" in dmp:
                        r[f"{zn}_max"] = float(dmp[f"{s}_max_{zn}"][i])
                        r[f"{zn}_cnt"] = float(np.log1p(dmp[f"{s}_cnt_{zn}"][i]))
                        r[f"top_{zn}"] = dmp[f"{s}_top_{zn}"][i]
                img[int(c)] = r
        for s in ("val", "test"):
            take(base, s)
        tr = np.array(groups["train"])
        tr_pos = np.array([any(case_y[c] for c in case_ids if case_pid[c] == p) for p in tr])
        fold = np.zeros(len(tr), int)
        for flag in (0, 1):
            j = np.where(tr_pos == flag)[0]
            fold[np.random.default_rng(seed + 7).permutation(j)] = np.arange(len(j)) % 3
        for f in range(3):
            sp = {"train": tr[fold != f], "val": groups["val"], "test": tr[fold == f]}
            dmp = run_patch(args, sname, sp, f"cf{f}", seed)
            take(dmp, "test")

        # records
        hv_rng = np.random.default_rng(1000 + seed)
        D = args.D
        cases = {k: [c for c in case_ids if split_of.get(case_pid[int(c)]) == k] for k in ("train", "val", "test")}
        val_of = lambda c, f: (float(rows[int(c)][f]) if (rows[int(c)].get(f) or "").strip() else np.nan)
        sym_of = lambda c, f: (rows[int(c)].get("center", "NA") if f == "center"
                               else scan.get(int(c), ("NA", "NA"))[0 if f == "scanner" else 1])
        keys = {f: hv_rng.integers(0, 2, D, dtype=np.uint8) for f in NUMERIC + SYMBOL + IMG_NUM + list(IMG_TOP)}
        tie = hv_rng.integers(0, 2, D, dtype=np.uint8)
        enc = {}
        for f in NUMERIC:
            enc[f] = LevelEncoder(D, args.levels, hv_rng).fit(np.array([val_of(c, f) for c in cases["train"]]))
        for f in SYMBOL:
            enc[f] = SymbolEncoder(D, hv_rng).fit([sym_of(c, f) for c in cases["train"]])
        has_zones = "pz_max" in img[int(cases["train"][0])]
        for f in IMG_NUM:
            if f in ZONES and not has_zones:
                continue
            L = args.levels if f == "img_max" else args.img_levels
            enc[f] = LevelEncoder(D, L, hv_rng).fit(np.array([img[int(c)].get(f, np.nan) for c in cases["train"]]))

        def field_hv(f, cs):
            if f in NUMERIC:
                return enc[f].encode(np.array([val_of(c, f) for c in cs]))
            if f in SYMBOL:
                return enc[f].encode([sym_of(c, f) for c in cs])
            if f in IMG_NUM:
                return enc[f].encode(np.array([img[int(c)].get(f, np.nan) for c in cs]))
            tops = np.stack([img[int(c)][IMG_TOP[f]] for c in cs])  # (n, 5, bytes)
            if tops.shape[-1] * 8 < D:
                raise SystemExit("img_top needs a supervised/HDC patch variant whose hypervectors have >= D bits")
            bits = unpack_codes(tops.reshape(-1, tops.shape[-1]), D).reshape(len(cs), 5, D)
            return bundle([bits[:, i] for i in range(5)], tie)

        res_seed = {}
        y = {k: np.array([case_y[int(c)] for c in cases[k]]) for k in cases}
        res_seed["image_score_raw_auroc"] = float(roc_auc_score(y["test"], [img[int(c)]["img_max"] for c in cases["test"]]))
        for name, fields in FIELD_SETS.items():
            if not has_zones and any(f in ZONES or f in ("pz_top", "tz_top") for f in fields):
                continue
            H = {k: bundle([field_hv(f, cases[k]) ^ keys[f][None, :] for f in fields], tie) for k in ("train", "val", "test")}
            clf = HDCBinaryClassifier(epochs=args.epochs, seed=seed).fit(H["train"], y["train"])
            sc = clf.score(H["test"])
            au = float(roc_auc_score(y["test"], sc))
            r = {"patient_auroc": au, "val_auroc": float(roc_auc_score(y["val"], clf.score(H["val"]))),
                 "n_test_cases": int(len(sc)), "n_test_pos": int(y["test"].sum())}
            a_ap = (ap_by_name.get(sname) if ap_by_name else (ap_by_seed[si] if ap_by_seed is not None else None))
            if a_ap is not None:
                r["lesion_ap_from_A"] = float(a_ap)
                r["picai_score"] = (au + float(a_ap)) / 2
            res_seed[name] = r
            print(f"{sname} {name:28s} patient AUROC {au:.3f}" + (f"  PI-CAI {r['picai_score']:.3f}" if 'picai_score' in r else ""))
        print(f"{sname} image max score as-is (no record): AUROC {res_seed['image_score_raw_auroc']:.3f}")
        results[sname] = res_seed

    summary = {}
    for name in [n for n in FIELD_SETS if n in results[next(iter(results))]] + ["image_score_raw_auroc"]:
        if name == "image_score_raw_auroc":
            v = [results[s][name] for s in results]
            summary[name] = {"patient_auroc": [float(np.mean(v)), float(np.std(v))]}
            continue
        summary[name] = {m: [float(np.mean([results[s][name][m] for s in results])),
                             float(np.std([results[s][name][m] for s in results]))]
                         for m in results[next(iter(results))][name] if m in ("patient_auroc", "picai_score", "val_auroc")}
    out = os.path.join("results", args.root, "summary.json")
    json.dump({"args": vars(args), "per_seed": results, "summary": summary}, open(out, "w"), indent=2)
    for k, v in summary.items():
        print(k, {m: f"{a:.3f} ± {b:.3f}" for m, (a, b) in v.items()})
    print("saved", out)


if __name__ == "__main__":
    main()
