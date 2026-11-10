#!/usr/bin/env python3
"""
summarize_results.py - one table over every lesion-task method run on PI-CAI fold 0 (seeds 0, 1, 2).

Reads
  results/slice_<feature_set>_seed<s>/results.json         Eric's whole-slice WaferHDC gate (run_mri_hdc.py)
  results/<sweep root>/summary.json                         patch-map sweeps (patch_sweep.py)
and writes results/final_summary.md (+ .json). Settings for map methods were chosen on val by patch_sweep.py;
for the combo family the weight w is chosen here, also by mean val lesion AP.
"""
from __future__ import annotations

import json
import os

import numpy as np

SEEDS = (0, 1, 2)
ROWS = []  # (group, method, metrics dict mean/std, runtime, setting)
# The sweeps ran 3 seeds at once on a shared CPU, which inflated some timings ~20x. These are solo re-timings
# (seed 0, encode + index + score val and test); other rows use the fastest of the three seeds.
SOLO = {("patch_sweep", "hdc256"): "8 s", ("new_stats", "hdc256_signed"): "9 s",
        ("hdnn_l2l3", "hdc256"): "9 s", ("hdnn_l2l3", "float"): "9 s",
        ("hdnn_l2l3", "hdc256_signed"): "10 s", ("hdnn_l2l3", "float_signed"): "10 s",
        ("hdnn_l2", "hdc256"): "<= 10 s", ("hdnn_l2", "hdc1024"): "<= 15 s", ("hdnn_l2", "float"): "<= 10 s",
        ("hdnn_l2", "hdc256_signed"): "<= 10 s", ("hdnn_l2", "float_signed"): "<= 10 s",
        ("hdnn_l2l3", "hdc1024"): "<= 15 s"}


def ms(x):
    x = [v for v in x if v is not None and not np.isnan(v)]
    return (float(np.mean(x)), float(np.std(x))) if x else (None, None)


def slice_rows():
    for fs in ("stats", "pixels", "resnet"):
        R = [json.load(open(f"results/slice_{fs}_seed{s}/results.json")) for s in SEEDS]
        E = [r["extras"] for r in R]
        for name, sk, pk in (("HDC gate", "hdc_gate_auroc", "hdc_patient_auroc_picai_eval"),
                             ("float kNN", "baseline_float_knn_auroc", "baseline_patient_auroc_picai_eval")):
            ROWS.append(dict(group="Eric's whole-slice WaferHDC (Stage 1)", method=f"{name}, {fs} features",
                             slice_auroc=ms([e[sk] for e in E]), voxel_auroc=(None, None),
                             patient_auroc=ms([e[pk] for e in E]), lesion_ap=(None, None), picai=(None, None),
                             runtime="see note", setting="whole slice, max slice score per patient"))


def sweep_rows(root, group, rename, pick_combo=False):
    p = f"results/{root}/summary.json"
    if not os.path.exists(p):
        return
    S = json.load(open(p))
    if pick_combo:
        combos = {v: r for v, r in S.items() if v.startswith("combo")}
        if combos:
            best = max(combos, key=lambda v: combos[v]["val_lesion_ap_mean"])  # best w > 0 on val
            S = {v: r for v, r in S.items() if not v.startswith("combo") or v == best}
    for v, r in S.items():
        if v not in rename and not any(v.startswith(k) for k in rename):
            continue
        name = rename.get(v) or next(rename[k] for k in rename if v.startswith(k))
        if v.startswith("combo"):
            name = f"{name} (w = {v[5:].split('_')[0]})"
        t = r["test_per_seed"]
        rt = [x.get("score_maps") for x in r["runtime_s_per_seed"]]
        ROWS.append(dict(group=group, method=name,
                         slice_auroc=(None, None),
                         voxel_auroc=ms(t["voxel_auroc_in_gland"]), patient_auroc=ms(t["patient_auroc"]),
                         lesion_ap=ms(t["lesion_ap"]), picai=ms(t["picai_score"]),
                         runtime=(f"{SOLO.get((root, 'hdc256'), '~9 s')} (its HDC term)" if v.startswith("combo")
                                  else SOLO.get((root, v)) or (f"{min(rt):.0f} s" if all(x is not None for x in rt) else "")),
                         setting=f"{r['config'].replace('P', 'patch ').replace('_s', ' / stride ')}, {r['setting']}"
                                 f" (val AP {r['val_lesion_ap_mean']:.3f})"))


def fmt(m):
    return "" if m[0] is None else f"{m[0]:.3f} ± {m[1]:.3f}"


def main():
    slice_rows()
    sweep_rows("patch_sweep", "Patch maps, mean+std descriptor (unsigned kNN)",
               {"hdc256": "HDC 256 kNN, unsigned", "float": "float kNN, unsigned"})
    sweep_rows("patch_sweep", "Intensity baselines (no learning, no kNN)",
               {"raw_adc": "raw -ADC (pixel)", "raw_hbv": "raw HBV (pixel)", "raw_mix": "raw (HBV-ADC)/2 (pixel)",
                "smooth_adc": "smoothed -ADC (patch mean)", "smooth_hbv": "smoothed HBV (patch mean)",
                "smooth_mix": "smoothed (HBV-ADC)/2 (patch mean)"})
    sweep_rows("new_stats", "Patch maps, mean+std descriptor: signed / intensity channel / combined",
               {"hdc256_signed": "HDC 256 kNN, signed", "float_signed": "float kNN, signed",
                "hdc256_int": "HDC 256 kNN + intensity channel", "float_int": "float kNN + intensity channel",
                "combo": "smoothed (HBV-ADC)/2 z + w x HDC 256 z"}, pick_combo=True)
    for root, lab in (("hdnn_l2", "layer2"), ("hdnn_l2l3", "layer2+3")):
        sweep_rows(root, f"HDnn: frozen ResNet-18 {lab} features per patch",
                   {"hdc256_signed": f"HDnn {lab} HDC 256, signed", "float_signed": f"PatchCore-like {lab} float kNN, signed",
                    "hdc256": f"HDnn {lab} HDC 256, unsigned", "hdc1024": f"HDnn {lab} HDC 1024, unsigned",
                    "float": f"PatchCore-like {lab} float kNN, unsigned",
                    "combo": f"smoothed (HBV-ADC)/2 z + w x HDnn {lab} HDC 256 z"}, pick_combo=True)

    SOLO.update({(r, v): t for r in ("sup_stats_zone", "sup_stats_grid", "sup_resnet_l2l3_zone", "sup_resnet_l2l3_grid")
                 for v, t in (("proto256", "~10 s"), ("proto256_negplus", "~12 s"), ("vote256", "~9 s"),
                              ("vote256_negplus", "~11 s"), ("hdc256_signed", "~10 s"), ("smooth_mix", "1 s"))})
    for root, lab in (("sup_stats_zone", "mean+std, zone bins"), ("sup_stats_grid", "mean+std, grid bins"),
                      ("sup_resnet_l2l3_zone", "ResNet-18 layer2+3, zone bins"),
                      ("sup_resnet_l2l3_grid", "ResNet-18 layer2+3, grid bins")):
        sweep_rows(root, f"Design A: supervised two-memory HDC ({lab})",
                   {"proto256": f"HDC prototypes + retraining, normal = negatives ({lab})",
                    "proto256_negplus": f"HDC prototypes + retraining, normal = negatives + non-lesion of positives ({lab})",
                    "vote256": f"HDC kNN vote over labelled library ({lab})",
                    "vote256_negplus": f"HDC kNN vote, + non-lesion of positives ({lab})"})
    pb = "results/patient_hdc_stats_zone/summary.json"
    if os.path.exists(pb):
        B = json.load(open(pb))
        for name, r in B["summary"].items():
            pa = r["patient_auroc"]; pc = r.get("picai_score", [None, None])
            ROWS.append(dict(group="Design B: HDDB-style patient record HV (image field from A: proto256_negplus, zone)",
                             method=("image max score as-is (no record), for reference" if name == "image_score_raw_auroc"
                                     else f"record fields: {name}"),
                             slice_auroc=(None, None), voxel_auroc=(None, None), patient_auroc=tuple(pa),
                             lesion_ap=(None, None), picai=tuple(pc) if pc[0] is not None else (None, None),
                             runtime="~5 min (incl. 3-fold cross-fit)", setting="D = 2048, 16 levels, 20 retraining epochs"))

    lines = ["| group | method | slice AUROC | voxel AUROC (gland) | patient AUROC | lesion AP | PI-CAI score | runtime / seed | chosen setting |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in ROWS:
        lines.append(f"| {r['group']} | {r['method']} | {fmt(r['slice_auroc'])} | {fmt(r['voxel_auroc'])} | "
                     f"{fmt(r['patient_auroc'])} | {fmt(r['lesion_ap'])} | {fmt(r['picai'])} | {r['runtime']} | {r['setting']} |")
    md = ("# Lesion task on PI-CAI fold 0: all methods (test, mean ± std over seeds 0, 1, 2)\n\n" + "\n".join(lines) +
          "\n\nRuntime = encode + index + score of val and test per seed (map methods); candidate extraction and "
          "picai_eval add ~30-60 s. ResNet-18 feature maps for all 4,383 slices: 17 s (layer2) / 21 s (layer2+3), once. "
          "Combined score: the table shows the best w > 0 on val; including w = 0 (intensity alone, val AP 0.663), "
          "val picks w = 0 in every case, so the HDC term is not supported by val. Design B PI-CAI = (B patient AUROC + "
          "lesion AP of design A proto256_negplus zone maps) / 2, which is what picai_eval gives with B as the case score. "
          "Whole-slice runs: 8 s (pixels), 32 s (stats), ~2 min (resnet) per seed, "
          "feature extraction included.\n")
    os.makedirs("results", exist_ok=True)
    open("results/final_summary.md", "w").write(md)
    json.dump(ROWS, open("results/final_summary.json", "w"), indent=2)
    print(md)


if __name__ == "__main__":
    main()
