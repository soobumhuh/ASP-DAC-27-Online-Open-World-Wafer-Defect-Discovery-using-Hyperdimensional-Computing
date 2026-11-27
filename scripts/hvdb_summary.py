#!/usr/bin/env python3
"""Hypervector database, step 4: tables -> results/hvdb/summary.md + summary.json.

Reads hvdb_eval.py output (hvdb_results.json, patient_vote_*.pkl), hvdb_qdrant.py output (qdrant.json), the native-grid
T7 results (native_eval.py) and paired bootstraps vs best_sota (bootstrap_ci.py); adds a paired patient-level bootstrap
of the T3 kNN-vote AUROC against best_sota's merged patient score (the same exams, 1000 resamples).

  python scripts/hvdb_summary.py --eval /data/hvdb/eval --qdrant /data/hvdb/smoke/qdrant.json --native /data/hvdb/native
"""
import argparse
import json
import os
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--eval", default="/data/hvdb/eval")
ap.add_argument("--qdrant", default="/data/hvdb/smoke/qdrant.json")
ap.add_argument("--native", default="/data/hvdb/native")
ap.add_argument("--best", default="/data/sota/base/cv_clean_hknn_cases.pkl")
ap.add_argument("--out", default="results/hvdb")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
R = json.load(open(os.path.join(a.eval, "hvdb_results.json")))
Q = json.load(open(a.qdrant))
K = pickle.load(open(a.best, "rb"))
best = {s: v for f in range(5) for s, v in K[("merged", f)]["case_pred"].items()}
fmt = lambda v: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" if isinstance(v, (list, tuple)) and len(v) == 3 else str(v)
f3 = lambda v: f"{v[0]:.3f}"


def paired_vote(path, n=1000):
    d = pickle.load(open(path, "rb"))
    s, v, y = d["study"], d["vote"], d["y"]
    b = np.array([best[x] for x in s])
    rng = np.random.default_rng(0)
    pt = (roc_auc_score(y, v), roc_auc_score(y, b))
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            diffs.append(roc_auc_score(y[i], v[i]) - roc_auc_score(y[i], b[i]))
    return {"vote_auroc": pt[0], "best_sota_auroc": pt[1], "diff": pt[0] - pt[1],
            "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


out = {"regions": R["regions"], "patients": R["patients"], "sizes_bytes_per_item": R["sizes"],
       "n_items_per_fold": R["n_items"], "gt_isup_exact_fraction": R["gt_isup_exact_fraction"], "qdrant": Q}
L = ["# Hypervector database on best_sota (prostate MRI, PI-CAI public 1,500 exams)", "",
     "All tests out-of-fold over the picai_baseline 5 folds (database side = other folds' exams, candidates and maps of "
     "the nested models nest<f><g>; same patient excluded; encoders fitted on the database side only; labels payload "
     "only). Pooled over folds; 95% CIs = 1,000 patient-level bootstrap resamples. Code: mri_hdc/hvdb.py, "
     "scripts/hvdb_{features,build,eval,qdrant,summary}.py.", ""]
# ---------------- headline table
L += ["## Headline (D = 2,048 bits; patients: k = 3 regions)", "",
      "| item type | T1 p@10 | T1 mAP | T1 Spearman ISUP | T2 AUROC | T3 p@10 (csPCa q) | T3 kNN-vote AUROC | "
      "same-centre@10 (base) regions / patients | T7 PI-CAI (native pooled) |", "|---|---|---|---|---|---|---|---|---|"]
t7 = {}
for t in ("record", "content", "both"):
    p = os.path.join(a.native, f"t7_{t}_merged.json")
    if os.path.exists(p):
        t7[t] = json.load(open(p))["native_pooled"]
for t in ("record", "content", "both", "random"):
    r, p = R["regions"][f"{t}|D2048"], R["patients"][f"{t}|D2048|k3"]
    L.append(f"| {t} | {fmt(r['p@10'])} | {fmt(r['mAP'])} | {fmt(r['spearman_isup'])} | {fmt(r['T2_auroc'])} | "
             f"{fmt(p['p@10_cspca'])} | {fmt(p['vote_auroc'])} | {f3(r['same_centre'])} ({f3(r['base_centre'])}) / "
             f"{f3(p['same_centre'])} ({f3(p['base_centre'])}) | "
             + (f"{t7[t]['picai_score']:.3f} (AUROC {t7[t]['patient_auroc']:.3f}, AP {t7[t]['lesion_ap']:.3f})"
                if t in t7 else "-") + " |")
L += ["", "best_sota reference (native pooled, extras_tier1): PI-CAI 0.655, AP 0.475, AUROC 0.836. "
      "Random rows = random bits (chance level / base rates).", ""]
# ---------------- T1 / T2 / T4 / T5 regions all dims
L += ["## Regions: T1 lesion grade, T2 lesion vs normal, T4 shortcut, T5 binarising cost", "",
      "Queries: ground-truth lesions of the test fold (T1); database = GT lesions + best_sota candidates of the other "
      "folds; relevant = csPCa. T2: GT + sampled normals, same zone, 10-NN vote. float_* = the same descriptors "
      "before binarising (cosine). content_joint = cached joint-RGB ResNet features instead of per-channel.", "",
      "| item | D | p@1 | p@10 | p@40 | mAP | Spearman | T2 AUROC | same-centre@10 | same-scanner@10 | LOCO p@10 | LOCO mAP | recall@10 HV vs float |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for k, r in R["regions"].items():
    t, D = k.split("|")
    g = lambda m: fmt(r[m]) if m in r else "-"
    L.append(f"| {t} | {D[1:]} | {g('p@1')} | {g('p@10')} | {g('p@40')} | {g('mAP')} | {g('spearman_isup')} | "
             f"{g('T2_auroc')} | {f3(r['same_centre'])} (base {f3(r['base_centre'])}) | {f3(r['same_scanner'])} "
             f"(base {f3(r['base_scanner'])}) | {g('loco_p@10')} | {g('loco_mAP')} | {g('recall@10_vs_float')} |")
# ---------------- T3 patients
L += ["", "## Patients: T3 retrieval, T4 shortcut (grid over D and k bundled regions)", "",
      "Patient HV = majority bundle of top-k candidate region HVs x rank roles + gland content x GLAND (content) + "
      "clinical record x CLIN; 'both' = content || record patient HVs. Vote k from {5,10,20,40} by leave-one-database-"
      "fold-out. Paired diff = kNN-vote AUROC minus best_sota merged patient AUROC on the same exams.", "",
      "| item | D | k reg | p@10 csPCa | agree@10 | GG agree@10 | vote AUROC | diff vs best_sota [95% CI] | LOCO p@10 | LOCO vote AUROC | same-centre@10 (base) | same-scanner@10 (base) |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
pdiff = {}
for k, r in R["patients"].items():
    parts = k.split("|")
    t, D = parts[0], parts[1]
    kr = parts[2] if len(parts) > 2 else "k3"
    pv = os.path.join(a.eval, f"patient_vote_{t}_{D}_{kr}.pkl")
    if os.path.exists(pv):
        pdiff[k] = paired_vote(pv)
    dd = pdiff.get(k)
    ds = f"{dd['diff']:+.3f} [{dd['diff_ci95'][0]:+.3f}, {dd['diff_ci95'][1]:+.3f}]" if dd else "-"
    L.append(f"| {t} | {D[1:]} | {kr[1:]} | {fmt(r['p@10_cspca'])} | {fmt(r['agree@10'])} | {fmt(r['gg_agree@10'])} | "
             f"{fmt(r['vote_auroc'])} | {ds} | {fmt(r['loco_p@10_cspca'])} | {fmt(r['loco_vote_auroc'])} | "
             f"{f3(r['same_centre'])} ({f3(r['base_centre'])}) | {f3(r['same_scanner'])} ({f3(r['base_scanner'])}) |")
out["patients_paired_vs_best_sota"] = pdiff
fl = R["patients"].get("float|D2048|k3", {})
if fl:
    L += ["", "T5 (patients): recall@10 of patient-HV neighbours vs the float patient descriptor (cosine on block-"
          "normalised region mean / gland / clinical z): " + ", ".join(
              f"{t} {fmt(fl[f'recall@10_{t}_vs_float'])}" for t in ("record", "content", "both")
              if f"recall@10_{t}_vs_float" in fl)]
# ---------------- T7
L += ["", "## T7: candidates scored only from database neighbours' labels (official picai_eval, native grid, pooled)", "",
      "Score = (TP neighbours + prior) / (k + 1) over the zone-filtered k nearest database candidates (k by nested "
      "LOFO on the database folds; no prototype re-scorer, no detector confidence). Patient score: max candidate "
      "score (image only) or merged with the clinical record (w = 0.25, best_sota's weight). Paired bootstrap vs "
      "best_sota over exams (bootstrap_ci.py).", "",
      "| run | PI-CAI | lesion AP | patient AUROC | Δ PI-CAI vs best_sota [95% CI] | Δ AP | Δ AUROC |", "|---|---|---|---|---|---|---|"]
bp = os.path.join(a.native, "boot_t7.json")
B = json.load(open(bp)) if os.path.exists(bp) else {}
out["t7_bootstrap"] = B
for name, r in B.items():
    c = r["cv_pooled"]
    ref = [k for k in c["picai_score"] if k.startswith("diff_vs_")][0]
    cell = lambda m: f"{c[m][ref]:+.3f} [{c[m]['diff_ci95'][0]:+.3f}, {c[m]['diff_ci95'][1]:+.3f}]"
    L.append(f"| {name} | {fmt([c['picai_score']['value']] + c['picai_score']['ci95'])} | "
             f"{fmt([c['lesion_ap']['value']] + c['lesion_ap']['ci95'])} | "
             f"{fmt([c['patient_auroc']['value']] + c['patient_auroc']['ci95'])} | {cell('picai_score')} | "
             f"{cell('lesion_ap')} | {cell('patient_auroc')} |")
cp = os.path.join(a.eval, "t7_crop.json")
if os.path.exists(cp):
    out["t7_crop_per_fold"] = json.load(open(cp))
    L += ["", "T7 chosen k per fold: " + "; ".join(f"{t} {[r['k'] for r in v]}" for t, v in out["t7_crop_per_fold"].items())]
# ---------------- Qdrant
L += ["", "## Store: exact numpy index vs Qdrant 1.18.3 (D = 2,048, k = 10)", "",
      f"Regions: {Q['collections']['sohuh_hvdb_regions_bq']['points']} points (database items of all 5 outer folds, "
      f"filtered by outer_fold); patients: {Q['collections']['sohuh_hvdb_patients_bq']['points']}. Recall is tie-aware "
      "(an item counts if its exact Hamming distance <= the exact 10th distance). Latency = single REST query from "
      "this container, median / p95 ms. Collections were deleted after the test.", "",
      "| collection | vector | search mode | recall@10 | latency median | p95 |", "|---|---|---|---|---|---|"]
for c, v in Q["collections"].items():
    for k, x in v.items():
        if isinstance(x, dict) and "recall@k" in x:
            vn, mode = k.split("/")
            L.append(f"| {c} | {vn} | {mode} | {x['recall@k']:.3f} | {x['latency_ms_median']:.2f} | {x['latency_ms_p95']:.2f} |")
    L.append(f"| {c} | content | numpy exact (in-process) | 1.000 | {v['numpy_exact_latency_ms_median']:.2f} | - |")
L += ["", "Memory per item (vectors): numpy packed D/8 = 256 B per 2,048-bit HV (both = 512 B); Qdrant BQ layout "
      "float16 +-1 on disk 4,096 B + 256 B 1-bit code in RAM; uint8 / Manhattan layout 2,048 B. Mean payload JSON: "
      f"regions {Q['collections']['sohuh_hvdb_regions_bq']['payload_bytes_mean']:.0f} B, patients "
      f"{Q['collections']['sohuh_hvdb_patients_bq']['payload_bytes_mean']:.0f} B.", "",
      f"Items per fold (fold 0): {R['n_items']['0']}. Ground-truth lesion ISUP unambiguous for "
      f"{R['gt_isup_exact_fraction']:.0%} of GT queries (others: ISUPs assigned to components by volume rank)."]
open(os.path.join(a.out, "summary.md"), "w").write("\n".join(L) + "\n")
json.dump(out, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
print("\n".join(L[:20]))
