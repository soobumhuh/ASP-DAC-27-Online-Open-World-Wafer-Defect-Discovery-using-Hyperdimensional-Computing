#!/usr/bin/env python3
"""Collect results/frontend_screen/rows/*.json into screen.csv + screen.md and apply the pass rule:
HD-probe AUROC >= ResNet-18 (A0, same adapter, same label rule) + 0.01 overall, and not worse on lesions < 500 voxels."""
import glob
import json
import os

import pandas as pd

OUT = "results/frontend_screen"
BASE = "A0_resnet18_l2"
LEAK = {"A1": "supervised on PI-CAI lesion labels (fold-1 cases may be in its training set)",
        "A3": "self-supervised on PI-CAI images (no labels)"}


def main():
    rows = [json.load(open(p)) for p in sorted(glob.glob(f"{OUT}/rows/*.json"))]
    df = pd.DataFrame(rows)
    cols = ["name", "rule", "probe", "auroc", "auroc_lt150", "auroc_150to500", "auroc_gt500", "auroc_lt500",
            "auroc_pz", "auroc_tz", "auroc_posexam", "n_pos", "n_pos_lt150", "n_pos_150to500", "n_pos_gt500",
            "raw_dim", "dim", "gpu_s_per_exam", "cpu_s_per_exam", "cache_gb", "fit_s"]
    df = df[[c for c in cols if c in df]].sort_values(["rule", "probe", "name"])
    hd = df[df.probe == "hd"].set_index(["rule", "name"])
    verdict = []
    for (rule, name), r in hd.iterrows():
        if name == BASE or (rule, BASE) not in hd.index:
            continue
        b = hd.loc[(rule, BASE)]
        d_all, d_small = r.auroc - b.auroc, r.auroc_lt500 - b.auroc_lt500
        verdict.append(dict(rule=rule, name=name, d_auroc=round(d_all, 4), d_auroc_lt500=round(d_small, 4),
                            passed=bool(d_all >= 0.01 and d_small >= 0)))
    vd = pd.DataFrame(verdict)
    df.to_csv(f"{OUT}/screen.csv", index=False)
    with open(f"{OUT}/screen.md", "w") as f:
        f.write("# Front-end patch-level screen (train folds 2-4, test fold 1; fold 0 unused)\n\n")
        f.write("Patch AUROC, lesion patches (rule) vs normal patches (no lesion pixel). Size = estimated native T2W "
                "voxels of the 3D lesion; zone = AI zone at the patch centre; posexam = csPCa-positive exams only.\n\n")
        f.write(df.round(4).to_markdown(index=False) + "\n\n")
        if len(vd):
            f.write(f"## Pass rule (HD probe vs {BASE}: +0.01 overall and not worse on lesions < 500 voxels)\n\n")
            f.write(vd.to_markdown(index=False) + "\n\n")
        f.write("Pretraining / label exposure: " + "; ".join(f"{k}: {v}" for k, v in LEAK.items()) + "\n")
    print(open(f"{OUT}/screen.md").read())


if __name__ == "__main__":
    main()
