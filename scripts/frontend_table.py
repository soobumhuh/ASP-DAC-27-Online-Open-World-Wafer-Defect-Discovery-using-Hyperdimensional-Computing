#!/usr/bin/env python3
"""Collect results/frontend_screen/rows/*.json into screen.csv + screen.md and apply the pass rule:
HD-probe AUROC >= ResNet-18 (A0, same adapter, same label rule) + 0.01 overall, and not worse on lesions < 500 voxels."""
import glob
import json
import os

import pandas as pd

OUT = "results/frontend_screen"
BASES = ("A0_resnet18_l2", "A0orig_resnet18_l2")  # adapter and current-pipeline descriptor: must beat both
LEAK = {"A1": "supervised on PI-CAI lesion labels (fold-1 cases may be in its training set)",
        "A3": "self-supervised on PI-CAI images (no labels)"}


NOTES = """## Notes

- Fold-1 test lesions (3D components on the crop, estimated native T2W voxels): 83 in total; 4 < 150, 7 at 150-500, 72 > 500.
  The < 150 and 150-500 columns rest on 4 and 7 lesions (centre rule: 79 and 435 patches; >= 25% rule: 20 and 271), so
  the "not worse on lesions < 500 voxels" part of the pass rule is noisy.
- A0 = ResNet-18 layer 2 through the shared adapter (extra 3 x 3 pooling + bilinear read-out + PCA 128); A0orig = the
  current pipeline's descriptor (nearest cell, no PCA, 134 numbers). The pass rule requires beating both.
- A3 stage 1 has 96 channels, so its PCA keeps 96 (descriptor 102 numbers).
- Extraction time per exam (~15 slices): GPU (RTX PRO 6000 Blackwell) ResNet-18 0.001 s, ProViCNet (3 sequences x
  ViT-S, 448 px) 0.13 s, ProFound (64 x 224 x 224 3D) 0.04 s; CPU (16 threads) ResNet-18 0.07 s, ProViCNet 4.9 s,
  ProFound 0.50 s; reading + resampling the MHA volumes 0.65 s (single process) to 2.2 s (12 parallel workers).
- Cache size (folds 1-4, float16): ResNet-18 layer 2 (all folds, 128 x 28 x 28) 4.4 GB; ProViCNet 1152 x 16 x 16
  10.5 GB per block; ProFound stage 1 96 x 24 x 24 2.0 GB, stage 2 192 x 16 x 16 1.7 GB.

"""


def md(df) -> str:
    """Markdown table without the tabulate dependency."""
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


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
        if name in BASES or any((rule, b) not in hd.index for b in BASES):
            continue
        v = dict(rule=rule, name=name)
        for b in BASES:
            v[f"d_auroc_vs_{b[:6]}"] = round(r.auroc - hd.loc[(rule, b)].auroc, 4)
            v[f"d_lt500_vs_{b[:6]}"] = round(r.auroc_lt500 - hd.loc[(rule, b)].auroc_lt500, 4)
        v["passed"] = all(v[f"d_auroc_vs_{b[:6]}"] >= 0.01 and v[f"d_lt500_vs_{b[:6]}"] >= 0 for b in BASES)
        v["label_exposure"] = LEAK.get(name[:2], "")[:40]
        verdict.append(v)
    vd = pd.DataFrame(verdict)
    df.to_csv(f"{OUT}/screen.csv", index=False)
    with open(f"{OUT}/screen.md", "w") as f:
        f.write("# Front-end patch-level screen (train folds 2-4, test fold 1; fold 0 unused)\n\n")
        f.write("Patch AUROC, lesion patches (rule) vs normal patches (no lesion pixel). Size = estimated native T2W "
                "voxels of the 3D lesion; zone = AI zone at the patch centre; posexam = csPCa-positive exams only.\n\n")
        f.write(md(df) + "\n\n")
        if len(vd):
            f.write("## Pass rule (HD probe vs both A0 variants: +0.01 overall and not worse on lesions < 500 voxels)\n\n")
            f.write(md(vd) + "\n\n")
        f.write(NOTES)
        f.write("Pretraining / label exposure: " + "; ".join(f"{k}: {v}" for k, v in LEAK.items()) + "\n")
    print(open(f"{OUT}/screen.md").read())


if __name__ == "__main__":
    main()
