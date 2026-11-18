#!/usr/bin/env python3
"""Dataset503_PicaiLesionsNested: the eval plan's Dataset502 (T2W/ADC/HBV on the T2W grid, expert csPCa lesion
labels, official folds) with 25 splits for nested stacking: 0-4 = outer folds (train = the other four folds,
val = fold f), 5-24 = nest<f><g> (train = the three folds that are neither f nor g, val = fold g). Raw images and
preprocessed data are symlinked from Dataset502 (read-only); only plans name and splits change.
"""
import json
import os

S502 = "/data/sota/nnunet/preprocessed/Dataset502_PicaiLesions"
RAW502 = "/data/sota/nnunet/raw/Dataset502_PicaiLesions"
T = "/data/auroc/nnunet/preprocessed/Dataset503_PicaiLesionsNested"
R = "/data/auroc/nnunet/raw/Dataset503_PicaiLesionsNested"
os.makedirs(T, exist_ok=True)
os.makedirs(R, exist_ok=True)
for src, dst in ((f"{S502}/nnUNetPlans_3d_fullres", f"{T}/nnUNetPlans_3d_fullres"), (f"{S502}/gt_segmentations", f"{T}/gt_segmentations"),
                 (f"{RAW502}/imagesTr", f"{R}/imagesTr"), (f"{RAW502}/labelsTr", f"{R}/labelsTr")):
    if not os.path.exists(dst):
        os.symlink(src, dst)
for f in ("dataset.json", "dataset_fingerprint.json"):
    open(f"{T}/{f}", "w").write(open(f"{S502}/{f}").read())
open(f"{R}/dataset.json", "w").write(open(f"{S502}/dataset.json").read())
p = json.load(open(f"{S502}/nnUNetPlans.json"))
p["dataset_name"] = "Dataset503_PicaiLesionsNested"
json.dump(p, open(f"{T}/nnUNetPlans.json", "w"), indent=1)
fold = [set(x["val"]) for x in json.load(open(f"{S502}/splits_final.json"))]
allc = set().union(*fold)
out = [{"train": sorted(allc - fold[f]), "val": sorted(fold[f])} for f in range(5)]
names = [f"fold{f}" for f in range(5)]
for f in range(5):
    for g in range(5):
        if g != f:
            out.append({"train": sorted(allc - fold[f] - fold[g]), "val": sorted(fold[g])})
            names.append(f"nest{f}{g}")
json.dump(out, open(f"{T}/splits_final.json", "w"))
json.dump(names, open(f"{T}/split_names.json", "w"))
