#!/usr/bin/env python3
"""Gland + zone segmenter (evaluation plan step b): nnU-Net v2 dataset from the PI-CAI public exams, so the detector
can run on exams that come without the PI-CAI AI masks (hidden test, Prostate158).

Input channels: T2W, ADC, HBV, with ADC / HBV resampled onto the T2W grid (linear), as in data.build_slices.
Label (T2W grid, nearest): 1 = PZ, 2 = TZ (AI zones HeviAI23), 3 = gland without a zone label (AI gland Bosma22b);
gland = label > 0. These are AI-derived anatomy masks only; no lesion or csPCa information enters the segmenter.
splits_final.json = the five public folds (identical to picai_baseline), so nnU-Net fold f never sees fold f and its
validation predictions are out-of-fold masks for every exam; the 'all' model is the final segmenter.

  uv run python scripts/seg_dataset.py --out /data/sota/nnunet/raw/Dataset501_PicaiGlandZones
"""
import argparse
import json
import os
from multiprocessing import get_context

import numpy as np

import _path  # noqa: F401
from mri_hdc import data as D

A = {}


def one(key):
    import SimpleITK as sitk
    f, L, out = A["files"][key], A["labels"], A["out"]
    t2 = sitk.ReadImage(f["t2w"], sitk.sitkFloat32)
    for i, s in enumerate(("t2w", "adc", "hbv")):
        im = t2 if s == "t2w" else sitk.Resample(sitk.ReadImage(f[s], sitk.sitkFloat32), t2, sitk.Transform(),
                                                 sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        p = os.path.join(out, "imagesTr", f"{key}_{i:04d}.nii.gz")
        if not os.path.exists(p):
            sitk.WriteImage(im, p)
    g = D.on_grid(sitk.ReadImage(os.path.join(L, D.GLAND_DIR, key + ".nii.gz"), sitk.sitkUInt8), t2, True) > 0
    zp = os.path.join(L, D.ZONE_DIR, key + ".nii.gz")
    z = D.on_grid(sitk.ReadImage(zp, sitk.sitkUInt8), t2, True) if os.path.exists(zp) else np.zeros(g.shape, np.uint8)
    lab = np.where(z == 1, 1, np.where(z == 2, 2, np.where(g, 3, 0))).astype(np.uint8)
    li = sitk.GetImageFromArray(lab)
    li.CopyInformation(t2)
    sitk.WriteImage(li, os.path.join(out, "labelsTr", key + ".nii.gz"))
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="/data/picai_images")
    ap.add_argument("--labels", default="/data/picai_labels")
    ap.add_argument("--slices", default="/data/picai_all_masks.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    for s in ("imagesTr", "labelsTr"):
        os.makedirs(os.path.join(a.out, s), exist_ok=True)
    d = np.load(a.slices, mmap_mode="r")
    fold = {f"{p}_{s}": int(f) for p, s, f in zip(d["patient_id"], d["study_id"], d["fold"])}
    A.update(files=D.index_mha_files(a.images), labels=a.labels, out=a.out)
    keys = sorted(fold)
    with get_context("fork").Pool(a.workers) as pool:
        for i, k in enumerate(pool.imap_unordered(one, keys, chunksize=4)):
            if i % 100 == 0:
                print(i, k, flush=True)
    json.dump({"channel_names": {"0": "T2W", "1": "ADC", "2": "HBV"},
               "labels": {"background": 0, "PZ": 1, "TZ": 2, "gland_other": 3},
               "numTraining": len(keys), "file_ending": ".nii.gz"}, open(os.path.join(a.out, "dataset.json"), "w"), indent=1)
    splits = [{"train": [k for k in keys if fold[k] != f], "val": [k for k in keys if fold[k] == f]} for f in range(5)]
    json.dump(splits, open(os.path.join(a.out, "splits_final.json"), "w"))
    print("done", len(keys))


if __name__ == "__main__":
    main()
