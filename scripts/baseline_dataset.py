#!/usr/bin/env python3
"""Evaluation plan (g): nnU-Net v2 csPCa detection baseline on exactly our setup: same 1,500 exams, same three input
sequences (the images of Dataset501, T2W grid), labels = expert lesion masks (human_expert/Pooch25, else
human_expert/resampled; empty for csPCa-negative exams), same five folds (= picai_baseline). Detection maps are
turned into lesion candidates with the same report_guided_annotation extraction as picai_baseline.

  uv run python scripts/baseline_dataset.py --src /data/sota/nnunet/raw/Dataset501_PicaiGlandZones \
      --out /data/sota/nnunet/raw/Dataset502_PicaiLesions
"""
import argparse
import glob
import json
import os
import shutil
from multiprocessing import get_context

import numpy as np

import _path  # noqa: F401
from mri_hdc import data as D

A = {}


def one(key):
    import SimpleITK as sitk
    ref = sitk.ReadImage(os.path.join(A["src"], "labelsTr", key + ".nii.gz"))
    p = D.lesion_mask_path(A["labels"], key)
    lab = (D.on_grid(sitk.ReadImage(p, sitk.sitkUInt8), ref, True) > 0).astype(np.uint8) if p and A["pos"][key] \
        else np.zeros(sitk.GetArrayFromImage(ref).shape, np.uint8)
    li = sitk.GetImageFromArray(lab)
    li.CopyInformation(ref)
    sitk.WriteImage(li, os.path.join(A["out"], "labelsTr", key + ".nii.gz"))
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default="/data/picai_labels")
    ap.add_argument("--slices", default="/data/picai_all_masks.npz")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "labelsTr"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "imagesTr"), exist_ok=True)
    for p in glob.glob(os.path.join(a.src, "imagesTr", "*.nii.gz")):  # same images, linked
        q = os.path.join(a.out, "imagesTr", os.path.basename(p))
        if not os.path.exists(q):
            os.symlink(p, q)
    d = np.load(a.slices, mmap_mode="r")
    pos = {f"{p}_{s}": int(c) for p, s, c in zip(d["patient_id"], d["study_id"], d["case_cspca"])}
    A.update(src=a.src, out=a.out, labels=a.labels, pos=pos)
    keys = sorted(pos)
    with get_context("fork").Pool(12) as pool:
        pool.map(one, keys, chunksize=8)
    json.dump({"channel_names": {"0": "T2W", "1": "ADC", "2": "HBV"}, "labels": {"background": 0, "csPCa": 1},
               "numTraining": len(keys), "file_ending": ".nii.gz"}, open(os.path.join(a.out, "dataset.json"), "w"), indent=1)
    shutil.copy(os.path.join(a.src, "splits_final.json"), os.path.join(a.out, "splits_final.json"))
    print("done", len(keys))


if __name__ == "__main__":
    main()
