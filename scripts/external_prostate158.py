#!/usr/bin/env python3
"""External test on Prostate158 (Adams et al. 2022; 139 + 19 exams, 3T, Charite Berlin), FROZEN final model, run once.

Stages
  nnunet_in   T2W / ADC / high-b DWI as nnU-Net input (<key>_0000/1/2.nii.gz) for our gland + zone segmenter
              (scripts/seg_dataset.py; the 'all' model trained on the 1,500 PI-CAI exams). Prostate158's own expert
              anatomy masks are NOT used: at inference the detector only gets what a new exam would get.
  build       slices exactly as data.build_slices (gland-masked percentile clip + z-score per sequence, square gland
              box + 25% margin + 4 px, slices with >= 2% gland, 96 x 96), anatomy from the segmenter predictions;
              absolute ADC crops (re-scorer fields); reference lesion masks on the T2W grid for native evaluation.
Reference standard: t2_tumor_reader1 (radiologist outline of PI-RADS >= 4 lesions; Prostate158 has no per-lesion
histopathology label like PI-CAI's ISUP >= 2), empty when the exam has none; reader 2 (ADC outlines, test subset) is
kept for an inter-reader check. Keys: p158_<ID>; study ids 9000000 + ID (no overlap with PI-CAI).

  uv run python scripts/external_prostate158.py nnunet_in --out /data/sota/external/prostate158/nnunet_in
  uv run python scripts/external_prostate158.py build --seg /data/sota/external/prostate158/seg \
      --out /data/sota/external/prostate158/slices.npz
"""
import argparse
import csv
import os

import numpy as np

import _path  # noqa: F401
from mri_hdc import data as D

ROOT = "/data/sota/external/prostate158"


def cases(root=ROOT):
    out = []
    for part, csvp in (("train", "train/prostate158_train/train.csv"), ("train", "train/prostate158_train/valid.csv"),
                       ("test", "test/prostate158_test/test.csv")):
        base = os.path.join(root, part, f"prostate158_{part}")
        for r in csv.DictReader(open(os.path.join(root, csvp))):
            f = {k: (os.path.join(base, v) if v else "") for k, v in r.items() if k != "ID"}
            out.append((f"p158_{int(r['ID']):03d}", int(r["ID"]), part, f))
    return sorted(out)


def read(p, ref=None, nearest=False):
    import SimpleITK as sitk
    im = sitk.ReadImage(p, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(im) if ref is None else D.on_grid(im, ref, nearest)


def lesion(f, t2, key="t2_tumor_reader1"):
    p = f.get(key, "")
    if not p or not os.path.exists(p) or p.endswith("empty.nii.gz"):
        return None
    return read(p, t2, True) > 0


def nnunet_in(a):
    import SimpleITK as sitk
    os.makedirs(a.out, exist_ok=True)
    for key, _, _, f in cases():
        t2 = sitk.ReadImage(f["t2"], sitk.sitkFloat32)
        for i, s in enumerate(("t2", "adc", "dwi")):
            im = t2 if s == "t2" else sitk.Resample(sitk.ReadImage(f[s], sitk.sitkFloat32), t2, sitk.Transform(),
                                                    sitk.sitkLinear, 0.0, sitk.sitkFloat32)
            sitk.WriteImage(im, os.path.join(a.out, f"{key}_{i:04d}.nii.gz"))
    print("wrote", len(cases()), "cases")


def build(a):
    import SimpleITK as sitk
    from skimage.transform import resize
    S = 96
    keys = ["images", "patient_id", "study_id", "slice_z", "case_cspca", "lesion_source", "case_isup", "lesion_pixels",
            "gland_pixels", "lesion_mask", "gland_mask", "rel_z", "zone_mask"]
    out = {k: [] for k in keys}
    adc_raw = []
    os.makedirs(a.labels, exist_ok=True)
    for key, cid, part, f in cases():
        t2 = sitk.ReadImage(f["t2"], sitk.sitkFloat32)
        vols = [sitk.GetArrayFromImage(t2), read(f["adc"], t2), read(f["dwi"], t2)]
        gland, zone = D.read_anatomy("", key, t2, seg_dir=a.seg)
        les = lesion(f, t2)
        les = np.zeros(gland.shape, bool) if les is None else les
        lab = sitk.GetImageFromArray(les.astype(np.uint8))
        lab.CopyInformation(t2)
        sitk.WriteImage(lab, os.path.join(a.labels, key + ".nii.gz"))  # native-grid reference (reader 1)
        r2 = lesion(f, t2, "adc_tumor_reader2")
        if r2 is not None:
            lab = sitk.GetImageFromArray(r2.astype(np.uint8))
            lab.CopyInformation(t2)
            sitk.WriteImage(lab, os.path.join(a.labels, key + "_reader2.nii.gz"))
        if gland.sum() < 100:
            print("skip (segmenter found no gland):", key, flush=True)
            continue
        raw_adc = vols[1].copy()
        vols = [D.normalise(v, gland) for v in vols]
        y0, y1, x0, x1 = D.square_box(gland, 0.25, gland.shape)
        area = (y1 - y0) * (x1 - x0)
        kept = [z for z in range(gland.shape[0]) if gland[z, y0:y1, x0:x1].sum() >= 0.02 * area]
        pos = int(les.any())
        rs = lambda m: resize(m[y0:y1, x0:x1].astype(np.float32), (S, S), order=0, preserve_range=True,
                              anti_aliasing=False).astype(np.uint8)
        for z in kept:
            out["images"].append(np.stack([resize(v[z, y0:y1, x0:x1], (S, S), order=1, preserve_range=True,
                                                  anti_aliasing=True) for v in vols]).astype(np.float16))
            adc_raw.append(resize(raw_adc[z, y0:y1, x0:x1], (S, S), order=1, preserve_range=True,
                                  anti_aliasing=True).astype(np.float32))
            out["patient_id"].append(900000 + cid)
            out["study_id"].append(9000000 + cid)
            out["slice_z"].append(z)
            out["case_cspca"].append(pos)
            out["lesion_source"].append(1 if pos else 0)
            out["case_isup"].append(0)
            out["lesion_pixels"].append(int(les[z, y0:y1, x0:x1].sum()))
            out["gland_pixels"].append(int(gland[z, y0:y1, x0:x1].sum()))
            out["lesion_mask"].append(rs(les[z]))
            out["gland_mask"].append(rs(gland[z]))
            out["zone_mask"].append(rs(zone[z]))
        out["rel_z"].extend(np.linspace(0.0, 1.0, len(kept)) if len(kept) > 1 else [0.5] * len(kept))
        print(key, part, "pos" if pos else "neg", len(kept), "slices", flush=True)
    arr = {k: np.asarray(v) for k, v in out.items()}
    arr["fold"] = np.full(len(arr["patient_id"]), -1)
    np.savez(a.out, **arr)
    np.save(a.out.replace(".npz", "_adc_raw.npy"), np.asarray(adc_raw, np.float32))
    print("saved", a.out, len(arr["patient_id"]), "slices,", len(np.unique(arr["study_id"])), "exams,",
          len(np.unique(arr["study_id"][arr["case_cspca"] == 1])), "positive")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["nnunet_in", "build"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg", default=os.path.join(ROOT, "seg"))
    ap.add_argument("--labels", default=os.path.join(ROOT, "labels"))
    a = ap.parse_args()
    {"nnunet_in": nnunet_in, "build": build}[a.stage](a)
