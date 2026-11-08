#!/usr/bin/env python3
"""
picai_slices.py - turn PI-CAI 3D MRI cases into 2D slice arrays.

Input (download separately):
  --images_dir   folder holding the unzipped picai_public_images_fold*/ folders
                 (each case is <pid>/<pid>_<sid>_t2w.mha, _adc.mha, _hbv.mha)
  --labels_dir   a clone of https://github.com/DIAGNijmegen/picai_labels

Output: one .npz with
  images        (N, 3, S, S) float16   channels = T2W, ADC, HBV, prostate-cropped
  patient_id, study_id, slice_z        which case and slice each image came from
  case_cspca    1 if the case is csPCa-positive (marksheet case_csPCa == YES)
  case_isup     case ISUP grade from the marksheet
  lesion_pixels csPCa lesion pixels in this slice (expert mask, on the T2W grid)
  gland_pixels  prostate pixels in this slice (AI whole-gland mask)
  with --save_masks, also:
  lesion_mask   (N, S, S) uint8  csPCa lesion mask on the crop (nearest-neighbour resize)
  gland_mask    (N, S, S) uint8  prostate mask on the crop
  rel_z         slice position among the case's kept slices, 0 = first (lowest z) .. 1 = last
  with --zonal_dir as well:
  zone_mask     (N, S, S) uint8  AI zonal anatomy on the crop: 1 = peripheral zone (PZ), 2 = transition zone (TZ),
                0 = outside / not labelled (picai_labels anatomical_delineations/zonal_pz_tz/AI/<set>)

What it does per case:
  1. read T2W and resample ADC + HBV + masks onto the T2W grid
  2. normalise each channel inside the prostate: clip to 1st-99th percentile, then z-score
  3. crop a square box around the prostate (plus a margin) and resize to S x S
  4. keep only slices that contain enough prostate
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import SimpleITK as sitk
from skimage.transform import resize


def index_mha_files(images_dir: str) -> dict:
    """Map '<pid>_<sid>' -> {'t2w': path, 'adc': path, 'hbv': path}."""
    out: dict = {}
    for root, _, files in os.walk(images_dir):
        for f in files:
            if not f.endswith(".mha"):
                continue
            parts = f[:-4].split("_")
            if len(parts) != 3:
                continue
            key, seq = f"{parts[0]}_{parts[1]}", parts[2]
            out.setdefault(key, {})[seq] = os.path.join(root, f)
    return out


def on_grid(img: sitk.Image, ref: sitk.Image, nearest: bool) -> np.ndarray:
    """Resample img onto ref's voxel grid and return a (z, y, x) array."""
    interp = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
    out = sitk.Resample(img, ref, sitk.Transform(), interp, 0.0, img.GetPixelID())
    return sitk.GetArrayFromImage(out)


def normalise(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = vol[mask] if mask.sum() > 100 else vol.ravel()
    lo, hi = np.percentile(vals, [1, 99])
    vol = np.clip(vol, lo, hi)
    vals = np.clip(vals, lo, hi)
    return ((vol - vals.mean()) / (vals.std() + 1e-6)).astype(np.float32)


def square_box(mask: np.ndarray, margin: float, shape) -> tuple:
    ys, xs = np.where(mask.any(axis=0))
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 * (1 + margin) + 4
    y0, y1 = int(max(cy - half, 0)), int(min(cy + half, shape[1]))
    x0, x1 = int(max(cx - half, 0)), int(min(cx + half, shape[2]))
    return y0, y1, x0, x1


def lesion_mask_path(labels_dir: str, key: str):
    for sub in ("csPCa_lesion_delineations/human_expert/Pooch25",
                "csPCa_lesion_delineations/human_expert/resampled"):
        p = os.path.join(labels_dir, sub, key + ".nii.gz")
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--out", default="./picai_slices.npz")
    ap.add_argument("--size", type=int, default=64, help="output slice size S")
    ap.add_argument("--margin", type=float, default=0.25, help="extra border around the prostate box")
    ap.add_argument("--min_gland_frac", type=float, default=0.02,
                    help="keep a slice only if prostate covers this fraction of the crop")
    ap.add_argument("--max_neg", type=int, default=0, help="cap on csPCa-negative cases (0 = all found)")
    ap.add_argument("--max_pos", type=int, default=0, help="cap on csPCa-positive cases (0 = all found)")
    ap.add_argument("--zonal_dir", default=None,
                    help="with --save_masks: folder of AI zonal masks (e.g. <labels>/anatomical_delineations/"
                         "zonal_pz_tz/AI/HeviAI23); adds zone_mask (1 = PZ, 2 = TZ)")
    ap.add_argument("--save_masks", action="store_true",
                    help="also save per-slice lesion/gland masks on the crop and the relative slice position")
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; refusing to overwrite")

    files = index_mha_files(args.images_dir)
    with open(os.path.join(args.labels_dir, "clinical_information", "marksheet.csv")) as f:
        rows = [r for r in csv.DictReader(f) if f"{r['patient_id']}_{r['study_id']}" in files]
    print(f"{len(files)} cases on disk, {len(rows)} with marksheet rows")

    n_seen = {0: 0, 1: 0}
    caps = {0: args.max_neg, 1: args.max_pos}
    out = {k: [] for k in ("images", "patient_id", "study_id", "slice_z", "case_cspca",
                           "case_isup", "lesion_pixels", "gland_pixels")}
    if args.save_masks:
        out.update(lesion_mask=[], gland_mask=[], rel_z=[])
        if args.zonal_dir:
            out["zone_mask"] = []
    S = args.size
    for r in rows:
        key = f"{r['patient_id']}_{r['study_id']}"
        pos = int(r["case_csPCa"].strip().upper() == "YES")
        if caps[pos] and n_seen[pos] >= caps[pos]:
            continue
        seqs = files[key]
        if not all(s in seqs for s in ("t2w", "adc", "hbv")):
            print("skip (missing sequence):", key)
            continue
        gland_p = os.path.join(args.labels_dir, "anatomical_delineations/whole_gland/AI/Bosma22b", key + ".nii.gz")
        lesion_p = lesion_mask_path(args.labels_dir, key)
        if not os.path.exists(gland_p) or (pos and lesion_p is None):
            print("skip (missing mask):", key)
            continue

        t2w = sitk.ReadImage(seqs["t2w"], sitk.sitkFloat32)
        vols = [sitk.GetArrayFromImage(t2w)]
        for s in ("adc", "hbv"):
            vols.append(on_grid(sitk.ReadImage(seqs[s], sitk.sitkFloat32), t2w, nearest=False))
        gland = on_grid(sitk.ReadImage(gland_p, sitk.sitkUInt8), t2w, nearest=True) > 0
        if lesion_p:
            lesion = on_grid(sitk.ReadImage(lesion_p, sitk.sitkUInt8), t2w, nearest=True) > 0
        else:
            lesion = np.zeros_like(gland)
        zone = None
        if args.save_masks and args.zonal_dir:
            zp = os.path.join(args.zonal_dir, key + ".nii.gz")
            zone = (on_grid(sitk.ReadImage(zp, sitk.sitkUInt8), t2w, nearest=True) if os.path.exists(zp)
                    else np.zeros(gland.shape, np.uint8))
        if gland.sum() < 100:
            print("skip (empty prostate mask):", key)
            continue

        vols = [normalise(v, gland) for v in vols]
        y0, y1, x0, x1 = square_box(gland, args.margin, gland.shape)
        crop_area = (y1 - y0) * (x1 - x0)
        n_kept = 0
        for z in range(gland.shape[0]):
            g = gland[z, y0:y1, x0:x1]
            if g.sum() < args.min_gland_frac * crop_area:
                continue
            img = np.stack([resize(v[z, y0:y1, x0:x1], (S, S), order=1, preserve_range=True, anti_aliasing=True)
                            for v in vols]).astype(np.float16)
            out["images"].append(img)
            out["patient_id"].append(int(r["patient_id"]))
            out["study_id"].append(int(r["study_id"]))
            out["slice_z"].append(z)
            out["case_cspca"].append(pos)
            out["case_isup"].append(int(r["case_ISUP"] or 0))
            out["lesion_pixels"].append(int(lesion[z, y0:y1, x0:x1].sum()))
            out["gland_pixels"].append(int(g.sum()))
            if args.save_masks:
                pairs = [("lesion_mask", lesion), ("gland_mask", gland)] + ([("zone_mask", zone)] if zone is not None else [])
                for name, m in pairs:
                    out[name].append(resize(m[z, y0:y1, x0:x1].astype(np.float32), (S, S), order=0,
                                            preserve_range=True, anti_aliasing=False).astype(np.uint8))
            n_kept += 1
        if args.save_masks:
            out["rel_z"].extend(np.linspace(0.0, 1.0, n_kept) if n_kept > 1 else [0.5] * n_kept)
        n_seen[pos] += 1
        print(f"{key}: csPCa={'YES' if pos else 'NO '} kept {n_kept} slices")

    arrays = {k: np.asarray(v) for k, v in out.items()}
    np.savez_compressed(args.out, **arrays)
    print(f"saved {len(arrays['images'])} slices from {n_seen[0]} negative + {n_seen[1]} positive cases -> {args.out}")


if __name__ == "__main__":
    main()
