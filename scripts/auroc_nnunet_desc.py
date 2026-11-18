#!/usr/bin/env python3
"""Patient descriptors from the csPCa nnU-Net detectors (Dataset503, 25 splits: 0-4 outer folds, 5-24 nested
nest<f><g>), read from each split's validation softmax (--npz; the validation cases of split s are exactly the
held-out fold of that split, so every record comes from a detector that never saw the case).

Per case (csPCa probability p on the native T2W grid, voxel volume from the prediction's spacing):
  max, top-k voxel means (k = 10, 100, 1000), suspicious volume above p = 0.1 / 0.3 / 0.5 / 0.7 (log mm^3),
  largest component above 0.5 (log mm^3), number of components above 0.5, 3D-smoothed peak, mean p inside the gland,
  per zone (AI zones HeviAI23, resampled to the T2W grid): max and top-100 mean.
Writes <out>/<split name>.pkl = {'val': {}, 'test': {study_id: {name: value}}} (no 'val' group: the probabilities
are used raw, without ECDF scaling).

  python scripts/auroc_nnunet_desc.py --procs 12
"""
import argparse
import glob
import json
import os
import pickle
from multiprocessing import get_context

import numpy as np
from scipy import ndimage

R = "/data/auroc/nnunet/results/Dataset503_PicaiLesionsNested/nnUNetTrainerCELoss_250epochs__nnUNetPlans__3d_fullres"
LABELS = "/data/picai_labels"
ZONE_DIR = "anatomical_delineations/zonal_pz_tz/AI/HeviAI23"
GLAND_DIR = "anatomical_delineations/whole_gland/AI/Bosma22b"


def topm(v, k):
    if len(v) == 0:
        return 0.0
    k = min(k, len(v))
    return float(np.partition(v, len(v) - k)[len(v) - k:].mean())


def one(npz):
    import SimpleITK as sitk
    key = os.path.basename(npz)[:-4]
    p = np.load(npz)["probabilities"][1].astype(np.float32)
    ref = sitk.ReadImage(npz[:-4] + ".nii.gz")
    vv = float(np.prod(ref.GetSpacing()))

    def on_ref(path):
        if not os.path.exists(path):
            return np.zeros(p.shape, np.uint8)
        im = sitk.Resample(sitk.ReadImage(path, sitk.sitkUInt8), ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
        return sitk.GetArrayFromImage(im)
    zone, gland = on_ref(os.path.join(LABELS, ZONE_DIR, key + ".nii.gz")), on_ref(os.path.join(LABELS, GLAND_DIR, key + ".nii.gz")) > 0
    v = p.ravel()
    out = {"nn_max": float(v.max())}
    for k in (10, 100, 1000):
        out[f"nn_top{k}"] = topm(v, k)
    for t in (0.1, 0.3, 0.5, 0.7):
        out[f"nn_vol{t:g}"] = float(np.log1p((v > t).sum() * vv))
    lab, n = ndimage.label(p > 0.5)
    out["nn_cc_max"] = float(np.log1p(np.bincount(lab.ravel())[1:].max() * vv)) if n else 0.0
    out["nn_cc_n"] = float(n)
    out["nn_smax"] = float(ndimage.gaussian_filter(p, sigma=(0.5, 3, 3)).max())
    out["nn_gland_mean"] = float(p[gland].mean()) if gland.any() else float(v.mean())
    for zc in (1, 2):
        vz = p[zone == zc]
        out[f"nn_z{zc}_max"] = float(vz.max()) if len(vz) else 0.0
        out[f"nn_z{zc}_top100"] = topm(vz, 100)
    return int(key.split("_")[1]), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=R)
    ap.add_argument("--names", default="/data/auroc/nnunet/preprocessed/Dataset503_PicaiLesionsNested/split_names.json")
    ap.add_argument("--out", default="/data/auroc/desc/nnunet")
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    names = json.load(open(a.names))
    os.makedirs(a.out, exist_ok=True)
    with get_context("fork").Pool(a.procs) as pool:
        for i, name in enumerate(names):
            dst = os.path.join(a.out, f"{name}.pkl")
            vd = os.path.join(a.results, f"fold_{i}", "validation")
            if os.path.exists(dst) or not os.path.exists(os.path.join(vd, "summary.json")):
                continue
            res = dict(pool.imap_unordered(one, sorted(glob.glob(os.path.join(vd, "*.npz")))))
            pickle.dump({"val": {}, "test": res}, open(dst, "wb"))
            print(name, len(res), flush=True)


if __name__ == "__main__":
    main()
