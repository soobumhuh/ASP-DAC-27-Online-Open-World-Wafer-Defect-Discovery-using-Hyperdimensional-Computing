#!/usr/bin/env python3
"""Absolute ADC (um^2/s) on the crop grid of every slice of the slice file, in the same order: same T2W resampling,
gland box and kept slices as data.build_slices, but without the per-exam z-scoring. Used only for re-scorer fields.

  uv run python scripts/build_adc_raw.py --slices /data/picai_all_masks.npz --out mri_feature_cache/adc_raw.npy
"""
import argparse
import os
from multiprocessing import get_context

import numpy as np

import _path  # noqa: F401
from mri_hdc import data as D

A = {}


def one(item):
    import SimpleITK as sitk
    from skimage.transform import resize
    key, n = item
    files = A["files"][key]
    t2w = sitk.ReadImage(files["t2w"], sitk.sitkFloat32)
    adc = D.on_grid(sitk.ReadImage(files["adc"], sitk.sitkFloat32), t2w, False)
    gl = D.on_grid(sitk.ReadImage(os.path.join(A["labels"], D.GLAND_DIR, key + ".nii.gz"), sitk.sitkUInt8), t2w, True) > 0
    y0, y1, x0, x1 = D.square_box(gl, 0.25, gl.shape)
    area = (y1 - y0) * (x1 - x0)
    kept = [z for z in range(gl.shape[0]) if gl[z, y0:y1, x0:x1].sum() >= 0.02 * area]
    assert len(kept) == n, (key, len(kept), n)
    return key, np.stack([resize(adc[z, y0:y1, x0:x1], (96, 96), order=1, preserve_range=True, anti_aliasing=True)
                          for z in kept]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="/data/picai_all_masks.npz")
    ap.add_argument("--images_dir", default="/data/picai_images")
    ap.add_argument("--labels_dir", default="/data/picai_labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    d = np.load(args.slices, mmap_mode="r")
    sid, pid, z = np.asarray(d["study_id"]), np.asarray(d["patient_id"]), np.asarray(d["slice_z"])
    A.update(files=D.index_mha_files(args.images_dir), labels=args.labels_dir)
    studies = list(dict.fromkeys(sid.tolist()))
    items = [(f"{int(pid[np.where(sid == s)[0][0]])}_{s}", int((sid == s).sum())) for s in studies]
    out = np.zeros((len(sid), 96, 96), np.float32)
    with get_context("fork").Pool(args.workers) as pool:
        for key, arr in pool.imap_unordered(one, items, chunksize=4):
            s = int(key.split("_")[1])
            j = np.where(sid == s)[0]
            out[j[np.argsort(z[j])]] = arr
    np.save(args.out, out)
    print("saved", args.out, out.shape, "median gland-slice ADC", float(np.median(out)))


if __name__ == "__main__":
    main()
