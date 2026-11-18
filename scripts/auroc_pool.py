#!/usr/bin/env python3
"""Frozen ResNet-18 layer-2 features pooled per case and AI zone (mean and max over the zone's pixels of all gland
slices; 2 zones x 2 poolings x 128 channels = 512 values). Label-free and split-free (the front end is frozen).

  python scripts/auroc_pool.py --out /data/auroc/resnet_pool.pkl
"""
import argparse
import pickle

import numpy as np
from skimage.transform import resize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="/data/picai_all_masks.npz")
    ap.add_argument("--features", default="/workspace/wafer-hdc/mri_feature_cache/resnet_l2.npy")
    ap.add_argument("--out", default="/data/auroc/resnet_pool.pkl")
    a = ap.parse_args()
    d = np.load(a.slices)
    F = np.load(a.features, mmap_mode="r")
    sid, zone, gland = d["study_id"], d["zone_mask"], d["gland_mask"]
    out = {}
    for k in np.unique(sid):
        j = np.where(sid == k)[0]
        f = np.asarray(F[j], np.float32)                      # (n, 128, 28, 28)
        zm = np.stack([resize(np.where(gland[i] > 0, np.maximum(zone[i], 3 * (zone[i] == 0)), 0), f.shape[-2:], order=0, preserve_range=True, anti_aliasing=False) for i in j])
        v = []
        for zc in (1, 2):
            m = zm == zc
            if m.sum() == 0:
                m = zm > 0 if (zm > 0).any() else np.ones_like(zm, bool)
            x = f.transpose(1, 0, 2, 3)[:, m]               # (128, npix)
            v += [x.mean(1), x.max(1)]
        out[int(k)] = np.concatenate(v).astype(np.float32)
    pickle.dump(out, open(a.out, "wb"))
    print(len(out))


if __name__ == "__main__":
    main()
