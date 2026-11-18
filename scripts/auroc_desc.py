#!/usr/bin/env python3
"""Per-case patient descriptors (mri_hdc.patient_desc) of the saved seed-ensemble maps of one or more map families.

For every split (fold0-4, nest<f><g>) and both of its case groups (val, test) writes
  <out>/<family>/<split>.pkl = {'val': {study_id: {name: value}}, 'test': {...}}
Reads only saved maps; trains nothing.

  python scripts/auroc_desc.py --families base dwi perm rawpix --out results/desc --procs 8
"""
import argparse
import os
import pickle
from multiprocessing import get_context

import numpy as np

import _path  # noqa: F401
from mri_hdc.patient_desc import describe

G = {}


def one(job):
    fam, split, src, out = job
    dst = os.path.join(out, fam, f"{split}.pkl")
    if os.path.exists(dst):
        return dst
    mp = pickle.load(open(src, "rb"))
    res = {}
    for grp in ("val", "test"):
        res[grp] = {}
        for k, c in mp[grp].items():
            s = G["rows"][int(k)]
            res[grp][int(k)] = describe(c, G["adc"][s], G["hbv"][s])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pickle.dump(res, open(dst + ".tmp", "wb"))
    os.replace(dst + ".tmp", dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["base"])
    ap.add_argument("--maps_root", default="/data/sota")
    ap.add_argument("--slices", default="/data/picai_all_masks.npz")
    ap.add_argument("--adc_raw", default="/workspace/wafer-hdc/mri_feature_cache/adc_raw.npy")
    ap.add_argument("--out", default="/data/auroc/desc")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    d = np.load(a.slices)
    sid, z = d["study_id"], d["slice_z"]
    rows = {}
    for k in np.unique(sid):
        j = np.where(sid == k)[0]
        rows[int(k)] = j[np.argsort(z[j])]
    G.update(rows=rows, adc=np.load(a.adc_raw, mmap_mode="r"), hbv=d["images"][:, 2])
    jobs = []
    for fam in a.families:
        ens = os.path.join(a.maps_root, fam, "maps", "ens")
        for f in range(5):
            for split in [f"fold{f}"] + [f"nest{f}{g}" for g in range(5) if g != f]:
                jobs.append((fam, split, os.path.join(ens, f"{split}.pkl"), a.out))
    with get_context("fork").Pool(a.procs) as pool:
        for p in pool.imap_unordered(one, jobs):
            print(p, flush=True)


if __name__ == "__main__":
    main()
