#!/usr/bin/env python3
"""Inner-split screen of fused maps: the voxel-wise mean of the z maps of several model configs x seeds
(e.g. a 20 px and a 10 px patch memory = multi-scale fusion), scored with one fixed extraction setting.

  uv run python scripts/fuse_screen.py --pp results/round6/pp_t110.json \
      --maps 'results/lesion_ap/maps/{m}/{n}/maps.pkl:t110,t110s1,t110s2' 'results/d2sweep/p10/maps/seed{s}/{n}.pkl:0,1,2'
"""
import argparse
import json

import numpy as np

import _path  # noqa: F401
from mri_hdc import evaluate as E, extraction as X, maps as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", required=True, help="'path pattern with {n} and {m} or {s}:id1,id2,...'")
    ap.add_argument("--pp", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    pp = {k: v for k, v in {**X.DEFAULT, **json.load(open(args.pp))}.items() if k in X.DEFAULT}
    r = []
    for n in ("inner1", "inner2"):
        ms = []
        for spec in args.maps:
            pat, ids = spec.rsplit(":", 1)
            for i in ids.split(","):
                ms.append(M.load(pat.format(n=n, m=i, s=i))["val"])
        fused = {}
        for k, c in ms[0].items():
            fused[k] = {**c, "map": np.mean([x[k]["map"].astype(np.float32) for x in ms], axis=0).astype(np.float16)}
        r.append(E.picai(E.detection_maps(X.extract(fused, fused, pp, None, args.workers)), fused, workers=args.workers))
    print("RESULT", {k: round(float(np.mean([x[k] for x in r])), 4) for k in r[0]}, flush=True)


if __name__ == "__main__":
    main()
