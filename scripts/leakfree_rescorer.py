#!/usr/bin/env python3
"""Fold-0 check of the re-scorer's training data.

Standard protocol: the fold-0 re-scorer is trained on the CV test candidates of folds 1-4, which come from fold models
that were trained with fold 0 among their training data (an indirect leak: no fold-0 labels are used). Here it is
trained only on candidates from the inner-split models (train = folds 1-4 minus v, validation = fold v), which never
saw fold 0; everything else is unchanged.

  uv run python scripts/leakfree_rescorer.py --config results/review/pick/config.yaml \
      --inner results/lesion_ap/maps/t110ens3/inner{v}/maps.pkl
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import _path  # noqa: F401
from mri_hdc import data as D, evaluate as E, extraction as X, maps as M, patient as PT
from mri_hdc.pipeline import ensemble_maps, load_config, out_path, splits
from mri_hdc.rescorer import Rescorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--inner", required=True, help="path pattern of the inner-split maps, '{v}' = 1, 2")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pp, rc, w, workers = cfg["extraction"], cfg["rescorer"], cfg["patient"]["merge_weight"], cfg["run"]["workers"]
    d = D.load_slices(cfg["paths"]["slices"])
    mp = {f: ensemble_maps(cfg, f"fold{f}") for f in range(5)}
    ext = {f: X.extract(mp[f]["test"], mp[f]["val"], pp, d, workers) for f in range(5)}
    inner = [M.load(args.inner.format(v=v))["val"] for v in (1, 2)]
    ext_in = [X.extract(m, m, pp, d, workers) for m in inner]
    train = {"standard (folds 1-4 CV test candidates)": [c for g in range(1, 5) for c in X.flat(ext[g])],
             "leak-free (inner-split validation candidates)": [c for e in ext_in for c in X.flat(e)]}
    sp = splits(cfg)["fold0"]
    pid = {int(s): int(p) for s, p in zip(d["study_id"], d["patient_id"])}
    yv = {int(s): int(c) for s, c in zip(d["study_id"], d["case_cspca"])}
    groups = {k: set(int(p) for p in sp[k]) for k in ("train", "val", "test")}
    cases = {k: sorted(s for s in pid if pid[s] in groups[k]) for k in groups}
    y = {k: [yv[c] for c in cases[k]] for k in cases}
    rows = PT.read_marksheet(os.path.join(cfg["paths"]["labels_dir"], "clinical_information", "marksheet.csv"))
    clin = PT.clinical_scores(rows, D.scanner_info(cfg["paths"]["images_dir"]), cases, y, seed=0,
                              fields=cfg["patient"].get("fields"))
    ev = X.extract(mp[0]["val"], mp[0]["val"], pp, None, workers)
    img_val = [max([c["conf"] for c in ev[k][0]], default=0.0) for k in cases["val"]]
    res = {}
    for name, cands in train.items():
        rs = Rescorer(seed=rc["seed"], epochs=rc["epochs"]).fit(cands)
        conf = rs.confidences(ext[0], rc["alpha"])
        r = E.picai(E.detection_maps(ext[0], conf), mp[0]["test"], workers=workers)
        img = [max(conf[k], default=0.0) for k in cases["test"]]
        auc = float(roc_auc_score(y["test"], PT.merge(img_val, img, clin["val"], clin["test"], w)))
        res[name] = {"training_candidates": len(cands), "true_positives": int(sum(c["tp"] for c in cands)),
                     "image_only": r, "merged": {"patient_auroc": auc, "lesion_ap": r["lesion_ap"],
                                                 "picai_score": (auc + r["lesion_ap"]) / 2}}
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(out_path(cfg, "leakfree_rescorer_fold0.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
