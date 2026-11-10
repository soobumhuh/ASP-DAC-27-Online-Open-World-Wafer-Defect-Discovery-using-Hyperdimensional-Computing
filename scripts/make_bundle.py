#!/usr/bin/env python3
"""Collect everything a trained pipeline needs to score a new exam into one file (training side; not timed).

  uv run python scripts/make_bundle.py --config configs/test_fresh.yaml --split fold0 \
      --single_pp results/summary/round3_t110_final.json

Needs: models/seed<s>_<split>.pkl for every ensemble seed (scripts/fit_model.py) and the CV maps of the run
(scripts/run_cv.py): the re-scorer of fold f is trained on the other folds' test candidates, the calibration and the
patient-score standardisation use the split's validation cases. Writes <paths.out>/models/bundle_<split>.pkl with
  models      PatchModel per seed
  single      #110 single model: extraction setting + calibration (seed-0 maps)
  ensemble    extraction setting + calibration of the seed ensemble, re-scorer, clinical record, z statistics
"""
import argparse
import json
import os
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score

import _path  # noqa: F401
from mri_hdc import data as D, extraction as X, maps as M, patient as PT
from mri_hdc.memory import HDCBinaryClassifier
from mri_hdc.pipeline import ensemble_maps, load_config, map_path, out_path, splits
from mri_hdc.rescorer import Rescorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="fold0")
    ap.add_argument("--single_pp", required=True, help="json with the single model's tuned extraction ('tuned'/'pp')")
    args = ap.parse_args()
    cfg = load_config(args.config)
    f = int(args.split[4:])
    pp_ens = {**X.DEFAULT, **cfg["extraction"]}
    pp_one = {**X.DEFAULT, **json.load(open(args.single_pp))["tuned"]["pp"]}
    pp_one = {k: v for k, v in pp_one.items() if k in X.DEFAULT}
    models = {s: pickle.load(open(out_path(cfg, "models", f"seed{s}_{args.split}.pkl"), "rb"))
              for s in cfg["ensemble_seeds"]}
    d = D.load_slices(cfg["paths"]["slices"])
    w = cfg["run"]["workers"]

    one_val = M.load(map_path(cfg, cfg["ensemble_seeds"][0], args.split))["val"]
    ens = {g: ensemble_maps(cfg, f"fold{g}") for g in range(5)}
    rc = cfg["rescorer"]
    cands = [c for g in range(5) if g != f for c in X.flat(X.extract(ens[g]["test"], ens[g]["val"], pp_ens, d, w))]
    rescorer = Rescorer(seed=rc["seed"], epochs=rc["epochs"]).fit(cands)

    # clinical record of the split (epochs chosen on its validation cases, as in patient.clinical_scores)
    sp = splits(cfg)[args.split]
    pid = {int(s): int(p) for s, p in zip(d["study_id"], d["patient_id"])}
    yv = {int(s): int(c) for s, c in zip(d["study_id"], d["case_cspca"])}
    groups = {k: set(int(p) for p in sp[k]) for k in ("train", "val")}
    cases = {k: sorted(s for s in pid if pid[s] in groups[k]) for k in groups}
    y = {k: np.array([yv[c] for c in cases[k]]) for k in cases}
    rows = PT.read_marksheet(os.path.join(cfg["paths"]["labels_dir"], "clinical_information", "marksheet.csv"))
    scan = D.scanner_info(cfg["paths"]["images_dir"])
    best = None
    for ep in (5, 20, 50):
        R = PT.ClinicalRecords(rows, scan, cases["train"], f)
        clf = HDCBinaryClassifier(epochs=ep, seed=f).fit(R.hv(cases["train"]), y["train"])
        v = roc_auc_score(y["val"], clf.score(R.hv(cases["val"])))
        if best is None or v > best[0]:
            best = (v, R, clf)
    _, R, clf = best
    clin_val = clf.score(R.hv(cases["val"]))
    R.rows, R.scan = {}, {}  # the marksheet / scanner tables are not shipped; a new exam brings its own fields
    ev = X.extract(ens[f]["val"], ens[f]["val"], pp_ens, None, w)
    img_val = np.array([max([c["conf"] for c in ev[k][0]], default=0.0) for k in cases["val"]])

    bundle = {"model_cfg": cfg["model"], "models": models,
              "single": {"pp": pp_one, "calibration": X.calibration(one_val, pp_one)},
              "ensemble": {"pp": pp_ens, "calibration": X.calibration(ens[f]["val"], pp_ens), "rescorer": rescorer,
                           "alpha": rc["alpha"], "records": R, "clinical": clf,
                           "img_stats": (float(img_val.mean()), float(img_val.std())),
                           "clin_stats": (float(clin_val.mean()), float(clin_val.std())),
                           "merge_weight": cfg["patient"]["merge_weight"]}}
    out = out_path(cfg, "models", f"bundle_{args.split}.pkl")
    pickle.dump(bundle, open(out, "wb"), protocol=4)
    print("saved", out, f"{os.path.getsize(out) / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
