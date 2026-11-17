#!/usr/bin/env python3
"""Leave-one-centre-out check (evaluation plan, tier 2) with FROZEN settings: everything that nested CV chose is
fixed (extraction from the config, re-scorer kind / k / alpha, merge weight w); nothing is tuned on the held-out
centre or on anything else here.

Per held-out centre C (data.centre_splits):
  maps         seed-ensemble maps of the loco_C model (trained on the other two centres)
  re-scorer    trained on the out-of-sample test candidates of locoin_C_D (train = third centre, test = centre D) for
               both D != C, so its training candidates come from models that never saw them (and never saw C)
  patient      z(image) + w x z(clinical record trained on loco_C's training exams), z from loco_C's validation exams
Writes <out>/loco<tag>_cases.pkl in the cv_clean_cases.pkl layout with the centre as fold key (for native_eval.py and
bootstrap_ci.py) and <out>/loco<tag>.json.

  uv run python scripts/run_loco.py --config configs/sota/best_sota.yaml --k 40 --alpha 0.5 --tag _best_sota
"""
import argparse
import json
import os
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score

import _path  # noqa: F401
from mri_hdc import data as D, evaluate as E, extraction as X, patient as PT
from mri_hdc.clean import _cases
from mri_hdc.pipeline import build_maps, ensemble_maps, load_config, out_path, splits
from mri_hdc.rescorer import make as make_rescorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=40, help="re-scorer grid value (k for knn kinds, epochs otherwise)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--w", type=float, default=0.25)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = load_config(a.config)
    sp = splits(cfg)
    centres = sorted(k[5:] for k in sp if k.startswith("loco_"))
    names = [f"loco_{c}" for c in centres] + [k for k in sp if k.startswith("locoin_")]
    build_maps(cfg, names)
    pp, rc, workers = cfg["extraction"], cfg["rescorer"], cfg["run"]["workers"]
    d = D.load_slices(cfg["paths"]["slices"])
    if cfg["paths"].get("adc_raw"):
        d["adc_raw"] = np.load(cfg["paths"]["adc_raw"], mmap_mode="r")
    rows_ms = PT.read_marksheet(os.path.join(cfg["paths"]["labels_dir"], "clinical_information", "marksheet.csv"))
    scan = D.scanner_info(cfg["paths"]["images_dir"])
    study_pid = {int(s): int(p) for s, p in zip(d["study_id"], d["patient_id"])}
    study_y = {int(s): int(c) for s, c in zip(d["study_id"], d["case_cspca"])}
    K, rows = {}, []
    for c in centres:
        mp = ensemble_maps(cfg, f"loco_{c}")
        ext = X.extract(mp["test"], mp["val"], pp, d, workers)
        tr = []
        for k in sp:
            if k.startswith(f"locoin_{c}_"):
                m = ensemble_maps(cfg, k)
                tr += X.flat(X.extract(m["test"], m["val"], pp, d, workers))
        rs = make_rescorer(rc, a.k).fit(tr)
        conf = rs.confidences(ext, a.alpha)
        K[("candidates", c)] = {k: (shape, [(cd["mask"], v) for cd, v in zip(cands, conf[k])])
                                for k, (cands, shape) in ext.items()}
        r = {"centre": c, "rescore": E.picai(E.detection_maps(ext, conf), mp["test"], workers=workers,
                                             keep=K.setdefault(("rescore", c), {}))}
        cases, y = _cases(sp, study_pid, study_y, f"loco_{c}")
        clin = PT.clinical_scores(rows_ms, scan, cases, y, seed=0, fields=cfg["patient"].get("fields"),
                                  kind=cfg["patient"].get("kind", "binary"))
        ev = X.extract(mp["val"], mp["val"], pp, None, workers)
        img_val = [max([cd["conf"] for cd in ev[k][0]], default=0.0) for k in cases["val"]]
        img_test = [max(conf[k], default=0.0) for k in cases["test"]]
        merged = PT.merge(img_val, img_test, clin["val"], clin["test"], a.w)
        ap_ = r["rescore"]["lesion_ap"]
        auc = float(roc_auc_score(y["test"], merged))
        r["merged"] = {"patient_auroc": auc, "lesion_ap": ap_, "picai_score": (auc + ap_) / 2,
                       "n_exams": len(cases["test"]), "n_pos": int(sum(y["test"]))}
        K[("merged", c)] = {**K[("rescore", c)], "case_pred": dict(zip(cases["test"], map(float, merged)))}
        rows.append(r)
        print(c, json.dumps(r["merged"]), flush=True)
    res = {"settings": {"k": a.k, "alpha": a.alpha, "w": a.w, "rescorer": rc, "extraction": pp}, "per_centre": rows}
    json.dump(res, open(out_path(cfg, f"loco{a.tag}.json"), "w"), indent=2)
    pickle.dump(K, open(out_path(cfg, f"loco{a.tag}_cases.pkl"), "wb"))


if __name__ == "__main__":
    main()
