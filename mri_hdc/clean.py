"""Nested ("clean") 5-fold CV of the full pipeline: nothing used to score test fold f was trained or tuned on fold f.

Per outer test fold f (configs/best_clean.yaml):
  maps         seed-ensemble maps of fold f's test exams from the fold-f models (train = the other four folds), as
               in the standard CV
  re-scorer    trained on the test candidates of the nested models nest<f><g> (g != f; train = the three folds that
               are neither f nor g), so its training candidates come from models that never saw fold f; its epochs and
               alpha are chosen by leave-one-fold-out over those four g (cfg.rescorer grid)
  merge weight w chosen by leave-one-fold-out over the four g: image score of fold g = the nested model's max
               candidate confidence, clinical record trained on nest<f><g>'s training exams, both z-standardised on
               nest<f><g>'s validation exams; w maximises the mean AUROC over the four g
  patient      z(image) + w x z(clinical record of fold f's training exams), z from fold f's validation exams
Still fixed from the earlier tuning on folds 1-4: the HDC model settings and the extraction setting (stated in the
report as the remaining optimism).
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score

from . import data as D
from . import evaluate as E
from . import extraction as X
from . import patient as PT
from .pipeline import build_maps, ensemble_maps, out_path, splits
from .rescorer import make as make_rescorer


def _cases(sp, study_pid, study_y, name):
    groups = {k: set(int(p) for p in sp[name][k]) for k in ("train", "val", "test")}
    cases = {k: sorted(s for s in study_pid if study_pid[s] in groups[k]) for k in groups}
    return cases, {k: [study_y[c] for c in cases[k]] for k in cases}


def run_cv_clean(cfg, folds=(0, 1, 2, 3, 4)) -> dict:
    sp = splits(cfg)
    nest = [f"nest{f}{g}" for f in folds for g in range(5) if g != f]
    build_maps(cfg, [f"fold{f}" for f in folds] + nest)
    pp, rc, workers = cfg["extraction"], cfg["rescorer"], cfg["run"]["workers"]
    fields = cfg["patient"].get("fields")
    d = D.load_slices(cfg["paths"]["slices"])
    rows_ms = PT.read_marksheet(os.path.join(cfg["paths"]["labels_dir"], "clinical_information", "marksheet.csv"))
    scan = D.scanner_info(cfg["paths"]["images_dir"])
    study_pid = {int(s): int(p) for s, p in zip(d["study_id"], d["patient_id"])}
    study_y = {int(s): int(c) for s, c in zip(d["study_id"], d["case_cspca"])}
    grid_ep, grid_a = rc.get("epochs_grid", [5, 20, 50]), rc.get("alpha_grid", [0.0, 0.1, 0.25, 0.5, 0.75])
    grid_w = cfg["patient"].get("merge_grid", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
    rows, K = [], {}
    for f in folds:
        mp = ensemble_maps(cfg, f"fold{f}")
        test = mp["test"]
        ext = X.extract(test, mp["val"], pp, d, workers)
        inner = {g: ensemble_maps(cfg, f"nest{f}{g}") for g in range(5) if g != f}
        ext_in = {g: X.extract(m["test"], m["val"], pp, d, workers) for g, m in inner.items()}
        # re-scorer epochs / alpha: leave-one-fold-out over the nested folds
        tune = {}
        for ep in grid_ep:
            fits = {v: make_rescorer(rc, ep).fit([c for g in inner if g != v for c in X.flat(ext_in[g])])
                    for v in inner}
            for a in grid_a:
                r = [E.picai(E.detection_maps(ext_in[v], fits[v].confidences(ext_in[v], a)), inner[v]["test"],
                             workers=workers)["picai_score"] for v in inner]
                tune[(ep, a)] = float(np.mean(r))
        ep, a = max(tune, key=lambda k: (tune[k], -k[1]))
        rs = make_rescorer(rc, ep).fit([c for g in inner for c in X.flat(ext_in[g])])
        conf = rs.confidences(ext, a)
        K[("candidates", f)] = {k: (shape, [(c["mask"], v) for c, v in zip(cands, conf[k])])
                                for k, (cands, shape) in ext.items()}  # for native-space evaluation
        r = {"fold": f, "rescorer": {"epochs": ep, "alpha": a}, "voxel_auroc": E.voxel_auroc(test),
             "base": E.picai(E.detection_maps(ext), test, workers=workers, keep=K.setdefault(("base", f), {})),
             "rescore": E.picai(E.detection_maps(ext, conf), test, workers=workers,
                                keep=K.setdefault(("rescore", f), {}))}
        # merge weight: leave-one-fold-out over the nested folds (un-re-scored image score, as in the original choice)
        aucs = {w: [] for w in grid_w}
        for g, m in inner.items():
            cs, y = _cases(sp, study_pid, study_y, f"nest{f}{g}")
            cl = PT.clinical_scores(rows_ms, scan, cs, y, seed=10 * f + g, fields=fields)
            ev = X.extract(m["val"], m["val"], pp, None, workers)
            iv = [max([c["conf"] for c in ev[k][0]], default=0.0) for k in cs["val"]]
            it = [max([c["conf"] for c in ext_in[g][k][0]], default=0.0) for k in cs["test"]]
            for w in grid_w:
                aucs[w].append(roc_auc_score(y["test"], PT.merge(iv, it, cl["val"], cl["test"], w)))
        w = max(grid_w, key=lambda x: (np.mean(aucs[x]), -x))
        cases, y = _cases(sp, study_pid, study_y, f"fold{f}")
        clin = PT.clinical_scores(rows_ms, scan, cases, y, seed=f, fields=fields)
        ev = X.extract(mp["val"], mp["val"], pp, None, workers)
        img_val = [max([c["conf"] for c in ev[k][0]], default=0.0) for k in cases["val"]]
        img_test = [max(conf[k], default=0.0) for k in cases["test"]]
        merged = PT.merge(img_val, img_test, clin["val"], clin["test"], w)
        ap = r["rescore"]["lesion_ap"]
        auc_i, auc_m = float(roc_auc_score(y["test"], img_test)), float(roc_auc_score(y["test"], merged))
        r.update(merge_weight=w, merge_lofo_auroc={str(k): float(np.mean(v)) for k, v in aucs.items()},
                 image_only={"patient_auroc": auc_i, "lesion_ap": ap, "picai_score": (auc_i + ap) / 2},
                 merged={"patient_auroc": auc_m, "lesion_ap": ap, "picai_score": (auc_m + ap) / 2})
        for stage, sc in (("image_only", img_test), ("merged", merged)):
            K[(stage, f)] = {**K[("rescore", f)], "case_pred": dict(zip(cases["test"], map(float, sc)))}
        rows.append(r)
        print(f"fold {f}: re-scorer {ep}/{a}, w {w} | " + " | ".join(
            f"{k} {v['patient_auroc']:.3f}/{v['lesion_ap']:.3f}/{v['picai_score']:.3f}"
            for k, v in r.items() if isinstance(v, dict) and "picai_score" in v), flush=True)
    stages = ["base", "rescore", "image_only", "merged"]
    res = {"per_fold": rows, "cv": {s: E.summary([r[s] for r in rows]) for s in stages} if len(rows) == 5 else None}
    json.dump(res, open(out_path(cfg, "cv_clean.json"), "w"), indent=2)
    pickle.dump(K, open(out_path(cfg, "cv_clean_cases.pkl"), "wb"))
    if res["cv"]:
        lines = ["| stage | fold-0 AUROC / AP / PI-CAI | CV AUROC | CV lesion AP | CV PI-CAI |", "|---|---|---|---|---|"]
        for s in stages:
            f0, cv = rows[0][s], res["cv"][s]
            lines.append(f"| {s} | {f0['patient_auroc']:.3f} / {f0['lesion_ap']:.3f} / {f0['picai_score']:.3f} | "
                         f"{cv['patient_auroc'][0]:.3f} ± {cv['patient_auroc'][1]:.3f} | {cv['lesion_ap'][0]:.3f} ± "
                         f"{cv['lesion_ap'][1]:.3f} | {cv['picai_score'][0]:.3f} ± {cv['picai_score'][1]:.3f} |")
        open(out_path(cfg, "cv_clean.md"), "w").write("\n".join(lines) + "\n")
        print("\n".join(lines))
    return res
