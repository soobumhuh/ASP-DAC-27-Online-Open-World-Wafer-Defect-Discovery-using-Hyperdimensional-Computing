"""End-to-end runs from a config (configs/best.yaml): maps -> ensemble -> extraction -> re-scorer -> patient merge.

Output layout (cfg.paths.out):
  splits.json                      the patient splits (data.cv_splits)
  maps/seed<s>/<split>.pkl         maps of one seed copy (val and test cases of the split)
  maps/ens/<split>.pkl             seed ensemble (mean of the seed maps)
  logs/                            one log per map job
  cv.json, cv.md                   fold-0 and 5-fold CV results
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from . import ROOT
from . import data as D
from . import evaluate as E
from . import extraction as X
from . import maps as M
from . import patient as PT
from .rescorer import Rescorer


def load_config(path: str) -> dict:
    cfg = yaml.safe_load(open(path))
    cfg["_path"] = os.path.abspath(path)
    return cfg


def out_path(cfg, *parts) -> str:
    return os.path.join(cfg["paths"]["out"], *parts)


def splits(cfg) -> dict:
    p = out_path(cfg, "splits.json")
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump({**D.cv_splits(cfg["paths"]["slices"]), **D.nested_splits(cfg["paths"]["slices"])}, open(p, "w"))
    return json.load(open(p))


def map_path(cfg, seed, split) -> str:
    return out_path(cfg, "maps", f"seed{seed}", f"{split}.pkl")


def build_maps(cfg, names: list, seeds: list = None):
    """Map jobs (seed x split) not yet on disk, each in its own process, cfg.run.parallel_jobs at a time."""
    seeds = cfg["ensemble_seeds"] if seeds is None else seeds
    splits(cfg)
    jobs = [(s, n) for s in seeds for n in names if not os.path.exists(map_path(cfg, s, n))]
    os.makedirs(out_path(cfg, "logs"), exist_ok=True)

    def run(job):
        s, n = job
        t0 = time.time()
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "train_predict.py"), "--config", cfg["_path"],
               "--split", n, "--seed", str(s)]
        env = {**os.environ, "OMP_NUM_THREADS": str(cfg["run"]["threads"]),
               "OPENBLAS_NUM_THREADS": str(cfg["run"]["threads"])}
        with open(out_path(cfg, "logs", f"maps_seed{s}_{n}.log"), "w") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True, env=env, cwd=ROOT)
        print(f"maps seed {s} {n}: {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(cfg["run"]["parallel_jobs"]) as ex:
        for f in [ex.submit(run, j) for j in jobs]:
            f.result()


def ensemble_maps(cfg, split) -> dict:
    p = out_path(cfg, "maps", "ens", f"{split}.pkl")
    if not os.path.exists(p):
        M.save(M.ensemble([M.load(map_path(cfg, s, split)) for s in cfg["ensemble_seeds"]]), p)
    return M.load(p)


def run_cv(cfg, folds=(0, 1, 2, 3, 4)) -> dict:
    """Fold-f test: ensemble maps -> extraction -> re-scorer trained on the other folds' test candidates ->
    lesion AP; patient AUROC from the image case score alone and merged with the clinical record.
    The re-scorer of every fold needs the candidates of all other folds, so all five folds are always built."""
    sp = splits(cfg)
    names = [f"fold{f}" for f in range(5)]
    build_maps(cfg, names)
    pp, rc, w = cfg["extraction"], cfg["rescorer"], cfg["patient"]["merge_weight"]
    workers = cfg["run"]["workers"]
    d = D.load_slices(cfg["paths"]["slices"])
    if cfg["paths"].get("adc_raw"):  # absolute ADC crops (scripts/build_adc_raw.py) for re-scorer fields
        d["adc_raw"] = np.load(cfg["paths"]["adc_raw"], mmap_mode="r")
    rows_ms = PT.read_marksheet(os.path.join(cfg["paths"]["labels_dir"], "clinical_information", "marksheet.csv"))
    scan = D.scanner_info(cfg["paths"]["images_dir"])
    study_pid = {int(s): int(p) for s, p in zip(d["study_id"], d["patient_id"])}
    study_y = {int(s): int(c) for s, c in zip(d["study_id"], d["case_cspca"])}
    mp = {f: ensemble_maps(cfg, f"fold{f}") for f in range(5)}
    ext = {f: X.extract(mp[f]["test"], mp[f]["val"], pp, d, workers) for f in range(5)}
    rows, K = [], {}
    for f in folds:
        name = f"fold{f}"
        test, val = mp[f]["test"], mp[f]["val"]
        r = {"fold": f, "voxel_auroc": E.voxel_auroc(test),
             "base": E.picai(E.detection_maps(ext[f]), test, workers=workers, keep=K.setdefault(("base", f), {}))}
        rs = Rescorer(seed=rc["seed"], epochs=rc["epochs"]).fit([c for g in range(5) if g != f for c in X.flat(ext[g])])
        conf = rs.confidences(ext[f], rc["alpha"])
        det_rs = E.detection_maps(ext[f], conf)
        r["rescore"] = E.picai(det_rs, test, workers=workers, keep=K.setdefault(("rescore", f), {}))
        K[("candidates", f)] = {k: (shape, [(c["mask"], v) for c, v in zip(cands, conf[k])])
                                for k, (cands, shape) in ext[f].items()}  # for native-space evaluation
        # patient score: image case score (max re-scored candidate) merged with the clinical record
        groups = {k: set(int(p) for p in sp[name][k]) for k in ("train", "val", "test")}
        cases = {k: sorted(s for s in study_pid if study_pid[s] in groups[k]) for k in groups}
        y = {k: [study_y[c] for c in cases[k]] for k in cases}
        clin = PT.clinical_scores(rows_ms, scan, cases, y, seed=f, fields=cfg["patient"].get("fields"))
        ev = X.extract(val, val, pp, None, workers)
        img_val = [max([c["conf"] for c in ev[k][0]], default=0.0) for k in cases["val"]]
        img_test = [max(conf[k], default=0.0) for k in cases["test"]]
        merged = PT.merge(img_val, img_test, clin["val"], clin["test"], w)
        auc_img = float(roc_auc_score(y["test"], img_test))
        auc_m = float(roc_auc_score(y["test"], merged))
        ap = r["rescore"]["lesion_ap"]
        r["clinical_epochs"] = clin["epochs"]
        r["merged"] = {"patient_auroc": auc_m, "lesion_ap": ap, "picai_score": (auc_m + ap) / 2}
        r["image_only"] = {"patient_auroc": auc_img, "lesion_ap": ap, "picai_score": (auc_img + ap) / 2}
        for stage, sc in (("image_only", img_test), ("merged", merged)):  # per-case results for bootstrap CIs
            K[(stage, f)] = {**K[("rescore", f)], "case_pred": dict(zip(cases["test"], map(float, sc)))}
        rows.append(r)
        print(f"fold {f}: " + " | ".join(f"{k} {v['patient_auroc']:.3f}/{v['lesion_ap']:.3f}/{v['picai_score']:.3f}"
                                         for k, v in r.items() if isinstance(v, dict)), flush=True)
    stages = ["base", "rescore", "image_only", "merged"]
    res = {"config": {k: v for k, v in cfg.items() if not k.startswith("_")}, "per_fold": rows,
           "cv": {s: E.summary([r[s] for r in rows]) for s in stages} if len(rows) == 5 else None}
    tag = "cv" if len(rows) == 5 else "fold" + "".join(str(f) for f in folds)
    import pickle
    pickle.dump(K, open(out_path(cfg, f"{tag}_cases.pkl"), "wb"))
    json.dump(res, open(out_path(cfg, f"{tag}.json"), "w"), indent=2)
    lines = ["| stage | fold-0 AUROC / AP / PI-CAI | CV AUROC | CV lesion AP | CV PI-CAI |", "|---|---|---|---|---|"]
    for s in stages:
        f0 = rows[0][s]
        cv = res["cv"][s] if res["cv"] else None
        lines.append(f"| {s} | {f0['patient_auroc']:.3f} / {f0['lesion_ap']:.3f} / {f0['picai_score']:.3f} | " +
                     (f"{cv['patient_auroc'][0]:.3f} ± {cv['patient_auroc'][1]:.3f} | {cv['lesion_ap'][0]:.3f} ± "
                      f"{cv['lesion_ap'][1]:.3f} | {cv['picai_score'][0]:.3f} ± {cv['picai_score'][1]:.3f} |"
                      if cv else "- | - | - |"))
    open(out_path(cfg, f"{tag}.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res
