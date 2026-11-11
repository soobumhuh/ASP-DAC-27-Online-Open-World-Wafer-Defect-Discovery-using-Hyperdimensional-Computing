"""Scoring with picai_eval (the PI-CAI challenge metrics).

  lesion AP       average precision of lesion detection: a candidate is a true positive if it overlaps an expert
                  lesion with IoU >= 0.1; candidates ranked by confidence across all cases
  patient AUROC   ROC AUC of the case-level score (csPCa yes / no)
  PI-CAI score    (patient AUROC + lesion AP) / 2
  voxel AUROC     ROC AUC of the raw map values vs the lesion mask inside the gland (diagnostic only)
"""
from __future__ import annotations

import numpy as np


def detection_maps(ext: dict, conf: dict = None) -> dict:
    """Candidates -> detection maps (each candidate's voxels = its confidence). conf: optional {case: [conf, ...]}."""
    out = {}
    for k, (cands, shape) in ext.items():
        det = np.zeros(int(np.prod(shape)), np.float32)
        cs = conf[k] if conf is not None else [c["conf"] for c in cands]
        for c, v in zip(cands, cs):
            det[c["mask"]] = v
        out[k] = det.reshape(shape)
    return out


def picai(det: dict, maps: dict, case_scores: dict = None, workers: int = 16, keep: dict = None) -> dict:
    """picai_eval on detection maps; case_scores replaces the default case score (max of the detection map) for the
    patient AUROC only (lesion AP never depends on it)."""
    from picai_eval import evaluate as pe
    from sklearn.metrics import roc_auc_score
    keys = sorted(det)
    m = pe(y_det=[det[k] for k in keys], y_true=[maps[k]["lesion"].astype(np.int32) for k in keys],
           subject_list=[str(k) for k in keys], num_parallel_calls=workers)
    r = {"patient_auroc": float(m.auroc), "lesion_ap": float(m.AP), "picai_score": float(m.score)}
    if case_scores is not None:
        y = [int(maps[k]["cspca"]) for k in keys]
        r["patient_auroc"] = float(roc_auc_score(y, [case_scores[k] for k in keys]))
        r["picai_score"] = (r["patient_auroc"] + r["lesion_ap"]) / 2
    if keep is not None:  # per-case results for bootstrap confidence intervals
        keep.update(lesion_results={k: m.lesion_results[str(k)] for k in keys},
                    case_target={k: int(maps[k]["cspca"]) for k in keys},
                    case_pred={k: float(case_scores[k]) if case_scores is not None else float(m.case_pred[str(k)])
                               for k in keys})
    return r


def voxel_auroc(maps: dict) -> float:
    from sklearn.metrics import roc_auc_score
    m = np.concatenate([c["map"].astype(np.float32).ravel() for c in maps.values()])
    y = np.concatenate([c["lesion"].ravel() for c in maps.values()])
    ok = ~np.isnan(m)
    return float(roc_auc_score(y[ok] > 0, m[ok]))


def summary(rows: list, keys=("patient_auroc", "lesion_ap", "picai_score")) -> dict:
    return {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in keys}
