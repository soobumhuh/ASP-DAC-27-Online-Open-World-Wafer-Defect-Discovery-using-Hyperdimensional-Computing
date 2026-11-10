#!/usr/bin/env python3
"""
round4_gating.py - idea 1: patient-gated candidate confidence.

About 80% of false-positive candidates sit in csPCa-negative patients, and removing them (an oracle) lifts lesion AP
by 0.15-0.20 on the inner folds. Here every candidate's confidence is multiplied by the patient's csPCa probability
raised to a power: conf' = conf x p_patient ** gamma, so candidates in low-risk patients drop in the global ranking
that lesion AP uses, while the order inside a patient is unchanged.

p_patient = the empirical CDF (over the split's validation cases) of a patient score:
  img    the #110 image score (map maximum)
  merge  z(image) + w x z(clinical HDC record)  (patient_round3.py, w chosen per split on its own val)
gamma (and the score kind) is chosen on the fold-0 split's internal validation patients (all from folds 1-4), then
applied unchanged to fold 0's test and to the other CV test folds. Candidates come from the model's maps with its
tuned extraction setting.

  uv run python round4_gating.py --model t110 --pp results/lesion_ap2/final_t110.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import postproc_search as pps
import rescore_hdc as rh

GAMMAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def gated(ext, p_of, gamma):
    out = {}
    for k, (cands, shape) in ext.items():
        g = p_of.get(k, 1.0) ** gamma
        out[k] = ([{**c, "conf": c["conf"] * g} for c in cands], shape)
    return out


def p_from(scores_split, kind, ref):
    ref = np.sort(np.asarray(ref))
    s = np.asarray(scores_split[kind])
    p = (np.searchsorted(ref, s, side="right") + 0.5) / (len(ref) + 1)
    return {int(c): float(v) for c, v in zip(scores_split["cases"], p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="t110")
    ap.add_argument("--pp", default="results/lesion_ap2/final_t110.json", help="final_*.json (uses its tuned pp)")
    ap.add_argument("--scores", default="results/round4/patient_scores/case_scores.json")
    ap.add_argument("--out", default="results/round4/gating")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pp = {**pps.DEFAULT_PP, **json.load(open(args.pp))["tuned"]["pp"]}
    S = json.load(open(args.scores))

    # tune on the fold-0 split's validation patients (folds 1-4 only)
    m0 = pps.load("fold0", args.model)
    ext_val = rh.extract_all(m0["val"], m0["val"], pp, args.workers)
    tune = {}
    for kind in ("img", "merge_z"):
        p_val = p_from(S["fold0"]["val"], kind, S["fold0"]["val"][kind])
        for g in GAMMAS:
            tune[(kind, g)] = rh.evaluate(gated(ext_val, p_val, g), m0["val"], None, 0, args.workers)
            print("tune", kind, g, tune[(kind, g)], flush=True)
    kind, g = max(tune, key=lambda k: (tune[k]["picai_score"], -k[1]))
    best_img_g = max(GAMMAS, key=lambda x: (tune[("img", x)]["picai_score"], -x))
    print("chosen", kind, g, "| best image-only gamma", best_img_g, flush=True)

    # test: fold 0 once, then the other CV folds, with the chosen setting (and the image-only variant for reference)
    rows = []
    for f in range(5):
        mp = pps.load(f"fold{f}", args.model)
        ext = rh.extract_all(mp["test"], mp["val"], pp, args.workers)
        r = {"fold": f, "no_gate": rh.evaluate(ext, mp["test"], None, 0, args.workers)}
        for label, kd, gg in (("gate_img", "img", best_img_g), ("gate_chosen", kind, g)):
            pt = p_from(S[f"fold{f}"]["test"], kd, S[f"fold{f}"]["val"][kd])
            r[label] = rh.evaluate(gated(ext, pt, gg), mp["test"], None, 0, args.workers)
        r["voxel_auroc"] = pps.voxel_auroc(mp["test"])
        rows.append(r)
        print(f, {k: v for k, v in r.items() if k != "fold"}, flush=True)
    summ = {lab: {m: [float(np.mean([r[lab][m] for r in rows])), float(np.std([r[lab][m] for r in rows]))]
                  for m in ("patient_auroc", "lesion_ap", "picai_score")} for lab in ("no_gate", "gate_img", "gate_chosen")}
    out = {"pp": pp, "tune": {f"{k[0]}/{k[1]}": v for k, v in tune.items()}, "chosen": [kind, g],
           "image_only_gamma": best_img_g, "per_fold": rows, "cv": summ}
    json.dump(out, open(os.path.join(args.out, f"gating_{args.model}.json"), "w"), indent=2)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
