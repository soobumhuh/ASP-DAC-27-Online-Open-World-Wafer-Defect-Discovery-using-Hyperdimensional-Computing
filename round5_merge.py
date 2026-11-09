#!/usr/bin/env python3
"""round5_merge.py - merge the clinical HDC record into the round-4 pick's patient score.

Pick = 3 seed copies of #110 (t110ens3) + HDC re-scorer v2. Its case score is the maximum re-scored candidate
confidence. Patient score = z(case score) + w x z(clinical HDC record score) (patient_round3.py; record trained on the
split's training cases). z uses the split's validation cases. w is chosen on the fold-0 split's validation patients
(all from folds 1-4) with the un-re-scored case score (the re-scorer of a split is trained on candidates that include
its validation patients), then applied unchanged to every test fold. Lesion AP is unchanged; PI-CAI uses the pick's
per-fold lesion AP (results/round4/stack/stack_t110ens3.json, 'rescore')."""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

import postproc_search as pps
import rescore_hdc as rh
from round4_rescore import Rescorer2, add_contrast

BASE, PP = "t110ens3", "results/round4/pp_ens3.json"
pp = {**pps.DEFAULT_PP, **json.load(open(PP))}
S = json.load(open("results/round5/patient_scores/case_scores.json"))
stack = json.load(open("results/round4/stack/stack_t110ens3.json"))
d = np.load(pps.NPZ, mmap_mode="r")
maps = {f: pps.load(f"fold{f}", BASE) for f in range(5)}
ext = {}
for f in range(5):
    ext[f] = rh.extract_all(maps[f]["test"], maps[f]["val"], pp, 32)
    add_contrast(ext[f], d)
case = lambda e, k, conf=None: max([c["conf"] for c in e[k][0]], default=0.0) if conf is None else conf


def zs(ref, x):
    ref = np.asarray(ref, float)
    return (np.asarray(x, float) - ref.mean()) / (ref.std() + 1e-9)


# tune w on the fold-0 split's validation patients (un-re-scored case score)
ev = rh.extract_all(maps[0]["val"], maps[0]["val"], pp, 32)
sv = S["fold0"]["val"]
img_v = np.array([max([c["conf"] for c in ev[k][0]], default=0.0) for k in sv["cases"]])
yv = np.array(sv["y"])
grid = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
val_auc = {w: float(roc_auc_score(yv, zs(img_v, img_v) + w * zs(sv["clin"], sv["clin"]))) for w in grid}
w = max(grid, key=lambda g: (val_auc[g], -g))
print("val AUROC by w:", val_auc, "chosen w", w, flush=True)

rows = []
for f in range(5):
    rs = Rescorer2(epochs=50).fit([c for g in range(5) if g != f for c in rh.flat(ext[g])])
    st, svf = S[f"fold{f}"]["test"], S[f"fold{f}"]["val"]
    e = ext[f]
    img_t = []
    for k in st["cases"]:
        cands = e[k][0]
        if cands:
            conf = np.array([c["conf"] for c in cands]) ** 0.75 * rs.rank(cands) ** 0.25
            img_t.append(float(conf.max()))
        else:
            img_t.append(0.0)
    ev_f = rh.extract_all(maps[f]["val"], maps[f]["val"], pp, 32)
    img_vf = [max([c["conf"] for c in ev_f[k][0]], default=0.0) for k in svf["cases"]]
    yt = np.array(st["y"])
    merged = zs(img_vf, img_t) + w * zs(svf["clin"], st["clin"])
    ap = stack["per_fold"][f]["rescore"]["lesion_ap"]
    r = {"fold": f, "auroc_image": float(roc_auc_score(yt, img_t)), "auroc_merged": float(roc_auc_score(yt, merged)),
         "lesion_ap": ap}
    r["picai_image"] = (r["auroc_image"] + ap) / 2
    r["picai_merged"] = (r["auroc_merged"] + ap) / 2
    rows.append(r)
    print(r, flush=True)
summ = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))]
        for k in ("auroc_image", "auroc_merged", "lesion_ap", "picai_image", "picai_merged")}
json.dump({"w": w, "val_auc": val_auc, "per_fold": rows, "cv": summ}, open("results/round5/merge_pick.json", "w"), indent=2)
print(json.dumps(summ, indent=1))
