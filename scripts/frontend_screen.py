#!/usr/bin/env python3
"""Patch-level screen of one front end / layer (see mri_hdc/frontend.py for the adapter and the protocol).

  CUDA_VISIBLE_DEVICES=3 uv run python scripts/frontend_screen.py --name A0_resnet18_l2 \
      --maps mri_feature_cache/resnet_l2.npy --rows full

--maps: (N, C, h, w) float16 memmap; --rows full = rows of the full slice file (all folds), sub = rows of the
folds-1-4 subset (caches written by scripts/frontend_features.py). --orig: the current pipeline's descriptor
(nearest cell of the already pooled map, no PCA) instead of the adapter (A0 reference).
For each lesion rule (frac25 = >= 25% lesion pixels; centre = frac25 or centre in lesion):
  (a) class-balanced logistic regression on the 134 standardised numbers; (b) HD probe = the existing per-bin
  memories (thermometer HV, 2048 bits, 20 / 5 sub-prototypes, 20 epochs, seed 0), z-scored per bin on validation
  normals. Test = fold 1. Writes results/frontend_screen/rows/<name>_<rule>_<probe>.json.
"""
import argparse
import json
import os
import time

import numpy as np

import _path  # noqa: F401
from mri_hdc import data as D
from mri_hdc import frontend as FE
from mri_hdc.encoding import PatchEncoder
from mri_hdc.maps import PatchModel, split_indices
from mri_hdc.pipeline import load_config

OUT = "results/frontend_screen"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--rows", choices=["full", "sub"], default="sub")
    ap.add_argument("--orig", action="store_true")
    ap.add_argument("--config", default="configs/best_clean_adc.yaml")
    ap.add_argument("--rules", default="frac25,centre")
    ap.add_argument("--probes", default="lr,hd")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(f"{OUT}/rows", exist_ok=True)
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    cfg = load_config(args.config)
    m = dict(cfg["model"])
    P, stride = m["patch"], m["stride"]

    d = FE.drop_fold0(D.load_slices(cfg["paths"]["slices"]))
    assert not (d["fold"] == 0).any()
    split = D.nested_splits(cfg["paths"]["slices"])["nest01"]  # train folds 2-4 (- 15% val), test fold 1
    idx = split_indices(d, split)
    log(" ".join(f"{k}: {len(v)} slices" for k, v in idx.items()))

    fm = np.load(args.maps, mmap_mode="r")
    rowmap = d["_row"] if args.rows == "full" else np.arange(len(d["fold"]))
    assert fm.shape[0] == (len(rowmap) if args.rows == "sub" else int(rowmap.max()) + 1) or args.rows == "full"
    get_maps = lambda s: fm[rowmap[s]]
    log(f"feature maps {args.maps}: {fm.shape}")

    # patches + raw descriptors of every set
    sets = {}
    t0 = time.time()
    for k in ("train_normal", "train_pos", "val", "test"):
        sl, ys, xs = FE.centres(d, idx[k], P, stride)
        if args.orig:  # the current descriptor: nearest cell of the (already 3x3-pooled) map
            raw = np.zeros((len(sl), fm.shape[1]), np.float16)
            fh = fm.shape[-1]
            for i in np.unique(sl):
                q = np.where(sl == i)[0]
                fy = np.clip(((ys[q] + 0.5) * fh / 96).astype(int), 0, fh - 1)
                fx = np.clip(((xs[q] + 0.5) * fh / 96).astype(int), 0, fh - 1)
                raw[q] = np.asarray(fm[rowmap[i]], np.float32)[:, fy, fx].T
        else:
            raw = FE.sample_features(get_maps, sl, ys, xs, device=args.device)
        st = FE.intensity_stats(d, sl, ys, xs, P)
        frac, centre = FE.patch_labels(d, sl, ys, xs, P)
        sets[k] = dict(sl=sl, ys=ys, xs=xs, raw=raw, st=st, frac=frac, centre=centre,
                       bins=FE.position_bins(d, sl, ys, xs, m["slice_levels"]))
        log(f"{k}: {len(sl)} patches, raw dim {raw.shape[1]} ({time.time() - t0:.0f}s)")
    t_patch = time.time() - t0

    comp, sizes = FE.lesion_components(d)
    te = sets["test"]
    te_size = FE.patch_lesion_size(comp, sizes, te["sl"], te["ys"], te["xs"], P)
    te_zone = d["zone_mask"][te["sl"], te["ys"], te["xs"]]
    te_case = d["case_cspca"][te["sl"]] == 1
    meta_path = args.maps + ".json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    tn, tp = sets["train_normal"], sets["train_pos"]
    for rule in args.rules.split(","):
        lesion = lambda s: (s["frac"] >= m["lesion_frac"]) | ((s["centre"]) if rule == "centre" else False)
        # adapter step 3: PCA on training patches (lesion = rule on positive slices; normal = negative exams + clean)
        if args.orig:
            proj = lambda s: s["raw"].astype(np.float32)
        else:
            Xtr = np.concatenate([tn["raw"], tp["raw"]])
            is_l = np.concatenate([np.zeros(len(tn["raw"]), bool), lesion(tp)])
            is_n = np.concatenate([np.ones(len(tn["raw"]), bool), tp["frac"] == 0])
            mu, V = FE.fit_pca(Xtr, is_l, is_n, seed=args.seed)
            del Xtr
            proj = lambda s: FE.apply_pca(s["raw"], mu, V)
        F = {k: np.concatenate([proj(s), s["st"]], 1).astype(np.float32) for k, s in sets.items()}
        pos, neg = lesion(te), te["frac"] == 0
        log(f"rule {rule}: descriptor dim {F['test'].shape[1]}; test {pos.sum()} lesion / {neg.sum()} normal patches")

        for probe in args.probes.split(","):
            t1 = time.time()
            if probe == "lr":
                from sklearn.linear_model import LogisticRegression
                pe = PatchEncoder().fit_scaler(F["train_normal"])
                Xl = F["train_pos"][lesion(tp)]
                Xn = np.concatenate([F["train_normal"], F["train_pos"][tp["frac"] == 0]])
                rng = np.random.default_rng(args.seed)
                nl = min(len(Xl), 200_000)
                Xl = Xl[rng.choice(len(Xl), nl, replace=False)]
                Xn = Xn[rng.choice(len(Xn), nl, replace=False)]
                clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
                clf.fit(pe.prep(np.concatenate([Xn, Xl])), np.r_[np.zeros(nl), np.ones(nl)])
                score = clf.decision_function(pe.prep(F["test"]))
            else:
                mm = {**m, "lesion_rule": "frac_or_centre" if rule == "centre" else None}
                pat = {k: (F[k], sets[k]["sl"], sets[k]["ys"], sets[k]["xs"], sets[k]["bins"])
                       for k in ("train_normal", "train_pos", "val")}
                model = PatchModel().fit(d, None, split, mm, args.seed, log=log, pat=pat)
                score = model.patch_z(F["test"], te["bins"])
            r = FE.breakdown(score, pos, neg, te_size, te_zone, te_case)
            r.update(name=args.name, rule=rule, probe=probe, maps=args.maps, raw_dim=int(te["raw"].shape[1]),
                     dim=int(F["test"].shape[1]), orig=bool(args.orig), fit_s=round(time.time() - t1, 1),
                     patch_s=round(t_patch, 1), cache_gb=round(os.path.getsize(args.maps) / 1e9, 2),
                     **{k: meta[k] for k in ("gpu_s_per_exam", "cpu_s_per_exam", "map_shape", "source") if k in meta})
            json.dump(r, open(f"{OUT}/rows/{args.name}_{rule}_{probe}.json", "w"), indent=1)
            log(f"{args.name} {rule} {probe}: AUROC {r['auroc']:.4f} | <150 {r['auroc_lt150']:.4f} "
                f"150-500 {r['auroc_150to500']:.4f} >500 {r['auroc_gt500']:.4f} | PZ {r['auroc_pz']:.4f} "
                f"TZ {r['auroc_tz']:.4f} | pos-exam {r['auroc_posexam']:.4f} ({r['fit_s']}s)")


if __name__ == "__main__":
    main()
