#!/usr/bin/env python3
"""Hypervector database, step 1: region and patient items (float descriptors + payload; no hypervectors yet).

Per outer fold f (picai_baseline folds) the database side and the query side come from different models, exactly as
in the best_sota retrieval re-scorer (mri_hdc.rescorer.KNNRescorer):
  query  fold f's exams, maps of the fold-f models (role 'query'); also fold f's validation exams (role 'val', used
         only for the clinical-merge z-scores of test T7)
  db     fold g's exams (g != f), maps of the nested models nest<f><g> (trained on neither f nor g) (role 'db')
Items per map set: every best_sota candidate (same extraction as the pipeline), every ground-truth lesion (3D
connected component of the crop lesion mask) and 3 sampled normal regions per zone (PZ, TZ) per exam (no lesion
within 3 px; size drawn from the ground-truth lesion volumes). Each item gets
  record   the KNNRescorer record fields computed on the item's region (log size, peak / mean probability, ... ,
           absolute ADC, rim distance) from that map set's calibrated probability map
  content  frozen ResNet-18 layer-2 features inside the region: per channel (T2W, ADC, HBV fed alone, 3 x 128) the
           mean over the region's feature cells + the max over its core (1 px in-plane erosion) -> 768 floats; and
           the same from the cached joint RGB features (128 x 2 = 256 floats)
  payload  study, patient, exam fold, zone at the peak, size, detector score, IoU / TP, lesion ISUP, case ISUP,
           csPCa, grade group, centre, scanner, PSA, PSAD (labels are payload only; never in a descriptor)
Patient-level descriptors: whole-gland content (mean + max over the gland) and the clinical numbers.

  python scripts/hvdb_build.py --config configs/sota/best_sota.yaml --out /data/hvdb/items
"""
import argparse
import os
import pickle
import time
from multiprocessing import get_context

import numpy as np
from scipy import ndimage

import _path  # noqa: F401
from mri_hdc import data as D
from mri_hdc import extraction as X
from mri_hdc import patient as PT
from mri_hdc.pipeline import ensemble_maps, load_config

G = {}  # fork-shared state
NORMAL_PER_ZONE = 3


def case_index(d):
    idx = {}
    order = np.lexsort((d["slice_z"], d["study_id"]))
    for i in order:
        idx.setdefault(int(d["study_id"][i]), []).append(int(i))
    return {k: np.array(v) for k, v in idx.items()}


def content(rows: np.ndarray, K: np.ndarray) -> np.ndarray:
    """rows: the exam's slice indices; K (Z, 96, 96) bool region -> 768 per-channel + 256 joint floats."""
    FP, FJ = G["fp"], G["fj"]
    core = ndimage.binary_erosion(K, np.ones((1, 3, 3), bool))
    if not core.any():
        core = K
    S, h = K.shape[-1], 28
    acc_p, acc_j, n = np.zeros((3, 128)), np.zeros(128), 0
    mx_p, mx_j = np.full((3, 128), -np.inf), np.full(128, -np.inf)
    for j in np.where(K.any((1, 2)))[0]:
        ys, xs = np.nonzero(K[j])
        cell = ((ys + 0.5) * h / S).astype(int) * h + ((xs + 0.5) * h / S).astype(int)
        w = np.bincount(cell, minlength=h * h).astype(np.float64)
        u = np.nonzero(w)[0]
        fp = np.asarray(FP[rows[j]], dtype=np.float32).reshape(3, 128, h * h)
        fj = np.asarray(FJ[rows[j]], dtype=np.float32).reshape(128, h * h)
        acc_p += fp[:, :, u] @ w[u]
        acc_j += fj[:, u] @ w[u]
        n += w.sum()
        if core[j].any():
            cy, cx = np.nonzero(core[j])
            cu = np.unique(((cy + 0.5) * h / S).astype(int) * h + ((cx + 0.5) * h / S).astype(int))
            mx_p = np.maximum(mx_p, fp[:, :, cu].max(-1))
            mx_j = np.maximum(mx_j, fj[:, cu].max(-1))
    if not np.isfinite(mx_j).all():  # core on no slice with region (cannot happen, core subset of K) - guard
        mx_p, mx_j = acc_p / n, acc_j / n
    return np.concatenate([(acc_p / n).ravel(), mx_p.ravel(), acc_j / n, mx_j]).astype(np.float32)


def record_fields(prob, K, c, img, gland, cache):
    """The record fields of mri_hdc.extraction._extract for an arbitrary region K (same definitions)."""
    v = prob[K]
    pk = np.unravel_index(np.argmax(np.where(K, prob, -1)), K.shape)
    zz, yy, xx = np.nonzero(K)
    f = {"logsize": float(np.log(K.sum())), "peak": float(v.max()), "mean": float(v.mean()),
         "nslices": float(len(np.unique(zz))), "level": float(c["rel_z"][pk[0]]), "zone": int(c["zone"][pk]),
         "compact": float(K.sum() / ((np.ptp(zz) + 1) * (np.ptp(yy) + 1) * (np.ptp(xx) + 1)))}
    R = ndimage.binary_dilation(K, np.ones((1, 7, 7), bool)) & ~K & gland
    for ch, name in enumerate(("c_t2w", "c_adc", "c_hbv")):
        a, b = img[:, ch][K], img[:, ch][R]
        f[name] = float(a.mean() - (b.mean() if len(b) else 0.0))
    a = img[:, 3][K]
    f["a_mean"], f["a_p10"] = float(a.mean()), float(np.percentile(a, 10))
    if "dist" not in cache:
        cache["dist"] = ndimage.distance_transform_edt(gland, sampling=(100, 1, 1))
    f["rim"] = float(np.median(cache["dist"][K]))
    conf = float(np.sort(v)[-max(1, len(v) // 10):].mean())  # top10pct, as the best_sota extraction
    return f, conf


def fixed_regions(study):
    """Ground-truth lesions and sampled normal regions of one exam (model-independent) with their content."""
    d, rows = G["d"], G["idx"][study]
    L = d["lesion_mask"][rows] > 0
    Z = d["zone_mask"][rows]
    gt, n = ndimage.label(L)
    out = []
    comps = [(gt == i) for i in range(1, n + 1)]
    for K in comps:
        out.append({"kind": "gt", "region": np.flatnonzero(K), "content": content(rows, K)})
    rng = np.random.default_rng(10_000 + study)
    forbid = ndimage.binary_dilation(L, np.ones((1, 7, 7), bool)) if L.any() else L
    vols = G["gt_vols"]
    for zone in (1, 2):
        allowed = (Z == zone) & ~forbid
        if allowed.sum() < 10:
            continue
        cand_vox = np.argwhere(allowed)
        for r in range(NORMAL_PER_ZONE):
            for _ in range(30):
                V = float(rng.choice(vols))
                zc, yc, xc = cand_vox[rng.integers(len(cand_vox))]
                ns = 3 if V >= 60 else 1
                rad = max(2.0, np.sqrt(V / ns / np.pi))
                yy, xx = np.mgrid[:L.shape[1], :L.shape[2]]
                disk = (yy - yc) ** 2 + (xx - xc) ** 2 <= rad ** 2
                K = np.zeros_like(L)
                for z in range(max(0, zc - ns // 2), min(L.shape[0], zc + ns // 2 + 1)):
                    K[z] = disk
                K &= allowed
                if K.sum() >= 10:
                    out.append({"kind": "normal", "region": np.flatnonzero(K), "content": content(rows, K),
                                "zone_sampled": zone})
                    break
    G_ = d["gland_mask"][rows] > 0
    gland_c = content(rows, G_)
    return study, out, gland_c


def _mapset_case(item):
    """One exam of one map set: record fields of its fixed regions, content of its candidates."""
    study, c, cands, off, temp, qgrid = item
    d, rows = G["d"], G["idx"][study]
    pp = G["pp"]
    prob = X.probability(c["map"], off, temp, qgrid, pp)
    img = np.asarray(d["images"][rows], dtype=np.float32)
    img = np.concatenate([img, np.asarray(d["adc_raw"][rows], dtype=np.float32)[:, None]], 1)
    gland = d["gland_mask"][rows] > 0
    shape = c["map"].shape
    cache, fixed = {}, []
    for r in G["fixed"][study]:
        K = np.zeros(int(np.prod(shape)), bool)
        K[r["region"]] = True
        f, conf = record_fields(prob, K.reshape(shape), c, img, gland, cache)
        fixed.append((f, conf))
    assert np.array_equal(c["lesion"] > 0, d["lesion_mask"][rows] > 0), study  # same ground truth as fixed regions
    gt, _ = ndimage.label(c["lesion"] > 0)
    cc = []
    for cd in cands:
        K = np.zeros(int(np.prod(shape)), bool)
        K[cd["mask"]] = True
        K = K.reshape(shape)
        best = (0.0, 0)
        for g in np.unique(gt[K]):
            if g > 0:
                Gm = gt == g
                best = max(best, (float((Gm & K).sum() / (Gm | K).sum()), int(g)))
        cc.append((content(rows, K), best))
    return study, fixed, cc


def lesion_isups(row, n):
    """csPCa ISUP per ground-truth component (components sorted by volume, descending); exact when unambiguous."""
    vals = sorted([int(float(v)) for v in (row.get("lesion_ISUP") or "").split(",") if v.strip() != ""], reverse=True)
    vals = [v for v in vals if v >= 2]
    case = int(float(row.get("case_ISUP") or 0))
    if not vals:
        vals = [max(case, 2)]
    exact = n <= 1 and len(vals) == 1 or len(set(vals)) == 1
    return [vals[i] if i < len(vals) else vals[-1] for i in range(n)], exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sota/best_sota.yaml")
    ap.add_argument("--out", default="/data/hvdb/items")
    ap.add_argument("--perchan", default="/data/hvdb/fe/resnet_l2_perchan.npy")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--folds", default="0,1,2,3,4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = load_config(a.config)
    pp = {**X.DEFAULT, **cfg["extraction"]}
    d = D.load_slices(cfg["paths"]["slices"])
    d["adc_raw"] = np.load(cfg["paths"]["adc_raw"], mmap_mode="r")
    G.update(d=d, idx=case_index(d), pp=pp, fp=np.load(a.perchan, mmap_mode="r"),
             fj=np.load(cfg["paths"]["feature_cache"], mmap_mode="r"))
    ms = PT.read_marksheet(os.path.join(cfg["paths"]["labels_dir"], "clinical_information", "marksheet.csv"))
    t0 = time.time()
    meta_p = os.path.join(a.out, "fixed.pkl")
    if os.path.exists(meta_p):
        meta = pickle.load(open(meta_p, "rb"))
    else:
        scan = D.scanner_info(cfg["paths"]["images_dir"])
        vols = []
        for s, rows in G["idx"].items():
            gt, n = ndimage.label(d["lesion_mask"][rows] > 0)
            vols += list(np.bincount(gt.ravel())[1:])
        G["gt_vols"] = np.array(vols, float)
        with get_context("fork").Pool(a.workers) as pool:
            res = pool.map(fixed_regions, sorted(G["idx"]), chunksize=4)
        fixed, gland = {}, {}
        for s, out, gc in res:
            fixed[s], gland[s] = out, gc
        exams = {}
        for s, rows in G["idx"].items():
            r = ms[s]
            gts = [x for x in fixed[s] if x["kind"] == "gt"]
            order = np.argsort([-len(x["region"]) for x in gts])
            isups, exact = lesion_isups(r, len(gts))
            for rank, i in enumerate(order):
                gts[i]["isup"], gts[i]["isup_exact"] = isups[rank], exact
            num = lambda k: float(r[k]) if (r.get(k) or "").strip() else np.nan
            exams[s] = {"study": s, "patient": int(d["patient_id"][rows[0]]), "efold": int(d["fold"][rows[0]]),
                        "case_isup": int(float(r.get("case_ISUP") or 0)), "cspca": int(d["case_cspca"][rows[0]]),
                        "gg": PT.grade_group(ms, s), "centre": r["center"], "scanner": scan.get(s, ("NA", "NA"))[0],
                        "psa": num("psa"), "psad": num("psad"), "volume": num("prostate_volume")}
        meta = {"fixed": fixed, "gland": gland, "exams": exams, "gt_vols": G["gt_vols"]}
        pickle.dump(meta, open(meta_p, "wb"))
        print(f"fixed regions: {sum(len(v) for v in fixed.values())} in {time.time() - t0:.0f}s", flush=True)
    G["fixed"] = meta["fixed"]
    for f in [int(x) for x in a.folds.split(",")]:
        p = os.path.join(a.out, f"fold{f}.pkl")
        if os.path.exists(p):
            continue
        sets = [("query", f"fold{f}", "test", None), ("val", f"fold{f}", "val", None)] + \
               [("db", f"nest{f}{g}", "test", g) for g in range(5) if g != f]
        items, cont = [], []
        for role, split, part, g in sets:
            t1 = time.time()
            mp = ensemble_maps(cfg, split)
            maps = mp[part]
            ext = X.extract(maps, mp["val"], cfg["extraction"], d, a.workers)
            off, temp, qgrid = X.calibration(mp["val"], pp)
            work = [(s, maps[s], ext[s][0], off, temp, qgrid) for s in sorted(maps)]
            with get_context("fork").Pool(a.workers) as pool:
                res = pool.map(_mapset_case, work, chunksize=2)
            for s, fixed, cc in res:
                base = {"role": role, "split": split, "study": s}
                for r, (fe, conf) in zip(G["fixed"][s], fixed):
                    it = {**base, "kind": r["kind"], "feats": fe, "conf": conf, "size": len(r["region"]),
                          "zone": int(r.get("zone_sampled", fe["zone"])), "isup": r.get("isup", 0),
                          "isup_exact": r.get("isup_exact", True), "tp": r["kind"] == "gt", "iou": float(r["kind"] == "gt"),
                          "cidx": len(cont), "mask": r["region"]}
                    cont.append(r["content"])
                    items.append(it)
                gts = [r for r in G["fixed"][s] if r["kind"] == "gt"]
                for cd, (cv, (iou, gi)) in zip(ext[s][0], cc):
                    isup = gts[gi - 1]["isup"] if iou >= 0.1 and gi > 0 and gi <= len(gts) else 0
                    items.append({**base, "kind": "cand", "feats": cd["feats"], "conf": cd["conf"],
                                  "size": len(cd["mask"]), "zone": int(cd["feats"]["zone"]), "isup": isup,
                                  "isup_exact": True, "tp": bool(cd["tp"]), "iou": iou, "cidx": len(cont),
                                  "mask": cd["mask"]})
                    cont.append(cv)
            del mp, maps
            print(f"fold {f} {split} {part}: {len(work)} exams, {len(items)} items total, {time.time() - t1:.0f}s",
                  flush=True)
        pickle.dump({"items": items, "content": np.stack(cont)}, open(p + ".tmp", "wb"), protocol=4)
        os.replace(p + ".tmp", p)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
