"""Map -> lesion candidates.

  calibration  prob = sigmoid((z - offset) / temp); offset = the q-th percentile and temp = tf x std of the
               reference split's csPCa-negative voxels (the split's validation cases, never test)
  smoothing    optional extra in-plane (sxy px) / through-slice (sz slices) Gaussian inside the gland
  extraction   report_guided_annotation.extract_lesion_candidates (threshold mode, dynamic factor, minimum size,
               number of candidates, removal of adjacent candidates)
  confidence   per candidate: its peak ('max'), mean ('mean') or mean of its top 10% voxels ('top10pct')
Every candidate also gets the fields the re-scorer reads: log size, peak, mean, number of slices, slice level, zone
at the peak, compactness, and inside-vs-ring intensity contrast of T2W / ADC / HBV.
"""
from __future__ import annotations

from multiprocessing import get_context

import numpy as np
from scipy import ndimage
from scipy.special import expit

DEFAULT = dict(sxy=0.0, sz=0.0, q=97.0, tf=0.5, mapping="sigmoid", mode="dynamic", factor=2.5, minvox=10, nles=5,
               adj=True, conf="max", gamma=0.0)
_PP = {}


def calibration(ref_maps: dict, pp: dict):
    ref = np.concatenate([c["map"][~np.isnan(c["map"])].astype(np.float32).ravel()
                          for c in ref_maps.values() if c["cspca"] == 0])
    return (float(np.percentile(ref, pp["q"])), float(pp["tf"] * ref.std() + 1e-6),
            np.percentile(ref, np.linspace(0, 100, 1001)))


def probability(m: np.ndarray, off: float, temp: float, qgrid, p: dict) -> np.ndarray:
    m = m.astype(np.float32)
    if p["sxy"] > 0 or p["sz"] > 0:
        w = (~np.isnan(m)).astype(np.float32)
        num = ndimage.gaussian_filter(np.nan_to_num(m), sigma=(p["sz"], p["sxy"], p["sxy"]))
        den = ndimage.gaussian_filter(w, sigma=(p["sz"], p["sxy"], p["sxy"]))
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(w > 0, num / np.maximum(den, 1e-6), np.nan)
    if p["mapping"] == "rank":  # empirical CDF of the reference normal voxels, sharpened
        prob = np.interp(np.nan_to_num(m, nan=-1e3), qgrid, np.linspace(0, 1, len(qgrid))) ** (1.0 / max(p["tf"], 1e-3))
        prob[np.isnan(m)] = 0
        return prob.astype(np.float64)
    return expit((np.nan_to_num(m, nan=-1e3) - off) / temp).astype(np.float64)


def _extract(item):
    from report_guided_annotation import extract_lesion_candidates
    key, c, img, gland, off, temp, qgrid = item
    p = _PP
    prob = probability(c["map"], off, temp, qgrid, p)
    thr = p["mode"] if p["mode"] in ("dynamic", "dynamic-fast") else float(p["mode"])
    _, conf, idx = extract_lesion_candidates(prob, threshold=thr, min_voxels_detection=int(p["minvox"]),
                                             num_lesions_to_extract=int(p["nles"]),
                                             dynamic_threshold_factor=float(p["factor"]),
                                             remove_adjacent_lesion_candidates=bool(p["adj"]))
    gt, _ = ndimage.label(c["lesion"] > 0)
    ring_st = np.ones((1, 7, 7), bool)  # 3 px in-plane ring
    cands = []
    for k, cf in conf:
        K = idx == k
        v = prob[K]
        if p["conf"] == "mean":
            cf = float(v.mean())
        elif p["conf"] == "top10pct":
            cf = float(np.sort(v)[-max(1, len(v) // 10):].mean())
        iou = 0.0  # evaluation only (true / false positive label for training the re-scorer)
        for g in np.unique(gt[K]):
            if g > 0:
                G = gt == g
                iou = max(iou, float((G & K).sum() / (G | K).sum()))
        pk = np.unravel_index(np.argmax(np.where(K, prob, -1)), K.shape)
        zz, yy, xx = np.nonzero(K)
        f = {"logsize": float(np.log(K.sum())), "peak": float(v.max()), "mean": float(v.mean()),
             "nslices": float(len(np.unique(zz))), "level": float(c["rel_z"][pk[0]]), "zone": int(c["zone"][pk]),
             "compact": float(K.sum() / ((np.ptp(zz) + 1) * (np.ptp(yy) + 1) * (np.ptp(xx) + 1)))}
        if img is not None:
            R = ndimage.binary_dilation(K, ring_st) & ~K & gland
            for ch, name in enumerate(("c_t2w", "c_adc", "c_hbv")):
                a, b = img[:, ch][K], img[:, ch][R]
                f[name] = float(a.mean() - (b.mean() if len(b) else 0.0))
        cands.append({"conf": float(cf), "tp": iou >= 0.1, "feats": f, "mask": np.flatnonzero(K)})
    if p.get("gamma", 0) > 0 and cands:  # case gating (not used by the best setting)
        top = max(cd["conf"] for cd in cands)
        for cd in cands:
            cd["conf"] *= top ** p["gamma"]
    return key, cands, c["lesion"].shape


def _init(pp):
    global _PP
    _PP = pp


def case_images(d: dict, study: int, n_slices: int):
    j = np.where(d["study_id"] == study)[0]
    j = j[np.argsort(d["slice_z"][j])]
    assert len(j) == n_slices
    return np.asarray(d["images"][j], dtype=np.float32), np.asarray(d["gland_mask"][j]) > 0


def extract(maps: dict, ref_maps: dict, pp: dict, d: dict = None, workers: int = 16) -> dict:
    """{case: (candidates, shape)}. With the slice data d, candidates also get the image-contrast fields."""
    pp = {**DEFAULT, **pp}
    off, temp, qgrid = calibration(ref_maps, pp)
    items = []
    for k in sorted(maps):
        img, gland = case_images(d, k, maps[k]["map"].shape[0]) if d is not None else (None, None)
        items.append((k, maps[k], img, gland, off, temp, qgrid))
    with get_context("fork").Pool(workers, initializer=_init, initargs=(pp,)) as pool:
        out = pool.map(_extract, items, chunksize=4)
    return {k: (cands, shape) for k, cands, shape in out}


def flat(ext: dict) -> list:
    return [c for cands, _ in ext.values() for c in cands]
