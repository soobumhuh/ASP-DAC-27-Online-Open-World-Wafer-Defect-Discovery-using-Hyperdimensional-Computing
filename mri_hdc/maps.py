"""Train the per-bin HDC patch memories on one split and score its validation / test cases into z-scored maps.

  1. patches (features.extract_patches) of train csPCa-negative slices (normal memory), train csPCa-positive slices
     (lesion patches + 'negplus' clean patches), val and test slices
  2. clip / standardise on train normals; cap every position bin at bin_cap normal patches (seeded)
  3. encode to hypervectors (encoding.PatchEncoder)
  4. labels on positive slices: lesion if >= lesion_frac of the patch is expert lesion; clean ('negplus' normal) if no
     lesion pixel; up to half as many clean patches as normal patches join the normal memory
  5. per position bin: HDCPrototypeClassifier on its normal and lesion patches (lesions pooled from the same zone when
     the bin has < 50); score the bin's val / test patches
  6. z-score per bin with the val csPCa-negative patches (never test), average overlapping patches per pixel, mask to
     the gland, Gaussian smoothing (normalised convolution) -> one (Z, S, S) map per case
Random streams (seed s): encoder s, bin cap s + 1, negplus sample s + 2, classifier of bin b: s + b.
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np

from .encoding import PatchEncoder, bipolar, unpack
from .features import bin_zone_group, extract_patches, n_bins, patch_fraction, position_bins
from .memory import HDCPrototypeClassifier


def split_indices(d: dict, split: dict) -> dict:
    lut = {int(p): k for k in ("train", "val", "test") for p in split[k]}
    s = np.array([lut.get(int(p), "none") for p in d["patient_id"]])
    c = d["case_cspca"]
    return {"train_normal": np.where((s == "train") & (c == 0))[0], "train_pos": np.where((s == "train") & (c == 1))[0],
            "val": np.where(s == "val")[0], "test": np.where(s == "test")[0]}


def zscore_per_bin(ref_scores, ref_bins, scores, bins, nb):
    mu = np.full(nb, ref_scores.mean()); sd = np.full(nb, ref_scores.std() + 1e-6)
    for b in np.unique(ref_bins):
        mu[b], sd[b] = ref_scores[ref_bins == b].mean(), ref_scores[ref_bins == b].std() + 1e-6
    return (scores - mu[bins]) / sd[bins]


def assemble_maps(d, slices, sl, ys, xs, z, P, sigma):
    """Average overlapping patch scores per pixel; NaN outside the gland / where no patch covers; in-plane Gaussian
    smoothing inside the gland."""
    from scipy.ndimage import gaussian_filter
    S = d["images"].shape[-1]
    h = P // 2
    row = np.full(int(slices.max()) + 1, -1, np.int64)
    row[slices] = np.arange(len(slices))
    j = row[sl]
    acc = np.zeros((len(slices), S, S)); cnt = np.zeros((len(slices), S, S))
    for dy in range(P):
        for dx in range(P):
            np.add.at(acc, (j, ys - h + dy, xs - h + dx), z)
            np.add.at(cnt, (j, ys - h + dy, xs - h + dx), 1.0)
    with np.errstate(invalid="ignore"):
        m = acc / cnt
    m[~(d["gland_mask"][slices] > 0) | (cnt == 0)] = np.nan
    if sigma > 0:
        w = (~np.isnan(m)).astype(np.float64)
        num = gaussian_filter(np.nan_to_num(m), sigma=(0, sigma, sigma))
        den = gaussian_filter(w, sigma=(0, sigma, sigma))
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(w > 0, num / den, np.nan)
    return m


def case_volumes(d, slices, maps) -> dict:
    """{study_id: {map (Z, S, S) float16, lesion, cspca, zone, rel_z}}, slices sorted by z."""
    out = {}
    sid = d["study_id"][slices]
    for k in np.unique(sid):
        j = np.where(sid == k)[0]
        j = j[np.argsort(d["slice_z"][slices[j]])]
        s = slices[j]
        out[int(k)] = {"map": maps[j].astype(np.float16), "lesion": (d["lesion_mask"][s] > 0).astype(np.uint8),
                       "cspca": int(d["case_cspca"][s[0]]), "zone": d["zone_mask"][s].astype(np.uint8),
                       "rel_z": d["rel_z"][s].astype(np.float32)}
    return out


def train_and_map(d: dict, fmaps: np.ndarray, split: dict, m: dict, seed: int, log=print) -> dict:
    """m: the 'model' section of the config. Returns {'val': case volumes, 'test': case volumes}."""
    t0 = time.time()
    P, levels = m["patch"], m["slice_levels"]
    NB = n_bins(levels)
    idx = split_indices(d, split)
    pat = {}
    for k, ix in idx.items():
        F, sl, ys, xs = extract_patches(d, ix, P, m["stride"], fmaps)
        pat[k] = (F, sl, ys, xs, position_bins(d, sl, ys, xs, levels))
    log(" ".join(f"{k}: {len(v[0])} patches" for k, v in pat.items()) + f" ({time.time() - t0:.0f}s)")

    # normal memory: scaler on all train normals, then a seeded cap per bin
    Ftr, btr = pat["train_normal"][0], pat["train_normal"][4]
    pe = PatchEncoder(m["num_proj"], m["num_levels"], m["clip_value"], seed).fit_scaler(Ftr)
    keep = np.zeros(len(Ftr), bool)
    cap_rng = np.random.default_rng(seed + 1)
    for b in np.unique(btr):
        j = np.where(btr == b)[0]
        if len(j) > m["bin_cap"]:
            j = cap_rng.choice(j, m["bin_cap"], replace=False)
        keep[j] = True
    Xtr, btr = pe.prep(Ftr[keep]), btr[keep]
    pe.fit_hdc(Xtr)
    D = pe.D

    # lesion / clean labels on csPCa-positive training slices
    Fp, slp, yp, xp, bp = pat["train_pos"]
    frac = patch_fraction(d["lesion_mask"], slp, yp, xp, P)
    les, clean = frac >= m["lesion_frac"], frac == 0
    C = {"n": pe.codes(Xtr), "l": pe.codes(pe.prep(Fp[les])),
         "val": pe.codes(pe.prep(pat["val"][0])), "test": pe.codes(pe.prep(pat["test"][0]))}
    B = {"n": btr, "l": bp[les], "val": pat["val"][4], "test": pat["test"][4]}
    nl = np.where(clean)[0]
    nl = np.random.default_rng(seed + 2).choice(nl, min(len(nl), len(Xtr) // 2), replace=False)
    C["n"] = np.concatenate([C["n"], pe.codes(pe.prep(Fp[nl]))])
    B["n"] = np.concatenate([B["n"], bp[nl]])
    log(f"memory: {len(C['n'])} normal, {len(C['l'])} lesion patch HVs, D = {D}")

    # one prototype memory per bin
    sc = {s: np.zeros(len(C[s])) for s in ("val", "test")}
    for b in range(NB):
        qs = {s: B[s] == b for s in ("val", "test")}
        if not any(q.any() for q in qs.values()):
            continue
        mn = B["n"] == b
        if mn.sum() == 0:
            mn = np.ones(len(B["n"]), bool)
        ml = B["l"] == b
        if ml.sum() < 50:
            ml = bin_zone_group(B["l"]) == bin_zone_group(b)
            if ml.sum() < 50:
                ml = np.ones(len(B["l"]), bool)
        Xn, Xl = bipolar(unpack(C["n"][mn], D)), bipolar(unpack(C["l"][ml], D))
        clf = HDCPrototypeClassifier(n_sub=tuple(m["n_sub"]), epochs=m["epochs"], seed=seed + b, lr=m["lr"],
                                     softmax_t=m["softmax_t"]).fit(
            np.concatenate([Xn, Xl]), np.r_[np.zeros(len(Xn), int), np.ones(len(Xl), int)])
        for s, q in qs.items():
            if q.any():
                sc[s][q] = clf.score(bipolar(unpack(C[s][q], D)))

    # z-score per bin on val csPCa-negative patches, maps per case
    vneg = d["case_cspca"][pat["val"][1]] == 0
    out = {}
    for s in ("val", "test"):
        z = zscore_per_bin(sc["val"][vneg], B["val"][vneg], sc[s], B[s], NB)
        mp = assemble_maps(d, idx[s], pat[s][1], pat[s][2], pat[s][3], z, P, m["smooth_sigma"])
        out[s] = case_volumes(d, idx[s], mp)
    log(f"maps done ({time.time() - t0:.0f}s)")
    return out


def ensemble(maps: list) -> dict:
    """Average the z maps of models that differ only in their seed (same split)."""
    out = {}
    for s in ("val", "test"):
        out[s] = {}
        for k, c in maps[0][s].items():
            avg = np.mean([x[s][k]["map"].astype(np.float32) for x in maps], axis=0)
            out[s][k] = {**{a: v for a, v in c.items() if a != "map"}, "map": avg.astype(np.float16)}
    return out


def save(maps: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "wb") as fh:
        pickle.dump(maps, fh, protocol=4)
    os.replace(path + ".tmp", path)


def load(path: str) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)
