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

from .encoding import PatchEncoder, bipolar, bipolar_int8, unpack
from .features import bin_zone_group, extract_patches, n_bins, patch_fraction, position_bins
from .memory import HDCPrototypeClassifier


def prepare_slices(d: dict, m: dict) -> dict:
    """Model-specific view of the slice data. gland_dilate > 0: the AI gland mask is dilated in-plane by that many
    pixels, so patch centres and map values also cover tissue just outside it (lesions extending past the AI mask)."""
    if m.get("context") == "25d":  # descriptors of the slices above and below are appended (2.5D)
        from .features import neighbour_slices
        d = {**d, "_context": neighbour_slices(d)}
    k = int(m.get("gland_dilate", 0) or 0)
    if k > 0:
        from scipy.ndimage import binary_dilation
        st = np.zeros((1, 2 * k + 1, 2 * k + 1), bool)
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        st[0] = yy ** 2 + xx ** 2 <= k * k
        d = {**d, "gland_mask": binary_dilation(d["gland_mask"] > 0, st).astype(np.uint8)}
    return d


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


class PatchModel:
    """The trained per-bin HDC memories of one split and seed: scaler + thermometer encoder, one
    HDCPrototypeClassifier per position bin, and the per-bin z-score statistics (from the split's validation
    csPCa-negative patches). fit() = training; patch_z() / maps() = inference on any slices."""

    def fit(self, d: dict, fmaps: np.ndarray, split: dict, m: dict, seed: int, log=print, pat: dict = None):
        """pat (optional): precomputed {'train_normal' | 'train_pos' | 'val': (F, sl, ys, xs, bins)} patch descriptors
        (front-end screen, mri_hdc.frontend); fmaps is then unused."""
        t0 = time.time()
        self.m, self.seed = m, seed
        P, levels = m["patch"], m["slice_levels"]
        NB = self.NB = n_bins(levels)
        idx = split_indices(d, split)
        given, pat = pat is not None, dict(pat or {})
        for k in ([] if given else ("train_normal", "train_pos", "val")):
            F, sl, ys, xs = extract_patches(d, idx[k], P, m["stride"], fmaps, context=d.get("_context"),
                                            pix=m.get("pixels", 0), tex=m.get("texture", False),
                                            stats=m.get("stats", True), readout=m.get("readout", "nearest"))
            pat[k] = (F, sl, ys, xs, position_bins(d, sl, ys, xs, levels))
        # ablations: 'none' = one global memory; 'bound' = one global memory, position HV XOR-bound into each patch HV
        self.mode = m.get("bins", "position")
        log(" ".join(f"{k}: {len(v[0])} patches" for k, v in pat.items()) + f" ({time.time() - t0:.0f}s)")

        if m.get("pca"):  # front-end adapter: PCA of the feature-map columns, fitted on this split's training patches
            self._fit_pca(d, pat, m, seed, log)
            pat = {k: (self._pca(v[0]),) + tuple(v[1:]) for k, v in pat.items()}

        # normal memory: scaler on all train normals, then a seeded cap per bin
        Ftr, btr = pat["train_normal"][0], pat["train_normal"][4]
        pe = self.enc = PatchEncoder(m["num_proj"], m["num_levels"], m["clip_value"], seed,
                                     groups=m.get("encoder_groups")).fit_scaler(Ftr)
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
        if self.mode == "bound":  # HDDB-style: position HV = level HV xor zone HV xor grid-cell HV
            prng = np.random.default_rng(seed + 999)
            Lh, Zh, Gh = (prng.integers(0, 2, (n, D), dtype=np.uint8) for n in (levels, 3, 9))
            b = np.arange(NB)
            self.poscodes = np.packbits(Lh[b // 27] ^ Zh[(b // 9) % 3] ^ Gh[b % 9], axis=1, bitorder="little")

        # lesion / clean labels on csPCa-positive training slices
        Fp, slp, yp, xp, bp = pat["train_pos"]
        frac = patch_fraction(d["lesion_mask"], slp, yp, xp, P)
        les, clean = frac >= m["lesion_frac"], frac == 0
        if m.get("lesion_rule") == "frac_or_centre":  # small lesions: the patch centre inside a lesion also counts
            les = les | (d["lesion_mask"][slp, yp, xp] > 0)
        C = {"n": self._codes(Xtr, btr), "l": self._codes(pe.prep(Fp[les]), bp[les])}
        B = {"n": self._sel(btr), "l": self._sel(bp[les])}
        nl = np.where(clean)[0]
        nl = np.random.default_rng(seed + 2).choice(nl, min(len(nl), len(Xtr) // 2), replace=False)
        C["n"] = np.concatenate([C["n"], self._codes(pe.prep(Fp[nl]), bp[nl])])
        B["n"] = np.concatenate([B["n"], self._sel(bp[nl])])
        log(f"memory: {len(C['n'])} normal, {len(C['l'])} lesion patch HVs, D = {D}")

        # one prototype memory per bin (classifier of bin b seeded with seed + b)
        self.clf = {}
        for b in (range(NB) if self.mode == "position" else [0]):
            mn = B["n"] == b
            if mn.sum() == 0:
                mn = np.ones(len(B["n"]), bool)
            ml = B["l"] == b
            if ml.sum() < 50:
                ml = bin_zone_group(B["l"]) == bin_zone_group(b)
                if ml.sum() < 50:
                    ml = np.ones(len(B["l"]), bool)
            Xn = bipolar_int8(C["n"] if mn.all() else C["n"][mn], D)
            Xl = bipolar_int8(C["l"] if ml.all() else C["l"][ml], D)
            ns = m["n_sub_tz"] if m.get("n_sub_tz") and bin_zone_group(b) == 2 else m["n_sub"]  # TZ bins: more capacity
            c = HDCPrototypeClassifier(n_sub=tuple(ns), epochs=m["epochs"], seed=seed + b, lr=m["lr"],
                                       softmax_t=m["softmax_t"]).fit(Xn, Xl)
            del Xn, Xl
            del c.counts, c.rng  # keep only the bipolar prototypes
            self.clf[b] = c

        # per-bin z statistics from the validation csPCa-negative patches
        vb = self._sel(pat["val"][4])
        sv = self.raw_scores(self._codes(pe.prep(pat["val"][0]), pat["val"][4]), vb)
        vneg = d["case_cspca"][pat["val"][1]] == 0
        ref, rb = sv[vneg], vb[vneg]
        self.mu, self.sd = np.full(NB, ref.mean()), np.full(NB, ref.std() + 1e-6)
        for b in np.unique(rb):
            self.mu[b], self.sd[b] = ref[rb == b].mean(), ref[rb == b].std() + 1e-6
        log(f"fit done ({time.time() - t0:.0f}s)")
        return self

    def _fit_pca(self, d, pat, m, seed, log=print):
        """Class-balanced sample (pca_sample, default 200k) of training patches: lesion = the label rule on csPCa-positive
        slices, normal = csPCa-negative slices + clean (no lesion pixel) positive patches; the first pca_cols columns
        (the feature map) are projected to min(pca, pca_cols) components; the remaining columns pass through."""
        from .frontend import fit_pca
        nc = int(m["pca_cols"])
        Fp, slp, yp, xp = pat["train_pos"][:4]
        frac = patch_fraction(d["lesion_mask"], slp, yp, xp, m["patch"])
        les = frac >= m["lesion_frac"]
        if m.get("lesion_rule") == "frac_or_centre":
            les = les | (d["lesion_mask"][slp, yp, xp] > 0)
        Fn = pat["train_normal"][0]
        X = np.concatenate([Fn[:, :nc], Fp[:, :nc]])
        is_l = np.r_[np.zeros(len(Fn), bool), les]
        is_n = np.r_[np.ones(len(Fn), bool), frac == 0]
        self.pca_mu, self.pca_V = fit_pca(X, is_l, is_n, n=m.get("pca_sample", 200_000), k=m["pca"], seed=seed)
        self.pca_nc = nc
        log(f"PCA {nc} -> {len(self.pca_V)} on {min(is_l.sum(), m.get('pca_sample', 200_000) // 2)} lesion + normal")

    def _pca(self, F):
        if getattr(self, "pca_V", None) is None:
            return F
        from .frontend import apply_pca
        return np.concatenate([apply_pca(F[:, :self.pca_nc], self.pca_mu, self.pca_V), F[:, self.pca_nc:]], 1)

    def _sel(self, bins):
        """Memory index of each patch: its position bin, or 0 for a single global memory (ablations)."""
        return bins if getattr(self, "mode", "position") == "position" else np.zeros_like(bins)

    def _codes(self, X, bins):
        c = self.enc.codes(X)
        return c ^ self.poscodes[bins] if getattr(self, "mode", "position") == "bound" else c

    def raw_scores(self, codes: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """Hamming-similarity score of each patch HV against the prototypes of its bin."""
        sc = np.zeros(len(codes))
        for b in np.unique(bins):
            q = bins == b
            sc[q] = self.clf[int(b)].score(bipolar(unpack(codes[q], self.enc.D)))
        return sc

    def patches(self, d: dict, fmaps: np.ndarray, idx: np.ndarray):
        F, sl, ys, xs = extract_patches(d, idx, self.m["patch"], self.m["stride"], fmaps, context=d.get("_context"),
                                        pix=self.m.get("pixels", 0), tex=self.m.get("texture", False),
                                        stats=self.m.get("stats", True), readout=self.m.get("readout", "nearest"))
        F = self._pca(F)
        return F, sl, ys, xs, position_bins(d, sl, ys, xs, self.m["slice_levels"])

    def patch_z(self, F: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """Descriptors -> hypervectors -> per-bin z-scored patch scores."""
        mb = self._sel(bins)
        sc = self.raw_scores(self._codes(self.enc.prep(F), bins), mb)
        return (sc - self.mu[mb]) / self.sd[mb]

    def maps(self, d: dict, fmaps: np.ndarray, idx: np.ndarray) -> dict:
        F, sl, ys, xs, bins = self.patches(d, fmaps, idx)
        mp = assemble_maps(d, idx, sl, ys, xs, self.patch_z(F, bins), self.m["patch"], self.m["smooth_sigma"])
        return case_volumes(d, idx, mp)


def train_and_map(d: dict, fmaps: np.ndarray, split: dict, m: dict, seed: int, log=print, return_model=False):
    """m: the 'model' section of the config. Returns {'val': case volumes, 'test': case volumes} (and the model)."""
    model = PatchModel().fit(d, fmaps, split, m, seed, log)
    idx = split_indices(d, split)
    out = {s: model.maps(d, fmaps, idx[s]) for s in ("val", "test")}
    return (out, model) if return_model else out


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
