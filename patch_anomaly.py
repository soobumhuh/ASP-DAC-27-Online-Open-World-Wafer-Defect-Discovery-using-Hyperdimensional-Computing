#!/usr/bin/env python3
"""
patch_anomaly.py - patch-level, position-aware anomaly maps for csPCa lesion localisation.

Whole-slice anomaly scores were at chance on the lesion task: slice position and anatomy
dominate a whole-slice descriptor and lesions are small. Here each small patch is compared
only with normal patches from the SAME position in the gland.

Pipeline
  1. Patches: P x P (default 12) with stride 4, kept when the patch centre is inside the gland.
     Descriptor = the 3-channel pixels (T2W, ADC, HBV) flattened, plus per-channel mean/std
     (--descriptor pixels+stats, the default; or pixels / stats alone).
  2. Position bin = relative slice position (3 bins: apex / mid / base) x in-plane 3 x 3 grid
     of the prostate crop = 27 bins.
  3. Normal memory = patches of the TRAIN split's csPCa-negative patients only (same patient split
     as picai_bundle.split_patients, same seed). Clip + standardise fitted on those patches.
     Variants:
       hdc<N>  Eric's BinaryThermometerHDC (N random projections x 8 bins) -> binary hypervectors in a
               faiss IndexBinaryFlat per bin; score = mean Hamming distance to the k nearest normals
       float   same, Euclidean kNN on the standardised descriptors (faiss IndexFlatL2), no HDC
       smooth_adc / smooth_hbv / smooth_mix   no kNN: patch mean of -ADC, of HBV, or of (HBV - ADC)/2,
               z-scored per bin like the anomaly variants (raw-intensity baselines, same patch footprint)
       raw_adc / raw_hbv / raw_mix            the same intensities per pixel, no patches; z-scored with
               the val normals' gland pixels
     Patch scores are z-scored per bin with the val split's normal (csPCa-negative) patches.
  4. Maps: overlapping patch scores averaged per pixel, zero outside the gland, slices stacked into
     one 3D volume per patient (crop space). Detection maps from report_guided_annotation's
     extract_lesion_candidates on sigmoid((z - offset) / temp), offset = a percentile and temp = a fraction
     of the std of the val split's normal voxel map values (scale-free across variants); evaluated with picai_eval.evaluate against the
     stacked lesion masks (patient AUROC, lesion AP, PI-CAI score). Voxel AUROC inside the gland as a
     sanity check. Every variant is evaluated on the VAL split (all its patients) and the TEST split for
     each (offset, temp) setting, so settings can be chosen on val (see patch_sweep.py).

Example:
  uv run python patch_anomaly.py --slices_npz /data/picai_fold0_masks.npz --descriptor stats \
      --variants hdc256,float,smooth_adc,raw_adc --patch 8 --stride 2 --figure q99/t0.25
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from picai_bundle import split_patients
from wafer_hdc_encoding import BinaryThermometerHDC, apply_clip, apply_standardize, fit_clip_stats, fit_standardize


def n_bins(mode: str, levels: int = 3) -> int:
    return levels * {"grid": 9, "zone": 3, "zone_grid": 27}[mode]


def rebin(d: dict, sl: np.ndarray, ys: np.ndarray, xs: np.ndarray, mode: str, levels: int = 3):
    """Position bins. grid: slice level x in-plane 3 x 3 grid of the crop (27 for 3 levels).
    zone: slice level x zone at the patch centre (0 = unlabelled, 1 = PZ, 2 = TZ; AI zonal mask) (9 for 3 levels).
    zone_grid: slice level x zone x 3 x 3 grid."""
    S = d["images"].shape[-1]
    zl = np.minimum((d["rel_z"][sl] * levels).astype(int), levels - 1)
    cell = np.minimum(ys * 3 // S, 2) * 3 + np.minimum(xs * 3 // S, 2)
    if mode == "zone":
        return zl * 3 + d["zone_mask"][sl, ys, xs].astype(np.int64)
    if mode == "zone_grid":
        return (zl * 3 + d["zone_mask"][sl, ys, xs].astype(np.int64)) * 9 + cell
    return zl * 9 + cell


def bin_group(b, mode: str):
    """Coarser group used to pool lesion patches when a bin has too few: same zone (zone modes) or grid cell."""
    return {"zone": b % 3, "zone_grid": (b // 9) % 3, "grid": b % 9}[mode]


def patch_fraction(mask: np.ndarray, sl, ys, xs, P: int) -> np.ndarray:
    """Fraction of mask pixels inside each P x P patch (summed-area table per slice)."""
    h = P // 2
    out = np.zeros(len(sl))
    for i in np.unique(sl):
        m = sl == i
        c = np.pad(mask[i].astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        y0, x0 = ys[m] - h, xs[m] - h
        out[m] = (c[y0 + P, x0 + P] - c[y0, x0 + P] - c[y0 + P, x0] + c[y0, x0]) / (P * P)
    return out


# ---------------------------------------------------------------- patches
def patch_grid(S: int, P: int, stride: int) -> np.ndarray:
    h = P // 2
    return np.arange(h, S - (P - h) + 1, stride)


def extract_patches(d: dict, idx: np.ndarray, P: int, stride: int, add_stats: bool, add_pixels: bool = True,
                    fmaps: np.ndarray = None, add_pct: bool = False, add_tex: bool = False, fch: slice = None):
    """Patches of slices idx whose centre lies in the gland.
    Returns descriptors (n, D) float32, slice index (n,), centre y, x (n,), position bin (n,),
    per-channel patch means (n, C) (used by the smoothed raw-intensity baselines).
    With fmaps (N, D, h, w): the feature vector at the patch centre (HDnn / PatchCore; channels fch) is part of it.
    add_pct: per-channel 10/50/90th percentiles; add_tex: T2W mean |d/dy|, mean |d/dx|, mean |Laplacian|."""
    S = d["images"].shape[-1]
    cs = patch_grid(S, P, stride)
    h = P // 2
    zbin = np.minimum((d["rel_z"] * 3).astype(int), 2)
    feats, sl, ys, xs, bins, means = [], [], [], [], [], []
    for i in idx:
        img = d["images"][i].astype(np.float32)
        g = d["gland_mask"][i]
        cy, cx = np.meshgrid(cs, cs, indexing="ij")
        cy, cx = cy.ravel(), cx.ravel()
        ok = g[cy, cx] > 0
        cy, cx = cy[ok], cx[ok]
        if len(cy) == 0:
            continue
        # (n, C, P, P) via fancy indexing
        oy = cy[:, None] - h + np.arange(P)[None, :]
        ox = cx[:, None] - h + np.arange(P)[None, :]
        pat = img[:, oy[:, :, None], ox[:, None, :]].transpose(1, 0, 2, 3)  # (n, C, P, P)
        parts = []
        if fmaps is not None:
            fh = fmaps.shape[-1]
            fy = np.clip(((cy + 0.5) * fh / S).astype(int), 0, fh - 1)
            fx = np.clip(((cx + 0.5) * fh / S).astype(int), 0, fh - 1)
            fm = np.asarray(fmaps[i] if fch is None else fmaps[i][fch], dtype=np.float32)
            parts.append(fm[:, fy, fx].T)
        if add_pixels:
            parts.append(pat.reshape(len(cy), -1))
        if add_stats:
            parts += [pat.mean(axis=(2, 3)), pat.std(axis=(2, 3))]
        if add_pct:
            parts.append(np.percentile(pat.reshape(len(cy), pat.shape[1], -1), [10, 50, 90], axis=2)
                         .transpose(1, 2, 0).reshape(len(cy), -1))
        if add_tex:
            t = pat[:, 0]
            lap = np.abs(4 * t[:, 1:-1, 1:-1] - t[:, :-2, 1:-1] - t[:, 2:, 1:-1] - t[:, 1:-1, :-2] - t[:, 1:-1, 2:])
            parts.append(np.stack([np.abs(np.diff(t, axis=1)).mean((1, 2)), np.abs(np.diff(t, axis=2)).mean((1, 2)),
                                   lap.mean((1, 2))], 1))
        f = np.concatenate(parts, axis=1)
        feats.append(f.astype(np.float32))
        means.append(pat.mean(axis=(2, 3)).astype(np.float32))
        sl.append(np.full(len(cy), i)); ys.append(cy); xs.append(cx)
        bins.append(zbin[i] * 9 + np.minimum(cy * 3 // S, 2) * 3 + np.minimum(cx * 3 // S, 2))
    cat = lambda a, dt: np.concatenate(a).astype(dt) if a else np.zeros(0, dt)
    F = np.concatenate(feats) if feats else np.zeros((0, 1), np.float32)
    M = np.concatenate(means) if means else np.zeros((0, 3), np.float32)
    return F, cat(sl, np.int64), cat(ys, np.int64), cat(xs, np.int64), cat(bins, np.int64), M


# ---------------------------------------------------------------- kNN memories (the HDDB part)
class BinMemory:
    """One faiss index per position bin; score = mean distance to the k nearest stored normals."""

    def __init__(self, kind: str, dim: int, k: int):
        self.kind, self.dim, self.k = kind, dim, k
        self.index = {}

    def _new(self):
        import faiss
        return faiss.IndexBinaryFlat(self.dim) if self.kind == "binary" else faiss.IndexFlatL2(self.dim)

    def add(self, X: np.ndarray, bins: np.ndarray):
        self.ids = {}
        self._X, self._all = X, np.arange(len(X))
        for b in np.unique(bins):
            ix = self._new()
            ix.add(np.ascontiguousarray(X[bins == b]))
            self.index[int(b)] = ix
            self.ids[int(b)] = np.where(bins == b)[0]

    def knn(self, X: np.ndarray, bins: np.ndarray):
        """Distances (n, k) (Hamming bits, or Euclidean) and global row ids (n, k) of the k nearest normals
        in the query's own position bin."""
        n = len(X)
        D_out = np.zeros((n, self.k), np.float64)
        I_out = np.zeros((n, self.k), np.int64)
        for b in np.unique(bins):
            m = bins == b
            ix, ids = self.index.get(int(b)), self.ids.get(int(b))
            if ix is None:  # no normal reference at this position: fall back to all bins pooled
                ix, ids = self._pooled(), self._all
            k = min(self.k, ix.ntotal)
            D, I = ix.search(np.ascontiguousarray(X[m]), k)
            D = D.astype(np.float64)
            if self.kind == "float":
                D = np.sqrt(np.maximum(D, 0))
            if k < self.k:  # pad by repeating the last neighbour
                D = np.concatenate([D, np.repeat(D[:, -1:], self.k - k, 1)], 1)
                I = np.concatenate([I, np.repeat(I[:, -1:], self.k - k, 1)], 1)
            D_out[m], I_out[m] = D, ids[I]
        return D_out, I_out

    def score(self, X: np.ndarray, bins: np.ndarray) -> np.ndarray:
        return self.knn(X, bins)[0].mean(axis=1)

    def _pooled(self):
        if not hasattr(self, "_pool"):
            self._pool = self._new()
            self._pool.add(np.ascontiguousarray(self._X))
        return self._pool


def hdc_codes(enc: BinaryThermometerHDC, X: np.ndarray) -> np.ndarray:
    """Eric's packed uint64 hypervectors viewed as bytes for faiss (bit order does not change Hamming)."""
    return np.ascontiguousarray(enc.transform(X, batch_size=4096)).view(np.uint8)


# ---------------------------------------------------------------- maps
def score_maps(d, slices, sl, ys, xs, z, P, sigma: float = 0.0):
    """Average overlapping patch scores per pixel; NaN where no patch covers / outside gland.
    sigma > 0: in-plane Gaussian smoothing inside the gland (normalised convolution)."""
    S = d["images"].shape[-1]
    h = P // 2
    row = np.full(int(slices.max()) + 1, -1, np.int64)
    row[slices] = np.arange(len(slices))
    j = row[sl]
    acc = np.zeros((len(slices), S, S), np.float64)
    cnt = np.zeros((len(slices), S, S), np.float64)
    for dy in range(P):
        for dx in range(P):
            np.add.at(acc, (j, ys - h + dy, xs - h + dx), z)
            np.add.at(cnt, (j, ys - h + dy, xs - h + dx), 1.0)
    with np.errstate(invalid="ignore"):
        m = acc / cnt
    gland = d["gland_mask"][slices] > 0
    m[~gland | (cnt == 0)] = np.nan
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        w = (~np.isnan(m)).astype(np.float64)
        num = gaussian_filter(np.nan_to_num(m), sigma=(0, sigma, sigma))
        den = gaussian_filter(w, sigma=(0, sigma, sigma))
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(w > 0, num / den, np.nan)
    return m


def patient_volumes(d, slices, maps):
    """Stack each case's slices (sorted by z) into a (Z, S, S) volume, keyed by study_id
    (a few PI-CAI patients have more than one study; the patient split keeps them together)."""
    out = {}
    pid = d["study_id"][slices]
    for p in np.unique(pid):
        j = np.where(pid == p)[0]
        j = j[np.argsort(d["slice_z"][slices[j]])]
        out[int(p)] = dict(rows=j, slices=slices[j], map=maps[j],
                           lesion=d["lesion_mask"][slices[j]].astype(np.int32),
                           cspca=int(d["case_cspca"][slices[j][0]]))
    return out


# ---------------------------------------------------------------- figure
def save_figure(d, vols, path, offset, n_each=3, seed=0, title="", temp=1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    pos = [p for p, v in vols.items() if v["cspca"] == 1 and v["lesion"].any()]
    neg = [p for p, v in vols.items() if v["cspca"] == 0]
    picks = list(rng.permutation(pos)[:n_each]) + list(rng.permutation(neg)[:n_each])
    fig, axes = plt.subplots(len(picks), 3, figsize=(7.5, 2.5 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, p in enumerate(picks):
        v = vols[p]
        if v["lesion"].any():
            j = int(v["lesion"].reshape(len(v["lesion"]), -1).sum(1).argmax())
        else:
            j = int(np.argmin(np.abs(d["rel_z"][v["slices"]] - 0.5)))
        s = v["slices"][j]
        from scipy.special import expit
        heat = expit((np.nan_to_num(v["map"][j], nan=-1e3) - offset) / temp)
        heat[np.isnan(v["map"][j])] = np.nan
        for c, (im, col_title, cmap) in enumerate([(d["images"][s][0], "T2W", "gray"), (d["images"][s][1], "ADC", "gray"),
                                               (heat, "anomaly map", "magma")]):
            ax = axes[r, c]
            if c == 2:
                ax.imshow(d["images"][s][0].astype(np.float32), cmap="gray")
                ax.imshow(heat, cmap=cmap, vmin=0, vmax=1, alpha=0.75)
            else:
                ax.imshow(im.astype(np.float32), cmap=cmap)
            ax.contour(d["gland_mask"][s], levels=[0.5], colors="c", linewidths=0.5)
            if v["lesion"][j].any():
                ax.contour(v["lesion"][j], levels=[0.5], colors="lime", linewidths=1.0)
            ax.set_title(f"{'csPCa+' if v['cspca'] else 'csPCa-'} {p} z={d['slice_z'][s]}  {col_title}" if c == 0 else col_title,
                         fontsize=7)
            ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- detection + picai_eval
_OT = (0.0, 1.0, "dynamic-fast")


def _detect(m):
    from report_guided_annotation import extract_lesion_candidates
    from scipy.special import expit
    prob = expit((np.nan_to_num(m, nan=-1e3) - _OT[0]) / _OT[1])
    return extract_lesion_candidates(prob.astype(np.float64), threshold=_OT[2])[0].astype(np.float32)


def _set_ot(o, t, mode="dynamic-fast"):
    global _OT
    _OT = (o, t, mode)


def evaluate_maps(vols, settings, pool_size, threshold_mode: str = "dynamic-fast"):
    """voxel AUROC + picai_eval metrics for each setting {label: (offset, temperature)}:
    detection input = sigmoid((z - offset) / temperature)."""
    from multiprocessing import get_context
    from picai_eval import evaluate
    from sklearn.metrics import roc_auc_score
    keys = sorted(vols)
    allm = np.concatenate([vols[p]["map"].ravel() for p in keys])
    alll = np.concatenate([vols[p]["lesion"].ravel() for p in keys])
    ok = ~np.isnan(allm)
    out = {"voxel_auroc_in_gland": float(roc_auc_score(alll[ok] > 0, allm[ok])) if alll[ok].any() else float("nan"),
           "patients_pos": int(sum(vols[p]["cspca"] for p in keys)),
           "patients_neg": int(sum(1 - vols[p]["cspca"] for p in keys)),
           "by_offset": {}}
    y_true = [(vols[p]["lesion"] > 0).astype(np.int32) for p in keys]
    out["num_lesion_patients_by_mask"] = int(sum(y.any() for y in y_true))
    for label, (o, t) in settings.items():
        with get_context("fork").Pool(pool_size, initializer=_set_ot, initargs=(o, t, threshold_mode)) as pool:
            y_det = pool.map(_detect, [vols[p]["map"] for p in keys])
        m = evaluate(y_det=y_det, y_true=y_true, subject_list=[str(p) for p in keys], num_parallel_calls=pool_size)
        out["by_offset"][label] = {"offset": float(o), "temp": float(t), "patient_auroc": float(m.auroc), "lesion_ap": float(m.AP),
                                    "picai_score": float(m.score), "num_lesions": int(m.num_lesions)}
    return out


# ---------------------------------------------------------------- main
RAW = {"adc": lambda im: -im[:, 1], "hbv": lambda im: im[:, 2], "mix": lambda im: (im[:, 2] - im[:, 1]) / 2}
RAW_M = {"adc": lambda M: -M[:, 1], "hbv": lambda M: M[:, 2], "mix": lambda M: (M[:, 2] - M[:, 1]) / 2}
RESNET_LAYERS = {"resnet_l2": ("layer2",), "resnet_l3": ("layer3",), "resnet_l2l3": ("layer2", "layer3")}


def resnet_feature_maps(images: np.ndarray, layers, cache: str = None, batch_size: int = 64) -> np.ndarray:
    """HDnn / PatchCore-style local features: frozen ImageNet ResNet-18, the 3-channel crop (T2W, ADC, HBV as RGB)
    upsampled to 224, layer2 (28 x 28, 128 ch) and/or layer3 (14 x 14, 256 ch, upsampled to 28 x 28),
    3 x 3 average-pooled for local context (as in PatchCore). Returns (N, D, 28, 28) float16."""
    if cache and os.path.exists(cache):
        return np.load(cache, mmap_mode="r")
    import torch
    import torch.nn.functional as F
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            x = torch.from_numpy(images[i:i + batch_size, :3].astype(np.float32))
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
            x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            l2 = m.layer2(m.layer1(x))
            feats = {"layer2": l2}
            if "layer3" in layers:
                feats["layer3"] = F.interpolate(m.layer3(l2), size=l2.shape[-1], mode="bilinear", align_corners=False)
            f = torch.cat([F.avg_pool2d(feats[k], 3, 1, 1) for k in layers], 1)
            outs.append(f.numpy().astype(np.float16))
    out = np.concatenate(outs)
    if cache:
        np.save(cache, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices_npz", required=True, help="picai_slices.py output made with --save_masks")
    ap.add_argument("--variants", default="hdc256,float,smooth_adc,smooth_hbv,smooth_mix,raw_adc,raw_hbv,raw_mix",
                    help="comma list of: hdc<num_proj> | float  (unsigned kNN anomaly), each optionally with "
                         "_signed (only lesion-direction intensity deviations from the k nearest normals count) or "
                         "_int (smoothed (HBV-ADC)/2 per-bin z appended to the descriptor before encoding); "
                         "combo<w>_<hdcN|float> (smoothed (HBV-ADC)/2 z + w x kNN anomaly z); "
                         "smooth_{adc,hbv,mix}; raw_{adc,hbv,mix}")
    ap.add_argument("--save_root", default="results", help="results go to <save_root>/patch_<variant><tag>/")
    ap.add_argument("--tag", default="", help="suffix for the results folder")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.25)
    ap.add_argument("--min_lesion_pixels", type=int, default=10, help="only used to reproduce picai_bundle's split")
    ap.add_argument("--patch", type=int, default=12)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--descriptor", default="stats",
                    help="patch descriptor: flattened 3-channel pixels and/or per-channel mean/std, or frozen "
                         "ResNet-18 feature maps at the patch centre (resnet_l2 | resnet_l3 | resnet_l2l3)")
    ap.add_argument("--feature_cache", default="./mri_feature_cache")
    ap.add_argument("--slice_levels", type=int, default=3, help="slice-level bins (apex .. base)")
    ap.add_argument("--sup_lr", type=float, default=1.0, help="retraining step for proto variants")
    ap.add_argument("--sup_balance", type=int, default=1, help="class-balanced retraining epochs (1) or all samples (0)")
    ap.add_argument("--label_rule", default="centre",
                    help="lesion patch = centre in the lesion mask ('centre') or lesion fraction >= f ('frac0.25')")
    ap.add_argument("--score_mode", default="max", help="proto score per class: max | top2 | lse<T> (soft max)")
    ap.add_argument("--smooth_sigma", type=float, default=0.0, help="Gaussian smoothing of the maps (px)")
    ap.add_argument("--det_threshold", default="dynamic-fast",
                    help="extract_lesion_candidates threshold mode: dynamic-fast | dynamic")
    ap.add_argument("--bin_mode", choices=["grid", "zone", "zone_grid"], default="grid",
                    help="grid: 3 slice levels x 3x3 in-plane grid (27 bins); zone: 3 slice levels x AI zone "
                         "(PZ / TZ / unlabelled) at the patch centre (9 bins; needs zone_mask in the npz)")
    ap.add_argument("--sup_k", type=int, default=25, help="neighbours for the supervised kNN vote")
    ap.add_argument("--ctx3d", type=int, default=0, help="proto/vote: bundle patch HVs with the slices above/below")
    ap.add_argument("--split_file", default=None,
                    help="json {train: [patient ids], val: [...], test: [...]} overriding the seeded split "
                         "(used by patient_hdc.py to cross-fit image scores on train patients)")
    ap.add_argument("--dump_cases", default=None,
                    help="npz path: per case (study) of val and test, the max of the map and the hypervectors of "
                         "its top-5 patches, for the first variant (patient_hdc.py)")
    ap.add_argument("--skip_eval", action="store_true", help="no detection / picai_eval (with --dump_cases)")
    ap.add_argument("--save_maps", default=None,
                    help="pickle path: per case (study) of val and test, the z-scored map volume (float16), the lesion "
                         "mask and the csPCa label, for the first variant (postproc_search.py)")
    ap.add_argument("--eval_splits", default="val,test",
                    help="splits to run detection + picai_eval on (hyperparameter search: val only)")
    ap.add_argument("--sup_epochs", type=int, default=5, help="HDC retraining epochs for proto variants")
    ap.add_argument("--sup_nsub", default="10,5", help="sub-prototypes per class (normal,lesion) for proto variants")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--bin_cap", type=int, default=50000, help="random cap on normal patches stored per bin")
    ap.add_argument("--num_bins", type=int, default=8, help="HDC thermometer levels")
    ap.add_argument("--clip_value", type=float, default=3.0)
    ap.add_argument("--offset_q", default="90,97,99,99.7",
                    help="comma list of percentiles of the val-normal voxel map values used as sigmoid offsets; "
                         "detection input = sigmoid((z - offset) / temp). Scale-free, so it works for any variant.")
    ap.add_argument("--temp_f", default="0.5,2,8",
                    help="comma list; temp = temp_f x std of the val-normal voxel map values")
    ap.add_argument("--figure", default=None,
                    help="setting label 'q<offset_q>/t<temp_f>' (e.g. q99/t0.25): save test_patients.png with it")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    qs = [float(q) for q in args.offset_q.split(",")]
    tfs = [float(t) for t in args.temp_f.split(",")]

    t_all = time.time()
    d = dict(np.load(args.slices_npz))
    for key in ("gland_mask", "lesion_mask", "rel_z"):
        if key not in d:
            raise SystemExit(f"{args.slices_npz} has no '{key}': rebuild it with picai_slices.py --save_masks")

    # same patient split as picai_bundle.prepare_picai_bundle (first use of the seeded rng)
    rng = np.random.default_rng(args.seed)
    pids, cspca = d["patient_id"], d["case_cspca"]
    split_keep = (d["lesion_pixels"] >= args.min_lesion_pixels) | (cspca == 0)
    split_of = split_patients(pids[split_keep], cspca[split_keep], args.val_frac, args.test_frac, rng)
    split = np.array([split_of.get(int(p), "none") for p in pids])
    if args.split_file:
        sf = json.load(open(args.split_file))
        lut = {int(p): k for k in ("train", "val", "test") for p in sf[k]}
        split = np.array([lut.get(int(p), "none") for p in pids])
    has_lesion = d["lesion_mask"].reshape(len(pids), -1).sum(1) > 0
    counts = {s: {"patients": int(len(np.unique(pids[split == s]))),
                  "patients_pos": int(len(np.unique(pids[(split == s) & (cspca == 1)]))),
                  "slices": int((split == s).sum()), "lesion_slices": int(((split == s) & has_lesion).sum())}
              for s in ("train", "val", "test")}
    counts["unsplit_patients"] = sorted({int(p) for p in pids[split == "none"]})
    print("split:", json.dumps(counts))
    idx = {"train_normal": np.where((split == "train") & (cspca == 0))[0],
           "val": np.where(split == "val")[0], "test": np.where(split == "test")[0]}
    val_neg_slice = cspca[idx["val"]] == 0
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    supervised = any(v.startswith(("proto", "vote")) for v in variants)
    if supervised:
        idx["train_pos"] = np.where((split == "train") & (cspca == 1))[0]
    if args.bin_mode.startswith("zone") and "zone_mask" not in d:
        raise SystemExit("--bin_mode zone needs zone_mask: rebuild with picai_slices.py --save_masks --zonal_dir ...")
    NB = n_bins(args.bin_mode, args.slice_levels)

    t0 = time.time()
    toks = set(args.descriptor.split("+"))
    add_stats, add_pixels, add_pct, add_tex = ("stats" in toks), ("pixels" in toks), ("pct" in toks), ("tex" in toks)
    fmaps, fch = None, None
    rn = [t for t in toks if t in RESNET_LAYERS]
    if rn:
        # one cached layer2+3 map (memory-mapped); layer2 = channels 0-127, layer3 = 128-383
        os.makedirs(args.feature_cache, exist_ok=True)
        cache = os.path.join(args.feature_cache, f"{os.path.basename(args.slices_npz)}_resnet_l2l3.npy")
        fmaps = resnet_feature_maps(d["images"], RESNET_LAYERS["resnet_l2l3"], cache)
        fch = {"resnet_l2": slice(0, 128), "resnet_l3": slice(128, 384), "resnet_l2l3": None}[rn[0]]
    P = {name: extract_patches(d, ix, args.patch, args.stride, add_stats, add_pixels, fmaps, add_pct, add_tex, fch)
         for name, ix in idx.items()}
    del fmaps
    P = {k: (*v[:4], rebin(d, v[1], v[2], v[3], args.bin_mode, args.slice_levels), *v[5:]) for k, v in P.items()}
    t_extract = time.time() - t0
    for name, v in P.items():
        print(f"{name}: {len(v[0])} patches, dim {v[0].shape[1]}")
    counts["patches"] = {name: int(len(v[0])) for name, v in P.items()}
    row_of = np.full(len(pids), -1, np.int64)
    row_of[idx["val"]] = np.arange(len(idx["val"]))
    val_normal_patch = val_neg_slice[row_of[P["val"][1]]]

    # normal memory: clip + standardise on train normal patches, then cap each bin
    Ftr, btr = P["train_normal"][0], P["train_normal"][4]
    lo, hi = fit_clip_stats(Ftr, 1.0, 99.0)
    mu, sd = fit_standardize(apply_clip(Ftr, lo, hi))
    prep = lambda F: apply_standardize(apply_clip(F, lo, hi), mu, sd)
    keep = np.zeros(len(Ftr), bool)
    cap_rng = np.random.default_rng(args.seed + 1)
    capped = {}
    for b in np.unique(btr):
        j = np.where(btr == b)[0]
        if len(j) > args.bin_cap:
            capped[int(b)] = int(len(j))
            j = cap_rng.choice(j, args.bin_cap, replace=False)
        keep[j] = True
    if capped:
        print(f"capped bins to {args.bin_cap} (original sizes): {capped}")
    bin_sizes = {int(b): int((btr[keep] == b).sum()) for b in range(NB)}
    Xtr, btr = prep(Ftr[keep]), btr[keep]
    X = {s: prep(P[s][0]) for s in ("val", "test")}
    # patch-mean intensities in standardised units (train-normal mean/std), for the signed and _int variants
    Mtr_raw = P["train_normal"][5][keep]
    m_mu, m_sd = Mtr_raw.mean(0), Mtr_raw.std(0) + 1e-6
    Mstd = {"train": (Mtr_raw - m_mu) / m_sd, **{s: (P[s][5] - m_mu) / m_sd for s in ("val", "test")}}
    mix = {k: (v[:, 2] - v[:, 1]) / 2 for k, v in Mstd.items()}  # (HBV - ADC) / 2, lesion direction = up
    mix_mu = np.full(NB, mix["train"].mean()); mix_sd = np.full(NB, mix["train"].std() + 1e-6)
    for b in np.unique(btr):
        mix_mu[b], mix_sd[b] = mix["train"][btr == b].mean(), mix["train"][btr == b].std() + 1e-6
    bins_of = {"train": btr, "val": P["val"][4], "test": P["test"][4]}
    mix_z = {k: (mix[k] - mix_mu[bins_of[k]]) / mix_sd[bins_of[k]] for k in mix}  # per-bin z vs train normals

    def zscore_per_bin(sv, bv, scores, bins):
        mu_b = np.full(NB, sv.mean()); sd_b = np.full(NB, sv.std() + 1e-6)
        for b in np.unique(bv):
            mu_b[b], sd_b[b] = sv[bv == b].mean(), sv[bv == b].std() + 1e-6
        return (scores - mu_b[bins]) / sd_b[bins]

    knn_cache = {}
    last_codes = {}

    def with_context(codes, sl, ys, xs):
        """Bundle (bitwise majority of 3) each patch HV with the byte-rotated HVs of the patch at the same (y, x)
        on the slice above and below in the same case (itself when that patch does not exist)."""
        S = d["images"].shape[-1]
        key = (sl.astype(np.int64) * S + ys) * S + xs
        order = np.argsort(key)
        ks = key[order]
        out = []
        for dz, sh in ((-1, 1), (1, 2)):
            nb = sl + dz
            ok = (nb >= 0) & (nb < len(d["study_id"]))
            ok[ok] &= d["study_id"][nb[ok]] == d["study_id"][sl[ok]]
            nk = (nb.astype(np.int64) * S + ys) * S + xs
            pos = np.clip(np.searchsorted(ks, nk), 0, len(ks) - 1)
            hit = ok & (ks[pos] == nk)
            src = np.where(hit, order[pos], np.arange(len(sl)))
            out.append(np.roll(codes[src], sh, axis=1))
        a, b, c = codes, out[0], out[1]
        return ((a & b) | (a & c) | (b & c)).astype(np.uint8)

    def sup_z(kind: str, nproj: int, negplus: bool):
        """Design A: supervised HDC per bin (proto = sub-prototypes + retraining; vote = labelled kNN)."""
        from hdc_classifier import HDCPrototypeClassifier, bipolar, knn_vote_scores, unpack_codes
        enc = BinaryThermometerHDC(in_dim=Xtr.shape[1], num_proj=nproj, num_bins=args.num_bins,
                                   seed=args.seed, clip_value=args.clip_value).fit(Xtr)
        Fp, slp, yp, xp, bp = P["train_pos"][:5]
        if args.label_rule == "centre":
            les = d["lesion_mask"][slp, yp, xp] > 0
            clean = ~les
        else:  # lesion if the patch is at least f lesion; 'normal' (negplus) only if it has no lesion pixel
            frac = patch_fraction(d["lesion_mask"], slp, yp, xp, args.patch)
            les, clean = frac >= float(args.label_rule[4:]), frac == 0
        if args.ctx3d:  # 3D context: bundle each patch HV with the permuted HVs of the slices above and below
            full_n = with_context(hdc_codes(enc, prep(Ftr)), *P["train_normal"][1:4])
            full_p = with_context(hdc_codes(enc, prep(Fp)), slp, yp, xp)
            C = {"n": full_n[keep], "l": full_p[les],
                 "val": with_context(hdc_codes(enc, X["val"]), *P["val"][1:4]),
                 "test": with_context(hdc_codes(enc, X["test"]), *P["test"][1:4])}
        else:
            full_p = None
            C = {"n": hdc_codes(enc, Xtr), "l": hdc_codes(enc, prep(Fp[les])),
                 "val": hdc_codes(enc, X["val"]), "test": hdc_codes(enc, X["test"])}
        B = {"n": btr, "l": bp[les], "val": bins_of["val"], "test": bins_of["test"]}
        if negplus:  # also non-lesion gland patches of csPCa-positive train patients as 'normal'
            nl = np.where(clean)[0]
            nl = np.random.default_rng(args.seed + 2).choice(nl, min(len(nl), len(Xtr) // 2), replace=False)
            C["n"] = np.concatenate([C["n"], full_p[nl] if full_p is not None else hdc_codes(enc, prep(Fp[nl]))])
            B["n"] = np.concatenate([B["n"], bp[nl]])
        group = lambda b: bin_group(b, args.bin_mode)  # same zone / same grid cell
        sc = {s: np.zeros(len(C[s])) for s in ("val", "test")}
        n_fallback = 0
        for b in range(NB):
            qs = {s: B[s] == b for s in ("val", "test")}
            if not any(q.any() for q in qs.values()):
                continue
            mn = B["n"] == b
            if mn.sum() == 0:
                mn = np.ones(len(B["n"]), bool)
            ml = B["l"] == b
            if ml.sum() < 50:  # too few lesion patches here: pool lesions from the same zone (or grid cell), then all
                n_fallback += 1
                ml = group(B["l"]) == group(b)
                if ml.sum() < 50:
                    ml = np.ones(len(B["l"]), bool)
            if kind == "proto":
                Xn, Xl = bipolar(unpack_codes(C["n"][mn], enc.bit_dim)), bipolar(unpack_codes(C["l"][ml], enc.bit_dim))
                clf = HDCPrototypeClassifier(n_sub=tuple(int(v) for v in args.sup_nsub.split(",")),
                                             epochs=args.sup_epochs, seed=args.seed + b, lr=args.sup_lr,
                                             balance=bool(args.sup_balance), score_mode=args.score_mode).fit(
                    np.concatenate([Xn, Xl]), np.r_[np.zeros(len(Xn), int), np.ones(len(Xl), int)])
                for s, q in qs.items():
                    if q.any():
                        sc[s][q] = clf.score(bipolar(unpack_codes(C[s][q], enc.bit_dim)))
            else:
                jn = np.where(mn)[0]
                if len(jn) > 20000:
                    jn = np.random.default_rng(args.seed + b).choice(jn, 20000, replace=False)
                lib = np.concatenate([C["n"][jn], C["l"][ml]])
                ly = np.r_[np.zeros(len(jn), int), np.ones(int(ml.sum()), int)]
                for s, q in qs.items():
                    if q.any():
                        sc[s][q] = knn_vote_scores(lib, ly, C[s][q], enc.bit_dim, args.sup_k)
        z = {s: zscore_per_bin(sc["val"][val_normal_patch], bins_of["val"][val_normal_patch], sc[s], bins_of[s])
             for s in sc}
        last_codes.update(val=C["val"], test=C["test"])
        meta = {**enc.meta(), "supervised": kind, "negplus": negplus, "train_lesion_patches": int(les.sum()),
                "bins_with_pooled_lesions": n_fallback, "bin_mode": args.bin_mode}
        return z, meta

    def knn_z(base: str, mod: str):
        """Per-bin z-scored patch scores {val, test} for a kNN variant (cached so combo variants reuse them)."""
        key = (base, mod)
        if key in knn_cache:
            return knn_cache[key]
        A = {"train": Xtr, **X}
        if mod == "int":  # intensity channel, weighted like the whole descriptor (sqrt(D))
            w = np.sqrt(Xtr.shape[1])
            A = {k: np.concatenate([v, w * mix_z[k][:, None]], 1).astype(np.float32) for k, v in A.items()}
        if base.startswith("hdc"):
            enc = BinaryThermometerHDC(in_dim=A["train"].shape[1], num_proj=int(base[3:]), num_bins=args.num_bins,
                                       seed=args.seed, clip_value=args.clip_value).fit(A["train"])
            codes = lambda Z: hdc_codes(enc, Z)
            mem = BinMemory("binary", enc.bit_dim, args.k)
            meta = enc.meta()
        elif base == "float":
            codes = lambda Z: np.ascontiguousarray(Z, dtype=np.float32)
            mem = BinMemory("float", A["train"].shape[1], args.k)
            meta = {"encoder": "none (float standardised descriptors)", "dim": int(A["train"].shape[1])}
        else:
            raise SystemExit(f"unknown kNN base {base}")
        mem.add(codes(A["train"]), btr)
        sc = {}
        for s in ("val", "test"):
            D, I = mem.knn(codes(A[s]), bins_of[s])
            if mod == "signed":
                # only deviations in the lesion direction (darker ADC, brighter HBV) than the k nearest normals
                nn = Mstd["train"][I]  # (n, k, 3)
                q = Mstd[s][:, None, :]
                sc[s] = (np.maximum(0, nn[..., 1] - q[..., 1]) + np.maximum(0, q[..., 2] - nn[..., 2])).mean(1)
            else:
                sc[s] = D.mean(1)
        z = {s: zscore_per_bin(sc["val"][val_normal_patch], bins_of["val"][val_normal_patch], sc[s], bins_of[s])
             for s in sc}
        knn_cache[key] = (z, meta)
        return knn_cache[key]

    for variant in variants:
        t0 = time.time()
        meta = {}
        if variant.startswith("raw_"):
            f = RAW[variant[4:]]
            maps = {s: f(d["images"][idx[s]].astype(np.float32)) for s in ("val", "test")}
            for s in maps:
                maps[s][d["gland_mask"][idx[s]] == 0] = np.nan
            ref = maps["val"][val_neg_slice]
            ref = ref[~np.isnan(ref)]
            for s in maps:
                maps[s] = (maps[s] - ref.mean()) / (ref.std() + 1e-6)
            meta = {"encoder": "none (raw pixel intensity)"}
        else:
            if variant.startswith("smooth_"):
                f = RAW_M[variant[7:]]
                raw = {s: f(P[s][5]).astype(np.float64) for s in ("val", "test")}
                sv, bv = raw["val"][val_normal_patch], P["val"][4][val_normal_patch]
                z = {s: zscore_per_bin(sv, bv, raw[s], P[s][4]) for s in raw}
                meta = {"encoder": "none (patch-mean intensity)"}
            elif variant.startswith(("proto", "vote")):
                kind = "proto" if variant.startswith("proto") else "vote"
                base, _, mod = variant.partition("_")
                z, meta = sup_z(kind, int(base[len(kind):]), mod == "negplus")
            elif variant.startswith("combo"):
                wstr, base = variant[5:].split("_", 1)
                zk, meta = knn_z(base, "")
                sv, bv = RAW_M["mix"](P["val"][5])[val_normal_patch].astype(np.float64), bins_of["val"][val_normal_patch]
                zs = {s: zscore_per_bin(sv, bv, RAW_M["mix"](P[s][5]).astype(np.float64), bins_of[s]) for s in zk}
                z = {s: zs[s] + float(wstr) * zk[s] for s in zk}
                meta = {**meta, "combo_weight": float(wstr)}
            else:
                base, _, mod = variant.partition("_")
                z, meta = knn_z(base, mod)
            maps = {s: score_maps(d, idx[s], P[s][1], P[s][2], P[s][3], z[s], args.patch, args.smooth_sigma) for s in z}
        t_score = time.time() - t0

        if args.dump_cases and variant == variants[0]:
            out = {}
            for s in ("val", "test"):
                st_slice = d["study_id"][idx[s]]
                st_patch = d["study_id"][P[s][1]]
                cases = np.unique(st_slice)
                mx = np.array([np.nanmax(maps[s][st_slice == c]) for c in cases])
                top = []
                for c in cases:
                    j = np.where(st_patch == c)[0]
                    j = j[np.argsort(-z[s][j])[:5]]
                    if "val" in last_codes and len(j):
                        cc = last_codes[s][j]
                        top.append(np.concatenate([cc, np.repeat(cc[-1:], 5 - len(cc), 0)]))
                    else:
                        top.append(np.zeros((5, 1), np.uint8))
                out[f"{s}_study"], out[f"{s}_max"], out[f"{s}_top"] = cases, mx, np.stack(top)
                # per-zone fields (zone of a patch = zone at its centre; of a voxel = the zone mask)
                if "zone_mask" in d:
                    zm = d["zone_mask"][idx[s]]
                    zp = d["zone_mask"][P[s][1], P[s][2], P[s][3]]
                    ref = maps["val"][val_neg_slice]
                    thr = float(np.nanpercentile(ref, 97))  # val csPCa-negative voxels, never test
                    for zz, zn in ((1, "pz"), (2, "tz")):
                        zmax, zcnt, ztop = [], [], []
                        for c in cases:
                            mc, zc = maps[s][st_slice == c], zm[st_slice == c] == zz
                            v = mc[zc & ~np.isnan(mc)]
                            zmax.append(float(v.max()) if len(v) else np.nan)
                            zcnt.append(float((v > thr).sum()))
                            j = np.where((st_patch == c) & (zp == zz))[0]
                            j = j[np.argsort(-z[s][j])[:5]]
                            if "val" in last_codes and len(j):
                                cc = last_codes[s][j]
                                ztop.append(np.concatenate([cc, np.repeat(cc[-1:], 5 - len(cc), 0)]))
                            else:
                                ztop.append(np.zeros((5, out[f"{s}_top"].shape[-1]), np.uint8))
                        out[f"{s}_max_{zn}"], out[f"{s}_cnt_{zn}"] = np.array(zmax), np.array(zcnt)
                        out[f"{s}_top_{zn}"] = np.stack(ztop)
            np.savez_compressed(args.dump_cases, **out)
            print("dumped case scores ->", args.dump_cases)
            if args.skip_eval:
                break
        t1 = time.time()
        res = {"variant": variant}
        # offsets / temperatures from the val split's csPCa-negative voxels (never test)
        ref = maps["val"][val_neg_slice]
        ref = ref[~np.isnan(ref)]
        settings = {f"q{q:g}/t{tf:g}": (float(np.percentile(ref, q)), float(tf * ref.std() + 1e-6))
                    for q in qs for tf in tfs}
        vols = {}
        if args.save_maps and variant == variants[0]:
            import pickle
            dump = {}
            for s in ("val", "test"):
                v = patient_volumes(d, idx[s], maps[s])
                dump[s] = {int(k): {"map": c["map"].astype(np.float16), "lesion": (c["lesion"] > 0).astype(np.uint8),
                                    "cspca": c["cspca"],
                                    "zone": (d["zone_mask"][c["slices"]] if "zone_mask" in d else
                                             np.zeros_like(c["lesion"])).astype(np.uint8),
                                    "rel_z": d["rel_z"][c["slices"]].astype(np.float32)} for k, c in v.items()}
                if s in last_codes:  # patch hypervectors + positions (row in the case volume, y, x) for re-scoring
                    st_patch = d["study_id"][P[s][1]]
                    for k, c in v.items():
                        j = np.where(st_patch == k)[0]
                        row = {int(sl): r for r, sl in enumerate(c["slices"])}
                        dump[s][int(k)]["codes"] = last_codes[s][j]
                        dump[s][int(k)]["pos"] = np.stack([[row[int(x)] for x in P[s][1][j]], P[s][2][j],
                                                           P[s][3][j]], 1).astype(np.int16)
                        dump[s][int(k)]["patch"] = args.patch
            os.makedirs(os.path.dirname(os.path.abspath(args.save_maps)), exist_ok=True)
            with open(args.save_maps, "wb") as fh:
                pickle.dump(dump, fh, protocol=4)
            print("saved maps ->", args.save_maps)
        eval_splits = [s for s in ("val", "test") if s in args.eval_splits.split(",")]
        for s in eval_splits:
            vols[s] = patient_volumes(d, idx[s], maps[s])
            res[s] = evaluate_maps(vols[s], settings, args.workers, args.det_threshold)
        t_eval = time.time() - t1
        res.update(runtime_s={"patch_extraction_shared": round(t_extract, 1), "score_maps": round(t_score, 1),
                              "detection_and_eval": round(t_eval, 1)},
                   encoder=meta, normal_patches_per_bin=bin_sizes, capped_bins=capped,
                   split_counts=counts, args=vars(args))
        out_dir = os.path.join(args.save_root, f"patch_{variant}{args.tag}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        if args.figure and "test" in vols:
            o, t = settings[args.figure]
            save_figure(d, vols["test"], os.path.join(out_dir, "test_patients.png"), o, seed=args.seed, temp=t,
                        title=f"{variant} {args.descriptor} P={args.patch} stride={args.stride} "
                              f"{args.figure} seed={args.seed}  heat = sigmoid((z-{o:.2f})/{t:.2f})")
        for s in eval_splits:
            best = max(res[s]["by_offset"].items(), key=lambda kv: kv[1]["lesion_ap"])
            print(f"[{variant} {s}] voxel AUROC {res[s]['voxel_auroc_in_gland']:.3f} | best-AP setting {best[0]}: "
                  f"patient AUROC {best[1]['patient_auroc']:.3f} AP {best[1]['lesion_ap']:.3f} "
                  f"PI-CAI {best[1]['picai_score']:.3f}")
        print(f"   score {t_score:.0f}s eval {t_eval:.0f}s -> {out_dir}")
    print(f"total {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
