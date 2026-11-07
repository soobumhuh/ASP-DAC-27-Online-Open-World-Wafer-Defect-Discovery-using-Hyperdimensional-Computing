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


# ---------------------------------------------------------------- patches
def patch_grid(S: int, P: int, stride: int) -> np.ndarray:
    h = P // 2
    return np.arange(h, S - (P - h) + 1, stride)


def extract_patches(d: dict, idx: np.ndarray, P: int, stride: int, add_stats: bool, add_pixels: bool = True,
                    fmaps: np.ndarray = None):
    """Patches of slices idx whose centre lies in the gland.
    Returns descriptors (n, D) float32, slice index (n,), centre y, x (n,), position bin (n,),
    per-channel patch means (n, C) (used by the smoothed raw-intensity baselines).
    With fmaps (N, D, h, w): the descriptor is the feature vector at the patch centre instead (HDnn / PatchCore)."""
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
        if fmaps is not None:
            fh = fmaps.shape[-1]
            fy = np.clip(((cy + 0.5) * fh / S).astype(int), 0, fh - 1)
            fx = np.clip(((cx + 0.5) * fh / S).astype(int), 0, fh - 1)
            f = fmaps[i][:, fy, fx].T.astype(np.float32)
        else:
            parts = [pat.reshape(len(cy), -1)] if add_pixels else []
            if add_stats:
                parts += [pat.mean(axis=(2, 3)), pat.std(axis=(2, 3))]
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
def score_maps(d, slices, sl, ys, xs, z, P):
    """Average overlapping patch scores per pixel; NaN where no patch covers / outside gland."""
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
_OT = (0.0, 1.0)


def _detect(m):
    from report_guided_annotation import extract_lesion_candidates
    from scipy.special import expit
    prob = expit((np.nan_to_num(m, nan=-1e3) - _OT[0]) / _OT[1])
    return extract_lesion_candidates(prob.astype(np.float64))[0].astype(np.float32)


def _set_ot(o, t):
    global _OT
    _OT = (o, t)


def evaluate_maps(vols, settings, pool_size):
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
        with get_context("fork").Pool(pool_size, initializer=_set_ot, initargs=(o, t)) as pool:
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
        return np.load(cache)
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
    ap.add_argument("--descriptor", choices=["pixels+stats", "pixels", "stats"] + list(RESNET_LAYERS), default="stats",
                    help="patch descriptor: flattened 3-channel pixels and/or per-channel mean/std, or frozen "
                         "ResNet-18 feature maps at the patch centre (resnet_l2 | resnet_l3 | resnet_l2l3)")
    ap.add_argument("--feature_cache", default="./mri_feature_cache")
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

    t0 = time.time()
    add_stats, add_pixels = "stats" in args.descriptor, "pixels" in args.descriptor
    fmaps = None
    if args.descriptor in RESNET_LAYERS:
        os.makedirs(args.feature_cache, exist_ok=True)
        cache = os.path.join(args.feature_cache, f"{os.path.basename(args.slices_npz)}_{args.descriptor}.npy")
        fmaps = resnet_feature_maps(d["images"], RESNET_LAYERS[args.descriptor], cache)
    P = {name: extract_patches(d, ix, args.patch, args.stride, add_stats, add_pixels, fmaps)
         for name, ix in idx.items()}
    del fmaps
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
    bin_sizes = {int(b): int((btr[keep] == b).sum()) for b in range(27)}
    Xtr, btr = prep(Ftr[keep]), btr[keep]
    X = {s: prep(P[s][0]) for s in ("val", "test")}
    # patch-mean intensities in standardised units (train-normal mean/std), for the signed and _int variants
    Mtr_raw = P["train_normal"][5][keep]
    m_mu, m_sd = Mtr_raw.mean(0), Mtr_raw.std(0) + 1e-6
    Mstd = {"train": (Mtr_raw - m_mu) / m_sd, **{s: (P[s][5] - m_mu) / m_sd for s in ("val", "test")}}
    mix = {k: (v[:, 2] - v[:, 1]) / 2 for k, v in Mstd.items()}  # (HBV - ADC) / 2, lesion direction = up
    mix_mu = np.full(27, mix["train"].mean()); mix_sd = np.full(27, mix["train"].std() + 1e-6)
    for b in np.unique(btr):
        mix_mu[b], mix_sd[b] = mix["train"][btr == b].mean(), mix["train"][btr == b].std() + 1e-6
    bins_of = {"train": btr, "val": P["val"][4], "test": P["test"][4]}
    mix_z = {k: (mix[k] - mix_mu[bins_of[k]]) / mix_sd[bins_of[k]] for k in mix}  # per-bin z vs train normals

    def zscore_per_bin(sv, bv, scores, bins):
        mu_b = np.full(27, sv.mean()); sd_b = np.full(27, sv.std() + 1e-6)
        for b in np.unique(bv):
            mu_b[b], sd_b[b] = sv[bv == b].mean(), sv[bv == b].std() + 1e-6
        return (scores - mu_b[bins]) / sd_b[bins]

    knn_cache = {}

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

    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
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
            maps = {s: score_maps(d, idx[s], P[s][1], P[s][2], P[s][3], z[s], args.patch) for s in z}
        t_score = time.time() - t0

        t1 = time.time()
        res = {"variant": variant}
        # offsets / temperatures from the val split's csPCa-negative voxels (never test)
        ref = maps["val"][val_neg_slice]
        ref = ref[~np.isnan(ref)]
        settings = {f"q{q:g}/t{tf:g}": (float(np.percentile(ref, q)), float(tf * ref.std() + 1e-6))
                    for q in qs for tf in tfs}
        vols = {}
        for s in ("val", "test"):
            vols[s] = patient_volumes(d, idx[s], maps[s])
            res[s] = evaluate_maps(vols[s], settings, args.workers)
        t_eval = time.time() - t1
        res.update(runtime_s={"patch_extraction_shared": round(t_extract, 1), "score_maps": round(t_score, 1),
                              "detection_and_eval": round(t_eval, 1)},
                   encoder=meta, normal_patches_per_bin=bin_sizes, capped_bins=capped,
                   split_counts=counts, args=vars(args))
        out_dir = os.path.join(args.save_root, f"patch_{variant}{args.tag}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        if args.figure:
            o, t = settings[args.figure]
            save_figure(d, vols["test"], os.path.join(out_dir, "test_patients.png"), o, seed=args.seed, temp=t,
                        title=f"{variant} {args.descriptor} P={args.patch} stride={args.stride} "
                              f"{args.figure} seed={args.seed}  heat = sigmoid((z-{o:.2f})/{t:.2f})")
        for s in ("val", "test"):
            best = max(res[s]["by_offset"].items(), key=lambda kv: kv[1]["lesion_ap"])
            print(f"[{variant} {s}] voxel AUROC {res[s]['voxel_auroc_in_gland']:.3f} | best-AP setting {best[0]}: "
                  f"patient AUROC {best[1]['patient_auroc']:.3f} AP {best[1]['lesion_ap']:.3f} "
                  f"PI-CAI {best[1]['picai_score']:.3f}")
        print(f"   score {t_score:.0f}s eval {t_eval:.0f}s -> {out_dir}")
    print(f"total {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
