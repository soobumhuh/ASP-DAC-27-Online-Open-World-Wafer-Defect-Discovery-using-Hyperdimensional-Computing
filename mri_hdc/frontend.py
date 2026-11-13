"""Front-end screen: swap the patch descriptor's CNN part for another frozen encoder and probe it at patch level.

Shared feature adapter (every front end goes through the same steps):
  1. encoder feature map of a slice (C, h, w), cached per slice as a float16 memmap aligned to the screen's slices
  2. 3 x 3 cell average (avg-pool, stride 1), then bilinear read-out at each patch centre (the 20 px patches on the
     stride-2 grid inside the gland, centres 10 .. 86 px on the 96 px crop); pixel-centre mapping
     u = (c + 0.5) / 96 * w - 0.5 (grid_sample, align_corners = False)
  3. PCA to 128 dimensions fitted on training patches only (class-balanced sample of 200k: lesion vs normal)
  4. + the existing 6 intensity statistics (per-channel patch mean and std) -> 134 numbers; the existing clip /
     standardise / thermometer encoding follows in the HD probe (maps.PatchModel)

Screen protocol (scripts/frontend_screen.py): split nest01 = train on folds 2-4 (minus a seeded 15% validation set,
used only for the per-bin z statistics), test on fold 1. Fold 0 is dropped at load and never read.
"""
from __future__ import annotations

import time

import numpy as np

from .features import patch_fraction, patch_grid, position_bins


# ------------------------------------------------------------------ data subset
def drop_fold0(d: dict) -> dict:
    """Keep folds 1-4 only; '_row' = row in the full slice file (to index caches built on all slices)."""
    keep = np.where(np.asarray(d["fold"]) != 0)[0]
    out = {k: (v[keep] if hasattr(v, "__len__") and len(v) == len(d["fold"]) else v) for k, v in d.items()}
    out["_row"] = keep
    return out


def centres(d: dict, idx, P: int = 20, stride: int = 2):
    """Patch centres inside the gland (same grid and rule as features.extract_patches)."""
    S = d["images"].shape[-1]
    cs = patch_grid(S, P, stride)
    cy, cx = np.meshgrid(cs, cs, indexing="ij")
    cy, cx = cy.ravel(), cx.ravel()
    sl, ys, xs = [], [], []
    for i in idx:
        ok = d["gland_mask"][i][cy, cx] > 0
        sl.append(np.full(ok.sum(), i)); ys.append(cy[ok]); xs.append(cx[ok])
    cat = lambda a: np.concatenate(a).astype(np.int64)
    return cat(sl), cat(ys), cat(xs)


def intensity_stats(d: dict, sl, ys, xs, P: int = 20) -> np.ndarray:
    """Per-channel mean and std of each P x P patch (the existing 6 statistics)."""
    h = P // 2
    out = np.zeros((len(sl), 6), np.float32)
    for i in np.unique(sl):
        m = np.where(sl == i)[0]
        im = d["images"][i].astype(np.float32)
        oy = ys[m, None] - h + np.arange(P)[None, :]
        ox = xs[m, None] - h + np.arange(P)[None, :]
        pat = im[:, oy[:, :, None], ox[:, None, :]].transpose(1, 0, 2, 3)
        out[m, :3], out[m, 3:] = pat.mean(axis=(2, 3)), pat.std(axis=(2, 3))
    return out


def sample_features(get_maps, sl, ys, xs, S: int = 96, device: str = "cpu", batch: int = 256, pool: int = 3):
    """Adapter steps 2: pooled + bilinear read-out of the feature maps at the patch centres.
    get_maps(slice_indices) -> (n, C, h, w) array. Returns (n_patches, C) float16."""
    import torch
    import torch.nn.functional as F
    order = np.argsort(sl, kind="stable")
    usl = np.unique(sl)
    out = None
    with torch.no_grad():
        for b0 in range(0, len(usl), batch):
            s = usl[b0:b0 + batch]
            f = torch.from_numpy(np.asarray(get_maps(s), dtype=np.float32)).to(device)
            if pool > 1:
                f = F.avg_pool2d(f, pool, 1, pool // 2, count_include_pad=False)
            if out is None:
                out = np.zeros((len(sl), f.shape[1]), np.float16)
            lo, hi = np.searchsorted(sl[order], [s[0], s[-1] + 1])
            pm = order[lo:hi]  # patches of these slices
            row = {int(v): k for k, v in enumerate(s)}
            r = np.array([row[int(v)] for v in sl[pm]])
            # pixel-centre normalised coordinates (align_corners=False): -1 / +1 = outer edges of the crop
            gx = torch.from_numpy(((xs[pm] + 0.5) / S * 2 - 1).astype(np.float32)).to(device)
            gy = torch.from_numpy(((ys[pm] + 0.5) / S * 2 - 1).astype(np.float32)).to(device)
            rr = torch.from_numpy(r).to(device)
            for k in range(len(s)):  # one grid_sample per slice: (1, C, h, w) x (1, 1, n, 2)
                q = (rr == k).nonzero()[:, 0]
                if len(q) == 0:
                    continue
                g = torch.stack([gx[q], gy[q]], 1)[None, None]
                v = F.grid_sample(f[k:k + 1], g, mode="bilinear", align_corners=False)[0, :, 0].T
                out[pm[q.cpu().numpy()]] = v.cpu().numpy().astype(np.float16)
    return out


# ------------------------------------------------------------------ labels and lesion sizes
def patch_labels(d: dict, sl, ys, xs, P: int = 20):
    """frac = lesion fraction of the patch; centre = patch centre inside an expert lesion."""
    frac = patch_fraction(d["lesion_mask"], sl, ys, xs, P)
    centre = d["lesion_mask"][sl, ys, xs] > 0
    return frac, centre


def lesion_components(d: dict):
    """3D connected lesion components per exam on the stacked crop masks (slices adjacent in z). Returns
    comp (N, S, S) int32 (0 = background, global ids) and size (n_comp + 1,) = estimated native T2W voxels
    (sum over slices of crop pixels x native pixels per crop pixel, from gland_pixels / crop gland pixels)."""
    from scipy.ndimage import label
    N, S = len(d["study_id"]), d["images"].shape[-1]
    comp = np.zeros((N, S, S), np.int32)
    scale = np.asarray(d["gland_pixels"], np.float64) / np.maximum((d["gland_mask"] > 0).sum((1, 2)), 1)
    sizes = [0.0]
    sid, z = np.asarray(d["study_id"]), np.asarray(d["slice_z"])
    for k in np.unique(sid[np.asarray(d["lesion_pixels"]) > 0]):
        j = np.where(sid == k)[0]
        j = j[np.argsort(z[j])]
        zz = z[j]
        vol = np.zeros((zz.max() - zz.min() + 1, S, S), bool)
        vol[zz - zz.min()] = d["lesion_mask"][j] > 0
        lab, n = label(vol)
        if n == 0:
            continue
        lab = lab[zz - zz.min()]
        base = len(sizes)
        for c in range(1, n + 1):
            sizes.append(float(((lab == c).sum((1, 2)) * scale[j]).sum()))
        comp[j] = np.where(lab > 0, lab + base - 1, 0)
    return comp, np.asarray(sizes)


def patch_lesion_size(comp, sizes, sl, ys, xs, P: int = 20) -> np.ndarray:
    """Size (native voxels) of the lesion a patch belongs to: the component at its centre, else the component
    with the most pixels in the patch; 0 when the patch has no lesion pixel."""
    h = P // 2
    out = np.zeros(len(sl))
    c0 = comp[sl, ys, xs]
    out[c0 > 0] = sizes[c0[c0 > 0]]
    for n in np.where(c0 == 0)[0]:
        w = comp[sl[n], ys[n] - h:ys[n] + P - h, xs[n] - h:xs[n] + P - h]
        w = w[w > 0]
        if len(w):
            out[n] = sizes[np.bincount(w).argmax()]
    return out


# ------------------------------------------------------------------ PCA
def fit_pca(X: np.ndarray, is_lesion: np.ndarray, is_normal: np.ndarray, n: int = 200_000, k: int = 128,
            seed: int = 0):
    """PCA (k components) on a class-balanced sample of training patches: up to n / 2 lesion patches, the rest
    normal. Returns (mean, components (k, C))."""
    rng = np.random.default_rng(seed)
    les, nor = np.where(is_lesion)[0], np.where(is_normal)[0]
    nl = min(len(les), n // 2)
    pick = np.concatenate([rng.choice(les, nl, replace=False), rng.choice(nor, min(len(nor), n - nl), replace=False)])
    A = X[np.sort(pick)].astype(np.float32)
    mu = A.mean(0)
    k = min(k, A.shape[1])
    _, _, Vt = np.linalg.svd(A - mu, full_matrices=False)
    return mu, Vt[:k].astype(np.float32)


def apply_pca(X, mu, V, block: int = 500_000) -> np.ndarray:
    return np.concatenate([(X[i:i + block].astype(np.float32) - mu) @ V.T for i in range(0, len(X), block)])


# ------------------------------------------------------------------ metrics
def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, bool)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


SIZE_BINS = (("lt150", 0, 150), ("150to500", 150, 500), ("gt500", 500, np.inf), ("lt500", 0, 500))


def breakdown(score, pos, neg, size, zone, case_pos) -> dict:
    """Patch AUROC overall and by lesion size (lesion patches of that size vs all normal patches), zone (both
    classes restricted to the zone at the patch centre) and within csPCa-positive exams."""
    r = {"auroc": auroc(pos[pos | neg], score[pos | neg]), "n_pos": int(pos.sum()), "n_neg": int(neg.sum())}
    for name, lo, hi in SIZE_BINS:
        p = pos & (size >= lo) & (size < hi)
        r[f"auroc_{name}"] = auroc(p[p | neg], score[p | neg])
        r[f"n_pos_{name}"] = int(p.sum())
    for name, zv in (("pz", 1), ("tz", 2)):
        m = (pos | neg) & (zone == zv)
        r[f"auroc_{name}"] = auroc(pos[m], score[m])
    m = (pos | neg) & case_pos
    r["auroc_posexam"] = auroc(pos[m], score[m])
    return r


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, key):
        return _T(self, key)


class _T:
    def __init__(self, tm, key):
        self.tm, self.key = tm, key

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, *a):
        self.tm.t[self.key] = self.tm.t.get(self.key, 0.0) + time.time() - self.t0


__all__ = ["drop_fold0", "centres", "intensity_stats", "sample_features", "patch_labels", "lesion_components",
           "patch_lesion_size", "fit_pca", "apply_pca", "breakdown", "position_bins"]
