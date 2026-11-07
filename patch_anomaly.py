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
     Patch scores are z-scored per bin with the val split's normal (csPCa-negative) patches.
  4. Maps: overlapping patch scores averaged per pixel, zero outside the gland, slices stacked into
     one 3D volume per patient (crop space). Detection maps from report_guided_annotation's
     extract_lesion_candidates on sigmoid(z - offset); evaluated with picai_eval.evaluate against the
     stacked lesion masks (patient AUROC, lesion AP, PI-CAI score). Voxel AUROC inside the gland as a
     sanity check.

Example:
  uv run python patch_anomaly.py --slices_npz /data/picai_small_masks.npz --variants hdc256,hdc1024,float
  uv run python patch_anomaly.py --slices_npz /data/picai_small_masks.npz --descriptor stats --tag _stats
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


def extract_patches(d: dict, idx: np.ndarray, P: int, stride: int, add_stats: bool, add_pixels: bool = True):
    """Patches of slices idx whose centre lies in the gland.
    Returns descriptors (n, D) float32, slice index (n,), centre y, x (n,), position bin (n,)."""
    S = d["images"].shape[-1]
    cs = patch_grid(S, P, stride)
    h = P // 2
    zbin = np.minimum((d["rel_z"] * 3).astype(int), 2)
    feats, sl, ys, xs, bins = [], [], [], [], []
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
        parts = [pat.reshape(len(cy), -1)] if add_pixels else []
        if add_stats:
            parts += [pat.mean(axis=(2, 3)), pat.std(axis=(2, 3))]
        f = np.concatenate(parts, axis=1)
        feats.append(f.astype(np.float32))
        sl.append(np.full(len(cy), i)); ys.append(cy); xs.append(cx)
        bins.append(zbin[i] * 9 + np.minimum(cy * 3 // S, 2) * 3 + np.minimum(cx * 3 // S, 2))
    cat = lambda a, dt: np.concatenate(a).astype(dt) if a else np.zeros(0, dt)
    F = np.concatenate(feats) if feats else np.zeros((0, 1), np.float32)
    return F, cat(sl, np.int64), cat(ys, np.int64), cat(xs, np.int64), cat(bins, np.int64)


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
        for b in np.unique(bins):
            ix = self._new()
            ix.add(np.ascontiguousarray(X[bins == b]))
            self.index[int(b)] = ix

    def score(self, X: np.ndarray, bins: np.ndarray) -> np.ndarray:
        out = np.full(len(X), np.nan, dtype=np.float64)
        for b in np.unique(bins):
            m = bins == b
            ix = self.index.get(int(b))
            if ix is None:  # no normal reference at this position: fall back to all bins pooled
                ix = self._pooled()
            k = min(self.k, ix.ntotal)
            D, _ = ix.search(np.ascontiguousarray(X[m]), k)
            D = D.astype(np.float64)
            out[m] = (np.sqrt(np.maximum(D, 0)) if self.kind == "float" else D).mean(axis=1)
        return out

    def _pooled(self):
        if not hasattr(self, "_pool"):
            self._pool = self._new()
            for ix in self.index.values():
                if self.kind == "binary":
                    self._pool.add(np.vstack([ix.reconstruct(j) for j in range(ix.ntotal)]))
                else:
                    self._pool.add(ix.reconstruct_n(0, ix.ntotal))
        return self._pool


def hdc_codes(enc: BinaryThermometerHDC, X: np.ndarray) -> np.ndarray:
    """Eric's packed uint64 hypervectors viewed as bytes for faiss (bit order does not change Hamming)."""
    return np.ascontiguousarray(enc.transform(X, batch_size=4096)).view(np.uint8)


# ---------------------------------------------------------------- maps
def score_maps(d, slices, sl, ys, xs, z, P):
    """Average overlapping patch scores per pixel; NaN where no patch covers / outside gland."""
    S = d["images"].shape[-1]
    h = P // 2
    pos = {int(s): j for j, s in enumerate(slices)}
    acc = np.zeros((len(slices), S, S), np.float64)
    cnt = np.zeros((len(slices), S, S), np.float64)
    for s, y, x, v in zip(sl, ys, xs, z):
        j = pos[int(s)]
        acc[j, y - h:y - h + P, x - h:x - h + P] += v
        cnt[j, y - h:y - h + P, x - h:x - h + P] += 1
    with np.errstate(invalid="ignore"):
        m = acc / cnt
    gland = d["gland_mask"][slices] > 0
    m[~gland | (cnt == 0)] = np.nan
    return m


def patient_volumes(d, slices, maps):
    """Stack a patient's slices (sorted by z) into (Z, S, S) volumes."""
    out = {}
    pid = d["patient_id"][slices]
    for p in np.unique(pid):
        j = np.where(pid == p)[0]
        j = j[np.argsort(d["slice_z"][slices[j]])]
        out[int(p)] = dict(rows=j, slices=slices[j], map=maps[j],
                           lesion=d["lesion_mask"][slices[j]].astype(np.int32),
                           cspca=int(d["case_cspca"][slices[j][0]]))
    return out


# ---------------------------------------------------------------- figure
def save_figure(d, vols, path, offset, n_each=3, seed=0):
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
        heat = 1 / (1 + np.exp(-(np.nan_to_num(v["map"][j], nan=-10) - offset)))
        heat[np.isnan(v["map"][j])] = np.nan
        for c, (im, title, cmap) in enumerate([(d["images"][s][0], "T2W", "gray"), (d["images"][s][1], "ADC", "gray"),
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
            ax.set_title(f"{'csPCa+' if v['cspca'] else 'csPCa-'} {p} z={d['slice_z'][s]}  {title}" if c == 0 else title,
                         fontsize=7)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices_npz", required=True, help="picai_slices.py output made with --save_masks")
    ap.add_argument("--variants", default="hdc256,hdc1024,float", help="comma list of hdc<num_proj> | float")
    ap.add_argument("--save_root", default="results", help="results go to <save_root>/patch_<variant>/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.25)
    ap.add_argument("--min_lesion_pixels", type=int, default=10, help="only used to reproduce picai_bundle's split")
    ap.add_argument("--patch", type=int, default=12)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--descriptor", choices=["pixels+stats", "pixels", "stats"], default="pixels+stats",
                    help="patch descriptor: flattened 3-channel pixels and/or per-channel mean/std")
    ap.add_argument("--tag", default="", help="suffix for the results folder: <save_root>/patch_<variant><tag>/")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--bin_cap", type=int, default=50000, help="random cap on normal patches stored per bin")
    ap.add_argument("--num_bins", type=int, default=8, help="HDC thermometer levels")
    ap.add_argument("--clip_value", type=float, default=3.0)
    ap.add_argument("--sigmoid_offset", type=float, default=2.0,
                    help="detection input = sigmoid(z - offset): a normal patch (z~0) maps to ~0.12")
    ap.add_argument("--num_parallel_calls", type=int, default=16)
    args = ap.parse_args()

    from report_guided_annotation import extract_lesion_candidates
    from picai_eval import evaluate
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score

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
    idx = {
        "train_normal": np.where((split == "train") & (cspca == 0))[0],
        "val_normal": np.where((split == "val") & (cspca == 0))[0],
        "test": np.where(split == "test")[0],
    }
    counts = {s: {"patients": int(len(np.unique(pids[split == s]))),
                  "patients_pos": int(len(np.unique(pids[(split == s) & (cspca == 1)]))),
                  "slices": int((split == s).sum()),
                  "lesion_slices": int(((split == s) & (d["lesion_mask"].reshape(len(pids), -1).sum(1) > 0)).sum())}
              for s in ("train", "val", "test")}
    counts["unsplit_patients"] = sorted({int(p) for p in pids[split == "none"]})
    print("split:", json.dumps(counts))

    t0 = time.time()
    add_stats, add_pixels = "stats" in args.descriptor, "pixels" in args.descriptor
    P = {name: extract_patches(d, ix, args.patch, args.stride, add_stats, add_pixels) for name, ix in idx.items()}
    t_extract = time.time() - t0
    for name, (F, *_rest) in P.items():
        print(f"{name}: {len(F)} patches, dim {F.shape[1]}")
    counts["patches"] = {name: int(len(v[0])) for name, v in P.items()}

    # normal memory: clip + standardise on train normal patches, then cap each bin
    Ftr, _, _, _, btr = P["train_normal"]
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
    print("normal patches per bin:", bin_sizes)
    Xtr = prep(Ftr[keep]); btr = btr[keep]
    Xval, bval = prep(P["val_normal"][0]), P["val_normal"][4]
    Ft, slt, yt, xt, bt = P["test"]
    Xte = prep(Ft)
    test_slices = idx["test"]

    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        t0 = time.time()
        if variant.startswith("hdc"):
            nproj = int(variant[3:])
            enc = BinaryThermometerHDC(in_dim=Xtr.shape[1], num_proj=nproj, num_bins=args.num_bins,
                                       seed=args.seed, clip_value=args.clip_value).fit(Xtr)
            codes = lambda X: hdc_codes(enc, X)
            mem = BinMemory("binary", enc.bit_dim, args.k)
            enc_meta = enc.meta()
        elif variant == "float":
            codes = lambda X: np.ascontiguousarray(X, dtype=np.float32)
            mem = BinMemory("float", Xtr.shape[1], args.k)
            enc_meta = {"encoder": "none (float standardised descriptors)", "dim": int(Xtr.shape[1])}
        else:
            raise SystemExit(f"unknown variant {variant}")
        mem.add(codes(Xtr), btr)
        # per-bin z-score from val normal patches
        sv = mem.score(codes(Xval), bval)
        z_mu = {int(b): float(sv[bval == b].mean()) for b in np.unique(bval)}
        z_sd = {int(b): float(sv[bval == b].std() + 1e-6) for b in np.unique(bval)}
        g_mu, g_sd = float(sv.mean()), float(sv.std() + 1e-6)
        st = mem.score(codes(Xte), bt)
        mu_b = np.array([z_mu.get(b, g_mu) for b in range(27)]); sd_b = np.array([z_sd.get(b, g_sd) for b in range(27)])
        zt = (st - mu_b[bt]) / sd_b[bt]
        t_score = time.time() - t0

        maps = score_maps(d, test_slices, slt, yt, xt, zt, args.patch)
        vols = patient_volumes(d, test_slices, maps)

        # voxel AUROC inside the gland (pixels covered by at least one patch)
        allm = np.concatenate([v["map"].ravel() for v in vols.values()])
        alll = np.concatenate([v["lesion"].ravel() for v in vols.values()])
        ok = ~np.isnan(allm)
        voxel_auroc = float(roc_auc_score(alll[ok] > 0, allm[ok])) if alll[ok].any() else float("nan")

        # detection maps -> picai_eval
        t1 = time.time()
        keys = sorted(vols)
        y_det, y_true = [], []
        for p in keys:
            m = np.nan_to_num(vols[p]["map"], nan=-1e3)
            prob = expit(m - args.sigmoid_offset)
            det = extract_lesion_candidates(prob.astype(np.float64))[0]
            y_det.append(det.astype(np.float32))
            y_true.append((vols[p]["lesion"] > 0).astype(np.int32))
        metrics = evaluate(y_det=y_det, y_true=y_true, subject_list=[str(p) for p in keys],
                           num_parallel_calls=args.num_parallel_calls)
        # picai_eval's case labels come from the masks; check they agree with case_cspca
        case_label_mismatch = [p for p in keys if int(vols[p]["lesion"].any()) != vols[p]["cspca"]]
        t_eval = time.time() - t1

        res = {
            "variant": variant,
            "voxel_auroc_in_gland": voxel_auroc,
            "patient_auroc_picai_eval": float(metrics.auroc),
            "lesion_ap_picai_eval": float(metrics.AP),
            "picai_score": float(metrics.score),
            "num_lesions_test": int(metrics.num_lesions),
            "test_patients_pos": int(sum(vols[p]["cspca"] for p in keys)),
            "test_patients_neg": int(sum(1 - vols[p]["cspca"] for p in keys)),
            "case_label_mismatch_mask_vs_cspca": case_label_mismatch,
            "patient_max_detection": {str(p): float(y.max()) for p, y in zip(keys, y_det)},
            "runtime_s": {"patch_extraction_shared": round(t_extract, 1), "encode_index_score": round(t_score, 1),
                          "detection_and_eval": round(t_eval, 1)},
            "encoder": enc_meta, "normal_patches_per_bin": bin_sizes, "capped_bins": capped,
            "split_counts": counts, "args": vars(args),
        }
        out_dir = os.path.join(args.save_root, f"patch_{variant}{args.tag}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(res, f, indent=2)
        save_figure(d, vols, os.path.join(out_dir, "test_patients.png"), args.sigmoid_offset, seed=args.seed)
        print(f"[{variant}] voxel AUROC {voxel_auroc:.3f} | patient AUROC {metrics.auroc:.3f} | "
              f"lesion AP {metrics.AP:.3f} | PI-CAI {metrics.score:.3f} | score {t_score:.0f}s eval {t_eval:.0f}s "
              f"-> {out_dir}")
    print(f"total {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
