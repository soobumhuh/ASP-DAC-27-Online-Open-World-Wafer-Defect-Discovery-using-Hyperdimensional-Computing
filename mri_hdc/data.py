"""PI-CAI data: 3D cases -> prostate-cropped 2D slices, fold merging, CV splits, scanner metadata.

Per case (build_slices):
  1. read T2W; resample ADC, HBV and the masks onto the T2W grid (linear for images, nearest for masks)
  2. normalise each channel inside the AI prostate mask (Bosma22b): clip to its 1st-99th percentile, z-score
  3. crop a square box around the prostate (+25% margin + 4 px) and resize to S x S (96 in the paper)
  4. keep slices whose prostate covers >= 2% of the crop
Saved per slice: images (3, S, S) float16; patient_id, study_id, slice_z; case_cspca (label, never an input);
case_isup (never an input); lesion_pixels, gland_pixels; lesion_mask (expert csPCa lesions; AI Bosma22a only for
positive cases without an expert mask, which does not occur in the public 1,500); gland_mask; zone_mask (AI zonal
anatomy HeviAI23: 1 = PZ, 2 = TZ); rel_z (slice position among the kept slices, 0 .. 1); lesion_source.

The public PI-CAI folds 0-4 are built one by one (the Zenodo file list of each fold) and merged with a 'fold' key.
"""
from __future__ import annotations

import csv
import glob
import os

import numpy as np

GLAND_DIR = "anatomical_delineations/whole_gland/AI/Bosma22b"
ZONE_DIR = "anatomical_delineations/zonal_pz_tz/AI/HeviAI23"
AI_LESION_DIR = "csPCa_lesion_delineations/AI/Bosma22a"


# ------------------------------------------------------------------ slice building
def index_mha_files(images_dir: str) -> dict:
    """'<pid>_<sid>' -> {'t2w': path, 'adc': path, 'hbv': path}."""
    out: dict = {}
    for root, _, files in os.walk(images_dir):
        for f in files:
            if not f.endswith(".mha"):
                continue
            parts = f[:-4].split("_")
            if len(parts) != 3:
                continue
            out.setdefault(f"{parts[0]}_{parts[1]}", {})[parts[2]] = os.path.join(root, f)
    return out


def on_grid(img, ref, nearest: bool) -> np.ndarray:
    """Resample img onto ref's voxel grid; returns a (z, y, x) array."""
    import SimpleITK as sitk
    interp = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
    return sitk.GetArrayFromImage(sitk.Resample(img, ref, sitk.Transform(), interp, 0.0, img.GetPixelID()))


def normalise(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = vol[mask] if mask.sum() > 100 else vol.ravel()
    lo, hi = np.percentile(vals, [1, 99])
    vol, vals = np.clip(vol, lo, hi), np.clip(vals, lo, hi)
    return ((vol - vals.mean()) / (vals.std() + 1e-6)).astype(np.float32)


def square_box(mask: np.ndarray, margin: float, shape) -> tuple:
    ys, xs = np.where(mask.any(axis=0))
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 * (1 + margin) + 4
    return (int(max(cy - half, 0)), int(min(cy + half, shape[1])),
            int(max(cx - half, 0)), int(min(cx + half, shape[2])))


def lesion_mask_path(labels_dir: str, key: str):
    for sub in ("csPCa_lesion_delineations/human_expert/Pooch25", "csPCa_lesion_delineations/human_expert/resampled"):
        p = os.path.join(labels_dir, sub, key + ".nii.gz")
        if os.path.exists(p):
            return p
    return None


def read_patient_list(path: str) -> set:
    lines = [l.strip() for l in open(path) if l.strip()]
    return {os.path.basename(l).split("_")[0].split("/")[0] for l in lines} | {l.split("/")[0] for l in lines}


def build_slices(images_dir: str, labels_dir: str, out_path: str, size: int = 96, margin: float = 0.25,
                 min_gland_frac: float = 0.02, patient_list: str = None, zones: bool = True, ai_lesions: bool = True,
                 spacing_mm: float = None):
    """One .npz of slices (see module docstring) for the cases in images_dir (optionally only patient_list).
    spacing_mm: fixed in-plane pixel size instead of fitting the gland box into size x size: a box of
    size x spacing_mm millimetres centred on the gland box (padded where it leaves the image); the kept slices are
    still chosen with the gland box."""
    import SimpleITK as sitk
    from skimage.transform import resize
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists; refusing to overwrite")
    files = index_mha_files(images_dir)
    if patient_list:
        keep = read_patient_list(patient_list)
        files = {k: v for k, v in files.items() if k.split("_")[0] in keep}
    with open(os.path.join(labels_dir, "clinical_information", "marksheet.csv")) as f:
        rows = [r for r in csv.DictReader(f) if f"{r['patient_id']}_{r['study_id']}" in files]
    print(f"{len(files)} cases on disk, {len(rows)} with marksheet rows", flush=True)
    keys = ["images", "patient_id", "study_id", "slice_z", "case_cspca", "lesion_source", "case_isup",
            "lesion_pixels", "gland_pixels", "lesion_mask", "gland_mask", "rel_z"] + (["zone_mask"] if zones else [])
    out = {k: [] for k in keys}
    n_seen = {0: 0, 1: 0}
    S = size
    for r in rows:
        key = f"{r['patient_id']}_{r['study_id']}"
        pos = int(r["case_csPCa"].strip().upper() == "YES")
        seqs = files[key]
        if not all(s in seqs for s in ("t2w", "adc", "hbv")):
            print("skip (missing sequence):", key)
            continue
        gland_p = os.path.join(labels_dir, GLAND_DIR, key + ".nii.gz")
        lesion_p = lesion_mask_path(labels_dir, key)
        source = 1 if lesion_p else 0
        if pos and lesion_p is None and ai_lesions:
            ai_p = os.path.join(labels_dir, AI_LESION_DIR, key + ".nii.gz")
            if os.path.exists(ai_p):
                lesion_p, source = ai_p, 2
        if not os.path.exists(gland_p) or (pos and lesion_p is None):
            print("skip (missing mask):", key)
            continue
        t2w = sitk.ReadImage(seqs["t2w"], sitk.sitkFloat32)
        vols = [sitk.GetArrayFromImage(t2w)] + [on_grid(sitk.ReadImage(seqs[s], sitk.sitkFloat32), t2w, False)
                                                for s in ("adc", "hbv")]
        gland = on_grid(sitk.ReadImage(gland_p, sitk.sitkUInt8), t2w, True) > 0
        lesion = (on_grid(sitk.ReadImage(lesion_p, sitk.sitkUInt8), t2w, True) > 0) if lesion_p else np.zeros_like(gland)
        zone = None
        if zones:
            zp = os.path.join(labels_dir, ZONE_DIR, key + ".nii.gz")
            zone = on_grid(sitk.ReadImage(zp, sitk.sitkUInt8), t2w, True) if os.path.exists(zp) else \
                np.zeros(gland.shape, np.uint8)
        if gland.sum() < 100:
            print("skip (empty prostate mask):", key)
            continue
        vols = [normalise(v, gland) for v in vols]
        y0, y1, x0, x1 = square_box(gland, margin, gland.shape)
        crop_area = (y1 - y0) * (x1 - x0)
        keep_z = [z for z in range(gland.shape[0]) if gland[z, y0:y1, x0:x1].sum() >= min_gland_frac * crop_area]
        if spacing_mm:  # fixed physical pixel size: re-centre a size x spacing_mm box and pad the volumes
            sx, sy = t2w.GetSpacing()[:2]
            cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
            hy, hx = S * spacing_mm / sy / 2, S * spacing_mm / sx / 2
            y0, y1, x0, x1 = int(round(cy - hy)), int(round(cy + hy)), int(round(cx - hx)), int(round(cx + hx))
            pad = ((0, 0), (max(0, -y0), max(0, y1 - gland.shape[1])), (max(0, -x0), max(0, x1 - gland.shape[2])))
            vols = [np.pad(v, pad, constant_values=float(v.min())) for v in vols]
            gland, lesion = np.pad(gland, pad), np.pad(lesion, pad)
            zone = np.pad(zone, pad) if zone is not None else None
            y0, y1, x0, x1 = y0 + pad[1][0], y1 + pad[1][0], x0 + pad[2][0], x1 + pad[2][0]
        n_kept = 0
        for z in range(gland.shape[0]):
            g = gland[z, y0:y1, x0:x1]
            if z not in keep_z:
                continue
            out["images"].append(np.stack([resize(v[z, y0:y1, x0:x1], (S, S), order=1, preserve_range=True,
                                                  anti_aliasing=True) for v in vols]).astype(np.float16))
            out["patient_id"].append(int(r["patient_id"]))
            out["study_id"].append(int(r["study_id"]))
            out["slice_z"].append(z)
            out["case_cspca"].append(pos)
            out["lesion_source"].append(source if pos else 0)
            out["case_isup"].append(int(r["case_ISUP"] or 0))
            out["lesion_pixels"].append(int(lesion[z, y0:y1, x0:x1].sum()))
            out["gland_pixels"].append(int(g.sum()))
            for name, m in [("lesion_mask", lesion), ("gland_mask", gland)] + ([("zone_mask", zone)] if zones else []):
                out[name].append(resize(m[z, y0:y1, x0:x1].astype(np.float32), (S, S), order=0, preserve_range=True,
                                        anti_aliasing=False).astype(np.uint8))
            n_kept += 1
        out["rel_z"].extend(np.linspace(0.0, 1.0, n_kept) if n_kept > 1 else [0.5] * n_kept)
        n_seen[pos] += 1
        print(f"{key}: csPCa={'YES' if pos else 'NO '} kept {n_kept} slices", flush=True)
    arrays = {k: np.asarray(v) for k, v in out.items()}
    np.savez_compressed(out_path, **arrays)
    print(f"saved {len(arrays['images'])} slices from {n_seen[0]} negative + {n_seen[1]} positive cases -> {out_path}")


def merge_folds(parts: list, out_path: str):
    """Concatenate the per-fold slice files (fold 0 first) and add the 'fold' key."""
    ds = [dict(np.load(p)) for p in parts]
    keys = set.intersection(*[set(d) for d in ds])
    out = {k: np.concatenate([d[k] for d in ds]) for k in keys}
    out["fold"] = np.concatenate([np.full(len(d["patient_id"]), f) for f, d in enumerate(ds)])
    np.savez(out_path, **out)
    print(f"merged {len(out['patient_id'])} slices, {len(np.unique(out['study_id']))} cases -> {out_path}")


# ------------------------------------------------------------------ loading and splits
def load_slices(path: str) -> dict:
    """All arrays of the slice file in memory (the patch code indexes them many times)."""
    return dict(np.load(path))


def cv_splits(npz_path: str) -> dict:
    """Patient splits used everywhere.
    fold<f>  test = public fold f; train / val = the other four folds, val = a seeded 15% of their patients,
             stratified by csPCa (rng seed 100 + f). fold0 is the held-out test, used once per variant.
    inner<v> tuning splits inside folds 1-4: train = folds 1-4 except v, val = fold v (v = 1, 2)."""
    d = np.load(npz_path, mmap_mode="r")
    fold = {int(p): int(f) for p, f in zip(d["patient_id"], d["fold"])}
    pos = {}
    for p, c in zip(d["patient_id"], d["case_cspca"]):
        pos[int(p)] = max(pos.get(int(p), 0), int(c))
    pats = sorted(fold)
    out = {}
    for vf in (1, 2):
        out[f"inner{vf}"] = {"train": [q for q in pats if fold[q] not in (0, vf)],
                             "val": [q for q in pats if fold[q] == vf], "test": [q for q in pats if fold[q] == vf][:5]}
    for f in range(5):
        pool = np.array([q for q in pats if fold[q] != f])
        rng = np.random.default_rng(100 + f)
        val = []
        for flag in (0, 1):
            g = pool[np.array([pos[q] == flag for q in pool])]
            val += list(rng.permutation(g)[: int(round(0.15 * len(g)))])
        sv = set(int(v) for v in val)
        out[f"fold{f}"] = {"train": [int(q) for q in pool if int(q) not in sv], "val": sorted(sv),
                           "test": [q for q in pats if fold[q] == f]}
    return out


def nested_splits(npz_path: str) -> dict:
    """Nested splits for a clean CV: for outer test fold f and each other fold g, nest<f><g> trains on the three
    folds that are neither f nor g (minus a seeded, csPCa-stratified 15% validation set, rng 200 + 10 f + g) and
    tests on fold g. Their test candidates train fold f's re-scorer and choose its merge weight, so nothing
    used for fold f was ever trained on fold f."""
    d = np.load(npz_path, mmap_mode="r")
    fold = {int(p): int(f) for p, f in zip(d["patient_id"], d["fold"])}
    pos = {}
    for p, c in zip(d["patient_id"], d["case_cspca"]):
        pos[int(p)] = max(pos.get(int(p), 0), int(c))
    pats = sorted(fold)
    out = {}
    for f in range(5):
        for g in range(5):
            if g == f:
                continue
            pool = np.array([q for q in pats if fold[q] not in (f, g)])
            rng = np.random.default_rng(200 + 10 * f + g)
            val = []
            for flag in (0, 1):
                grp = pool[np.array([pos[q] == flag for q in pool])]
                val += list(rng.permutation(grp)[: int(round(0.15 * len(grp)))])
            sv = set(int(v) for v in val)
            out[f"nest{f}{g}"] = {"train": [int(q) for q in pool if int(q) not in sv], "val": sorted(sv),
                                  "test": [q for q in pats if fold[q] == g]}
    return out


def scanner_info(images_dir: str) -> dict:
    """study_id -> (scanner manufacturer + model, HBV b-value) from the .mha headers (acquisition metadata only)."""
    import SimpleITK as sitk
    out = {}
    for f in glob.glob(os.path.join(images_dir, "**", "*_hbv.mha"), recursive=True):
        r = sitk.ImageFileReader()
        r.SetFileName(f)
        r.ReadImageInformation()
        g = lambda k: r.GetMetaData(k) if r.HasMetaDataKey(k) else "NA"
        out[int(os.path.basename(f).split("_")[1])] = (f"{g('0008|0070')} {g('0008|1090')}", g("0018|9087"))
    return out


def preprocess_case(seqs: dict, gland_path: str, zone_path: str = None, size: int = 96, margin: float = 0.25,
                    min_gland_frac: float = 0.02) -> dict:
    """Inference-time preprocessing of one new exam (same steps as build_slices, no lesion mask):
    {'t2w', 'adc', 'hbv'} .mha paths + AI gland (and zone) mask -> {images (Z, 3, S, S) float16, gland_mask, zone_mask,
    rel_z, slice_z}."""
    import SimpleITK as sitk
    from skimage.transform import resize
    t2w = sitk.ReadImage(seqs["t2w"], sitk.sitkFloat32)
    vols = [sitk.GetArrayFromImage(t2w)] + [on_grid(sitk.ReadImage(seqs[s], sitk.sitkFloat32), t2w, False)
                                            for s in ("adc", "hbv")]
    gland = on_grid(sitk.ReadImage(gland_path, sitk.sitkUInt8), t2w, True) > 0
    zone = on_grid(sitk.ReadImage(zone_path, sitk.sitkUInt8), t2w, True) if zone_path and os.path.exists(zone_path) \
        else np.zeros(gland.shape, np.uint8)
    vols = [normalise(v, gland) for v in vols]
    y0, y1, x0, x1 = square_box(gland, margin, gland.shape)
    crop_area = (y1 - y0) * (x1 - x0)
    out = {"images": [], "gland_mask": [], "zone_mask": [], "slice_z": []}
    rs = lambda m: resize(m.astype(np.float32), (size, size), order=0, preserve_range=True,
                          anti_aliasing=False).astype(np.uint8)
    for z in range(gland.shape[0]):
        g = gland[z, y0:y1, x0:x1]
        if g.sum() < min_gland_frac * crop_area:
            continue
        out["images"].append(np.stack([resize(v[z, y0:y1, x0:x1], (size, size), order=1, preserve_range=True,
                                              anti_aliasing=True) for v in vols]).astype(np.float16))
        out["gland_mask"].append(rs(gland[z, y0:y1, x0:x1]))
        out["zone_mask"].append(rs(zone[z, y0:y1, x0:x1]))
        out["slice_z"].append(z)
    n = len(out["slice_z"])
    out = {k: np.asarray(v) for k, v in out.items()}
    out["rel_z"] = np.linspace(0.0, 1.0, n) if n > 1 else np.full(n, 0.5)
    return out
