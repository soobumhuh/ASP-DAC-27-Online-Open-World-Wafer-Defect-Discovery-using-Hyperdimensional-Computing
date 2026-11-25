#!/usr/bin/env python3
"""External run on the UCSD (Liss lab) RSI prostate export, FROZEN final model, run once.

Data: /data/liss/imagined_export/<PDSxxxx>/<date>/Series_*__RSI_*/  (MATLAB .mat volumes from the RSI pipeline +
DICOM_Output). Every volume of one series shares one LPH frame (struct field Mvxl2lph, 1-based voxel -> LPH mm), so
T2 / ADC / DWI / RSIrs / contours are already co-registered. One series per exam is used (see pick_series).

Inputs given to the model (exactly what a new exam would give it, nothing tuned):
  T2W   T2_corrected_GUW (0.25 mm in-plane, resampled here to 0.5 mm like PI-CAI)
  ADC   conventional_ADC_map (mm^2/s, x 1e6 -> PI-CAI units um^2/s)
  HBV   b = 1400 shell of DWI_vol_preprocessed_averaged (acquired); series without b = 1400 (older ep2d protocols)
        get S1400 = S_b * exp(-(1400 - b) * ADC) from the nearest acquired shell b
  anatomy  our own nnU-Net gland/zone segmenter (Dataset501, fold all), not the export's contours
Comparators kept per exam: max RSIrs inside the export's prostate contour (and inside our gland), prostate volume
from the export contour and from our segmenter.
The export has NO reference standard (no biopsy grade, no expert lesion outlines: its 'Lesion*_Autoseg_maxRSIrs'
RT structures are RSIrs threshold detections), so build writes empty lesion masks and apply_final.py --keymap only
writes per-exam scores until labels arrive.

  .venv/bin/python scripts/external_liss.py inventory --out /data/liss_work/inventory.csv
  .venv/bin/python scripts/external_liss.py nnunet_in --out /data/liss_work/nnunet_in
  (segmenter) nnUNetv2 predict, Dataset501 2d fold all -> /data/liss_work/seg
  .venv/bin/python scripts/external_liss.py build --seg /data/liss_work/seg --out /data/liss_work/slices.npz
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np

import _path  # noqa: F401
from mri_hdc import data as D

ROOT = "/data/liss/imagined_export"
WORK = "/data/liss_work"
SID_BASE = 8000000
PREF = [r"RSI_TE_[Ss]hort$", r"RSI_TE_[Ss]hort_interp$", r"RSI_ep2d_advdiff_bval0-600-1200-1800$",
        r"RSI_TE_[Ss]hort_abv$", r"RSI_TE_[Ll]ong$"]
NEED = ["T2_corrected_GUW", "DWI_vol_preprocessed_averaged", "conventional_ADC_map", "RSIrs", "bvals",
        "prostate_contour_DWI_space"]


def loadmat(p):
    import scipy.io as sio
    return sio.loadmat(p, squeeze_me=True, struct_as_record=False)


def vol(series, name):
    return loadmat(os.path.join(series, name + ".mat"))["vol"]


def to_sitk(v, arr=None):
    """mat 'vol' struct -> SimpleITK image. imgs is indexed (i, j, k) with LPH = M @ [i+1, j+1, k+1, 1]."""
    import SimpleITK as sitk
    a = np.asarray(v.imgs if arr is None else arr, np.float32)
    M = np.asarray(v.Mvxl2lph, float)
    sp = np.linalg.norm(M[:3, :3], axis=0)
    im = sitk.GetImageFromArray(np.ascontiguousarray(a.transpose(2, 1, 0)))  # sitk array is (k, j, i)
    im.SetSpacing([float(s) for s in sp])
    im.SetDirection([float(x) for x in (M[:3, :3] / sp).ravel()])
    im.SetOrigin([float(x) for x in (M @ np.array([1, 1, 1, 1.0]))[:3]])
    return im


def bvals(series):
    m = loadmat(os.path.join(series, "bvals.mat"))
    return np.ravel([v for k, v in m.items() if not k.startswith("__")][0]).astype(int)


def pick_series(pdir):
    ss = sorted(glob.glob(os.path.join(pdir, "*", "Series_*")))
    ok = [s for s in ss if all(os.path.exists(os.path.join(s, n + ".mat")) for n in NEED)]
    for pat in PREF:
        for s in ok:
            if re.search(pat, os.path.basename(s).split("__", 1)[1]):
                return s, ss
    return (ok[0] if ok else None), ss


def cases(root=ROOT):
    out = []
    for p in sorted(glob.glob(os.path.join(root, "PDS*"))):
        pid = os.path.basename(p)
        s, _ = pick_series(p)
        if s:
            out.append((f"liss_{pid[3:]}", SID_BASE + int(pid[3:]), s))
    return out


def hbv_and_adc(s):
    """(b=1400 image, ADC um^2/s) on the DWI grid; extrapolated from the nearest shell if 1400 was not acquired."""
    dw = vol(s, "DWI_vol_preprocessed_averaged")
    adc = np.asarray(vol(s, "conventional_ADC_map").imgs, np.float64)
    shells = sorted(set(bvals(s).tolist()))
    imgs = np.asarray(dw.imgs, np.float64)
    assert imgs.shape[-1] == len(shells), (s, imgs.shape, shells)
    if 1400 in shells:
        hb, how = imgs[..., shells.index(1400)], "acquired b1400"
    else:
        b = min((x for x in shells if x > 0), key=lambda x: abs(x - 1400))
        hb = imgs[..., shells.index(b)] * np.exp(-(1400 - b) * np.clip(adc, 0, 0.005))
        how = f"extrapolated from b{b}"
    return dw, hb, adc * 1e6, how


def t2_image(s):
    import SimpleITK as sitk
    t2 = to_sitk(vol(s, "T2_corrected_GUW"))
    sp = list(t2.GetSpacing())
    if sp[0] < 0.45:  # 0.25 mm -> 0.5 mm in-plane (PI-CAI T2W resolution), anti-aliased
        f = 0.5 / sp[0]
        sm = sitk.SmoothingRecursiveGaussian(t2, [0.5 * sp[0] * f, 0.5 * sp[1] * f, 0.01])
        size = [int(round(t2.GetSize()[0] / f)), int(round(t2.GetSize()[1] / f)), t2.GetSize()[2]]
        org = np.array(t2.GetOrigin()) + np.array(t2.GetDirection()).reshape(3, 3) @ np.array(
            [(f - 1) * sp[0] / 2, (f - 1) * sp[1] / 2, 0])
        ref = sitk.Image(size, sitk.sitkFloat32)
        ref.SetSpacing([sp[0] * f, sp[1] * f, sp[2]])
        ref.SetDirection(t2.GetDirection())
        ref.SetOrigin([float(x) for x in org])
        t2 = sitk.Resample(sm, ref, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    return t2


def inventory(a):
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "PDS*"))):
        s, ss = pick_series(p)
        r = {"exam": os.path.basename(p), "n_series": len(ss), "series_used": os.path.basename(s) if s else "",
             "all_series": ";".join(os.path.basename(x) for x in ss)}
        if s:
            r["bvals"] = "/".join(map(str, sorted(set(bvals(s).tolist()))))
            dw, hb, adc, how = hbv_and_adc(s)
            r["hbv"] = how
            r["dwi_grid"] = "x".join(map(str, np.asarray(dw.imgs).shape[:3]))
            t2 = vol(s, "T2_corrected_GUW")
            r["t2_grid"] = "x".join(map(str, np.asarray(t2.imgs).shape))
            r["t2_mm"] = f"{t2.vx:.2f}x{t2.vy:.2f}x{t2.vz:.2f}"
            pc = np.asarray(vol(s, "prostate_contour_DWI_space").imgs) > 0.5
            r["prostate_ml_export"] = round(pc.sum() * dw.vx * dw.vy * dw.vz / 1000, 1)
            rs = np.asarray(vol(s, "RSIrs").imgs)
            r["max_rsirs_prostate"] = round(float(rs[pc].max()), 1) if pc.any() else ""
            les = sorted(glob.glob(os.path.join(s, "DICOM_Output", "*", "*.dcm")))
            r["rtstruct_lesion_autoseg"] = int(any("Lesion" in x for x in les) or
                                               os.path.exists(os.path.join(s, "lesion_contours_DWI_space.mat")))
        rows.append(r)
        print(r, flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def nnunet_in(a):
    import SimpleITK as sitk
    os.makedirs(a.out, exist_ok=True)
    meta = {}
    for key, sid, s in cases():
        t2 = t2_image(s)
        dw, hb, adc, how = hbv_and_adc(s)
        ims = [t2] + [sitk.Resample(to_sitk(dw, x), t2, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
                      for x in (adc, hb)]
        for i, im in enumerate(ims):
            sitk.WriteImage(im, os.path.join(a.out, f"{key}_{i:04d}.nii.gz"))
        meta[key] = {"study_id": sid, "series": s, "hbv": how}
        print(key, os.path.basename(s), how, t2.GetSize(), flush=True)
    json.dump(meta, open(os.path.join(WORK, "keymap.json"), "w"), indent=1)


def build(a):
    import SimpleITK as sitk
    from skimage.transform import resize
    S = 96
    keys = ["images", "patient_id", "study_id", "slice_z", "case_cspca", "lesion_source", "case_isup", "lesion_pixels",
            "gland_pixels", "lesion_mask", "gland_mask", "rel_z", "zone_mask"]
    out = {k: [] for k in keys}
    adc_raw = []
    os.makedirs(a.labels, exist_ok=True)
    for key, sid, s in cases():
        t2 = sitk.ReadImage(os.path.join(a.nnunet_in, key + "_0000.nii.gz"), sitk.sitkFloat32)
        vols = [sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(a.nnunet_in, f"{key}_{i:04d}.nii.gz"),
                                                      sitk.sitkFloat32)) for i in range(3)]
        gland, zone = D.read_anatomy("", key, t2, seg_dir=a.seg)
        les = np.zeros(gland.shape, bool)  # no reference standard in this export
        lab = sitk.GetImageFromArray(les.astype(np.uint8))
        lab.CopyInformation(t2)
        sitk.WriteImage(lab, os.path.join(a.labels, key + ".nii.gz"))
        if gland.sum() < 100:
            print("skip (segmenter found no gland):", key, flush=True)
            continue
        raw_adc = vols[1].copy()
        vols = [D.normalise(v, gland) for v in vols]
        y0, y1, x0, x1 = D.square_box(gland, 0.25, gland.shape)
        area = (y1 - y0) * (x1 - x0)
        kept = [z for z in range(gland.shape[0]) if gland[z, y0:y1, x0:x1].sum() >= 0.02 * area]
        rs = lambda m: resize(m[y0:y1, x0:x1].astype(np.float32), (S, S), order=0, preserve_range=True,
                              anti_aliasing=False).astype(np.uint8)
        for z in kept:
            out["images"].append(np.stack([resize(v[z, y0:y1, x0:x1], (S, S), order=1, preserve_range=True,
                                                  anti_aliasing=True) for v in vols]).astype(np.float16))
            adc_raw.append(resize(raw_adc[z, y0:y1, x0:x1], (S, S), order=1, preserve_range=True,
                                  anti_aliasing=True).astype(np.float32))
            out["patient_id"].append(sid - SID_BASE + 800000)
            out["study_id"].append(sid)
            out["slice_z"].append(z)
            out["case_cspca"].append(0)
            out["lesion_source"].append(0)
            out["case_isup"].append(0)
            out["lesion_pixels"].append(0)
            out["gland_pixels"].append(int(gland[z, y0:y1, x0:x1].sum()))
            out["lesion_mask"].append(rs(les[z]))
            out["gland_mask"].append(rs(gland[z]))
            out["zone_mask"].append(rs(zone[z]))
        out["rel_z"].extend(np.linspace(0.0, 1.0, len(kept)) if len(kept) > 1 else [0.5] * len(kept))
        print(key, len(kept), "slices", flush=True)
    arr = {k: np.asarray(v) for k, v in out.items()}
    arr["fold"] = np.full(len(arr["patient_id"]), -1)
    np.savez(a.out, **arr)
    np.save(a.out.replace(".npz", "_adc_raw.npy"), np.asarray(adc_raw, np.float32))
    print("saved", a.out, len(arr["patient_id"]), "slices,", len(np.unique(arr["study_id"])), "exams")


def compare(a):
    """Per exam: our segmenter vs the export's prostate contour (Dice, volumes) and max RSIrs inside each."""
    import SimpleITK as sitk
    rows = []
    for key, sid, s in cases():
        t2 = sitk.ReadImage(os.path.join(a.nnunet_in, key + "_0000.nii.gz"), sitk.sitkFloat32)
        gl, zone = D.read_anatomy("", key, t2, seg_dir=a.seg)
        pc_v = vol(s, "prostate_contour_DWI_space")
        pc = D.on_grid(to_sitk(pc_v), t2, True) > 0.5
        rsv = vol(s, "RSIrs")
        rs = D.on_grid(to_sitk(rsv), t2, False)
        rs_dwi = np.asarray(rsv.imgs)
        pcd = np.asarray(pc_v.imgs) > 0.5
        vx = np.prod(t2.GetSpacing()) / 1000
        dice = 2 * (gl & pc).sum() / max(gl.sum() + pc.sum(), 1)
        rows.append({"key": key, "study_id": sid, "dice_seg_vs_export": round(float(dice), 3),
                     "vol_ml_seg": round(float(gl.sum() * vx), 1), "vol_ml_export": round(float(pc.sum() * vx), 1),
                     "pz_frac_seg": round(float((zone == 1).sum() / max(gl.sum(), 1)), 2),
                     "max_rsirs_export_contour": float(rs_dwi[pcd].max()) if pcd.any() else np.nan,
                     "max_rsirs_seg_gland": float(rs[gl].max()) if gl.any() else np.nan})
        print(rows[-1], flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["inventory", "nnunet_in", "build", "compare"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--nnunet_in", default=os.path.join(WORK, "nnunet_in"))
    ap.add_argument("--seg", default=os.path.join(WORK, "seg"))
    ap.add_argument("--labels", default=os.path.join(WORK, "labels"))
    a = ap.parse_args()
    {"inventory": inventory, "nnunet_in": nnunet_in, "build": build, "compare": compare}[a.stage](a)
