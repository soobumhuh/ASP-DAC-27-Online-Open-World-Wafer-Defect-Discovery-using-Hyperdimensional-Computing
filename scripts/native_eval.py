#!/usr/bin/env python3
"""Native-space evaluation: detections pasted back onto each exam's full T2W grid and scored with picai_eval against
the expert lesion masks on that grid (the official evaluation space), instead of the 96 x 96 crop grid.

Per exam: the crop box and the kept slices are recomputed exactly as when the slices were built (AI gland mask,
square box with 25% margin + 4 px, slices with >= 2% gland in the box); each candidate's voxels are resized back
into the box with nearest-neighbour interpolation, keeping its confidence. Labels: the expert csPCa lesion masks
(human_expert/Pooch25, else human_expert/resampled) resampled to the T2W grid, all slices, no crop; csPCa-negative
exams get empty labels. The patient score is the same as in the crop-grid run (e.g. the clinical merge), so only
the lesion AP and the detection-map case score can differ.

  uv run python scripts/native_eval.py --config configs/best.yaml --cases results/review/pick/cv_cases.pkl \
      --stage merged --out results/native/pick.json
"""
import argparse
import json
import os
import pickle
from multiprocessing import get_context

import numpy as np

import _path  # noqa: F401
from mri_hdc import data as D
from mri_hdc.pipeline import load_config

CFG = {}


def native_case(item):
    import SimpleITK as sitk
    from skimage.transform import resize
    key, shape, cands = item
    p = CFG
    files = p["files"][key]
    t2w = sitk.ReadImage(files["t2w"], sitk.sitkFloat32)
    gl = D.on_grid(sitk.ReadImage(os.path.join(p["labels"], D.GLAND_DIR, key + ".nii.gz"), sitk.sitkUInt8), t2w, True) > 0
    y0, y1, x0, x1 = D.square_box(gl, 0.25, gl.shape)
    area = (y1 - y0) * (x1 - x0)
    kept = [z for z in range(gl.shape[0]) if gl[z, y0:y1, x0:x1].sum() >= 0.02 * area]
    assert len(kept) == shape[0], (key, len(kept), shape)
    det = np.zeros(gl.shape, np.float32)
    for mask, conf in cands:
        m = np.zeros(int(np.prod(shape)), bool)
        m[mask] = True
        m = m.reshape(shape)
        for j, z in enumerate(kept):
            if m[j].any():
                up = resize(m[j].astype(np.float32), (y1 - y0, x1 - x0), order=0, preserve_range=True,
                            anti_aliasing=False) > 0.5
                det[z, y0:y1, x0:x1][up] = conf
    lp = D.lesion_mask_path(p["labels"], key)
    lab = (D.on_grid(sitk.ReadImage(lp, sitk.sitkUInt8), t2w, True) > 0).astype(np.int32) if lp and p["pos"][key] \
        else np.zeros(gl.shape, np.int32)
    from picai_eval.eval import evaluate_case
    lesion_results, *_ = evaluate_case(y_det=det, y_true=lab)
    return key, lesion_results, float(det.max()), int(lab.any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cases", required=True, help="cv_cases.pkl / cv_clean_cases.pkl with ('candidates', fold)")
    ap.add_argument("--stage", default="merged", help="stage whose patient score is kept (merged / image_only / rescore)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--save-cases", default="", help="pickle of the native per-exam results, in the cv_cases.pkl "
                    "layout {(stage, fold): {lesion_results, case_target, case_pred}} (for bootstrap_ci.py)")
    args = ap.parse_args()
    from picai_eval.metrics import Metrics
    cfg = load_config(args.config)
    K = pickle.load(open(args.cases, "rb"))
    d = np.load(cfg["paths"]["slices"], mmap_mode="r")
    sid2key = {int(s): f"{int(p)}_{int(s)}" for s, p in zip(d["study_id"], d["patient_id"])}
    pos = {sid2key[int(s)]: int(c) for s, c in zip(d["study_id"], d["case_cspca"])}
    CFG.update(files=D.index_mha_files(cfg["paths"]["images_dir"]), labels=cfg["paths"]["labels_dir"], pos=pos)
    out, KN = {"per_fold": []}, {}
    allr = {"lesion_results": {}, "case_target": {}, "case_pred": {}, "case_pred_det": {}}
    for f in range(5):
        cands = K[("candidates", f)]
        items = [(sid2key[k], shape, c) for k, (shape, c) in cands.items()]
        with get_context("fork").Pool(args.workers) as pool:
            res = pool.map(native_case, items, chunksize=4)
        case_pred = K[(args.stage, f)]["case_pred"]
        lr = {k: r for k, r, _, _ in res}
        tgt = {k: t for k, _, _, t in res}
        cp = {sid2key[s]: v for s, v in case_pred.items()}
        cpd = {k: m for k, _, m, _ in res}
        m = Metrics(lesion_results=lr, case_target=tgt, case_pred=cp)
        mdet = Metrics(lesion_results=lr, case_target=tgt, case_pred=cpd)
        crop = Metrics(lesion_results={sid2key[s]: v for s, v in K[(args.stage, f)]["lesion_results"].items()},
                       case_target={sid2key[s]: v for s, v in K[(args.stage, f)]["case_target"].items()}, case_pred=cp)
        row = {"fold": f, "native": {"patient_auroc": m.auroc, "lesion_ap": m.AP, "picai_score": (m.auroc + m.AP) / 2},
               "native_det_case_score": {"patient_auroc": mdet.auroc, "lesion_ap": mdet.AP,
                                         "picai_score": (mdet.auroc + mdet.AP) / 2},
               "crop_grid": {"patient_auroc": crop.auroc, "lesion_ap": crop.AP, "picai_score": (crop.auroc + crop.AP) / 2}}
        out["per_fold"].append(row)
        KN[(args.stage, f)] = {"lesion_results": lr, "case_target": tgt, "case_pred": cp}
        print(f, json.dumps({k: {a: round(b, 4) for a, b in v.items()} for k, v in row.items() if k != "fold"}), flush=True)
    for k in ("native", "native_det_case_score", "crop_grid"):
        out[k] = {m: [float(np.mean([r[k][m] for r in out["per_fold"]])), float(np.std([r[k][m] for r in out["per_fold"]]))]
                  for m in ("patient_auroc", "lesion_ap", "picai_score")}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.save_cases:
        pickle.dump(KN, open(args.save_cases, "wb"))
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("native", "native_det_case_score", "crop_grid")}, indent=1))


if __name__ == "__main__":
    main()
