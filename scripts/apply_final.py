#!/usr/bin/env python3
"""Apply the FROZEN final model (trained on all 1,500 PI-CAI exams, split 'final') to an external slice set and score
it on each exam's native T2W grid. Run once per external set; nothing here is tuned.

Frozen pieces (all from PI-CAI only):
  maps         the 'final' patch memories of every ensemble seed (and member config), *_model.pkl from train_predict
  calibration  the extraction calibration from the final model's PI-CAI validation maps; extraction = cfg.extraction
  re-scorer    cfg.rescorer kind, fitted on the out-of-fold test candidates of the five CV fold models (every
               PI-CAI exam once, each from a model that never saw it); k / epochs = --k, alpha = --alpha
  patient      image-only: max re-scored candidate confidence (external sets without PSA)
Metrics (picai_eval, native grid, lesion IoU >= 0.1): patient AUROC, lesion AP, PI-CAI score, FROC sensitivity at
0.5 / 1 FP per exam, with 95% bootstrap CIs over exams; optional subset (e.g. the 19 exams with a second reader).

  uv run python scripts/apply_final.py --config configs/sota/best_sota.yaml --slices /data/sota/external/prostate158/slices.npz \
      --seg /data/sota/external/prostate158/seg --labels /data/sota/external/prostate158/labels \
      --images /data/sota/external/prostate158 --out /data/sota/external/prostate158/eval_best_sota
"""
import argparse
import json
import os
import pickle
from multiprocessing import get_context

import numpy as np

import _path  # noqa: F401
from mri_hdc import data as D, extraction as X, features as Fe, maps as M
from mri_hdc.pipeline import ensemble_maps, load_config, map_path
from mri_hdc.rescorer import make as make_rescorer

G = {}


def member_maps(cfg, d, n_ext):
    """Seed-ensemble maps of the final model of one (non-member) config on the external slices."""
    m = cfg["model"]
    dd = M.prepare_slices(d, m)
    feats = m.get("features", "rgb")
    if feats == "external":
        fm = np.load(cfg["_ext_features"], mmap_mode="r")
    elif feats == "none":
        fm = None
    else:
        fm = Fe.resnet_l2_maps(dd["images"], cfg["_ext_resnet"])
    idx = np.arange(n_ext)
    outs = []
    for s in cfg["ensemble_seeds"]:
        model = pickle.load(open(map_path(cfg, s, "final").replace(".pkl", "_model.pkl"), "rb"))
        outs.append({"test": model.maps(dd, fm, idx), "val": {}})
        print("seed", s, "maps done", flush=True)
    return M.ensemble([{"val": {}, "test": o["test"]} for o in outs])["test"]


def native_case(item):
    import SimpleITK as sitk
    from picai_eval.eval import evaluate_case
    from skimage.transform import resize
    key, shape, cands = item
    t2 = sitk.ReadImage(G["t2"][key], sitk.sitkFloat32)
    gl = D.read_anatomy("", key, t2, seg_dir=G["seg"])[0]
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
    out = {}
    for tag, suffix in (("reader1", ""), ("reader2", "_reader2")):
        p = os.path.join(G["labels"], key + suffix + ".nii.gz")
        if os.path.exists(p):
            lab = (sitk.GetArrayFromImage(sitk.ReadImage(p)) > 0).astype(np.int32)
            lr, *_ = evaluate_case(y_det=det, y_true=lab)
            out[tag] = (lr, int(lab.any()))
    return key, out


def metrics(lr, ct, cp, draws=None):
    from picai_eval.metrics import Metrics
    keys = sorted(ct)
    idx = np.arange(len(keys)) if draws is None else draws
    m = Metrics(lesion_results={f"{j}": lr[keys[i]] for j, i in enumerate(idx)},
                case_target={f"{j}": ct[keys[i]] for j, i in enumerate(idx)},
                case_pred={f"{j}": cp[keys[i]] for j, i in enumerate(idx)})
    fpr, tpr = np.asarray(m.lesion_FPR), np.asarray(m.lesion_TPR)
    o = np.argsort(fpr, kind="stable")
    return [m.auroc, m.AP, (m.auroc + m.AP) / 2] + [float(np.interp(x, fpr[o], tpr[o])) for x in (0.5, 1.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--slices", required=True)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--t2", default="/data/sota/external/prostate158/nnunet_in", help="dir with <key>_0000.nii.gz")
    ap.add_argument("--ext_features", default="", help="external front-end cache for members with features: external")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--subset", default="", help="comma list of study ids for a subset row (e.g. Prostate158 test)")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = load_config(a.config)
    d = D.load_slices(a.slices)
    d["adc_raw"] = np.load(a.slices.replace(".npz", "_adc_raw.npy"), mmap_mode="r")
    n = len(d["study_id"])
    cfgs = [load_config(c) for c in cfg["members"]] if cfg.get("members") else [cfg]
    ms = []
    for c in cfgs:
        c["_ext_resnet"] = a.slices.replace(".npz", "_resnet_l2.npy")
        c["_ext_features"] = a.ext_features
        ms.append(member_maps(c, d, n))
    test = M.ensemble([{"val": {}, "test": x} for x in ms])["test"] if len(ms) > 1 else ms[0]
    ref = ensemble_maps(cfg, "final")["val"]  # PI-CAI validation maps of the final model (calibration)
    pp, rc, w = cfg["extraction"], cfg["rescorer"], cfg["run"]["workers"]
    ext = X.extract(test, ref, pp, d, w)
    # re-scorer on the out-of-fold PI-CAI candidates of the five CV fold models
    dp = D.load_slices(cfg["paths"]["slices"])
    if cfg["paths"].get("adc_raw"):
        dp["adc_raw"] = np.load(cfg["paths"]["adc_raw"], mmap_mode="r")
    tr = []
    for f in range(5):
        mp = ensemble_maps(cfg, f"fold{f}")
        tr += X.flat(X.extract(mp["test"], mp["val"], pp, dp, w))
    rs = make_rescorer(rc, a.k).fit(tr)
    conf = rs.confidences(ext, a.alpha)
    sid2key = {9000000 + int(k.split("_")[1]): k for k in (f"p158_{i:03d}" for i in range(1, 1000))}
    items = [(sid2key[s], shape, [(c["mask"], v) for c, v in zip(cands, conf[s])]) for s, (cands, shape) in ext.items()]
    G.update(t2={k: os.path.join(a.t2, k + "_0000.nii.gz") for _, (k) in sid2key.items()}, seg=a.seg, labels=a.labels)
    with get_context("fork").Pool(12) as pool:
        res = dict(pool.map(native_case, items, chunksize=2))
    cp = {sid2key[s]: max(conf[s], default=0.0) for s in ext}
    out = {"config": a.config, "k": a.k, "alpha": a.alpha, "n_exams": len(res), "rows": {}}
    rows = {"all_reader1": [k for k in res if "reader1" in res[k]]}
    if a.subset:
        sub = {sid2key[int(s)] for s in a.subset.split(",")}
        rows["subset_reader1"] = [k for k in rows["all_reader1"] if k in sub]
        rows["subset_reader2"] = [k for k in res if k in sub and "reader2" in res[k]]
    rng = np.random.default_rng(0)
    names = ["patient_auroc", "lesion_ap", "picai_score", "froc_sens_at_0.5fp", "froc_sens_at_1fp"]
    for name, ks in rows.items():
        rd = "reader2" if name.endswith("reader2") else "reader1"
        lr = {k: res[k][rd][0] for k in ks}
        ct = {k: res[k][rd][1] for k in ks}
        cpp = {k: cp[k] for k in ks}
        pt = metrics(lr, ct, cpp)
        b = np.array([metrics(lr, ct, cpp, rng.integers(0, len(ks), len(ks))) for _ in range(a.n)])
        out["rows"][name] = {"n": len(ks), "n_pos": int(sum(ct.values())),
                             **{m: {"value": float(pt[i]), "ci95": [float(x) for x in np.nanpercentile(b[:, i], [2.5, 97.5])]}
                                for i, m in enumerate(names)}}
        print(name, json.dumps(out["rows"][name]), flush=True)
    pickle.dump({"native": res, "case_pred": cp, "candidates": items}, open(os.path.join(a.out, "cases.pkl"), "wb"))
    json.dump(out, open(os.path.join(a.out, "result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
