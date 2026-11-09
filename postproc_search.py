#!/usr/bin/env python3
"""
postproc_search.py - raise lesion AP of the tuned design-A HDC maps by tuning only the map -> detection step.

The HDC model is fixed (tuned config #119). Its z-scored maps are computed once and saved; the search then varies
only cheap post-processing, scored with picai_eval:
  smoothing      extra in-plane Gaussian (px) and through-slice Gaussian (slices), inside the gland
  calibration    sigmoid((z - offset) / temp); offset = a percentile of val-normal voxels, temp = f x their std
  extraction     report_guided_annotation.extract_lesion_candidates: threshold mode, dynamic threshold factor,
                 minimum candidate size, number of candidates per case, removal of adjacent candidates
  case gating    candidate confidence x (case peak probability)^gamma: weak extra blobs in a case are pushed down
Protocol as in hp_search.py: fold 0 untouched; objective = mean val PI-CAI over inner rounds (train on three of
folds 1-4, validate on folds 1 and 2). The best setting is then applied once to fold 0 (model trained on folds 1-4)
and to the 5-fold CV test folds (slightly optimistic: folds 1-4 were used for tuning).

  uv run python postproc_search.py maps                 # compute + save maps (7 model runs)
  uv run python postproc_search.py search --trials 200  # Optuna over post-processing
  uv run python postproc_search.py final                # fold-0 test and 5-fold CV with the best setting
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = "results/lesion_ap"          # saved maps live here
OUT = os.environ.get("LESION_AP_OUT", ROOT)  # searches / finals (results/lesion_ap2 for the second round)
NPZ = "/data/picai_all_masks.npz"
TUNED = ("--variants proto512_negplus --descriptor stats+resnet_l2 --patch 20 --stride 4 --slice_levels 5 "
         "--bin_mode zone_grid --num_bins 4 --clip_value 4.0 --sup_nsub 20,5 --sup_epochs 20 --sup_lr 2.0 "
         "--sup_balance 1 --label_rule frac0.25 --bin_cap 50000 --score_mode lse0.02 --smooth_sigma 2.0").split()
DEFAULT_PP = dict(sxy=0.0, sz=0.0, q=97.0, tf=0.5, mode="dynamic", factor=2.5, minvox=10, nles=5, adj=True, gamma=0.0,
                  mapping="sigmoid", conf="max", q_pz=None, q_tz=None)  # q_pz / q_tz: zone-specific calibration (idea 7)


# ------------------------------------------------------------------ maps
def splits():
    d = np.load(NPZ, mmap_mode="r")
    fold = {int(p): int(f) for p, f in zip(d["patient_id"], d["fold"])}
    pos = {}
    for p, c in zip(d["patient_id"], d["case_cspca"]):
        pos[int(p)] = max(pos.get(int(p), 0), int(c))
    pats = sorted(fold)
    out = {}
    for vf in (1, 2):  # inner rounds (same as hp_search.py)
        out[f"inner{vf}"] = {"train": [q for q in pats if fold[q] not in (0, vf)],
                             "val": [q for q in pats if fold[q] == vf], "test": [q for q in pats if fold[q] == vf][:5]}
    for f in range(5):  # CV folds with internal 15% val (same construction as cv_hdc.py; fold 0 = hp_search final)
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


MODELS = {"t119": TUNED, "t110": [a if a != "4" or TUNED[i - 1] != "--stride" else "2" for i, a in enumerate(TUNED)],
          "intensity": "--variants smooth_mix --descriptor stats --patch 12 --stride 4 --bin_mode zone".split(),
          "t119ctx": TUNED + ["--ctx3d", "1"],                                            # option b: 3D context
          "t119core": [a if a != "frac0.25" else "frac0.75" for a in TUNED]}             # option c: lesion-core labels


def _from_config(path):
    """CLI for a saved hp_search trial config (round 3: stride-2 search)."""
    import hp_search
    cfg = {**hp_search.DEFAULT, **json.load(open(path))}
    args, _ = hp_search.cli(cfg)
    return args


# round 4: seed copies of #110 (idea 3) and hard-negative mining on #110 (idea 2)
for _k in (1, 2, 3, 4):
    MODELS[f"t110s{_k}"] = MODELS["t110"] + ["--seed", str(_k)]
for _name, _f, _n in (("t110hn1", 0.02, 10), ("t110hn2", 0.05, 10), ("t110hn3", 0.01, 5)):
    MODELS[_name] = MODELS["t110"] + ["--hard_frac", str(_f), "--hard_nsub", str(_n)]

MODELS["t110isup"] = MODELS["t110"] + ["--isup_weight", "1.0"]                    # idea 8: grade-weighted lesions
MODELS["t110p8"] = [a if a != "20" or MODELS["t110"][i - 1] != "--patch" else "8"     # idea 5: small-patch boundary model
                    for i, a in enumerate(MODELS["t110"])]

# idea 6: the #110 recipe on 128 x 128 crops (patch 20 -> 26 px to keep its physical size; stride 2)
MODELS["t110r128"] = [("26" if a == "20" and MODELS["t110"][i - 1] == "--patch" else a) for i, a in enumerate(MODELS["t110"])]
MODEL_NPZ = {"t110r128": "/data/picai_all_masks128.npz"}
# idea 9: #110 recipe on a fine-tuned front end (one feature cache per split, mri_feature_cache/ft_<split>.npy)
MODELS["t110ft"] = MODELS["t110"]
for _k in (1, 2):
    MODELS[f"t110fts{_k}"] = MODELS["t110"] + ["--seed", str(_k)]

if os.path.exists("results/round3/config_t35.json"):
    MODELS["t35"] = _from_config("results/round3/config_t35.json")


def make_maps(args):
    sp = splits()
    os.makedirs(ROOT, exist_ok=True)
    keep = set(args.splits.split(",")) if args.splits else set(sp)
    jobs = [(m, n) for m in args.models.split(",") for n in sp if n in keep]

    def run(job):
        model, name = job
        out = os.path.join(ROOT, "maps", model, name)
        os.makedirs(out, exist_ok=True)
        pk = os.path.join(out, "maps.pkl")
        if os.path.exists(pk):
            return
        sf = os.path.join(out, "split.json")
        json.dump(sp[name], open(sf, "w"))
        cmd = [sys.executable, "patch_anomaly.py", "--slices_npz", MODEL_NPZ.get(model, NPZ), "--split_file", sf, "--save_root", out,
               "--save_maps", pk, "--eval_splits", "none", "--workers", "8"] + MODELS[model] + \
              (["--fmap_cache", f"mri_feature_cache/ft_{name}.npy"] if model.startswith("t110ft") else [])
        env = {**os.environ, "OMP_NUM_THREADS": "24", "OPENBLAS_NUM_THREADS": "24"}
        with open(os.path.join(out, "log.txt"), "w") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True, env=env)
        print("maps done:", model, name, flush=True)

    with ThreadPoolExecutor(args.parallel) as ex:
        for f in [ex.submit(run, j) for j in jobs]:
            f.result()


# ------------------------------------------------------------------ post-processing + evaluation
_PP = {}


def _process(item):
    from report_guided_annotation import extract_lesion_candidates
    from scipy.ndimage import gaussian_filter
    from scipy.special import expit
    m, off, temp, qgrid = item[:4]
    zoff = item[4] if len(item) > 4 else None  # (zone volume, {zone: offset}) for zone-specific calibration
    p = {**DEFAULT_PP, **_PP}
    m = m.astype(np.float32)
    if p["sxy"] > 0 or p["sz"] > 0:
        w = (~np.isnan(m)).astype(np.float32)
        num = gaussian_filter(np.nan_to_num(m), sigma=(p["sz"], p["sxy"], p["sxy"]))
        den = gaussian_filter(w, sigma=(p["sz"], p["sxy"], p["sxy"]))
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(w > 0, num / np.maximum(den, 1e-6), np.nan)
    if p["mapping"] == "rank":  # empirical CDF of val-normal voxels, sharpened: p = ecdf ** (1 / temp_f)
        e = np.interp(np.nan_to_num(m, nan=-1e3), qgrid, np.linspace(0, 1, len(qgrid)))
        prob = (e ** (1.0 / max(p["tf"], 1e-3))).astype(np.float64)
        prob[np.isnan(m)] = 0
    else:
        o = off
        if zoff is not None:
            zv, offs = zoff
            o = np.full(m.shape, off, np.float32)
            for zz, oz in offs.items():
                o[zv == zz] = oz
        prob = expit((np.nan_to_num(m, nan=-1e3) - o) / temp).astype(np.float64)
    thr = p["mode"] if p["mode"] in ("dynamic", "dynamic-fast") else float(p["mode"])
    det, conf, idx = extract_lesion_candidates(prob, threshold=thr, min_voxels_detection=int(p["minvox"]),
                                               num_lesions_to_extract=int(p["nles"]),
                                               dynamic_threshold_factor=float(p["factor"]),
                                               remove_adjacent_lesion_candidates=bool(p["adj"]))
    if p["conf"] != "max":  # candidate confidence from the whole blob instead of its peak
        det = np.zeros_like(det)
        for k, _ in conf:
            v = prob[idx == k]
            det[idx == k] = v.mean() if p["conf"] == "mean" else np.sort(v)[-max(1, len(v) // 10):].mean()
    if p["gamma"] > 0 and det.max() > 0:
        det = det * det.max() ** p["gamma"]
    return det.astype(np.float32)


def _init(pp):
    global _PP
    _PP = pp


def evaluate(maps: dict, ref_maps: dict, pp: dict, workers: int = 16) -> dict:
    """maps: {case: {map, lesion, cspca}}; ref_maps: the split whose csPCa-negative voxels set offset / temp."""
    from multiprocessing import get_context
    from picai_eval import evaluate as pe
    ref = np.concatenate([c["map"][~np.isnan(c["map"])].astype(np.float32).ravel()
                          for c in ref_maps.values() if c["cspca"] == 0])
    off, temp = float(np.percentile(ref, pp["q"])), float(pp["tf"] * ref.std() + 1e-6)
    qgrid = np.percentile(ref, np.linspace(0, 100, 1001))
    offs = None
    if pp.get("q_pz") is not None or pp.get("q_tz") is not None:  # zone-specific offsets from val-normal voxels
        offs = {}
        for zz, qk in ((1, "q_pz"), (2, "q_tz")):
            if pp.get(qk) is None:
                continue
            v = np.concatenate([c["map"][(c["zone"] == zz) & ~np.isnan(c["map"])].astype(np.float32)
                                for c in ref_maps.values() if c["cspca"] == 0])
            offs[zz] = float(np.percentile(v, pp[qk]))
    keys = sorted(maps)
    with get_context("fork").Pool(workers, initializer=_init, initargs=(pp,)) as pool:
        y_det = pool.map(_process, [(maps[k]["map"], off, temp, qgrid) + (((maps[k]["zone"], offs),) if offs else ())
                                    for k in keys])
    m = pe(y_det=y_det, y_true=[maps[k]["lesion"].astype(np.int32) for k in keys],
           subject_list=[str(k) for k in keys], num_parallel_calls=workers)
    return {"patient_auroc": float(m.auroc), "lesion_ap": float(m.AP), "picai_score": float(m.score)}


def load(name, model="t119"):
    return pickle.load(open(os.path.join(ROOT, "maps", model, name, "maps.pkl"), "rb"))


def suggest(trial):
    return dict(sxy=trial.suggest_categorical("sxy", [0.0, 1.0, 2.0, 3.0]),
                sz=trial.suggest_categorical("sz", [0.0, 0.5, 1.0, 2.0]),
                q=trial.suggest_categorical("q", [90.0, 95.0, 97.0, 99.0, 99.5, 99.9]),
                tf=trial.suggest_categorical("tf", [0.1, 0.25, 0.5, 1.0, 2.0]),
                mode=trial.suggest_categorical("mode", ["dynamic", "dynamic-fast", "0.3", "0.5", "0.7", "0.9"]),
                mapping=trial.suggest_categorical("mapping", ["sigmoid", "rank"]),
                conf=trial.suggest_categorical("conf", ["max", "mean", "top10pct"]),
                factor=trial.suggest_categorical("factor", [1.5, 2.0, 2.5, 3.0, 4.0, 6.0]),
                minvox=trial.suggest_categorical("minvox", [10, 25, 50, 100, 200]),
                nles=trial.suggest_categorical("nles", [1, 2, 3, 5, 10]),
                adj=trial.suggest_categorical("adj", [True, False]),
                gamma=trial.suggest_categorical("gamma", [0.0, 0.5, 1.0, 2.0]))


def search(args):
    import optuna
    inner = {n: load(n, args.model) for n in ("inner1", "inner2")}
    study = optuna.create_study(study_name=f"postproc_{args.model}", storage=f"sqlite:///{OUT}/optuna.db", direction="maximize",
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True))
    if len(study.trials) == 0:
        study.enqueue_trial(DEFAULT_PP)  # the setting used so far (q97/t0.5, dynamic, defaults)

    def objective(trial):
        pp = suggest(trial)
        r = [evaluate(inner[n]["val"], inner[n]["val"], pp, args.workers) for n in inner]
        for k in ("patient_auroc", "lesion_ap"):
            trial.set_user_attr(k, float(np.mean([x[k] for x in r])))
        return float(np.mean([x["picai_score"] for x in r]))

    study.optimize(objective, n_trials=args.trials, catch=(Exception,))


def status(args):
    import optuna
    st = optuna.load_study(study_name=f"postproc_{args.model}", storage=f"sqlite:///{OUT}/optuna.db")
    done = [t for t in st.trials if t.state.name == "COMPLETE"]
    print(f"{args.model}: {len(done)} complete of {len(st.trials)}")
    if done and done[0].number == 0 or any(t.number == 0 for t in done):
        t0 = [t for t in done if t.number == 0][0]
        print(f"current setting (trial 0): PI-CAI {t0.value:.3f} AP {t0.user_attrs['lesion_ap']:.3f} "
              f"AUROC {t0.user_attrs['patient_auroc']:.3f}")
    for t in sorted(done, key=lambda t: -t.value)[: args.top]:
        print(f"#{t.number} PI-CAI {t.value:.3f} AP {t.user_attrs['lesion_ap']:.3f} "
              f"AUROC {t.user_attrs['patient_auroc']:.3f} {t.params}")
    if len(done) >= 10:
        print("importance:", {k: round(v, 3) for k, v in optuna.importance.get_param_importances(st).items()})


def voxel_auroc(maps: dict) -> float:
    from sklearn.metrics import roc_auc_score
    m = np.concatenate([c["map"].astype(np.float32).ravel() for c in maps.values()])
    y = np.concatenate([c["lesion"].ravel() for c in maps.values()])
    ok = ~np.isnan(m)
    return float(roc_auc_score(y[ok] > 0, m[ok]))


def final(args):
    rows = {}
    if args.pp_json:  # a fixed setting (e.g. the best one found for t119), applied to another model's maps
        tuned = {**DEFAULT_PP, **json.load(open(args.pp_json))}
        best = None
    else:
        import optuna
        st = optuna.load_study(study_name=f"postproc_{args.model}", storage=f"sqlite:///{OUT}/optuna.db")
        best = max([t for t in st.trials if t.state.name == "COMPLETE"], key=lambda t: t.value)
        tuned = {**DEFAULT_PP, **best.params}
    for label, pp in (("current", DEFAULT_PP), ("tuned", tuned)):
        inner = [evaluate(load(n, args.model)["val"], load(n, args.model)["val"], pp, args.workers)
                 for n in ("inner1", "inner2")]
        rows.setdefault("inner_val", {})[label] = {k: float(np.mean([x[k] for x in inner])) for k in inner[0]}
        per = []
        for f in range(5):
            mp = load(f"fold{f}", args.model)
            r = evaluate(mp["test"], mp["val"], pp, args.workers)
            r["voxel_auroc"] = voxel_auroc(mp["test"])
            per.append({"fold": f, **r})
            print(label, f, r, flush=True)
        ks = ("voxel_auroc", "patient_auroc", "lesion_ap", "picai_score")
        rows[label] = {"pp": pp, "per_fold": per,
                       "mean": {k: float(np.mean([p[k] for p in per])) for k in ks},
                       "std": {k: float(np.std([p[k] for p in per])) for k in ks}}
    if best is not None:
        rows["best_trial"] = {"number": best.number, "inner_val_picai": best.value, **best.user_attrs}
    tag = "_pp-" + os.path.splitext(os.path.basename(args.pp_json))[0] if args.pp_json else ""
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, f"final_{args.model}{tag}.json"), "w"), indent=2)
    for label in ("current", "tuned"):
        m, s = rows[label]["mean"], rows[label]["std"]
        print(f"{label}: fold0 PI-CAI {rows[label]['per_fold'][0]['picai_score']:.3f} | CV PI-CAI "
              f"{m['picai_score']:.3f} ± {s['picai_score']:.3f}, AP {m['lesion_ap']:.3f} ± {s['lesion_ap']:.3f}, "
              f"AUROC {m['patient_auroc']:.3f} ± {s['patient_auroc']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["maps", "search", "status", "final"])
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--models", default="t119,intensity")
    ap.add_argument("--splits", default="", help="maps: only these splits (e.g. inner1,inner2)")
    ap.add_argument("--model", default="t119", help="which saved maps to search / finalise")
    ap.add_argument("--pp_json", default=None, help="final: use this post-processing setting instead of the search's best")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    {"maps": make_maps, "search": search, "status": status, "final": final}[args.mode](args)


if __name__ == "__main__":
    main()
