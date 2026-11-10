#!/usr/bin/env python3
"""
hp_search.py - Optuna (TPE) hyperparameter search for the design-A HDC patch classifier.

Protocol (no tuning on test): fold 0 of the official PI-CAI folds is held out and never touched by the search.
Each trial is scored by inner cross-validation on folds 1-4: train on three folds, validate on the fourth,
for --inner_rounds rounds (val fold 1, then 2, ...). Objective = mean val PI-CAI score (patient AUROC + lesion
AP) / 2, maximised over the detection offset / temperature grid; val patient AUROC and lesion AP are logged.
Round 1 is reported to a median pruner, so clearly bad trials stop after one round.

  search : uv run python hp_search.py search --workers 8 --trials 200
  status : uv run python hp_search.py status
  final  : uv run python hp_search.py final --top 3      # retrain top configs on folds 1-4, test fold 0 once

Trials run as separate patch_anomaly.py processes sharing an SQLite study (results/hp/optuna.db).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.environ.get("HP_ROOT", "results/hp")
NPZ = "/data/picai_all_masks.npz"
DEFAULT = dict(patch=12, stride=4, slice_levels=3, bin_mode="zone", descriptor="stats", num_proj=256, num_bins=8,
               clip_value=3.0, nsub_n=10, nsub_l=5, epochs=5, lr=1.0, balance=1, label_rule="centre", negplus=0,
               bin_cap=50000, score_mode="max", smooth_sigma=0.0, det_threshold="dynamic-fast")
DESCRIPTORS = ["stats", "stats+pct", "stats+tex", "stats+pct+tex", "resnet_l2", "resnet_l3", "resnet_l2l3",
               "stats+resnet_l2", "stats+pct+resnet_l3", "stats+pct+tex+resnet_l2l3"]


FIX = {}       # parameters held fixed for a study (e.g. stride=2), set from --fix
CHOICES = {}   # overridden choice lists, set from --patch_choices


def suggest(trial) -> dict:
    cat = lambda name, opts: trial.suggest_categorical(name, [FIX[name]] if name in FIX else CHOICES.get(name, opts))
    p = dict(
        patch=cat("patch", [8, 12, 16, 20]),
        stride=cat("stride", [2, 4, 6]),
        slice_levels=cat("slice_levels", [1, 3, 5]),
        bin_mode=cat("bin_mode", ["zone", "zone_grid"]),
        descriptor=cat("descriptor", DESCRIPTORS),
        num_proj=cat("num_proj", [128, 256, 512, 1024]),
        num_bins=cat("num_bins", [4, 8, 16]),
        clip_value=cat("clip_value", [2.0, 3.0, 4.0]),
        nsub_n=cat("nsub_n", [5, 10, 20, 40]),
        nsub_l=cat("nsub_l", [2, 5, 10, 20]),
        epochs=cat("epochs", [0, 3, 5, 10, 20]),
        lr=cat("lr", [0.25, 0.5, 1.0, 2.0]),
        balance=cat("balance", [0, 1]),
        label_rule=cat("label_rule", ["centre", "frac0.25", "frac0.5"]),
        negplus=cat("negplus", [0, 1]),
        bin_cap=cat("bin_cap", [20000, 50000, 100000]),
        score_mode=cat("score_mode", ["max", "top2", "lse0.02", "lse0.05"]),
        smooth_sigma=cat("smooth_sigma", [0.0, 1.0, 2.0, 3.0]),
        det_threshold=cat("det_threshold", ["dynamic-fast", "dynamic"]),
    )
    p["stride"] = min(p["stride"], p["patch"] // 2)  # stride <= patch / 2
    return p


def cli(p: dict) -> list:
    variant = f"proto{p['num_proj']}" + ("_negplus" if p["negplus"] else "")
    return ["--variants", variant, "--descriptor", p["descriptor"], "--patch", str(p["patch"]),
            "--stride", str(p["stride"]), "--slice_levels", str(p["slice_levels"]), "--bin_mode", p["bin_mode"],
            "--num_bins", str(p["num_bins"]), "--clip_value", str(p["clip_value"]),
            "--sup_nsub", f"{p['nsub_n']},{p['nsub_l']}", "--sup_epochs", str(p["epochs"]), "--sup_lr", str(p["lr"]),
            "--sup_balance", str(p["balance"]), "--label_rule", p["label_rule"], "--bin_cap", str(p["bin_cap"]),
            "--score_mode", p["score_mode"], "--smooth_sigma", str(p["smooth_sigma"]),
            "--det_threshold", p["det_threshold"]], variant


def folds_of() -> dict:
    d = np.load(NPZ, mmap_mode="r")
    return {int(pid): int(f) for pid, f in zip(d["patient_id"], d["fold"])}


def run_one(p: dict, split: dict, out: str, eval_splits: str, timeout: int, seed: int = 0) -> dict:
    os.makedirs(out, exist_ok=True)
    sf = os.path.join(out, "split.json")
    json.dump(split, open(sf, "w"))
    args, variant = cli(p)
    res = os.path.join(out, f"patch_{variant}", "results.json")
    if not os.path.exists(res):
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "patch_anomaly.py"), "--slices_npz", NPZ, "--split_file", sf, "--seed", str(seed),
               "--save_root", out, "--workers", "12", "--eval_splits", eval_splits] + args
        env = {**os.environ, "OMP_NUM_THREADS": "16", "OPENBLAS_NUM_THREADS": "16", "MKL_NUM_THREADS": "16"}
        with open(os.path.join(out, "log.txt"), "w") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True, timeout=timeout, env=env)
    return json.load(open(res))


def best_setting(r: dict, split: str, key="picai_score"):
    k = max(r[split]["by_offset"], key=lambda s: r[split]["by_offset"][s][key])
    return k, r[split]["by_offset"][k]


def search(args):
    import optuna
    os.makedirs(ROOT, exist_ok=True)
    storage = f"sqlite:///{ROOT}/optuna.db"
    study = optuna.create_study(study_name=args.study, storage=storage, direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=args.worker_seed, multivariate=True,
                                                                   n_startup_trials=20),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0))
    if args.worker_seed == 0 and len(study.trials) == 0:
        ref = dict(DEFAULT) if not args.reference else json.load(open(args.reference))
        study.enqueue_trial({**ref, **FIX})  # reference trial (the current defaults, or a given config)
    fold = folds_of()
    pats = sorted(fold)

    def objective(trial):
        p = suggest(trial)
        t0 = time.time()
        vals = []
        for r in range(args.inner_rounds):
            vf = 1 + r
            split = {"train": [q for q in pats if fold[q] not in (0, vf)], "val": [q for q in pats if fold[q] == vf],
                     "test": [q for q in pats if fold[q] == vf][:5]}  # test unused (val-only evaluation)
            try:
                tdir = f"trial{trial.number:04d}" if args.study == "designA" else f"{args.study}/trial{trial.number:04d}"
                res = run_one(p, split, os.path.join(ROOT, tdir, f"round{vf}"), "val",
                              args.timeout)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                trial.set_user_attr("error", type(e).__name__)
                raise optuna.TrialPruned()
            setting, m = best_setting(res, "val")
            vals.append(m)
            trial.set_user_attr(f"round{vf}", {"setting": setting, **m,
                                               "voxel_auroc": res["val"]["voxel_auroc_in_gland"]})
            trial.report(float(np.mean([v["picai_score"] for v in vals])), r)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("val_patient_auroc", float(np.mean([v["patient_auroc"] for v in vals])))
        trial.set_user_attr("val_lesion_ap", float(np.mean([v["lesion_ap"] for v in vals])))
        trial.set_user_attr("minutes", round((time.time() - t0) / 60, 1))
        trial.set_user_attr("config", p)
        return float(np.mean([v["picai_score"] for v in vals]))

    study.optimize(objective, n_trials=args.trials, catch=(Exception,))


def status(args):
    import optuna
    study = optuna.load_study(study_name=args.study, storage=f"sqlite:///{ROOT}/optuna.db")
    T = study.trials
    done = [t for t in T if t.state.name == "COMPLETE"]
    print(f"trials: {len(T)} total | complete {len(done)} | pruned {sum(t.state.name == 'PRUNED' for t in T)} | "
          f"running {sum(t.state.name == 'RUNNING' for t in T)} | failed {sum(t.state.name == 'FAIL' for t in T)}")
    ref = [t for t in done if t.number == 0]
    if ref:
        print(f"default config (trial 0): val PI-CAI {ref[0].value:.3f}")
    for t in sorted(done, key=lambda t: -t.value)[: args.top]:
        print(f"#{t.number} val PI-CAI {t.value:.3f} | AUROC {t.user_attrs.get('val_patient_auroc', float('nan')):.3f} "
              f"AP {t.user_attrs.get('val_lesion_ap', float('nan')):.3f} | {t.user_attrs.get('minutes')} min | {t.params}")
    if len(done) >= 10 and args.importance:
        try:
            imp = optuna.importance.get_param_importances(study)
            print("importance:", {k: round(v, 3) for k, v in imp.items()})
        except Exception as e:  # noqa
            print("importance failed:", e)


def final(args):
    """Retrain the top configs on folds 1-4 (15% of them as internal val for settings) and test fold 0 once."""
    import optuna
    study = optuna.load_study(study_name=args.study, storage=f"sqlite:///{ROOT}/optuna.db")
    done = sorted([t for t in study.trials if t.state.name == "COMPLETE"], key=lambda t: -t.value)
    picks = done[: args.top]
    if not any(t.number == 0 for t in picks):
        picks += [t for t in done if t.number == 0]  # the defaults, for reference
    fold = folds_of()
    d = np.load(NPZ, mmap_mode="r")
    pos = {}
    for pid, c in zip(d["patient_id"], d["case_cspca"]):
        pos[int(pid)] = max(pos.get(int(pid), 0), int(c))
    pool = np.array(sorted(q for q in fold if fold[q] != 0))
    rng = np.random.default_rng(100)
    val = []
    for flag in (0, 1):
        g = pool[np.array([pos[q] == flag for q in pool])]
        val += list(rng.permutation(g)[: int(round(0.15 * len(g)))])
    split = {"train": [int(q) for q in pool if q not in set(val)], "val": sorted(int(q) for q in val),
             "test": sorted(q for q in fold if fold[q] == 0)}
    rows = []
    for t in picks:
        p = {**DEFAULT, **t.params}
        p["stride"] = min(p["stride"], p["patch"] // 2)
        fdir = "final" if args.study == "designA" else f"final_{args.study}"
        res = run_one(p, split, os.path.join(ROOT, fdir, f"trial{t.number:04d}"), "val,test", 7200)
        setting, _ = best_setting(res, "val")
        m = res["test"]["by_offset"][setting]
        rows.append({"trial": t.number, "inner_val_picai": t.value, "setting": setting,
                     "test_voxel_auroc": res["test"]["voxel_auroc_in_gland"], **{f"test_{k}": v for k, v in m.items()},
                     "config": p, "runtime_s": res["runtime_s"]})
        print(json.dumps(rows[-1]))
    json.dump(rows, open(os.path.join(ROOT, fdir, "fold0_test.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["search", "status", "final"])
    ap.add_argument("--trials", type=int, default=25, help="trials for this worker")
    ap.add_argument("--worker_seed", type=int, default=0)
    ap.add_argument("--inner_rounds", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--importance", action="store_true")
    ap.add_argument("--study", default="designA")
    ap.add_argument("--fix", default="", help="comma list name=value held fixed, e.g. stride=2")
    ap.add_argument("--patch_choices", default="", help="comma list of patch sizes to search")
    ap.add_argument("--reference", default="", help="json config enqueued as trial 0")
    args = ap.parse_args()
    for kv in filter(None, args.fix.split(",")):
        k, v = kv.split("=")
        FIX[k] = type(DEFAULT[k])(v) if not isinstance(DEFAULT[k], float) else float(v)
    if args.patch_choices:
        CHOICES["patch"] = [int(x) for x in args.patch_choices.split(",")]
    {"search": search, "status": status, "final": final}[args.mode](args)


if __name__ == "__main__":
    main()
