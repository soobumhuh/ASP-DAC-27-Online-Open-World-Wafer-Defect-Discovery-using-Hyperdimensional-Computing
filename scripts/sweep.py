#!/usr/bin/env python3
"""Model-parameter sweep on the inner splits (folds 1-4 only; fold 0 is never used for choices).

  uv run python scripts/sweep.py --config configs/d8192.yaml --root results/d8sweep \
      --variants 'base: {}' 't005: {softmax_t: 0.005}' 'sub40: {n_sub: [40, 10]}' --pp results/round6/pp_t110.json

Every variant = the base config's 'model' section with the given overrides (YAML flow mappings), seed 0. Its inner1 /
inner2 maps go to <root>/<variant>/maps/seed0/; map jobs of all variants share one queue (run.parallel_jobs at a
time). Each variant is then scored with one fixed extraction setting (--pp) on the inner validation cases:
mean PI-CAI, lesion AP, patient AUROC and voxel AUROC over the two inner splits -> <root>/sweep.json and sweep.md.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import yaml

import _path  # noqa: F401
from mri_hdc import evaluate as E, extraction as X, maps as M
from mri_hdc.pipeline import build_maps, load_config, map_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--variants", nargs="+", required=True, help="'name: {param: value, ...}'")
    ap.add_argument("--pp", required=True, help="fixed extraction setting (json)")
    ap.add_argument("--build_only", action="store_true")
    ap.add_argument("--parallel", type=int, default=None, help="map jobs at a time (default: run.parallel_jobs)")
    args = ap.parse_args()
    base = load_config(args.config)
    pp = {**X.DEFAULT, **json.load(open(args.pp))}
    pp = {k: v for k, v in pp.items() if k in X.DEFAULT}
    cfgs = {}
    for v in args.variants:
        name, ov = v.split(":", 1)
        cfg = copy.deepcopy(base)
        cfg["model"].update(yaml.safe_load(ov) or {})
        cfg["paths"]["out"] = os.path.join(args.root, name.strip())
        os.makedirs(cfg["paths"]["out"], exist_ok=True)
        path = os.path.join(cfg["paths"]["out"], "config.yaml")
        yaml.safe_dump({k: c for k, c in cfg.items() if not k.startswith("_")}, open(path, "w"), sort_keys=False)
        cfgs[name.strip()] = load_config(path)

    # one queue over all variants: interleave so that each variant finishes early
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(n, s) for s in ("inner1", "inner2") for n in cfgs]
    with ThreadPoolExecutor(args.parallel or base["run"]["parallel_jobs"]) as ex:
        futs = [ex.submit(build_maps, cfgs[n], [s], [0]) for n, s in jobs]
        for f in futs:
            f.result()
    if args.build_only:
        return
    w = base["run"]["workers"]
    res = {}
    for n, cfg in cfgs.items():
        r, vox = [], []
        for s in ("inner1", "inner2"):
            mp = M.load(map_path(cfg, 0, s))["val"]
            r.append(E.picai(E.detection_maps(X.extract(mp, mp, pp, None, w)), mp, workers=w))
            vox.append(E.voxel_auroc(mp))
        logs = [os.path.join(cfg["paths"]["out"], "logs", f"maps_seed0_{s}.log") for s in ("inner1", "inner2")]
        secs = [float(l.split("(")[1].split("s)")[0]) for f in logs if os.path.exists(f)
                for l in open(f) if l.startswith("fit done")]
        res[n] = {**{k: float(np.mean([x[k] for x in r])) for k in r[0]}, "voxel_auroc": float(np.mean(vox)),
                  "model": cfg["model"], "fit_s": float(np.mean(secs)) if secs else None}
        print(n, {k: round(v, 4) for k, v in res[n].items() if isinstance(v, float)}, flush=True)
    old = json.load(open(os.path.join(args.root, "sweep.json"))) if os.path.exists(os.path.join(args.root, "sweep.json")) else {}
    old.update(res)
    json.dump(old, open(os.path.join(args.root, "sweep.json"), "w"), indent=2)
    lines = ["| variant | overrides | inner PI-CAI | lesion AP | patient AUROC | voxel AUROC | fit time / job |",
             "|---|---|---|---|---|---|---|"]
    bm = base["model"]
    for n, r in sorted(old.items(), key=lambda kv: -kv[1]["picai_score"]):
        diff = {k: v for k, v in r["model"].items() if bm.get(k) != v}
        lines.append(f"| {n} | {diff or '-'} | {r['picai_score']:.4f} | {r['lesion_ap']:.4f} | {r['patient_auroc']:.4f} | "
                     f"{r['voxel_auroc']:.4f} | {('%.0f s' % r['fit_s']) if r.get('fit_s') else '-'} |")
    open(os.path.join(args.root, "sweep.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
