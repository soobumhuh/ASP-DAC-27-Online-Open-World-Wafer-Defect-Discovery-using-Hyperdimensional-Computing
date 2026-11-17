#!/usr/bin/env python3
"""A small file-based queue for map jobs (scripts/train_predict.py), so several experiments share one CPU budget.

  uv run python scripts/jobq.py add --config configs/sota/x.yaml --splits inner1,inner2 --seeds 0,1,2 [--prio 5]
  uv run python scripts/jobq.py add --config configs/sota/x.yaml --nested           # fold0-4 + the 20 nest<f><g>
  uv run python scripts/jobq.py run --workers 4 --threads 12                       # long-running worker pool
  uv run python scripts/jobq.py status

Jobs are JSON files in <queue>/pending (named <prio>_<time>_<config>_<split>_<seed>; lower prio runs first); a
worker claims one by renaming it into running/, then moves it to done/ or failed/. Jobs whose map already exists are
skipped. Workers re-scan pending/ every 20 s, so jobs can be added while the pool runs.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time

import _path  # noqa: F401
from mri_hdc import ROOT
from mri_hdc.pipeline import load_config, map_path, out_path, splits

Q = os.environ.get("JOBQ", os.path.join(ROOT, "results", "sota", "queue"))
NESTED = [f"fold{f}" for f in range(5)] + [f"nest{f}{g}" for f in range(5) for g in range(5) if g != f]


def add(cfg_path, names, seeds, prio):
    cfg = load_config(cfg_path)
    splits(cfg)
    for s in ("pending", "running", "done", "failed"):
        os.makedirs(os.path.join(Q, s), exist_ok=True)
    tag = os.path.splitext(os.path.basename(cfg_path))[0]
    n = 0
    for sd in seeds:
        for name in names:
            if os.path.exists(map_path(cfg, sd, name)):
                continue
            job = {"config": os.path.abspath(cfg_path), "split": name, "seed": sd}
            fn = f"{prio:02d}_{time.time():.6f}_{tag}_{name}_{sd}.json"
            if glob.glob(os.path.join(Q, "*", f"*_{tag}_{name}_{sd}.json")):
                continue  # already queued / running
            json.dump(job, open(os.path.join(Q, "pending", fn), "w"))
            n += 1
    print(f"queued {n} jobs for {tag}")


def run(workers, threads, recover=False):
    lock = threading.Lock()

    def claim():
        with lock:
            for fn in sorted(os.listdir(os.path.join(Q, "pending"))):
                src = os.path.join(Q, "pending", fn)
                dst = os.path.join(Q, "running", fn)
                try:
                    os.rename(src, dst)
                    return fn
                except FileNotFoundError:
                    continue
        return None

    def worker(i):
        while True:
            fn = claim()
            if fn is None:
                time.sleep(20)
                continue
            job = json.load(open(os.path.join(Q, "running", fn)))
            cfg = load_config(job["config"])
            out = map_path(cfg, job["seed"], job["split"])
            ok = True
            if not os.path.exists(out):
                os.makedirs(out_path(cfg, "logs"), exist_ok=True)
                env = {**os.environ, "OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads),
                       "MKL_NUM_THREADS": str(threads)}
                t0 = time.time()
                with open(out_path(cfg, "logs", f"maps_seed{job['seed']}_{job['split']}.log"), "w") as fh:
                    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "train_predict.py"), "--config",
                                        job["config"], "--split", job["split"], "--seed", str(job["seed"])],
                                       stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
                ok = r.returncode == 0
                print(f"{time.strftime('%m-%d %H:%M')} worker {i}: {fn} {'ok' if ok else 'FAILED'} "
                      f"{time.time() - t0:.0f}s", flush=True)
            os.rename(os.path.join(Q, "running", fn), os.path.join(Q, "done" if ok else "failed", fn))

    for s in ("pending", "running", "done", "failed"):
        os.makedirs(os.path.join(Q, s), exist_ok=True)
    for fn in os.listdir(os.path.join(Q, "running")) if recover else []:  # jobs of a killed pool back to pending
        os.rename(os.path.join(Q, "running", fn), os.path.join(Q, "pending", fn))
    ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def status():
    for s in ("pending", "running", "done", "failed"):
        fs = sorted(os.listdir(os.path.join(Q, s))) if os.path.isdir(os.path.join(Q, s)) else []
        print(f"{s}: {len(fs)}")
        if s in ("running", "failed"):
            for f in fs:
                print("  ", f)


def wait(cfg_path, names, seeds):
    cfg = load_config(cfg_path)
    while not all(os.path.exists(map_path(cfg, s, n)) for s in seeds for n in names):
        time.sleep(60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["add", "run", "status", "wait"])
    ap.add_argument("--config")
    ap.add_argument("--splits", default="")
    ap.add_argument("--nested", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--prio", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--recover", action="store_true", help="move running/ jobs of a killed pool back to pending")
    a = ap.parse_args()
    names = (NESTED if a.nested else []) + [s for s in a.splits.split(",") if s]
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.cmd == "add":
        add(a.config, names, seeds, a.prio)
    elif a.cmd == "run":
        run(a.workers, a.threads, a.recover)
    elif a.cmd == "wait":
        wait(a.config, names, seeds)
    else:
        status()
