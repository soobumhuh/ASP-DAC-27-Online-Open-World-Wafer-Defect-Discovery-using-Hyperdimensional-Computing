#!/usr/bin/env python3
"""CPU time to score ONE new exam with an already-trained pipeline (scripts/make_bundle.py), stage by stage.

  uv run python scripts/time_inference.py --config configs/test_fresh.yaml --threads 1 --n 20
  uv run python scripts/time_inference.py --config configs/test_fresh.yaml --threads 24 --n 20

Exams: the first n + warmup fold-0 test cases (never seen in training). Per exam, from the raw .mha files:
  preprocess    read T2W / ADC / HBV + AI gland and zone masks, resample to T2W, normalise, crop, resize (+ scanner header)
  resnet        frozen ResNet-18 up to layer 2 on the exam's slices (CPU)
  patches       patch grid inside the gland, descriptors, position bins
  encode        clip / standardise + thermometer hypervectors (per seed)
  score         Hamming similarity to the 25 prototypes of each patch's bin, per-bin z (per seed)
  map           overlap averaging + Gaussian smoothing (per seed; + seed average)
  extract       calibration + extract_lesion_candidates (+ candidate fields)
  rescore       candidate record hypervectors -> re-scorer rank
  clinical      clinical record hypervector, score, merge with the image score
(a) #110 single model: preprocess, resnet, patches, encode / score / map for seed 0, extract (its own setting).
(b) full best pipeline: shared preprocess / resnet / patches, 3 x (encode, score, map), extract, rescore, clinical.
Also: one 2D slice (resnet + patches + encode + score + map, seed 0). Medians over n exams after `warmup` exams.
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--split", default="fold0")
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--warmup", type=int, default=2)
args = ap.parse_args()
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = str(args.threads)

import json  # noqa: E402
import pickle  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import _path  # noqa: F401,E402
from mri_hdc import data as D, extraction as X, features as Fe, maps as M, patient as PT  # noqa: E402
from mri_hdc.pipeline import load_config, out_path, splits  # noqa: E402

torch.set_num_threads(args.threads)


def resnet():
    import torch.nn.functional as F
    from torchvision.models import ResNet18_Weights, resnet18
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()

    def run(images):
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(images[:, :3], dtype=np.float32))
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
            x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            return F.avg_pool2d(m.layer2(m.layer1(x)), 3, 1, 1).numpy().astype(np.float16)
    return run


def main():
    cfg = load_config(args.config)
    p = cfg["paths"]
    B = pickle.load(open(out_path(cfg, "models", f"bundle_{args.split}.pkl"), "rb"))
    mcfg, models, seeds = B["model_cfg"], B["models"], sorted(B["models"])
    net = resnet()
    files = D.index_mha_files(p["images_dir"])
    rows = PT.read_marksheet(os.path.join(p["labels_dir"], "clinical_information", "marksheet.csv"))
    # the fold-0 test exams (study ids from the split's patients)
    import csv
    test_p = set(int(x) for x in splits(cfg)[args.split]["test"])
    exams = sorted(k for k in files if int(k.split("_")[0]) in test_p)[: args.n + args.warmup]
    base_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    T = {"single": [], "full": [], "slice": []}
    keep = {}
    for i, key in enumerate(exams):
        sid = int(key.split("_")[1])
        t = {}

        def tick(name, t0):
            t[name] = t.get(name, 0.0) + time.perf_counter() - t0
            return time.perf_counter()

        t0 = time.perf_counter()
        c = D.preprocess_case(files[key], os.path.join(p["labels_dir"], D.GLAND_DIR, key + ".nii.gz"),
                              os.path.join(p["labels_dir"], D.ZONE_DIR, key + ".nii.gz"), size=cfg["data"]["size"])
        import SimpleITK as sitk
        r = sitk.ImageFileReader(); r.SetFileName(files[key]["hbv"]); r.ReadImageInformation()
        g = lambda k: r.GetMetaData(k) if r.HasMetaDataKey(k) else "NA"
        scan = (f"{g('0008|0070')} {g('0008|1090')}", g("0018|9087"))
        t0 = tick("preprocess", t0)
        fm = net(c["images"])
        t0 = tick("resnet", t0)
        idx = np.arange(len(c["rel_z"]))
        F, sl, ys, xs = Fe.extract_patches(c, idx, mcfg["patch"], mcfg["stride"], fm)
        bins = Fe.position_bins(c, sl, ys, xs, mcfg["slice_levels"])
        t0 = tick("patches", t0)
        zmaps = {}
        for s in seeds:
            mod = models[s]
            codes = mod.enc.codes(mod.enc.prep(F))
            t0 = tick(f"encode{s}", t0)
            z = (mod.raw_scores(codes, bins) - mod.mu[bins]) / mod.sd[bins]
            t0 = tick(f"score{s}", t0)
            zmaps[s] = M.assemble_maps(c, idx, sl, ys, xs, z, mcfg["patch"], mcfg["smooth_sigma"])
            t0 = tick(f"map{s}", t0)
        ens = np.mean([zmaps[s].astype(np.float16).astype(np.float32) for s in seeds], axis=0).astype(np.float16)
        t0 = tick("map_avg", t0)
        case_one = {"map": zmaps[seeds[0]].astype(np.float16), "lesion": np.zeros(ens.shape, np.uint8),
                    "rel_z": c["rel_z"], "zone": c["zone_mask"]}
        X._init(B["single"]["pp"])
        off, temp, qg = B["single"]["calibration"]
        _, cands1, _ = X._extract((sid, case_one, None, None, off, temp, qg))
        t0 = tick("extract_single", t0)
        E = B["ensemble"]
        case = {**case_one, "map": ens}
        X._init(E["pp"])
        off, temp, qg = E["calibration"]
        _, cands, _ = X._extract((sid, case, c["images"].astype(np.float32), c["gland_mask"] > 0, off, temp, qg))
        t0 = tick("extract_full", t0)
        conf = E["rescorer"].confidences({sid: (cands, ens.shape)}, E["alpha"])[sid]
        img = max(conf, default=0.0)
        t0 = tick("rescore", t0)
        R = E["records"]
        R.rows, R.scan = {sid: rows.get(sid, {})}, {sid: scan}
        clin = float(E["clinical"].score(R.hv([sid]))[0])
        score = (img - E["img_stats"][0]) / (E["img_stats"][1] + 1e-9) + \
            E["merge_weight"] * (clin - E["clin_stats"][0]) / (E["clin_stats"][1] + 1e-9)
        t0 = tick("clinical", t0)
        # one 2D slice (middle slice), #110 seed 0
        j = len(idx) // 2
        c1 = {k: c[k][j:j + 1] for k in ("images", "gland_mask", "zone_mask")}
        c1["rel_z"] = c["rel_z"][j:j + 1]
        t1 = time.perf_counter()
        f1 = net(c1["images"])
        F1, sl1, ys1, xs1 = Fe.extract_patches(c1, np.arange(1), mcfg["patch"], mcfg["stride"], f1)
        b1 = Fe.position_bins(c1, sl1, ys1, xs1, mcfg["slice_levels"])
        mod = models[seeds[0]]
        z1 = (mod.raw_scores(mod.enc.codes(mod.enc.prep(F1)), b1) - mod.mu[b1]) / mod.sd[b1]
        M.assemble_maps(c1, np.arange(1), sl1, ys1, xs1, z1, mcfg["patch"], mcfg["smooth_sigma"])
        t_slice = time.perf_counter() - t1
        single = {k: t[k] for k in ("preprocess", "resnet", "patches")}
        single.update(encode=t[f"encode{seeds[0]}"], score=t[f"score{seeds[0]}"], map=t[f"map{seeds[0]}"],
                      extract=t["extract_single"])
        full = {k: t[k] for k in ("preprocess", "resnet", "patches")}
        full.update(encode=sum(t[f"encode{s}"] for s in seeds), score=sum(t[f"score{s}"] for s in seeds),
                    map=sum(t[f"map{s}"] for s in seeds) + t["map_avg"], extract=t["extract_full"],
                    rescore=t["rescore"], clinical=t["clinical"])
        single["total"], full["total"] = sum(single.values()), sum(full.values())
        print(f"{key}: {len(idx)} slices, {len(F)} patches, single {single['total']:.2f}s, full {full['total']:.2f}s, "
              f"slice {t_slice * 1000:.0f} ms, candidates {len(cands)}, score {score:.2f}", flush=True)
        if i >= args.warmup:
            T["single"].append(single); T["full"].append(full); T["slice"].append(t_slice)
            T.setdefault("slices", []).append(len(idx)); T.setdefault("patches", []).append(len(F))
            keep[sid] = zmaps[seeds[0]]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # GiB (ru_maxrss in KiB)
    med = {k: {s: float(np.median([x[s] for x in T[k]])) for s in T[k][0]} for k in ("single", "full")}
    res = {"threads": args.threads, "n_exams": len(T["slice"]), "median_s": med,
           "median_slice_ms": float(np.median(T["slice"]) * 1000), "median_slices": float(np.median(T["slices"])),
           "median_patches": float(np.median(T["patches"])), "peak_rss_gib": peak,
           "rss_before_exams_gib": base_rss / 2**20}
    # check: the seed-0 map of each timed exam equals the saved fold-0 test map of the same model
    ref = M.load(os.path.join(p["out"], "maps", f"seed{seeds[0]}", f"{args.split}.pkl"))["test"]
    res["maps_identical_to_saved"] = int(sum(np.array_equal(np.nan_to_num(ref[k]["map"].astype(np.float32)),
                                                            np.nan_to_num(v.astype(np.float16).astype(np.float32)))
                                             for k, v in keep.items()))
    json.dump(res, open(out_path(cfg, f"inference_time_t{args.threads}.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
