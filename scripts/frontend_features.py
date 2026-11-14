#!/usr/bin/env python3
"""Per-slice feature caches of the new front ends, on the 96 px prostate crop of every kept slice (folds 1-4 only;
fold 0 is never read). Rows are aligned to frontend.drop_fold0(slices).

A1 ProViCNet (pimed/ProViCNet, HF weights T2 / ADC / DWI, mri_hdc/encoders/provicnet.py): per sequence, the T2W
   grid resampled in-plane to 0.5 mm, 256 x 256 centred on the gland box (PI-CAI-style crop), z-scored over the
   gland voxels of the volume, resized to 448; channels = slices j-1, j, j+1 (edge-shifted); position (j, S).
   Patch tokens (32 x 32, 14 px = 4 mm) of blocks 7 (~2/3 deep) and 11 (last, after the final norm), the three
   sequences concatenated (3 x 384 = 1152 channels).
A3 ProFound-alpha (pipiwang/ProFound demo, ConvNeXt V2-Tiny 3D FCMAE, mri_hdc/encoders/profound.py): T2W, HBV, ADC
   on the T2W grid, in-plane anti-alias blur, resampled to 1 mm isotropic, 64 x 224 x 224 centred on the gland box
   (z: centre of the kept slices), each channel z-scored over nonzero voxels. Stage 1 (stride 4, 96 ch) and
   stage 2 (stride 8, 192 ch).
Every map is read out (bilinear / trilinear) on a G x G grid of cell centres covering exactly the slice's 96 px crop
(the same square_box as data.build_slices), so the shared adapter treats it like the ResNet map.

  CUDA_VISIBLE_DEVICES=3 uv run python scripts/frontend_features.py --workers 8
  uv run python scripts/frontend_features.py --cpu-timing 3        # CPU seconds per exam (no caches written)
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import _path  # noqa: F401
from mri_hdc import data as D
from mri_hdc import frontend as FE

CACHE = "mri_feature_cache"
W = "/data/weights/dl_tmp"
PV_CKPT = {s: f"{W}/ProViCNet_hf/{s}_best.pth" for s in ("T2", "ADC", "DWI")}
PF_CKPT = f"{W}/ProFound_weights/profound_alpha_checkpoint-799.pth"
OUTS = {  # name: (channels, grid)
    "provicnet_b7": (1152, 16), "provicnet_b11": (1152, 16), "profound_s1": (96, 24), "profound_s2": (192, 16)}
PV_LAYERS = (7, 11)


def _sample(vol, zi, yi, xi, order=1):
    from scipy.ndimage import map_coordinates
    return map_coordinates(vol, [zi, yi, xi], order=order, mode="constant", cval=0.0).astype(np.float32)


def prep_case(job):
    """CPU part of one exam: read + resample. Returns model inputs and the crop read-out coordinates."""
    key, images_dir, labels_dir, slice_z = job
    import SimpleITK as sitk
    from scipy.ndimage import gaussian_filter
    t0 = time.time()
    pid, sid = key.split("_")
    seqs = {s: os.path.join(images_dir, pid, f"{key}_{s}.mha") for s in ("t2w", "adc", "hbv")}
    t2 = sitk.ReadImage(seqs["t2w"], sitk.sitkFloat32)
    vols = {"t2w": sitk.GetArrayFromImage(t2)}
    for s in ("adc", "hbv"):
        vols[s] = D.on_grid(sitk.ReadImage(seqs[s], sitk.sitkFloat32), t2, False)
    gland = D.on_grid(sitk.ReadImage(os.path.join(labels_dir, D.GLAND_DIR, key + ".nii.gz"), sitk.sitkUInt8),
                      t2, True) > 0
    Z, H, Wd = gland.shape
    sx, sy, sz = t2.GetSpacing()
    y0, y1, x0, x1 = D.square_box(gland, 0.25, gland.shape)
    cy, cx = (y0 + y1) / 2 - 0.5, (x0 + x1) / 2 - 0.5  # box centre, index coordinates
    slice_z = np.asarray(slice_z)

    def crop_pts(G):  # cell centres of a G x G grid over the crop, T2W index coordinates (edge -> centre: -0.5)
        u = (np.arange(G) + 0.5) / G
        return y0 + u * (y1 - y0) - 0.5, x0 + u * (x1 - x0) - 0.5

    # check: the 96 px crop gland from this mapping (nearest)
    py, px = crop_pts(96)
    chk = gland[slice_z][:, np.clip(np.round(py).astype(int), 0, H - 1)][:, :, np.clip(np.round(px).astype(int), 0, Wd - 1)]

    # --- ProViCNet: 0.5 mm in-plane, 256 x 256 around the box centre, native slices
    r = (np.arange(256) - 127.5) * 0.5
    iy, ix = cy + r / sy, cx + r / sx
    zz, yy, xx = np.meshgrid(np.arange(Z), iy, ix, indexing="ij")
    pv_in, g256 = {}, _sample(gland.astype(np.float32), zz, yy, xx, 0) > 0.5
    for s, name in (("t2w", "T2"), ("adc", "ADC"), ("hbv", "DWI")):
        v = _sample(vols[s], zz, yy, xx)
        m = g256 if g256.sum() > 100 else np.ones_like(g256)
        pv_in[name] = ((v - v[m].mean()) / (v[m].std() + 1e-6)).astype(np.float32)
    pv_pts = {}  # crop grid points in normalised 256-grid coordinates (align_corners=False)
    for G in sorted({g for k, (c, g) in OUTS.items() if k.startswith("provicnet")}):
        qy, qx = crop_pts(G)
        pv_pts[G] = (((qy - cy) * sy / 0.5 + 127.5 + 0.5) / 256 * 2 - 1, ((qx - cx) * sx / 0.5 + 127.5 + 0.5) / 256 * 2 - 1)

    # --- ProFound: 1 mm isotropic, 64 x 224 x 224
    cz = float(slice_z.mean())
    rz, rp = np.arange(64) - 31.5, np.arange(224) - 111.5
    zz, yy, xx = np.meshgrid(cz + rz / sz, cy + rp / sy, cx + rp / sx, indexing="ij")
    pf_in = []
    for s in ("t2w", "hbv", "adc"):
        v = gaussian_filter(vols[s], sigma=(0, max(0.0, 0.4 * (1 / sy)), max(0.0, 0.4 * (1 / sx))))
        v = _sample(v, zz, yy, xx)
        nz = v != 0
        v[nz] = (v[nz] - v[nz].mean()) / (v[nz].std() + 1e-8)
        pf_in.append(v)
    pf_pts = {}
    for G in sorted({g for k, (c, g) in OUTS.items() if k.startswith("profound")}):
        qy, qx = crop_pts(G)
        pf_pts[G] = (((slice_z - cz) * sz + 31.5 + 0.5) / 64 * 2 - 1,
                     ((qy - cy) * sy + 111.5 + 0.5) / 224 * 2 - 1, ((qx - cx) * sx + 111.5 + 0.5) / 224 * 2 - 1)
    return dict(key=key, slice_z=slice_z, Z=Z, pv_in=pv_in, pv_pts=pv_pts, pf_in=np.stack(pf_in), pf_pts=pf_pts,
                chk=chk, cpu_s=time.time() - t0)


def run_models(c, pv, pf, device):
    """GPU part: returns {name: (n_kept, C, G, G) float16} and the seconds per model."""
    import torch
    import torch.nn.functional as F
    out, t = {}, {}
    sync = (lambda: torch.cuda.synchronize()) if device.startswith("cuda") else (lambda: None)
    # ProViCNet
    sync(); t0 = time.time()
    Z, zs = c["Z"], c["slice_z"]
    feats = {L: [] for L in PV_LAYERS}
    with torch.no_grad():
        for name in ("T2", "ADC", "DWI"):
            v = torch.from_numpy(c["pv_in"][name]).to(device)
            v = F.interpolate(v[:, None], size=(448, 448), mode="bilinear", align_corners=False)[:, 0]
            S = Z
            if S < 6:
                v = F.pad(v, (0, 0, 0, 0, 1, 1)); S += 2
            js = zs + (1 if Z < 6 else 0)
            cc = np.clip(js, 1, S - 2)
            x = torch.stack([v[k - 1:k + 2] for k in cc])
            pos = torch.tensor(np.stack([js, np.full(len(js), S)], 1), dtype=torch.float32, device=device)
            fs = pv[name](x, pos, layers=PV_LAYERS)
            for L, f in zip(PV_LAYERS, fs):
                feats[L].append(f)
        for L in PV_LAYERS:
            f = torch.cat(feats[L], 1)  # (n, 1152, 32, 32)
            G = OUTS[f"provicnet_b{L}"][1]
            gy, gx = (torch.tensor(a, dtype=torch.float32, device=device) for a in c["pv_pts"][G])
            grid = torch.stack(torch.meshgrid(gx, gy, indexing="xy"), -1)  # (G, G, 2) = (x, y), rows = y
            grid = grid[None].expand(len(zs), -1, -1, -1)
            out[f"provicnet_b{L}"] = F.grid_sample(f, grid, mode="bilinear", align_corners=False).half().cpu().numpy()
    sync(); t["provicnet"] = time.time() - t0
    # ProFound
    t0 = time.time()
    with torch.no_grad():
        s = pf(torch.from_numpy(c["pf_in"])[None].to(device))
        for k, f in (("profound_s1", s[0]), ("profound_s2", s[1])):
            G = OUTS[k][1]
            pz, py, px = c["pf_pts"][G]
            n = len(pz)
            Z_ = torch.tensor(pz, dtype=torch.float32, device=device)[:, None, None].expand(n, G, G)
            Y_ = torch.tensor(py, dtype=torch.float32, device=device)[None, :, None].expand(n, G, G)
            X_ = torch.tensor(px, dtype=torch.float32, device=device)[None, None, :].expand(n, G, G)
            grid = torch.stack([X_, Y_, Z_], -1)[None]  # (1, n, G, G, 3)
            o = F.grid_sample(f, grid, mode="bilinear", align_corners=False)[0]  # (C, n, G, G)
            out[k] = o.permute(1, 0, 2, 3).half().cpu().numpy()
    sync(); t["profound"] = time.time() - t0
    return out, t


def load_models(device):
    import torch
    from mri_hdc.encoders.profound import load_profound
    from mri_hdc.encoders.provicnet import ProViCNetEncoder
    torch.backends.cudnn.benchmark = False
    pv = {s: ProViCNetEncoder(p).to(device).eval() for s, p in PV_CKPT.items()}
    return pv, load_profound(PF_CKPT).to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="/data/picai_all_masks.npz")
    ap.add_argument("--images", default="/data/picai_images")
    ap.add_argument("--labels", default="/data/picai_labels")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cpu-timing", type=int, default=0, help="time N exams on CPU only; writes cpu_timing.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all-folds", action="store_true",
                    help="all five folds, rows = the full slice file (for the full pipeline); writes fe_<name>_all.npy")
    ap.add_argument("--only", default="", help="comma-separated outputs to write (default: all)")
    args = ap.parse_args()
    import torch
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    d = D.load_slices(args.slices)
    d = d if args.all_folds else FE.drop_fold0(d)
    keep = [k for k in OUTS if not args.only or k in args.only.split(",")]
    sfx = "_all" if args.all_folds else ""
    del d["images"], d["lesion_mask"], d["zone_mask"]
    N = len(d["fold"])
    sid, pid = np.asarray(d["study_id"]), np.asarray(d["patient_id"])
    keys = []
    for k in np.unique(sid):
        rows = np.where(sid == k)[0]
        rows = rows[np.argsort(d["slice_z"][rows])]
        keys.append((f"{pid[rows[0]]}_{k}", rows))
    if args.limit:
        keys = keys[:args.limit]
    if args.cpu_timing:
        torch.set_num_threads(16)
        pv, pf = load_models("cpu")
        rec = []
        for key, rows in keys[:args.cpu_timing]:
            c = prep_case((key, args.images, args.labels, d["slice_z"][rows]))
            _, t = run_models(c, pv, pf, "cpu")
            rec.append({"key": key, "prep_s": c["cpu_s"], **{f"{k}_s": v for k, v in t.items()}, "n_slices": len(rows)})
            log(rec[-1])
        json.dump({"threads": 16, "exams": rec,
                   "mean": {k: float(np.mean([r[k] for r in rec])) for k in rec[0] if k.endswith("_s")}},
                  open(f"{CACHE}/fe_cpu_timing.json", "w"), indent=1)
        return

    paths = {k: f"{CACHE}/fe_{k}{sfx}.npy" for k in keep}
    mm = {k: np.lib.format.open_memmap(paths[k] + ".tmp.npy", mode="w+", dtype=np.float16, shape=(N, c, g, g))
          for k, (c, g) in OUTS.items() if k in keep}
    pool = Pool(args.workers)  # fork before CUDA is initialised
    pv, pf = load_models(args.device)
    jobs = [(key, args.images, args.labels, d["slice_z"][rows]) for key, rows in keys]
    rowsof = dict(keys)
    T = {"provicnet": 0.0, "profound": 0.0, "prep": 0.0}
    agree, t_all = [], time.time()
    if True:
        for n, c in enumerate(pool.imap(prep_case, jobs, chunksize=1)):
            rows = rowsof[c["key"]]
            g = d["gland_mask"][rows] > 0
            agree.append(float((g == (c["chk"] > 0)).mean()))
            out, t = run_models(c, pv, pf, args.device)
            for k, v in out.items():
                if k in mm:
                    mm[k][rows] = v
            T["provicnet"] += t["provicnet"]; T["profound"] += t["profound"]; T["prep"] += c["cpu_s"]
            if n % 50 == 0 or agree[-1] < 0.97:
                log(f"{n + 1}/{len(jobs)} {c['key']}: crop gland agreement {agree[-1]:.3f}, "
                    f"gpu pv {t['provicnet']:.2f}s pf {t['profound']:.2f}s, prep {c['cpu_s']:.1f}s")
    pool.close()
    ne = len(jobs)
    if args.limit:  # smoke test: keep nothing
        for k in keep:
            del mm[k]
            os.remove(paths[k] + ".tmp.npy")
        log(f"limit run ok: {T}, agreement {agree}")
        return
    for k in keep:
        mm[k].flush()
        del mm[k]
        os.replace(paths[k] + ".tmp.npy", paths[k])
        model = k.split("_")[0]
        json.dump({"source": k, "map_shape": list(OUTS[k]), "gpu_s_per_exam": T[model] / ne,
                   "prep_cpu_s_per_exam": T["prep"] / ne, "n_exams": ne, "rows": "all" if args.all_folds else "drop_fold0",
                   "crop_gland_agreement_min": min(agree), "crop_gland_agreement_mean": float(np.mean(agree)),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
                  open(paths[k] + ".json", "w"), indent=1)
    log(f"done {ne} exams in {time.time() - t_all:.0f}s; gland agreement mean {np.mean(agree):.4f} min {min(agree):.4f}")


if __name__ == "__main__":
    main()
