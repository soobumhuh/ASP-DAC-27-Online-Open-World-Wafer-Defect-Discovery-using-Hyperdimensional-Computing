#!/usr/bin/env python3
"""Per-slice feature maps of SAM-Med3D-turbo (3D SAM ViT-B image encoder, 128^3 input at ~1.5 mm, trained on
SA-Med3D-140K incl. MRI; github.com/uni-medical/SAM-Med3D, HF blueyo0/SAM-Med3D), on the 96 px gland crops.

Per exam and sequence (single-channel model): crop stack (n, 96, 96) z-scored over the gland, resized to
(2n, 64, 64) (~1.5 mm voxels), centred in a zero 128^3 volume; encoder neck output (384 ch, 8^3 tokens) read back
trilinearly at every original slice on a 12 x 12 grid over the crop; three sequences concatenated -> 1152 ch.
Encoder source (image_encoder3D.py, common.py) fetched from the SAM-Med3D repository into /data/auroc/ext/samed3d.
"""
import argparse
import glob
import json
import os
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=glob.glob("/data/auroc/hf/hub/models--blueyo0--SAM-Med3D/snapshots/*/sam_med3d_turbo.pth")[0])
    ap.add_argument("--src", default="/data/auroc/ext")
    ap.add_argument("--slices", default="/data/picai_all_masks.npz")
    ap.add_argument("--out", default="/data/auroc/fe")
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    import importlib
    common = importlib.import_module("samed3d.common")
    sys.modules["common"] = common
    src = open(os.path.join(a.src, "samed3d", "image_encoder3D.py")).read().replace("from .common", "from samed3d.common")
    mod = type(sys)("samed3d.image_encoder3D")
    exec(compile(src, "image_encoder3D.py", "exec"), mod.__dict__)
    enc = mod.ImageEncoderViT3D(depth=12, embed_dim=768, img_size=128, mlp_ratio=4,
                                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), num_heads=12, patch_size=16,
                                qkv_bias=True, use_rel_pos=True, global_attn_indexes=[2, 5, 8, 11], window_size=14,
                                out_chans=384)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    enc.load_state_dict({k[len("image_encoder."):]: v for k, v in sd.items() if k.startswith("image_encoder.")})
    dev = "cuda"
    enc = enc.to(dev).eval()
    d = np.load(a.slices)
    imgs, gland, sid, z = d["images"], d["gland_mask"], d["study_id"], d["slice_z"]
    N, G = len(sid), 12
    mm = np.lib.format.open_memmap(os.path.join(a.out, "fe_samed3d_all.npy.tmp.npy"), "w+", np.float16, (N, 1152, G, G))
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ci, k in enumerate(np.unique(sid)):
            rows = np.where(sid == k)[0]
            rows = rows[np.argsort(z[rows])]
            n = len(rows)
            g = gland[rows] > 0
            Zs = min(2 * n, 128)
            z0 = (128 - Zs) // 2
            fs = []
            for ch in range(3):
                v = imgs[rows, ch].astype(np.float32)
                ref = v[g] if g.sum() > 50 else v.ravel()
                v = (v - ref.mean()) / (ref.std() + 1e-6)
                x = F.interpolate(torch.from_numpy(v).to(dev)[None, None], size=(Zs, 64, 64), mode="trilinear",
                                  align_corners=False)
                vol = torch.zeros(1, 1, 128, 128, 128, device=dev)
                vol[:, :, z0:z0 + Zs, 32:96, 32:96] = x
                f = enc(vol).float()                                         # (1, 384, 8, 8, 8)
                # sample at original slice centres / crop grid (normalised coords, align_corners=False)
                zc = (z0 + (np.arange(n) + 0.5) * Zs / n) / 128 * 2 - 1
                u = (32 + (np.arange(G) + 0.5) * 64 / G) / 128 * 2 - 1
                Zg, Yg, Xg = np.meshgrid(zc, u, u, indexing="ij")
                grid = torch.tensor(np.stack([Xg, Yg, Zg], -1)[None], dtype=torch.float32, device=dev)
                fs.append(F.grid_sample(f, grid, mode="bilinear", align_corners=False)[0])   # (384, n, G, G)
            mm[rows] = torch.cat(fs, 0).permute(1, 0, 2, 3).cpu().numpy().astype(np.float16)
            if ci % 100 == 0:
                print(f"{ci} {time.time() - t0:.0f}s", flush=True)
    mm.flush()
    del mm
    dst = os.path.join(a.out, "fe_samed3d_all.npy")
    os.replace(dst + ".tmp.npy", dst)
    json.dump({"source": "samed3d_turbo", "map_shape": [1152, G], "rows": "all"}, open(dst + ".json", "w"), indent=1)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
