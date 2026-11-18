#!/usr/bin/env python3
"""Per-slice feature maps of Triad (3D MRI foundation model, PlainConvUNet encoder, MAE on 131k 3D MRI from 36
datasets; github.com/wangshansong1/Triad, MIT), on the 96 px gland crops of the slice file.

Per exam and sequence (T2W, ADC, HBV separately, Triad is single-channel): the crop stack (n, 96, 96), z-scored
over the gland, slices up-sampled x3 (about 3 mm -> 1 mm, closer to the isotropic pre-training data), padded to a
multiple of 32 slices; encoder stages 2 (stride 4, 128 ch) and 3 (stride 8, 256 ch) read back at every original
slice (trilinear) on 24 x 24 / 12 x 12 grids; the three sequences concatenated -> 384 / 768 channels.

  CUDA_VISIBLE_DEVICES=3 python scripts/fe_triad.py --ckpt /data/auroc/weights/Triad-PlainConvUNet-MAE.pth
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def triad_encoder(ckpt):
    from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
    from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            conv = convert_dim_to_conv_op(3)
            self.encoder = PlainConvEncoder(1, 6, [min(32 * 2 ** i, 320) for i in range(6)], conv, [[3, 3, 3]] * 6,
                                            [[1, 1, 1]] + [[2, 2, 2]] * 5, [2] * 6, True,
                                            get_matching_instancenorm(conv), {"eps": 1e-5, "affine": True}, None,
                                            None, nn.LeakyReLU, {"inplace": True}, return_skips=True)
    m = Wrap()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if k.startswith("encoder.")], missing[:5]
    return m.encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/data/auroc/weights/Triad-PlainConvUNet-MAE.pth")
    ap.add_argument("--name", default="triad_mae")
    ap.add_argument("--slices", default="/data/picai_all_masks.npz")
    ap.add_argument("--out", default="/data/auroc/fe")
    ap.add_argument("--up", type=int, default=3)
    ap.add_argument("--stages", default="2,3")
    a = ap.parse_args()
    dev = "cuda"
    d = np.load(a.slices)
    imgs, gland, sid, z = d["images"], d["gland_mask"], d["study_id"], d["slice_z"]
    enc = triad_encoder(a.ckpt).to(dev).eval()
    N = len(sid)
    outs = {int(s): (3 * min(32 * 2 ** int(s), 320), 96 // 2 ** int(s)) for s in a.stages.split(",")}
    mm = {s: np.lib.format.open_memmap(os.path.join(a.out, f"fe_{a.name}_s{s}_all.npy.tmp.npy"), "w+", np.float16,
                                       (N, c, g, g)) for s, (c, g) in outs.items()}
    t0 = time.time()
    keys = np.unique(sid)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ci, k in enumerate(keys):
            rows = np.where(sid == k)[0]
            rows = rows[np.argsort(z[rows])]
            n = len(rows)
            feats = {s: [] for s in outs}
            g = gland[rows] > 0
            for ch in range(3):
                v = imgs[rows, ch].astype(np.float32)
                ref = v[g] if g.sum() > 50 else v.ravel()
                v = (v - ref.mean()) / (ref.std() + 1e-6)
                x = torch.from_numpy(v).to(dev)[None, None]                     # (1, 1, n, 96, 96)
                Zu = n * a.up
                x = F.interpolate(x, size=(Zu, 96, 96), mode="trilinear", align_corners=False)
                Zp = int(np.ceil(Zu / 32) * 32)
                x = F.pad(x, (0, 0, 0, 0, 0, Zp - Zu))
                sk = enc(x)
                for s, (c, gg) in outs.items():
                    f = sk[s].float()[:, :, : int(np.ceil(Zu / 2 ** s))]       # drop padded depth
                    f = F.interpolate(f, size=(Zu, gg, gg), mode="trilinear", align_corners=False)
                    feats[s].append(f[0, :, a.up // 2::a.up][:, :n])             # centre of each original slice
            for s in outs:
                mm[s][rows] = torch.cat(feats[s], 0).permute(1, 0, 2, 3).cpu().numpy().astype(np.float16)
            if ci % 100 == 0:
                print(f"{ci}/{len(keys)} {time.time() - t0:.0f}s", flush=True)
    for s, (c, g) in outs.items():
        mm[s].flush()
        dst = os.path.join(a.out, f"fe_{a.name}_s{s}_all.npy")
        os.replace(dst + ".tmp.npy", dst)
        json.dump({"source": a.name, "stage": s, "map_shape": [c, g], "rows": "all", "ckpt": a.ckpt,
                   "input": f"96 px crop stack per sequence, gland z-score, z x{a.up}", "seconds": time.time() - t0},
                  open(dst + ".json", "w"), indent=1)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
