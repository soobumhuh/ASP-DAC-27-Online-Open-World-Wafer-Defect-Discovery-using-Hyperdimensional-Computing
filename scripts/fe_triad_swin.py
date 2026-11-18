#!/usr/bin/env python3
"""Per-slice feature maps of Triad Swin-B (MONAI SwinViT v2, feature size 48, MAE pre-training on 131k 3D MRI;
github.com/wangshansong1/Triad), on the 96 px gland crops. Input per exam and sequence as scripts/fe_triad.py
(crop stack, gland z-score, slices x3, depth padded to a multiple of 32); hidden states 2 (stride 8, 192 ch) and
3 (stride 16, 384 ch) read back at every original slice; three sequences concatenated.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/data/auroc/weights/Triad-SwinB-MAE.pth")
    ap.add_argument("--name", default="triad_swin")
    ap.add_argument("--slices", default="/data/picai_all_masks.npz")
    ap.add_argument("--out", default="/data/auroc/fe")
    ap.add_argument("--up", type=int, default=3)
    a = ap.parse_args()
    from monai.networks.nets.swin_unetr import SwinTransformer
    m = SwinTransformer(in_chans=1, embed_dim=48, window_size=(7, 7, 7), patch_size=(2, 2, 2), depths=[2, 2, 2, 2],
                        num_heads=[3, 6, 12, 24], mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
                        drop_path_rate=0.0, norm_layer=torch.nn.LayerNorm, use_checkpoint=False, spatial_dims=3,
                        use_v2=True)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = {k[len("backbone.swinViT."):]: v for k, v in sd.items() if k.startswith("backbone.swinViT.")}
    m.load_state_dict(sd, strict=True)
    dev = "cuda"
    m = m.to(dev).eval()
    d = np.load(a.slices)
    imgs, gland, sid, z = d["images"], d["gland_mask"], d["study_id"], d["slice_z"]
    N = len(sid)
    outs = {2: (576, 12), 3: (1152, 6)}
    mm = {s: np.lib.format.open_memmap(os.path.join(a.out, f"fe_{a.name}_s{s}_all.npy.tmp.npy"), "w+", np.float16,
                                       (N, c, g, g)) for s, (c, g) in outs.items()}
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ci, k in enumerate(np.unique(sid)):
            rows = np.where(sid == k)[0]
            rows = rows[np.argsort(z[rows])]
            n = len(rows)
            g = gland[rows] > 0
            feats = {s: [] for s in outs}
            for ch in range(3):
                v = imgs[rows, ch].astype(np.float32)
                ref = v[g] if g.sum() > 50 else v.ravel()
                v = (v - ref.mean()) / (ref.std() + 1e-6)
                Zu = n * a.up
                x = F.interpolate(torch.from_numpy(v).to(dev)[None, None], size=(Zu, 96, 96), mode="trilinear",
                                  align_corners=False)
                Zp = int(np.ceil(Zu / 32) * 32)
                hs = m(F.pad(x, (0, 0, 0, 0, 0, Zp - Zu)))
                for s, (c, gg) in outs.items():
                    st = 2 ** (s + 1)
                    f = hs[s].float()[:, :, : int(np.ceil(Zu / st))]
                    f = F.interpolate(f, size=(Zu, gg, gg), mode="trilinear", align_corners=False)
                    feats[s].append(f[0, :, a.up // 2::a.up][:, :n])
            for s in outs:
                mm[s][rows] = torch.cat(feats[s], 0).permute(1, 0, 2, 3).cpu().numpy().astype(np.float16)
            if ci % 100 == 0:
                print(f"{ci} {time.time() - t0:.0f}s", flush=True)
    for s, (c, g) in outs.items():
        mm[s].flush()
        dst = os.path.join(a.out, f"fe_{a.name}_s{s}_all.npy")
        os.replace(dst + ".tmp.npy", dst)
        json.dump({"source": a.name, "stage": s, "map_shape": [c, g], "rows": "all"}, open(dst + ".json", "w"))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
