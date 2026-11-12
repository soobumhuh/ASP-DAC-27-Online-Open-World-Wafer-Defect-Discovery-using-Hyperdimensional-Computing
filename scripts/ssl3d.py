#!/usr/bin/env python3
"""Self-supervised 3D front end for the HDC detector (HDnn-style: learned features in front, HDC behind).

Encoder: a small 3D CNN (4 conv layers, 32-128 channels, about 7 GFLOPs per exam) reads a 5-slice stack (slices z-2 .. z+2, edge slices repeated) of the 3 sequences
(T2W, ADC, HBV; 3 x 5 x 96 x 96) and outputs a 128-channel feature map of the centre slice at 1/4 resolution
(24 x 24). It is trained WITHOUT labels by masked reconstruction: 60% of 8 x 8 px in-plane blocks are blanked in all
5 slices, and a light decoder must reconstruct the full stack; the loss is the mean squared error on the blanked
pixels (pixels inside the gland count twice). Only the slices of the training folds are used (--exclude_folds keeps a
test fold unseen, e.g. fold 0). Afterwards the encoder is frozen and run once on every slice; the cache
(N, 128, 24, 24) float16 replaces the ResNet-18 layer-2 maps (model.features: ssl3d, model.feature_path).

  CUDA_VISIBLE_DEVICES=3 uv run python scripts/ssl3d.py --exclude_folds 0 --out mri_feature_cache/ssl3d_ex0
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import _path  # noqa: F401
from mri_hdc import data as D, features as Fe


def block(cin, cout, k, stride=1, pad=1):
    return nn.Sequential(nn.Conv3d(cin, cout, k, stride, pad, bias=False), nn.GroupNorm(8, cout), nn.GELU())


class Encoder(nn.Module):
    def __init__(self, c=128):
        super().__init__()
        # about 0.24 GMAC per slice (about 7 GFLOPs per 15-slice exam): early in-plane downsampling keeps it CPU-cheap
        self.net = nn.Sequential(
            block(3, 32, 3, (1, 2, 2), 1),                # 5 x 48 x 48
            block(32, 64, 3, (1, 2, 2), (0, 1, 1)),       # 3 x 24 x 24
            block(64, 64, 3, 1, (0, 1, 1)),               # 1 x 24 x 24
        )
        self.head = nn.Sequential(nn.Conv2d(64, c, 3, 1, 1), nn.GroupNorm(8, c), nn.GELU(), nn.Conv2d(c, c, 1))

    def forward(self, x):  # x: (B, 3, 5, 96, 96)
        return self.head(self.net(x).squeeze(2))      # (B, c, 24, 24)


class Decoder(nn.Module):
    def __init__(self, c=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(c, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.Conv2d(32, 15, 3, 1, 1))

    def forward(self, f):
        return self.net(f).view(-1, 3, 5, 96, 96)


def stacks(images, idx, nb):
    """(B, 3, 5, 96, 96) float32: slices z-2 .. z+2 of each sequence (neighbours via the exam's slice order)."""
    prev, nxt = nb
    p1, n1 = prev[idx], nxt[idx]
    order = [prev[p1], p1, idx, n1, nxt[n1]]
    return np.stack([images[o] for o in order], 2).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="/data/picai_all_masks.npz")
    ap.add_argument("--exclude_folds", default="0", help="comma list of public folds never used for training")
    ap.add_argument("--out", required=True, help="output prefix: <out>.pt (encoder), <out>.npy (feature cache)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mask_ratio", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = D.load_slices(args.slices)
    images, gland = d["images"], d["gland_mask"]
    nb = Fe.neighbour_slices(d)
    excl = {int(f) for f in args.exclude_folds.split(",") if f != ""}
    train = np.where(~np.isin(d["fold"], list(excl)))[0]
    print(f"training slices {len(train)} (excluded folds {sorted(excl)}), device {dev}", flush=True)
    enc, dec = Encoder().to(dev), Decoder().to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * (len(train) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    t0, step, log = time.time(), 0, []
    for ep in range(args.epochs):
        perm = rng.permutation(train)
        tot = 0.0
        for b0 in range(0, len(perm) - args.batch + 1, args.batch):
            j = perm[b0:b0 + args.batch]
            x = torch.from_numpy(stacks(images, j, nb)).to(dev)
            if rng.random() < 0.5:
                x = x.flip(-1)  # left-right flip
                g = torch.from_numpy(gland[j][..., ::-1].copy()).to(dev)
            else:
                g = torch.from_numpy(np.ascontiguousarray(gland[j])).to(dev)
            m = (torch.rand(len(j), 1, 1, 12, 12, device=dev) < args.mask_ratio).float()
            m = F.interpolate(m, size=(1, 96, 96), mode="nearest").expand(-1, 1, 5, -1, -1)  # 8 x 8 blocks, all slices
            rec = dec(enc(x * (1 - m)))
            w = m * (1 + (g[:, None, None] > 0).float())
            loss = ((rec - x) ** 2 * w).sum() / (w.sum() * 3 + 1e-6)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
            step += 1
        n = len(perm) // args.batch
        log.append({"epoch": ep, "loss": tot / n, "minutes": (time.time() - t0) / 60})
        print(json.dumps(log[-1]), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(enc.state_dict(), args.out + ".pt")
    # frozen encoder on every slice -> feature cache
    enc.eval()
    out = np.lib.format.open_memmap(args.out + ".tmp.npy", mode="w+", dtype=np.float16, shape=(len(images), 128, 24, 24))
    with torch.no_grad():
        for b0 in range(0, len(images), 128):
            j = np.arange(b0, min(b0 + 128, len(images)))
            out[j] = enc(torch.from_numpy(stacks(images, j, nb)).to(dev)).cpu().numpy().astype(np.float16)
    out.flush()
    del out
    os.replace(args.out + ".tmp.npy", args.out + ".npy")
    json.dump({"args": vars(args), "log": log, "params": sum(p.numel() for p in enc.parameters())},
              open(args.out + ".json", "w"), indent=2)
    print("saved", args.out + ".npy", flush=True)


if __name__ == "__main__":
    main()
