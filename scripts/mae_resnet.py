#!/usr/bin/env python3
"""Idea 6: in-domain self-supervised front end. The frozen ImageNet ResNet-18 (conv1 .. layer2, the layers the HDC
detector reads) is further pretrained on PI-CAI slices WITHOUT labels by masked image modelling (SimMIM style):
60% of 16 x 16 blocks of the 224 px input (T2W, ADC, HBV as R, G, B, as in features.resnet_l2_maps) are set to zero,
a light decoder (1 x 1 conv + pixel shuffle) reconstructs the image from the layer-2 map (28 x 28), and the loss is
the L1 error on the masked pixels (gland pixels count twice). Only slices of the training folds are used
(--exclude_folds: the outer test fold), so per outer fold f an encoder that never saw fold f is made; labels are
never read. Afterwards the encoder is frozen and run once on every slice exactly like resnet_l2_maps (layer 2,
3 x 3 average-pooled) -> (N, 128, 28, 28) float16 cache, used through model.features: external with
model.feature_path = '..._ex{f}.npy' (train_predict.py picks the encoder of the split's outer fold).

  CUDA_VISIBLE_DEVICES=3 uv run python scripts/mae_resnet.py --exclude_folds 0 --out /data/sota/mae/mae_ex0
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
from mri_hdc import data as D


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2 = m.layer1, m.layer2

    def forward(self, x):
        return self.layer2(self.layer1(self.stem(x)))  # (B, 128, 28, 28)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="/data/picai_all_masks.npz")
    ap.add_argument("--exclude_folds", default="0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4, help="encoder lr (decoder 10x)")
    ap.add_argument("--mask_ratio", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"
    d = D.load_slices(a.slices)
    images, gland = d["images"], d["gland_mask"]
    excl = [int(f) for f in a.exclude_folds.split(",") if f != ""]
    train = np.where(~np.isin(d["fold"], excl))[0]
    print(f"training slices {len(train)} (excluded folds {excl})", flush=True)
    enc = Enc().to(dev)
    dec = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.GELU(), nn.Conv2d(256, 3 * 64, 1), nn.PixelShuffle(8)).to(dev)
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": a.lr}, {"params": dec.parameters(), "lr": 10 * a.lr}],
                            weight_decay=0.05)
    steps = a.epochs * (len(train) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr, 10 * a.lr], total_steps=steps, pct_start=0.05)
    t0, log = time.time(), []
    for ep in range(a.epochs):
        enc.train()
        perm = rng.permutation(train)
        tot = 0.0
        for b0 in range(0, len(perm) - a.batch + 1, a.batch):
            j = np.sort(perm[b0:b0 + a.batch])
            x = torch.from_numpy(np.asarray(images[j], np.float32)).to(dev)
            g = torch.from_numpy(np.asarray(gland[j], np.float32)).to(dev)[:, None]
            if rng.random() < 0.5:
                x, g = x.flip(-1), g.flip(-1)
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
            g = F.interpolate(g, size=224, mode="nearest")
            m = (torch.rand(len(j), 1, 14, 14, device=dev) < a.mask_ratio).float()
            m = F.interpolate(m, size=224, mode="nearest")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rec = dec(enc(x * (1 - m)))
            w = m * (1 + g)
            loss = ((rec.float() - x).abs() * w).sum() / (w.sum() * 3 + 1e-6)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        log.append({"epoch": ep, "loss": tot / (len(perm) // a.batch), "minutes": (time.time() - t0) / 60})
        print(json.dumps(log[-1]), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(enc.state_dict(), a.out + ".pt")
    enc.eval()
    out = np.lib.format.open_memmap(a.out + ".tmp.npy", mode="w+", dtype=np.float16, shape=(len(images), 128, 28, 28))
    with torch.no_grad():
        for b0 in range(0, len(images), 256):
            j = np.arange(b0, min(b0 + 256, len(images)))
            x = F.interpolate(torch.from_numpy(np.asarray(images[j], np.float32)).to(dev), size=224, mode="bilinear",
                              align_corners=False)
            out[j] = F.avg_pool2d(enc(x), 3, 1, 1).cpu().numpy().astype(np.float16)
    out.flush()
    del out
    os.replace(a.out + ".tmp.npy", a.out + ".npy")
    json.dump({"args": vars(a), "log": log}, open(a.out + ".json", "w"), indent=2)
    print("saved", a.out + ".npy", flush=True)


if __name__ == "__main__":
    main()
