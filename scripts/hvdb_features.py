#!/usr/bin/env python3
"""Per-channel frozen ResNet-18 layer-2 maps for the hypervector database (CPU).

The cached front end (mri_feature_cache/resnet_l2.npy) feeds T2W / ADC / HBV together as R / G / B. For per-channel
region descriptors each sequence is fed alone (the channel copied to R, G and B), through the same frozen ImageNet
ResNet-18 up to layer 2 with the same 3 x 3 average pooling (mri_hdc.features.resnet_l2_maps).
Output: (N, 3, 128, 28, 28) float16 memmap, channel order T2W, ADC, HBV.

  python scripts/hvdb_features.py --slices /data/picai_all_masks.npz --out /data/hvdb/fe/resnet_l2_perchan.npy
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

ap = argparse.ArgumentParser()
ap.add_argument("--slices", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--threads", type=int, default=32)
ap.add_argument("--batch", type=int, default=96)
a = ap.parse_args()
torch.set_num_threads(a.threads)
imgs = np.load(a.slices, mmap_mode="r")["images"]
N = len(imgs)
m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
tmp = a.out + ".tmp.npy"
out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(N, 3, 128, 28, 28))
with torch.no_grad():
    for i in range(0, N, a.batch):
        x = torch.from_numpy(np.asarray(imgs[i:i + a.batch], dtype=np.float32))  # (b, 3, 96, 96)
        b = x.shape[0]
        x = x.reshape(b * 3, 1, 96, 96).expand(-1, 3, -1, -1)
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        f = F.avg_pool2d(m.layer2(m.layer1(x)), 3, 1, 1)
        out[i:i + b] = f.reshape(b, 3, 128, 28, 28).numpy().astype(np.float16)
        if i % (a.batch * 20) == 0:
            print(f"{i}/{N}", flush=True)
out.flush()
del out
os.replace(tmp, a.out)
print("done", a.out)
