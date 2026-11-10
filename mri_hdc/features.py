"""Patches, position bins and the frozen ResNet-18 layer-2 descriptor."""
from __future__ import annotations

import os

import numpy as np


def patch_grid(S: int, P: int, stride: int, offset: int = 0) -> np.ndarray:
    """Patch-centre coordinates along one axis (patches fully inside the S x S crop)."""
    h = P // 2
    return np.arange(h + offset, S - (P - h) + 1, stride)


def resnet_l2_maps(images: np.ndarray, cache: str = None, batch_size: int = 64, device: str = "cpu") -> np.ndarray:
    """Frozen ImageNet ResNet-18 (torchvision IMAGENET1K_V1): the 3-channel crop (T2W, ADC, HBV as RGB) upsampled to
    224, layer-2 output (128 ch, 28 x 28), 3 x 3 average-pooled for local context (as in PatchCore).
    Returns (N, 128, 28, 28) float16; cached as .npy (memory-mapped when reloaded)."""
    if cache and os.path.exists(cache):
        return np.load(cache, mmap_mode="r")
    import torch
    import torch.nn.functional as F
    from torchvision.models import ResNet18_Weights, resnet18
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
    outs = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            x = torch.from_numpy(np.asarray(images[i:i + batch_size, :3], dtype=np.float32)).to(device)
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
            x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            f = F.avg_pool2d(m.layer2(m.layer1(x)), 3, 1, 1)
            outs.append(f.cpu().numpy().astype(np.float16))
    out = np.concatenate(outs)
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        np.save(cache + ".tmp.npy", out)
        os.replace(cache + ".tmp.npy", cache)
    return out


def extract_patches(d: dict, idx: np.ndarray, P: int, stride: int, fmaps: np.ndarray = None, stats: bool = True):
    """Patches of slices idx whose centre lies in the gland. Descriptor per patch = [feature vector of fmaps at the
    patch centre] + [per-channel mean, per-channel std] (stats). Returns descriptors (n, D) float32, slice index,
    centre y, centre x."""
    S = d["images"].shape[-1]
    cs = patch_grid(S, P, stride)
    h = P // 2
    feats, sl, ys, xs = [], [], [], []
    for i in idx:
        img = d["images"][i].astype(np.float32)
        cy, cx = np.meshgrid(cs, cs, indexing="ij")
        cy, cx = cy.ravel(), cx.ravel()
        ok = d["gland_mask"][i][cy, cx] > 0
        cy, cx = cy[ok], cx[ok]
        if len(cy) == 0:
            continue
        oy = cy[:, None] - h + np.arange(P)[None, :]
        ox = cx[:, None] - h + np.arange(P)[None, :]
        pat = img[:, oy[:, :, None], ox[:, None, :]].transpose(1, 0, 2, 3)  # (n, C, P, P)
        parts = []
        if fmaps is not None:
            fh = fmaps.shape[-1]
            fy = np.clip(((cy + 0.5) * fh / S).astype(int), 0, fh - 1)
            fx = np.clip(((cx + 0.5) * fh / S).astype(int), 0, fh - 1)
            parts.append(np.asarray(fmaps[i], dtype=np.float32)[:, fy, fx].T)
        if stats:
            parts += [pat.mean(axis=(2, 3)), pat.std(axis=(2, 3))]
        feats.append(np.concatenate(parts, axis=1).astype(np.float32))
        sl.append(np.full(len(cy), i)); ys.append(cy); xs.append(cx)
    cat = lambda a: np.concatenate(a).astype(np.int64) if a else np.zeros(0, np.int64)
    F = np.concatenate(feats) if feats else np.zeros((0, 1), np.float32)
    return F, cat(sl), cat(ys), cat(xs)


def n_bins(levels: int) -> int:
    return levels * 27


def position_bins(d: dict, sl, ys, xs, levels: int = 5) -> np.ndarray:
    """Position bin of each patch: slice level (rel_z in `levels` steps) x AI zone at the centre (0 = unlabelled,
    1 = PZ, 2 = TZ) x 3 x 3 in-plane grid cell of the crop."""
    S = d["images"].shape[-1]
    zl = np.minimum((d["rel_z"][sl] * levels).astype(int), levels - 1)
    cell = np.minimum(ys * 3 // S, 2) * 3 + np.minimum(xs * 3 // S, 2)
    return (zl * 3 + d["zone_mask"][sl, ys, xs].astype(np.int64)) * 9 + cell


def bin_zone_group(b):
    """Coarser group (the zone, over all slice levels and grid cells) used to pool lesion patches when a bin has
    too few."""
    return (b // 9) % 3


def patch_fraction(mask: np.ndarray, sl, ys, xs, P: int) -> np.ndarray:
    """Fraction of mask pixels inside each P x P patch (summed-area table per slice)."""
    h = P // 2
    out = np.zeros(len(sl))
    for i in np.unique(sl):
        m = sl == i
        c = np.pad(mask[i].astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        y0, x0 = ys[m] - h, xs[m] - h
        out[m] = (c[y0 + P, x0 + P] - c[y0, x0 + P] - c[y0 + P, x0] + c[y0, x0]) / (P * P)
    return out
