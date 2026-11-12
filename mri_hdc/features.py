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


def neighbour_slices(d: dict):
    """Index of the previous / next kept slice of the same exam (by slice_z); the first / last slice repeats itself."""
    n = len(d["study_id"])
    prev, nxt = np.arange(n), np.arange(n)
    sid, z = np.asarray(d["study_id"]), np.asarray(d["slice_z"])
    order = np.lexsort((z, sid))
    for a, b in zip(order[:-1], order[1:]):
        if sid[a] == sid[b]:
            nxt[a], prev[b] = b, a
    return prev, nxt


def resnet_l2_slices_maps(images: np.ndarray, prev: np.ndarray, nxt: np.ndarray, cache: str, batch_size: int = 64,
                          device: str = "cpu") -> np.ndarray:
    """2.5D variant of resnet_l2_maps: for each sequence (T2W, ADC, HBV) the slices z-1, z, z+1 go in as R, G, B;
    the three layer-2 maps are concatenated -> (N, 384, 28, 28) float16."""
    if cache and os.path.exists(cache):
        return np.load(cache, mmap_mode="r")
    import torch
    import torch.nn.functional as F
    from torchvision.models import ResNet18_Weights, resnet18
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
    out = np.lib.format.open_memmap(cache + ".tmp.npy", mode="w+", dtype=np.float16, shape=(len(images), 384, 28, 28))
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            j = np.arange(i, min(i + batch_size, len(images)))
            fs = []
            for c in range(3):
                x = np.stack([images[prev[j], c], images[j, c], images[nxt[j], c]], 1).astype(np.float32)
                x = F.interpolate(torch.from_numpy(x).to(device), size=224, mode="bilinear", align_corners=False)
                x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
                fs.append(F.avg_pool2d(m.layer2(m.layer1(x)), 3, 1, 1))
            out[j] = torch.cat(fs, 1).cpu().numpy().astype(np.float16)
    out.flush()
    del out
    os.replace(cache + ".tmp.npy", cache)
    return np.load(cache, mmap_mode="r")


def texture_maps(img: np.ndarray) -> np.ndarray:
    """Handcrafted texture per channel: gradient magnitude, |Laplacian| and a 10-bin uniform LBP (P = 8, R = 1)
    one-hot -> (3 x 12, S, S); averaged over each patch window in extract_patches (ablation H2)."""
    from scipy import ndimage
    from skimage.feature import local_binary_pattern
    out = []
    for ch in img:
        gy, gx = np.gradient(ch)
        lbp = local_binary_pattern(ch, 8, 1, method="uniform").astype(int)  # values 0 .. 9
        out += [np.hypot(gy, gx), np.abs(ndimage.laplace(ch))] + [(lbp == v).astype(np.float32) for v in range(10)]
    return np.stack(out).astype(np.float32)


def extract_patches(d: dict, idx: np.ndarray, P: int, stride: int, fmaps: np.ndarray = None, stats: bool = True,
                    context: tuple = None, pix: int = 0, tex: bool = False):
    """Patches of slices idx whose centre lies in the gland. Descriptor per patch = [feature vector of fmaps at the
    patch centre] + [per-channel mean, per-channel std] (stats). With context = (prev, next) slice indices (2.5D),
    the same descriptor of the slices above and below at the same position is appended (3 x the length).
    Returns descriptors (n, D) float32, slice index, centre y, centre x."""
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
        parts = []
        for j in ([i] if context is None else [i, context[0][i], context[1][i]]):
            im = img if j == i else d["images"][j].astype(np.float32)
            for fm in ([] if fmaps is None else fmaps if isinstance(fmaps, (list, tuple)) else [fmaps]):
                fh = fm.shape[-1]  # several feature maps may have different grids (e.g. 28 x 28 and 24 x 24)
                fy = np.clip(((cy + 0.5) * fh / S).astype(int), 0, fh - 1)
                fx = np.clip(((cx + 0.5) * fh / S).astype(int), 0, fh - 1)
                parts.append(np.asarray(fm[j], dtype=np.float32)[:, fy, fx].T)
            pat = im[:, oy[:, :, None], ox[:, None, :]].transpose(1, 0, 2, 3)  # (n, C, P, P)
            if pix:  # ablation H1: raw patch intensities, mean-pooled to pix x pix
                k = P // pix
                parts.append(pat[:, :, :pix * k, :pix * k].reshape(len(cy), -1, pix, k, pix, k).mean((3, 5))
                             .reshape(len(cy), -1))
            if tex:  # ablation H2: window means of handcrafted texture maps
                tm = texture_maps(im)
                parts.append(tm[:, oy[:, :, None], ox[:, None, :]].mean(axis=(2, 3)).T)
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
