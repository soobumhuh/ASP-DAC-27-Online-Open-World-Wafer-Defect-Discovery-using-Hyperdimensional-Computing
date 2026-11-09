#!/usr/bin/env python3
"""
hdnn_finetune.py - idea 9: HDnn-style front-end fine-tuning on the GPU.

HDnn-PIM (Dutta et al., GLSVLSI'22) puts a few CNN layers in front of an HDC classifier and tunes the CNN through
the HDC part. Here the frozen ImageNet ResNet-18 front end of the #110 descriptor (conv1..layer2, 3x3 average-pooled,
read at each patch centre) is fine-tuned so that its features separate lesion from normal patches *after* HDC
encoding:
  feature   128-d layer-2 vector at the patch centre (+ the 6 fixed intensity stats), standardised with statistics
            of the frozen network (fixed during training)
  encoder   the same random projection as BinaryThermometerHDC (fixed), robust-standardised with the frozen
            network's median / IQR, then a SOFT 4-level thermometer code: bit b = sigmoid((y - t_b) / 0.5) with
            the hard code's thresholds t_b = -2, 0, 2, 4, in bipolar form (2,048 soft bits)
  classifier 20 normal and 5 lesion learnable sub-prototypes; class score = soft maximum of HV . prototype / D
  loss      class-balanced cross-entropy on lesion vs normal patches of the split's TRAINING patients only
            (lesion = centre in an expert lesion of a csPCa-positive patient; normal = gland patch of a csPCa-negative
            patient); early stopping on the split's validation patients
After training, the fine-tuned conv1..layer2 computes 3x3-pooled layer-2 maps for every slice, saved as a feature
cache. The usual (hard, non-differentiable) HDC pipeline then runs unchanged on these features.

  CUDA_VISIBLE_DEVICES=3 uv run python hdnn_finetune.py --split inner1 --out mri_feature_cache/ft_inner1.npy
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import postproc_search as pps

S = 96
TH = torch.tensor([-2.0, 0.0, 2.0, 4.0])  # thermometer thresholds of the hard code (clip 4, 4 levels)


class FrontEnd(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2)

    def forward(self, x):  # x: (B, 3, 96, 96) z-scored crop -> (B, 128, 28, 28) pooled layer-2 map
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        return F.avg_pool2d(self.stem(x), 3, 1, 1)


def cells(cy, cx, fh=28):
    return (torch.clamp(((cy + 0.5) * fh / S).long(), 0, fh - 1), torch.clamp(((cx + 0.5) * fh / S).long(), 0, fh - 1))


def sample_batch(d, slices, rng, n_slices, per_slice, lesion):
    """Patch centres inside the gland; for lesion slices only centres inside the lesion."""
    pick = rng.choice(slices, n_slices, replace=len(slices) < n_slices)
    imgs, bi, ys, xs = [], [], [], []
    for j, s in enumerate(pick):
        g = d["gland_mask"][s] > 0
        if lesion:
            g = g & (d["lesion_mask"][s] > 0)
        yy, xx = np.nonzero(g[10:86, 10:86])
        if len(yy) == 0:
            yy, xx = np.nonzero(d["gland_mask"][s][10:86, 10:86] > 0)
        k = rng.choice(len(yy), min(per_slice, len(yy)), replace=False)
        imgs.append(np.asarray(d["images"][s], dtype=np.float32))
        bi += [j] * len(k); ys += list(yy[k] + 10); xs += list(xx[k] + 10)
    return (torch.from_numpy(np.stack(imgs)), torch.tensor(bi), torch.tensor(ys), torch.tensor(xs), pick)


def patch_stats(img, bi, ys, xs, P=20):
    """Per-channel mean / std of the 20 x 20 patch (fixed, not trained)."""
    h = P // 2
    out = []
    for b, y, x in zip(bi.tolist(), ys.tolist(), xs.tolist()):
        p = img[b, :, y - h:y - h + P, x - h:x - h + P]
        out.append(torch.cat([p.mean((1, 2)), p.std((1, 2))]))
    return torch.stack(out)


class SoftHDC(nn.Module):
    def __init__(self, in_dim, n_proj=512, n_sub=(20, 5), seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed + 12345)
        self.register_buffer("W", torch.randn(in_dim, n_proj, generator=g) / np.sqrt(in_dim))
        self.register_buffer("mu", torch.zeros(in_dim)); self.register_buffer("sd", torch.ones(in_dim))
        self.register_buffer("med", torch.zeros(n_proj)); self.register_buffer("scale", torch.ones(n_proj))
        self.protos = nn.ParameterList([nn.Parameter(0.01 * torch.randn(k, n_proj * 4)) for k in n_sub])

    def calibrate(self, feats):  # statistics of the frozen network, then fixed
        self.mu.copy_(feats.mean(0)); self.sd.copy_(feats.std(0) + 1e-6)
        y = ((feats - self.mu) / self.sd) @ self.W
        self.med.copy_(y.median(0).values)
        q = torch.quantile(y, torch.tensor([0.25, 0.75], device=y.device), dim=0)
        self.scale.copy_(((q[1] - q[0]) / 1.349).clamp_min(1e-3))

    def hv(self, feats):
        y = (((feats - self.mu) / self.sd) @ self.W - self.med) / self.scale
        y = y.clamp(-4, 4)
        bits = torch.sigmoid((y[:, :, None] - TH.to(y.device)) / 0.5)  # (n, 512, 4) soft thermometer
        return (2 * bits - 1).reshape(len(y), -1)

    def logits(self, feats, T=0.05):
        h = self.hv(feats)
        D = h.shape[1]
        s = [T * torch.logsumexp(h @ F.normalize(p, dim=1).T * (1 / np.sqrt(D)) / T, 1) for p in self.protos]
        return torch.stack(s, 1) / T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="inner1 | inner2 | fold0..fold4 (postproc_search.splits)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda"
    z = np.load(pps.NPZ)  # load each array once (npz members are re-read on every access otherwise)
    d = {k: z[k] for k in ("images", "gland_mask", "lesion_mask", "patient_id", "case_cspca")}
    sp = pps.splits()[args.split]
    pid, cs = np.asarray(d["patient_id"]), np.asarray(d["case_cspca"])
    has_les = np.asarray(d["lesion_mask"]).reshape(len(pid), -1).max(1) > 0
    sets = {}
    for part in ("train", "val"):
        inpart = np.isin(pid, sp[part])
        sets[part] = {"neg": np.where(inpart & (cs == 0))[0], "les": np.where(inpart & (cs == 1) & has_les)[0]}
    print({p: {k: len(v) for k, v in s.items()} for p, s in sets.items()}, flush=True)

    net = FrontEnd().to(dev)
    hdc = SoftHDC(128 + 6, seed=args.seed).to(dev)

    def feats_of(batch):
        img, bi, ys, xs, _ = batch
        img = img.to(dev)
        fm = net(img)
        fy, fx = cells(ys, xs)
        f = fm[bi.to(dev), :, fy.to(dev), fx.to(dev)]
        return torch.cat([f, patch_stats(img, bi, ys.to(dev), xs.to(dev))], 1)

    # calibrate standardisation / robust scaling on frozen features of training normals
    with torch.no_grad():
        net.eval()
        cal = torch.cat([feats_of(sample_batch(d, sets["train"]["neg"], rng, 32, 32, False)) for _ in range(20)])
        hdc.calibrate(cal)
    val_batches = [(sample_batch(d, sets["val"]["les"], np.random.default_rng(100 + i), 16, 16, True),
                    sample_batch(d, sets["val"]["neg"], np.random.default_rng(200 + i), 16, 16, False)) for i in range(12)]

    def val_loss():
        net.eval()
        tot, n, acc = 0.0, 0, 0
        with torch.no_grad():
            for bl, bn in val_batches:
                f = torch.cat([feats_of(bl), feats_of(bn)])
                y = torch.cat([torch.ones(len(bl[1])), torch.zeros(len(bn[1]))]).long().to(dev)
                lg = hdc.logits(f)
                w = torch.where(y == 1, 0.5 / (y == 1).sum(), 0.5 / (y == 0).sum())
                tot += float((F.cross_entropy(lg, y, reduction="none") * w).sum()); n += 1
                acc += float(((lg.argmax(1) == y).float() * w).sum())
        net.train()
        return tot / n, acc / n

    opt = torch.optim.Adam([{"params": net.parameters(), "lr": args.lr},
                            {"params": hdc.parameters(), "lr": 1e-2}])
    best, best_state, bad, log = (val_loss()[0], None, 0, [])
    print(f"start val loss {best:.4f}", flush=True)
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        bl = sample_batch(d, sets["train"]["les"], rng, 16, 16, True)
        bn = sample_batch(d, sets["train"]["neg"], rng, 16, 16, False)
        f = torch.cat([feats_of(bl), feats_of(bn)])
        y = torch.cat([torch.ones(len(bl[1])), torch.zeros(len(bn[1]))]).long().to(dev)
        w = torch.where(y == 1, 0.5 / (y == 1).sum(), 0.5 / (y == 0).sum())
        loss = (F.cross_entropy(hdc.logits(f), y, reduction="none") * w).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0:
            vl, va = val_loss()
            log.append({"step": step, "train_loss": float(loss), "val_loss": vl, "val_bal_acc": va, "min": (time.time() - t0) / 60})
            print(log[-1], flush=True)
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
    net.load_state_dict(best_state)
    net.eval()
    # feature cache for every slice (float16, (N, 128, 28, 28)), as patch_anomaly expects
    out = np.lib.format.open_memmap(args.out + ".tmp.npy", mode="w+", dtype=np.float16, shape=(len(pid), 128, 28, 28))
    with torch.no_grad():
        for i in range(0, len(pid), 128):
            x = torch.from_numpy(np.asarray(d["images"][i:i + 128], dtype=np.float32)).to(dev)
            out[i:i + 128] = net(x).cpu().numpy().astype(np.float16)
    out.flush(); del out
    os.replace(args.out + ".tmp.npy", args.out)
    json.dump({"split": args.split, "best_val_loss": best, "log": log, "minutes": (time.time() - t0) / 60},
              open(args.out + ".json", "w"), indent=1)
    print("saved", args.out, f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
