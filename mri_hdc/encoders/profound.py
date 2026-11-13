"""Self-contained ProFound-alpha (pipiwang/ProFound, demo branch) dense 3D ConvNeXtV2-Tiny encoder.

Mirrors models/convnextv2.py (ConvNeXtV2, Block) + models/util.py (LayerNorm, GRN) and the key remap
models/convnextv2.py::remap_checkpoint_keys (MinkowskiEngine sparse FCMAE ckpt -> dense Conv3d).
Input: (B, 3, D, H, W) with channels [T2w, high-b DWI, ADC], each z-scored over nonzero voxels.
Downsampling: stem /4, then /2, /2, /1  -> strides 4, 8, 16, 16 (in all three dims).
"""
import argparse, torch, torch.nn as nn, torch.nn.functional as F
from collections import OrderedDict


class LayerNormCF(nn.Module):  # channels_first LayerNorm over dim 1
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d)); self.bias = nn.Parameter(torch.zeros(d)); self.eps = eps
    def forward(self, x):
        u = x.mean(1, keepdim=True); s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None, None] * x + self.bias[:, None, None, None]


class GRN(nn.Module):
    def __init__(self, d):
        super().__init__(); self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, d)); self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, d))
    def forward(self, x):  # x channels-last (B,D,H,W,C); global over the whole input volume
        gx = torch.norm(x, p=2, dim=(1, 2, 3), keepdim=True)
        return self.gamma * (x * (gx / (gx.mean(-1, keepdim=True) + 1e-6))) + self.beta + x


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.dwconv = nn.Conv3d(d, d, 7, padding=3, groups=d); self.norm = nn.LayerNorm(d, eps=1e-6)
        self.pwconv1 = nn.Linear(d, 4 * d); self.act = nn.GELU(); self.grn = GRN(4 * d); self.pwconv2 = nn.Linear(4 * d, d)
    def forward(self, x):
        y = self.dwconv(x).permute(0, 2, 3, 4, 1)
        y = self.pwconv2(self.grn(self.act(self.pwconv1(self.norm(y)))))
        return x + y.permute(0, 4, 1, 2, 3)


class ConvNeXtV2_3D(nn.Module):
    def __init__(self, in_chans=3, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768)):
        super().__init__()
        self.downsample_layers = nn.ModuleList([nn.Sequential(nn.Conv3d(in_chans, dims[0], 4, stride=4), LayerNormCF(dims[0]))])
        for i, s in enumerate((2, 2, 1)):
            self.downsample_layers.append(nn.Sequential(LayerNormCF(dims[i]), nn.Conv3d(dims[i], dims[i + 1], s, stride=s)))
        self.stages = nn.ModuleList([nn.Sequential(*[Block(dims[i]) for _ in range(depths[i])]) for i in range(4)])
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # NOT in the pretrained ckpt (stays at init = identity affine)

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.stages[i](self.downsample_layers[i](x)); outs.append(x)
        return outs  # [(B,96,D/4,H/4,W/4), (B,192,/8), (B,384,/16), (B,768,/16)]


def remap_checkpoint_keys(sd):  # copy of ProFound models/convnextv2.py L178-239 (logic unchanged)
    new = OrderedDict()
    sd = {k: v for k, v in sd.items() if not any(s in k for s in ("decoder", "mask_token", "proj", "pred"))}
    for k, v in sd.items():
        if k.startswith("encoder"):
            k = ".".join(k.split(".")[1:])
        if k.endswith("kernel"):
            nk = ".".join(k.split(".")[:-1]) + ".weight"
            if v.dim() == 3:
                kv, ci, co = v.shape; ks = int(round(kv ** (1 / 3)))
                new[nk] = v.permute(2, 1, 0).reshape(co, ci, ks, ks, ks).permute(0, 1, 4, 3, 2)
            elif v.dim() == 2:
                if nk == "downsample_layers.3.1.weight":
                    new[nk] = v.permute(1, 0)[..., None, None, None]
                else:
                    kv, d = v.shape; ks = int(round(kv ** (1 / 3)))
                    new[nk] = v.permute(1, 0).reshape(d, 1, ks, ks, ks).permute(0, 1, 4, 3, 2)
            continue
        elif "ln" in k or "linear" in k:
            p = k.split("."); p.pop(-2); nk = ".".join(p)
        else:
            nk = k
        new[nk] = v
    for k, v in list(new.items()):
        if k.endswith("bias") and v.dim() != 1:
            new[k] = v.reshape(-1)
        elif "grn" in k:
            new[k] = v[None, None, None] if v.dim() == 2 else v  # (1,C) -> (1,1,1,1,C)
    return new


def load_profound(ckpt_path):
    torch.serialization.add_safe_globals([argparse.Namespace])
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m = ConvNeXtV2_3D()
    res = m.load_state_dict(remap_checkpoint_keys(ck["model"]), strict=False)
    assert set(res.missing_keys) <= {"norm.weight", "norm.bias"} and not res.unexpected_keys, res
    return m.eval()


def preprocess(t2w, dwi, adc):
    """Each (D,H,W) float tensor, co-registered on the same grid (paper: resampled to 1 mm isotropic;
    pretraining crops 64x224x224 voxels). MONAI NormalizeIntensityd(nonzero=True, channel_wise=True)."""
    x = torch.stack([t2w, dwi, adc]).float()
    for c in range(3):
        nz = x[c] != 0
        if nz.any():
            x[c][nz] = (x[c][nz] - x[c][nz].mean()) / x[c][nz].std(unbiased=False).clamp_min(1e-8)
    return x
