"""Self-contained ProViCNet (pimed/ProViCNet) frozen encoder, no repo code / torch.hub needed.

Re-implements ProViCNet_contrastive.prepare_tokens_with_axispos / forward_features_pos
(ProViCNet/ModelArchitectures/ProViCNet/ProViCNet.py L658-694) on top of a timm DINOv2 ViT-S/14-reg4.
Token order: [cls, reg x4, axis_pos, axis_max, 32x32 patches].
"""
import torch, torch.nn as nn, torch.nn.functional as F, timm


class ProViCNetEncoder(nn.Module):
    N_PREFIX = 1 + 4 + 2  # cls + 4 registers + 2 slice-position tokens

    def __init__(self, ckpt_path, img_size=448, with_seg_head=False):
        super().__init__()
        vit = timm.create_model("vit_small_patch14_reg4_dinov2", pretrained=False, img_size=img_size)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        self.patch_embed, self.blocks, self.norm = vit.patch_embed, vit.blocks, vit.norm
        sub = {k: v for k, v in bb.items() if k.split(".")[0] in ("patch_embed", "blocks", "norm")}
        missing, unexpected = nn.ModuleDict(
            {"patch_embed": self.patch_embed, "blocks": self.blocks, "norm": self.norm}
        ).load_state_dict(sub, strict=True), None
        self.cls_token = nn.Parameter(bb["cls_token"])            # (1,1,384)
        self.reg_tokens = nn.Parameter(bb["register_tokens"])     # (1,4,384)
        # DINOv2 interpolate_pos_encoding for *_reg models: bicubic, antialias=True, size=(w0,h0), offset 0
        pe = bb["pos_embed"].float()                               # (1,1370,384) = cls + 37x37
        g = img_size // 14
        patch_pe = F.interpolate(pe[:, 1:].reshape(1, 37, 37, -1).permute(0, 3, 1, 2),
                                 size=(g, g), mode="bicubic", antialias=True)
        self.register_buffer("pos_embed", torch.cat([pe[:, :1], patch_pe.permute(0, 2, 3, 1).reshape(1, g * g, -1)], 1))
        self.axis_pos, self.axis_max = nn.Linear(1, 384), nn.Linear(1, 384)
        self.axis_pos.load_state_dict({"weight": sd["axis_pos.weight"], "bias": sd["axis_pos.bias"]})
        self.axis_max.load_state_dict({"weight": sd["axis_max.weight"], "bias": sd["axis_max.bias"]})
        self.grid = g
        self.seg_head = None
        if with_seg_head:  # conv_head8 (ProViCNet.py L20-40): 4 classes bg/gland/indolent/csPCa
            self.seg_head = nn.Sequential(
                nn.Upsample(scale_factor=2), nn.Conv2d(384, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Upsample(scale_factor=2), nn.Conv2d(64, 4, 3, padding=1))
            self.seg_head.load_state_dict({k[len("head.segmentation_conv."):]: v for k, v in sd.items()
                                           if k.startswith("head.segmentation_conv.")})

    def forward(self, x, pos, layers=(11,), norm=True):
        """x: (B,3,448,448) = 3 consecutive axial slices of ONE sequence, gland-z-scored.
        pos: (B,2) float = (slice_index, n_slices_in_volume), raw (not normalised).
        Returns list of (B,384,32,32) patch-token maps for the requested block indices (0..11)."""
        B = x.shape[0]
        t = self.patch_embed(x)                                   # (B,1024,384)
        t = torch.cat([self.cls_token.expand(B, -1, -1), t], 1) + self.pos_embed
        t = torch.cat([t[:, :1], self.reg_tokens.expand(B, -1, -1),
                       self.axis_pos(pos[:, 0:1]).unsqueeze(1), self.axis_max(pos[:, 1:2]).unsqueeze(1),
                       t[:, 1:]], 1)
        outs = []
        for i, blk in enumerate(self.blocks):
            t = blk(t)
            if i in layers:
                o = self.norm(t) if norm else t
                outs.append(o[:, self.N_PREFIX:].transpose(1, 2).reshape(B, 384, self.grid, self.grid))
        return outs


def preprocess_volume(vol, gland):
    """vol, gland: (S,H,W) numpy at 0.5x0.5x3.0 mm, 256x256 centre crop (PI-CAI-style).
    Mirrors US_MRI_Generator (Prostate_DataGenerator.py L56-184): z-score using gland voxels,
    resize each slice to 448 (albumentations Resize = cv2 bilinear), 3-slice stacks with edge shift."""
    import numpy as np
    v = torch.from_numpy(vol.astype("float32") / 255.0)
    m = torch.from_numpy(gland) == 1
    v = (v - v[m].mean()) / v[m].std(unbiased=False)
    v = F.interpolate(v[:, None], size=(448, 448), mode="bilinear", align_corners=False)[:, 0]
    S = v.shape[0]
    if S < 6:  # getNChannel pads 1 slice each side when S < 2*nChannel
        v = F.pad(v, (0, 0, 0, 0, 1, 1)); S = v.shape[0]
    xs, ps = [], []
    for j in range(S):
        c = min(max(j, 1), S - 2)
        xs.append(v[c - 1:c + 2]); ps.append((j, S))
    return torch.stack(xs), torch.tensor(ps, dtype=torch.float32)
