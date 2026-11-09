#!/usr/bin/env python3
"""c_features.py - round 6 C: frozen descriptor upgrades, computed once on the GPU for every slice.

  radimagenet_l2  ResNet-50 trained on RadImageNet (radiology CT / MRI / US; third-party torchvision reproduction,
                  huggingface convergedmachine/RadImagenet resnet50.pth), layer 2 output (512 ch, 28 x 28 at 224 input),
                  3 x 3 average-pooled like the ResNet-18 descriptor
  dinov2_s        DINOv2 ViT-S/14 (Meta, self-supervised on natural images; timm architecture + hub weights), patch tokens
                  (384-d on a 16 x 16 grid at 224 input)
Input: the 3-channel z-scored crop (T2W, ADC, HBV as RGB) upsampled to 224, exactly as for the ResNet-18 descriptor.
Output: float16 (N, C, h, w) cache for patch_anomaly.py --fmap_cache.

  CUDA_VISIBLE_DEVICES=3 uv run python c_features.py --model dinov2_s --out mri_feature_cache/c_dinov2_s.npy
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

NPZ = "/data/picai_all_masks.npz"


def build(name):
    if name == "radimagenet_l2":
        import argparse as ap_mod
        from torchvision.models import resnet50
        with torch.serialization.safe_globals([ap_mod.Namespace]):
            ck = torch.load("/data/weights/radimagenet_resnet50.pth", map_location="cpu", weights_only=True)
        sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items() if not k.split(".")[-1] == "fc"}
        m = resnet50(num_classes=165)
        m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
        stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2)
        return lambda x: F.avg_pool2d(stem(x), 3, 1, 1), stem
    if name == "dinov2_s":
        import timm  # timm's DINOv2 ViT-S/14 architecture, Meta's weights from the Hugging Face hub (safetensors)
        m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, img_size=224, num_classes=0)

        def f(x):
            t = m.forward_features(x)[:, m.num_prefix_tokens:]  # normalised patch tokens (B, 256, 384)
            B, n, c = t.shape
            g = int(round(n ** 0.5))
            return t.transpose(1, 2).reshape(B, c, g, g)
        return f, m
    raise SystemExit(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["radimagenet_l2", "dinov2_s"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    fn, mod = build(args.model)
    mod.eval().cuda()
    images = np.load(NPZ)["images"]
    with torch.no_grad():
        probe = fn(torch.zeros(1, 3, 224, 224, device="cuda"))
        shape = (len(images),) + tuple(probe.shape[1:])
        out = np.lib.format.open_memmap(args.out + ".tmp.npy", mode="w+", dtype=np.float16, shape=shape)
        for i in range(0, len(images), args.batch):
            x = torch.from_numpy(np.asarray(images[i:i + args.batch], dtype=np.float32)).cuda()
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
            out[i:i + args.batch] = fn(x).float().cpu().numpy().astype(np.float16)
    out.flush(); del out
    os.replace(args.out + ".tmp.npy", args.out)
    print("saved", args.out, shape)


if __name__ == "__main__":
    main()
