#!/usr/bin/env python3
"""Per-slice feature maps of frozen 2D foundation encoders on the 96 px gland crops of the slice file.

Every encoder sees the same input: the crop's three channels (T2W, ADC, HBV; per-exam z-scored in the slice file)
clipped to [-3, 3] and mapped to [0, 1] as an RGB image, normalised with the encoder's own mean / std, resized to
the encoder's input size. Output: patch-token grid (C, G, G) per slice, float16, rows aligned to the slice file
(same format as the other fe_*_all.npy caches, so the map pipeline can read it as an external front end).

Encoders (all open weights, none trained on PI-CAI labels):
  dinov2_l    timm vit_large_patch14_dinov2.lvd142m (natural images, LVD-142M), 224 px -> 16 x 16 x 1024
  dinov2_b    timm vit_base_patch14_dinov2.lvd142m, 224 px -> 16 x 16 x 768
  rad_dino    microsoft/rad-dino (DINOv2 ViT-B/14 continued on chest X-ray), 224 px -> 16 x 16 x 768
  biomedclip  microsoft/BiomedCLIP ViT-B/16 image tower (PMC-15M figure-caption pairs), 224 px -> 14 x 14 x 768
  medsam      wanglab/medsam-vit-base (SAM ViT-B fine-tuned on 1.5M medical image-mask pairs), 1024 px -> 64 x 64
              x 256, average-pooled to 32 x 32

  CUDA_VISIBLE_DEVICES=3 python scripts/fe_zoo.py --model dinov2_l --out /data/auroc/fe
"""
import argparse
import json
import os
import time

import numpy as np

IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def build(name, device):
    import torch
    import torch.nn.functional as F
    if name in ("dinov2_l", "dinov2_b"):
        import timm
        m = timm.create_model({"dinov2_l": "vit_large_patch14_dinov2.lvd142m",
                               "dinov2_b": "vit_base_patch14_dinov2.lvd142m"}[name], pretrained=True,
                              img_size=224).to(device).eval()

        def fwd(x):
            t = m.forward_features(x)[:, m.num_prefix_tokens:]
            return t.transpose(1, 2).reshape(len(x), -1, 16, 16)
        return fwd, 224, IMNET
    if name == "rad_dino":
        from transformers import AutoModel
        m = AutoModel.from_pretrained("microsoft/rad-dino").to(device).eval()

        def fwd(x):
            t = m(pixel_values=x).last_hidden_state[:, 1:]
            return t.transpose(1, 2).reshape(len(x), -1, 16, 16)
        return fwd, 224, ((0.5307, 0.5307, 0.5307), (0.2583, 0.2583, 0.2583))
    if name == "biomedclip":
        import open_clip
        model, _ = open_clip.create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        trunk = model.visual.trunk.to(device).eval()

        def fwd(x):
            t = trunk.forward_features(x)[:, trunk.num_prefix_tokens:]
            return t.transpose(1, 2).reshape(len(x), -1, 14, 14)
        return fwd, 224, CLIP
    if name == "medsam":
        from transformers import SamModel
        m = SamModel.from_pretrained("wanglab/medsam-vit-base").to(device).eval()

        def fwd(x):
            e = m.get_image_embeddings(x)  # (n, 256, 64, 64)
            return F.avg_pool2d(e, 2)
        # MedSAM pre-processing: intensities to [0, 1], no mean / std normalisation
        return fwd, 1024, ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slices", default="/data/picai_all_masks.npz")
    ap.add_argument("--out", default="/data/auroc/fe")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, f"fe_{a.model}_all.npy")
    imgs = np.load(a.slices)["images"]  # (N, 3, 96, 96) float16
    fwd, size, (mu, sd) = build(a.model, a.device)
    mu = torch.tensor(mu, device=a.device)[None, :, None, None]
    sd = torch.tensor(sd, device=a.device)[None, :, None, None]
    t0, out = time.time(), None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(imgs), a.batch):
            x = torch.from_numpy(imgs[i:i + a.batch].astype(np.float32)).to(a.device)
            x = (x.clamp(-3, 3) + 3) / 6
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            f = fwd(((x - mu) / sd).float()).float()
            if out is None:
                out = np.lib.format.open_memmap(dst + ".tmp.npy", "w+", np.float16, (len(imgs),) + tuple(f.shape[1:]))
            out[i:i + len(f)] = f.cpu().numpy().astype(np.float16)
            if i % (20 * a.batch) == 0:
                print(f"{a.model} {i}/{len(imgs)} {time.time() - t0:.0f}s", flush=True)
    out.flush()
    del out
    os.replace(dst + ".tmp.npy", dst)
    json.dump({"source": a.model, "map_shape": list(np.load(dst, mmap_mode="r").shape[1:]), "rows": "all",
               "input": "96 px crop, T2W/ADC/HBV as RGB, clip [-3, 3] -> [0, 1]", "size": size,
               "seconds": time.time() - t0}, open(dst + ".json", "w"), indent=1)
    print("done", dst, time.time() - t0)


if __name__ == "__main__":
    main()
