# Front-end patch-level screen (train folds 2-4, test fold 1; fold 0 unused)

Patch AUROC, lesion patches (rule) vs normal patches (no lesion pixel). Size = estimated native T2W voxels of the 3D lesion; zone = AI zone at the patch centre; posexam = csPCa-positive exams only.

| name | rule | probe | auroc | auroc_lt150 | auroc_150to500 | auroc_gt500 | auroc_lt500 | auroc_pz | auroc_tz | auroc_posexam | n_pos | n_pos_lt150 | n_pos_150to500 | n_pos_gt500 | raw_dim | dim | gpu_s_per_exam | cache_gb | fit_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_resnet18_l2 | centre | hd | 0.8461 | 0.7530 | 0.7624 | 0.8467 | 0.7610 | 0.8993 | 0.8118 | 0.8513 | 70698 | 79 | 435 | 70184 | 128 | 134 | nan | 4.4400 | 582.7000 |
| A0orig_resnet18_l2 | centre | hd | 0.8317 | 0.7991 | 0.8370 | 0.8317 | 0.8311 | 0.8939 | 0.7912 | 0.8406 | 70698 | 79 | 435 | 70184 | 128 | 134 | nan | 4.4400 | 295.4000 |
| A1_provicnet_b11 | centre | hd | 0.9201 | 0.8327 | 0.8474 | 0.9207 | 0.8451 | 0.9433 | 0.9031 | 0.9054 | 70698 | 79 | 435 | 70184 | 1152 | 134 | 0.1270 | 10.4700 | 336.7000 |
| A1_provicnet_b7 | centre | hd | 0.9189 | 0.7918 | 0.8507 | 0.9194 | 0.8417 | 0.9452 | 0.9015 | 0.9077 | 70698 | 79 | 435 | 70184 | 1152 | 134 | 0.1270 | 10.4700 | 394.7000 |
| A3_profound_s1 | centre | hd | 0.8840 | 0.8423 | 0.8779 | 0.8841 | 0.8725 | 0.9304 | 0.8522 | 0.8829 | 70698 | 79 | 435 | 70184 | 96 | 102 | 0.0355 | 1.9600 | 326.1000 |
| A3_profound_s2 | centre | hd | 0.8695 | 0.7825 | 0.7605 | 0.8703 | 0.7639 | 0.9113 | 0.8407 | 0.8583 | 70698 | 79 | 435 | 70184 | 192 | 134 | 0.0355 | 1.7500 | 317.4000 |
| A0_resnet18_l2 | centre | lr | 0.8786 | 0.8542 | 0.7900 | 0.8791 | 0.7999 | 0.9192 | 0.8543 | 0.8914 | 70698 | 79 | 435 | 70184 | 128 | 134 | nan | 4.4400 | 25.8000 |
| A0orig_resnet18_l2 | centre | lr | 0.8541 | 0.8525 | 0.8003 | 0.8544 | 0.8083 | 0.9059 | 0.8209 | 0.8723 | 70698 | 79 | 435 | 70184 | 128 | 134 | nan | 4.4400 | 20.9000 |
| A1_provicnet_b11 | centre | lr | 0.9769 | 0.9036 | 0.9753 | 0.9770 | 0.9643 | 0.9796 | 0.9761 | 0.9690 | 70698 | 79 | 435 | 70184 | 1152 | 134 | 0.1270 | 10.4700 | 24.0000 |
| A1_provicnet_b7 | centre | lr | 0.9706 | 0.8683 | 0.9183 | 0.9710 | 0.9106 | 0.9708 | 0.9706 | 0.9625 | 70698 | 79 | 435 | 70184 | 1152 | 134 | 0.1270 | 10.4700 | 21.6000 |
| A3_profound_s1 | centre | lr | 0.9196 | 0.8489 | 0.8533 | 0.9201 | 0.8526 | 0.9475 | 0.8947 | 0.9141 | 70698 | 79 | 435 | 70184 | 96 | 102 | 0.0355 | 1.9600 | 19.7000 |
| A3_profound_s2 | centre | lr | 0.9306 | 0.8055 | 0.8812 | 0.9311 | 0.8696 | 0.9513 | 0.9145 | 0.9265 | 70698 | 79 | 435 | 70184 | 192 | 134 | 0.0355 | 1.7500 | 21.5000 |
| A0_resnet18_l2 | frac25 | hd | 0.8534 | 0.9025 | 0.7596 | 0.8537 | 0.7694 | 0.9112 | 0.8173 | 0.8595 | 68910 | 20 | 271 | 68619 | 128 | 134 | nan | 4.4400 | 453.8000 |
| A0orig_resnet18_l2 | frac25 | hd | 0.8326 | 0.8963 | 0.8144 | 0.8327 | 0.8201 | 0.9001 | 0.7890 | 0.8410 | 68910 | 20 | 271 | 68619 | 128 | 134 | nan | 4.4400 | 351.2000 |
| A1_provicnet_b11 | frac25 | hd | 0.9218 | 0.8446 | 0.9023 | 0.9219 | 0.8983 | 0.9440 | 0.9068 | 0.9058 | 68910 | 20 | 271 | 68619 | 1152 | 134 | 0.1270 | 10.4700 | 283.4000 |
| A1_provicnet_b7 | frac25 | hd | 0.9246 | 0.9652 | 0.9034 | 0.9247 | 0.9077 | 0.9481 | 0.9102 | 0.9126 | 68910 | 20 | 271 | 68619 | 1152 | 134 | 0.1270 | 10.4700 | 345.5000 |
| A3_profound_s1 | frac25 | hd | 0.8900 | 0.8286 | 0.7426 | 0.8906 | 0.7485 | 0.9286 | 0.8648 | 0.8859 | 68910 | 20 | 271 | 68619 | 96 | 102 | 0.0355 | 1.9600 | 344.6000 |
| A3_profound_s2 | frac25 | hd | 0.8660 | 0.7254 | 0.6223 | 0.8670 | 0.6294 | 0.9110 | 0.8331 | 0.8579 | 68910 | 20 | 271 | 68619 | 192 | 134 | 0.0355 | 1.7500 | 438.5000 |
| A0_resnet18_l2 | frac25 | lr | 0.8807 | 0.8386 | 0.7541 | 0.8812 | 0.7599 | 0.9239 | 0.8556 | 0.8933 | 68910 | 20 | 271 | 68619 | 128 | 134 | nan | 4.4400 | 32.6000 |
| A0orig_resnet18_l2 | frac25 | lr | 0.8558 | 0.8361 | 0.7625 | 0.8562 | 0.7676 | 0.9097 | 0.8222 | 0.8738 | 68910 | 20 | 271 | 68619 | 128 | 134 | nan | 4.4400 | 54.4000 |
| A1_provicnet_b11 | frac25 | lr | 0.9795 | 0.9932 | 0.9778 | 0.9795 | 0.9789 | 0.9832 | 0.9777 | 0.9719 | 68910 | 20 | 271 | 68619 | 1152 | 134 | 0.1270 | 10.4700 | 23.4000 |
| A1_provicnet_b7 | frac25 | lr | 0.9719 | 0.9520 | 0.9106 | 0.9722 | 0.9134 | 0.9730 | 0.9714 | 0.9641 | 68910 | 20 | 271 | 68619 | 1152 | 134 | 0.1270 | 10.4700 | 27.1000 |
| A3_profound_s1 | frac25 | lr | 0.9206 | 0.9864 | 0.8242 | 0.9210 | 0.8353 | 0.9496 | 0.8953 | 0.9150 | 68910 | 20 | 271 | 68619 | 96 | 102 | 0.0355 | 1.9600 | 18.6000 |
| A3_profound_s2 | frac25 | lr | 0.9333 | 0.9636 | 0.8691 | 0.9335 | 0.8756 | 0.9553 | 0.9164 | 0.9284 | 68910 | 20 | 271 | 68619 | 192 | 134 | 0.0355 | 1.7500 | 28.2000 |

## Pass rule (HD probe vs both A0 variants: +0.01 overall and not worse on lesions < 500 voxels)

| rule | name | d_auroc_vs_A0_res | d_lt500_vs_A0_res | d_auroc_vs_A0orig | d_lt500_vs_A0orig | passed | label_exposure |
|---|---|---|---|---|---|---|---|
| centre | A1_provicnet_b11 | 0.0740 | 0.0842 | 0.0885 | 0.0140 | True | supervised on PI-CAI lesion labels (fold |
| centre | A1_provicnet_b7 | 0.0728 | 0.0807 | 0.0872 | 0.0105 | True | supervised on PI-CAI lesion labels (fold |
| centre | A3_profound_s1 | 0.0379 | 0.1115 | 0.0524 | 0.0413 | True | self-supervised on PI-CAI images (no lab |
| centre | A3_profound_s2 | 0.0234 | 0.0029 | 0.0379 | -0.0673 | False | self-supervised on PI-CAI images (no lab |
| frac25 | A1_provicnet_b11 | 0.0685 | 0.1289 | 0.0892 | 0.0782 | True | supervised on PI-CAI lesion labels (fold |
| frac25 | A1_provicnet_b7 | 0.0713 | 0.1383 | 0.0920 | 0.0876 | True | supervised on PI-CAI lesion labels (fold |
| frac25 | A3_profound_s1 | 0.0367 | -0.0209 | 0.0574 | -0.0716 | False | self-supervised on PI-CAI images (no lab |
| frac25 | A3_profound_s2 | 0.0127 | -0.1400 | 0.0334 | -0.1907 | False | self-supervised on PI-CAI images (no lab |

## Notes

- Fold-1 test lesions (3D components on the crop, estimated native T2W voxels): 83 in total; 4 < 150, 7 at 150-500, 72 > 500.
  The < 150 and 150-500 columns rest on 4 and 7 lesions (centre rule: 79 and 435 patches; >= 25% rule: 20 and 271), so
  the "not worse on lesions < 500 voxels" part of the pass rule is noisy.
- A0 = ResNet-18 layer 2 through the shared adapter (extra 3 x 3 pooling + bilinear read-out + PCA 128); A0orig = the
  current pipeline's descriptor (nearest cell, no PCA, 134 numbers). The pass rule requires beating both.
- A3 stage 1 has 96 channels, so its PCA keeps 96 (descriptor 102 numbers).
- Extraction time per exam (~15 slices): GPU (RTX PRO 6000 Blackwell) ResNet-18 0.001 s, ProViCNet (3 sequences x
  ViT-S, 448 px) 0.13 s, ProFound (64 x 224 x 224 3D) 0.04 s; CPU (16 threads) ResNet-18 0.07 s, ProViCNet 4.9 s,
  ProFound 0.50 s; reading + resampling the MHA volumes 0.65 s (single process) to 2.2 s (12 parallel workers).
- Cache size (folds 1-4, float16): ResNet-18 layer 2 (all folds, 128 x 28 x 28) 4.4 GB; ProViCNet 1152 x 16 x 16
  10.5 GB per block; ProFound stage 1 96 x 24 x 24 2.0 GB, stage 2 192 x 16 x 16 1.7 GB.

Pretraining / label exposure: A1: supervised on PI-CAI lesion labels (fold-1 cases may be in its training set); A3: self-supervised on PI-CAI images (no labels)
