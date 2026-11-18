#!/bin/bash
# Dataset503 (25 splits: 0-4 outer folds, 5-24 nested) 3d_fullres csPCa nnU-Net, 250 epochs, on GPU 2.
# Worker $1 of $2 takes every $2-th split. Reuses Dataset502's preprocessed data (symlinked, read-only).
export nnUNet_raw=/data/auroc/nnunet/raw nnUNet_preprocessed=/data/auroc/nnunet/preprocessed nnUNet_results=/data/auroc/nnunet/results
export nnUNet_n_proc_DA=8 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2
cd /data/sota/nnunet
for f in $(seq $1 $2 24); do
  R=$nnUNet_results/Dataset503_PicaiLesionsNested/nnUNetTrainerCELoss_250epochs__nnUNetPlans__3d_fullres/fold_$f/validation/summary.json
  [ -f $R ] && continue
  C=""; [ -f $nnUNet_results/Dataset503_PicaiLesionsNested/nnUNetTrainerCELoss_250epochs__nnUNetPlans__3d_fullres/fold_$f/checkpoint_latest.pth ] && C="--c"
  /data/sota/envs/nnunet/bin/python /data/auroc/nnunet/train_ce.py 503 3d_fullres $f -tr nnUNetTrainerCELoss_250epochs --npz $C > /data/auroc/nnunet/logs/f$f.log 2>&1
done
echo done > /data/auroc/nnunet/worker$1.done
