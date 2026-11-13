#!/bin/bash
# waits for the A1/A3 caches, then runs the screens in two lanes on GPU 3 (24 threads each)
cd /workspace/wafer-hdc
until [ -f mri_feature_cache/fe_profound_s2.npy.json ]; do sleep 60; done
run() { CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 \
  .venv/bin/python scripts/frontend_screen.py --name $1 --maps $2 --rows $3 $4 > results/frontend_screen/logs/$1.log 2>&1; }
( run A3_profound_s2 mri_feature_cache/fe_profound_s2.npy sub; run A3_profound_s1 mri_feature_cache/fe_profound_s1.npy sub;
  run A0orig_resnet18_l2 mri_feature_cache/resnet_l2.npy full --orig ) &
( run A1_provicnet_b11 mri_feature_cache/fe_provicnet_b11.npy sub; run A1_provicnet_b7 mri_feature_cache/fe_provicnet_b7.npy sub ) &
wait
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/frontend_features.py --cpu-timing 3 > results/frontend_screen/logs/cpu_timing.log 2>&1
.venv/bin/python scripts/frontend_table.py > results/frontend_screen/logs/table.log 2>&1
echo QUEUE_DONE >> results/frontend_screen/logs/table.log
