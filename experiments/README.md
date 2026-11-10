# experiments/ — research scripts of rounds 1–6

These are the scripts behind every number in the round reports. They are kept as they were (only the subprocess
calls now locate their sibling scripts in this folder), for reproducibility; the
clean implementation of the best pipeline is `mri_hdc/` + `scripts/` (see `README_MRI.md`). The scripts import each
other by module name and use paths relative to the repo root, so run them from the root with:

    PYTHONPATH=.:experiments uv run python experiments/<script>.py ...

| script | round | what it does / tested |
|---|---|---|
| `picai_slices.py` | 1 | PI-CAI cases → prostate-cropped 2D slices (now `mri_hdc/data.py`) |
| `mri_features.py`, `mri_artifacts.py`, `picai_bundle.py`, `run_mri_hdc.py` | 1 | first slice-level HDC pipeline (image quality / lesion slices), patient split helper |
| `summarize_results.py` | 1 | tables of the round-1 runs |
| `patch_anomaly.py` | 1–6 | the research patch model: unsupervised kNN anomaly variants (HDC / float / intensity), supervised design A (`proto…`, `vote…`) and every later option (3D context, hard negatives, ISUP weights, bagging, TTA, bit errors, OnlineHD rule, external feature maps, lesion augmentation). `mri_hdc/maps.py` reimplements its `proto512_negplus` path |
| `patch_sweep.py` | 2 | grid of patch sizes / descriptors / variants |
| `hdc_classifier.py` | 2–6 | prototype classifier (+ hard negatives, weights, OnlineHD update), kNN vote, record encoders (now `mri_hdc/memory.py`) |
| `cv_hdc.py` | 2 | 5-fold CV runner for patch configs |
| `patient_hdc.py` | 2–3 | design B: patient-level HDDB-style records (clinical + image fields), cross-fitted image dumps |
| `hp_search.py` | 3 | Optuna search over the patch model (folds 1–4 only) → configs #119 / #110 |
| `postproc_search.py` | 3–6 | saved maps per model × split, map → candidate search, fold-0 / CV finals; model registry of all variants |
| `error_analysis.py` | 3 | false-positive / false-negative analysis of the maps |
| `rescore_hdc.py` | 3 | first HDC candidate re-scorer |
| `patient_round3.py` | 3 | clinical record + image merges (design B round 3); source of the clinical scores |
| `round4_*.py` | 4 | lesion-AP ideas: case gating, seed ensembles, inner-split scoring, shape refinement, re-scorer v2, stacking, zone calibration |
| `hdnn_finetune.py` | 5 | GPU fine-tuning of the ResNet-18 front end through a soft HDC head (HDnn) |
| `round5_merge.py` | 5 | clinical merge for the round-4 pick (current best) |
| `round6_cal.py` | 6 | scanner-group calibration × per-case normalisation |
| `round6_hw.py` | 6 | hardware table: hypervector size, bit errors, memory / popcounts |
| `c_features.py` | 6 | RadImageNet ResNet-50 and DINOv2 ViT-S/14 frozen descriptors |

Deleted when the code was cleaned up: `make_fake_picai.py` (synthetic test data generator, unused); it is in the git
history before branch `mri-hdc-clean`.
