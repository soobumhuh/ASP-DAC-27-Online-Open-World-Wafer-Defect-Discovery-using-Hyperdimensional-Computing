# HDC detection of clinically significant prostate cancer on PI-CAI MRI

This adds MRI to Eric's WaferHDC code. Everything is hyperdimensional computing: binary hypervectors from Eric's
`BinaryThermometerHDC` (`wafer_hdc_encoding.py`, unchanged), associative memories of prototypes, HDDB-style record
hypervectors. Eric's wafer files are untouched.

## Layout

| path | content |
|---|---|
| `mri_hdc/` | the best pipeline as a package: `data` (slices, folds, splits), `features` (patches, ResNet-18 layer 2), `encoding` (thermometer HVs), `memory` (prototype memories, record encoders), `maps` (train + score one split), `extraction` (calibration + candidates), `rescorer` (HDC candidate re-scorer), `patient` (clinical record + merge), `evaluate` (picai_eval), `pipeline` (end-to-end runs) |
| `configs/best.yaml` | every setting of the current best model |
| `configs/test_fresh.yaml` | the same, building data and caches from scratch into `/data/fresh` |
| `scripts/` | entry points: `build_data.py`, `train_predict.py`, `run_fold0.py`, `run_cv.py`, `tune.py` |
| `experiments/` | research scripts of rounds 1–6 (all ablations; see `experiments/README.md`) |
| `results/` | outputs (not in git) |

## Setup

    uv sync                     # Python 3.11, torch 2.11 cu128 (Blackwell GPUs), picai-eval, optuna, ...

Data (outside git; on the lab server `/data` lives on `/mnt/hdd/sohuh`, as do `results/` and `mri_feature_cache/`):

- `/data/picai_images`: the five public PI-CAI image folds (Zenodo 6624726), unzipped
- `/data/picai_labels`: clone of https://github.com/DIAGNijmegen/picai_labels (expert lesions, AI gland Bosma22b,
  AI zones HeviAI23, marksheet)
- `/data/picai_fold{0..4}_files.txt`: the patients of each public fold

## Reproduce the best result

    uv run python scripts/build_data.py --config configs/best.yaml   # slices (5 folds + merge) + ResNet cache
    uv run python scripts/run_cv.py     --config configs/best.yaml   # maps (3 seeds x 5 folds), extraction,
                                                                     # re-scorer, clinical merge -> results/pick/cv.md
    uv run python scripts/run_fold0.py  --config configs/best.yaml   # fold-0 row only (reuses the maps)

Map jobs run `run.parallel_jobs` (2) at a time; each takes about 10 min and 20–35 GB RAM on 24 CPU threads. The full
CV (15 map jobs + post-processing) takes about 1.5–2 h. Retuning (folds 1–4 only):

    uv run python scripts/tune.py --config configs/best.yaml extraction --trials 200   # needs inner-split maps
    uv run python scripts/tune.py --config configs/best.yaml rescorer
    uv run python scripts/tune.py --config configs/best.yaml merge

## Protocol

- PI-CAI public training data: 1,500 cases, 1,476 patients, 425 csPCa; folds 0–4 as published.
- Fold 0 is the held-out test fold: models trained on folds 1–4, tested once per variant.
- Every choice is made on folds 1–4 only: inner splits (train on folds 1–4 minus v, validate on fold v = 1, 2) or
  leave-one-fold-out over folds 1–4. Kept changes are then reported as 5-fold CV (each fold's test once).
- Never model inputs: case ISUP, lesion Gleason / ISUP, histopathology, case csPCa label, PI-RADS.
- Metrics (picai_eval): lesion AP (IoU ≥ 0.1), patient AUROC, PI-CAI score = (AUROC + AP) / 2.

## Results (fold 0 / 5-fold CV PI-CAI score)

| model | fold 0 | CV |
|---|---|---|
| #110 single HDC model, default extraction | 0.629 | 0.601 ± 0.023 |
| #110, tuned extraction | 0.659 | 0.624 ± 0.038 |
| 3-seed ensemble + tuned extraction + HDC re-scorer | 0.662 | 0.642 ± 0.036 |
| **+ clinical HDC record merge (current best, `configs/best.yaml`)** | **0.672** | **0.649 ± 0.036** |
| D = 8,192 single model (round 6, hardware study) | 0.684 | 0.631 ± 0.045 |
| U-Mamba MTL (3D CNN; PI-CAI hidden tuning set, not our CV) | 0.781 | – |

Reproduction check of this code (`configs/test_fresh.yaml`, everything rebuilt from the raw PI-CAI files):
- slice file: identical to the published one; maps of #110 fold 0 from the published feature cache: bit-identical
- post-processing of the saved maps: fold 0 0.672, CV 0.649 ± 0.036 (exact)
- fully fresh run (new ResNet cache, all 15 map jobs): fold 0 0.669, CV 0.648 ± 0.035. The new feature cache differs
  in ~0.1% of values by one float16 step (CPU thread order); thermometer quantisation and retraining turn that into
  seed-level noise in the maps.
- run time on the lab server (2 jobs at a time): slices 6 min, ResNet cache ~3 min (CPU), map jobs 11–14 min each,
  whole fresh run ~1.6 h.

Round-by-round results and everything that did not help are in the round reports (project docs) and in
`experiments/README.md`.
