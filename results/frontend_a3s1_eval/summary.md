# Front end A3 (ProFound-alpha stage 1) through the full pipeline

Nested 5-fold CV (run_cv_clean), 3 seeds, D = 2048, 135 bins, centre-in-lesion label rule, fixed extraction
(best_clean_adc), re-scorer / merge fixed (50 epochs, alpha 0.25, w 0.25). PI-CAI score = (patient AUROC + lesion AP) / 2,
native T2W grid (scripts/native_eval.py). Bootstrap: 1000 paired exam resamples (scripts/bootstrap_ci.py); "pooled CV" =
all five test folds pooled.

| arm | fold 0 | CV mean ± sd (folds 0-4) | per fold |
|---|---|---|---|
| best_clean_adc (ResNet-18, >= 25% rule) | 0.696 | 0.650 ± 0.042 | 0.696 0.698 0.615 0.651 0.592 |
| control best_clean_adc_foc (ResNet-18, centre rule) | 0.652 | 0.641 ± 0.039 | 0.652 0.705 0.616 0.646 0.588 |
| A3 ProFound s1 (centre rule) | 0.645 | 0.614 ± 0.038 | 0.645 0.645 0.557 0.643 0.579 |

A3 component means: patient AUROC 0.818 (best 0.832, control 0.829), lesion AP 0.409 (best 0.469, control 0.453).

Paired bootstrap, PI-CAI score difference (95% CI):

| comparison | fold 0 | pooled CV |
|---|---|---|
| A3 - best_clean_adc | -0.051 [-0.095, +0.002] | -0.038 [-0.059, -0.016] |
| A3 - control | -0.006 [-0.054, +0.043] | -0.027 [-0.049, -0.007] |
| control - best_clean_adc | -0.044 [-0.063, -0.018] | -0.010 [-0.018, -0.001] |

The patch-level gain from the screen (HD probe, fold 1: 0.884 vs 0.846) does not carry over: on fold 1 itself the full
pipeline gives 0.645 vs 0.705 (control). The loss is mostly lesion AP (pooled CV -0.044 vs control). Extraction was kept
fixed at the ResNet-18 setting (as in the control); ProFound stage-1 maps are coarser (4 mm cells, 3 x 3 pooled), so
a re-tuned extraction could recover part of the AP gap. That is the one untested confound.
