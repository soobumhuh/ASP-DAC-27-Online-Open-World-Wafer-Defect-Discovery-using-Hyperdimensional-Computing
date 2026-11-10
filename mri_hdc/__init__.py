"""mri_hdc - hyperdimensional-computing (HDC) detection of clinically significant prostate cancer (csPCa) on PI-CAI
bi-parametric MRI, built on Eric's WaferHDC encoder (wafer_hdc_encoding.BinaryThermometerHDC, repo root).

Pipeline of the current best model (configs/best.yaml):

  data        PI-CAI cases -> prostate-cropped 96 x 96 slices (T2W, ADC, HBV) + lesion / gland / zone masks
  features    20 px patches on a stride-2 grid inside the gland; descriptor = per-channel mean / std +
              frozen ImageNet ResNet-18 layer-2 features at the patch centre (3 x 3 average-pooled)
  encoding    clip + standardise, then BinaryThermometerHDC (512 random projections x 4 levels = 2,048 bits)
  memory      one HDC model per position bin (5 slice levels x zone x 3 x 3 grid = 135 bins): 20 normal and
              5 lesion sub-prototypes (Hamming k-means + class-balanced perceptron retraining)
  maps        patch score = soft-max similarity to lesion - to normal prototypes, z-scored per bin on validation
              normals, averaged over overlapping patches, smoothed; 3 projection seeds averaged (ensemble)
  extraction  sigmoid calibration + report_guided_annotation.extract_lesion_candidates
  rescorer    HDC record of each candidate (size, shape, intensity contrast, zone, ...) re-ranks candidates
  patient     case score = z(max candidate confidence) + w x z(clinical HDC record score)
  evaluate    picai_eval: patient AUROC, lesion AP, PI-CAI score = (AUROC + AP) / 2

Clinical outcome fields (ISUP, Gleason, csPCa label, PI-RADS, histopathology) are never model inputs.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:  # Eric's wafer_hdc_encoding.py lives at the repo root
    sys.path.insert(0, ROOT)
