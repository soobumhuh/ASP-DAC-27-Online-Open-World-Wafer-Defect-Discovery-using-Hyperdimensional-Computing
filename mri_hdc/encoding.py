"""Patch descriptors -> binary hypervectors with Eric's BinaryThermometerHDC (wafer_hdc_encoding.py, repo root).

PatchEncoder.fit (training normal patches only):
  clip every descriptor dimension to its 1st-99th percentile, standardise (mean / std), then fit the thermometer
  encoder: num_proj Gaussian random projections, each robust-standardised (median / IQR), clipped to +-clip_value
  and written as a num_bins-bit thermometer code -> D = num_proj x num_bins bits (2,048 for 512 x 4).
Hamming distance between two codes approximates the Euclidean distance of the descriptors.
"""
from __future__ import annotations

import numpy as np

import mri_hdc  # noqa: F401  (puts the repo root on sys.path)
from wafer_hdc_encoding import BinaryThermometerHDC, apply_clip, apply_standardize, fit_clip_stats, fit_standardize


class PatchEncoder:
    """groups (optional): lists of descriptor columns encoded by separate thermometer encoders (num_proj split evenly)
    whose codes are concatenated; Hamming distance then weighs every group equally, whatever its dimension."""

    def __init__(self, num_proj: int = 512, num_levels: int = 4, clip_value: float = 4.0, seed: int = 0,
                 groups: list = None):
        self.num_proj, self.num_levels, self.clip_value, self.seed = num_proj, num_levels, clip_value, seed
        self.groups = groups

    def fit_scaler(self, F_all: np.ndarray):
        """Clip limits from all training normal patches (before the per-bin cap)."""
        self.lo, self.hi = fit_clip_stats(F_all, 1.0, 99.0)
        self.mu, self.sd = fit_standardize(apply_clip(F_all, self.lo, self.hi))
        return self

    def prep(self, F: np.ndarray) -> np.ndarray:
        return apply_standardize(apply_clip(F, self.lo, self.hi), self.mu, self.sd)

    def fit_hdc(self, X: np.ndarray):
        """Thermometer encoder(s) fitted on the prepared (capped) training normal descriptors."""
        if not self.groups:
            self.enc = BinaryThermometerHDC(in_dim=X.shape[1], num_proj=self.num_proj, num_bins=self.num_levels,
                                            seed=self.seed, clip_value=self.clip_value).fit(X)
            self.D = self.enc.bit_dim
            return self
        n = self.num_proj // len(self.groups)
        self.encs = [BinaryThermometerHDC(in_dim=len(g), num_proj=n, num_bins=self.num_levels, seed=self.seed + 1000 * k,
                                          clip_value=self.clip_value).fit(X[:, g]) for k, g in enumerate(self.groups)]
        self.D = sum(e.bit_dim for e in self.encs)
        return self

    def codes(self, X: np.ndarray) -> np.ndarray:
        """Prepared descriptors -> packed hypervectors (n, D / 8) uint8 (little bit order)."""
        if not self.groups:
            return np.ascontiguousarray(self.enc.transform(X, batch_size=4096)).view(np.uint8)
        return np.ascontiguousarray(np.concatenate(
            [np.ascontiguousarray(e.transform(X[:, g], batch_size=4096)).view(np.uint8)
             for e, g in zip(self.encs, self.groups)], 1))


def unpack(codes_u8: np.ndarray, D: int) -> np.ndarray:
    """Packed bytes -> (n, D) {0, 1} uint8."""
    return np.unpackbits(codes_u8, axis=1, bitorder="little")[:, :D]


def bipolar(bits: np.ndarray) -> np.ndarray:
    """{0, 1} -> {-1, +1} float32; a dot product of bipolar vectors = D - 2 x Hamming distance."""
    return bits.astype(np.float32) * 2.0 - 1.0


def bipolar_int8(codes_u8: np.ndarray, D: int, block: int = 65536) -> np.ndarray:
    """Packed codes -> (n, D) int8 +-1, unpacked block by block (no full-size temporaries)."""
    out = np.empty((len(codes_u8), D), np.int8)
    for i in range(0, len(codes_u8), block):
        out[i:i + block] = unpack(codes_u8[i:i + block], D).astype(np.int8) * 2 - 1
    return out
