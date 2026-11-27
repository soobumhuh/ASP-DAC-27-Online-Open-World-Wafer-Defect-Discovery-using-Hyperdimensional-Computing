"""Hypervector database (HVDB) for prostate MRI: item encoders, an exact Hamming index and retrieval metrics.

Item hypervectors (labels never enter a hypervector; they are payload only):
  record   the detector-output record of the best_sota retrieval re-scorer (mri_hdc.rescorer: 12 fields as 16-level
           codes bound to field keys + zone symbol, majority bundle; level edges from the database candidates)
  content  frozen ResNet-18 region descriptor (per-channel mean + core max, 768 floats) standardised with the
           database items' mean / std, then sign random projection (Gaussian, fixed seed) to D bits
  both     content || record (2 D bits; Hamming distance = sum of the two)
Patient hypervector: majority bundle of the top-k candidate region HVs bound (XOR) to rank-role HVs, the whole-gland
content HV bound to GLAND and the clinical record HV (mri_hdc.patient.ClinicalRecords, the best_sota fields) bound to
CLIN; an OMICS role HV is drawn and reserved (unused). Ties broken by a fixed random HV.
HammingIndex: bits packed into uint64 words, distance = popcount(XOR) (np.bitwise_count), exact brute force.
"""
from __future__ import annotations

import numpy as np

from . import patient as PT
from . import rescorer as RS
from .memory import bundle

ROLE_SEED = 777
RP_SEED = 2026


def pack(bits: np.ndarray) -> np.ndarray:
    """(n, D) {0,1} uint8 -> (n, D / 64) uint64."""
    p = np.packbits(bits.astype(np.uint8), axis=1)
    return np.ascontiguousarray(p).view(np.uint64)


class HammingIndex:
    """Exact Hamming search over packed bit vectors (one or several concatenated blocks)."""

    def __init__(self, packed: np.ndarray):
        self.X = np.ascontiguousarray(packed)

    def dist(self, Q: np.ndarray, chunk: int = 32) -> np.ndarray:
        out = np.empty((len(Q), len(self.X)), np.int32)
        for i in range(0, len(Q), chunk):
            out[i:i + chunk] = np.bitwise_count(Q[i:i + chunk, None, :] ^ self.X[None]).sum(-1, dtype=np.int32)
        return out

    def nbytes_per_item(self) -> int:
        return self.X.shape[1] * 8


class ContentEncoder:
    def __init__(self, dim: int, D: int, seed: int = RP_SEED):
        self.R = np.random.default_rng(seed).standard_normal((dim, D)).astype(np.float32)

    def fit(self, X: np.ndarray):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        return self

    def z(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd).astype(np.float32)

    def encode(self, X: np.ndarray) -> np.ndarray:
        return (self.z(X) @ self.R > 0).astype(np.uint8)


def record_encoder(D: int, fields: list, db_feats: list, seed: int = 0) -> RS.Rescorer:
    """The KNNRescorer record encoder at dimension D (D = 2048 reproduces best_sota's record bits exactly)."""
    old = RS.D
    RS.D = D
    try:
        enc = RS.Rescorer(seed=seed, fields=fields)
        for f in enc.num:
            enc.lv[f].fit(np.array([x[f] for x in db_feats]))
    finally:
        RS.D = old
    return enc


def encode_record(enc: RS.Rescorer, feats: list) -> np.ndarray:
    return enc.encode([{"feats": x} for x in feats])


class Roles:
    def __init__(self, D: int, kmax: int = 5, seed: int = ROLE_SEED):
        rng = np.random.default_rng(seed)
        self.rank = rng.integers(0, 2, (kmax, D), dtype=np.uint8)
        self.gland, self.clin, self.omics = (rng.integers(0, 2, D, dtype=np.uint8) for _ in range(3))
        self.tie = rng.integers(0, 2, D, dtype=np.uint8)

    def patient(self, regions: np.ndarray, gland: np.ndarray = None, clin: np.ndarray = None) -> np.ndarray:
        parts = [regions[i] ^ self.rank[i] for i in range(len(regions))]
        if gland is not None:
            parts.append(gland ^ self.gland)
        if clin is not None:
            parts.append(clin ^ self.clin)
        return bundle([p[None] for p in parts], self.tie)[0]


def clinical_hv(rows, train_cases, cases, seed: int, fields, D: int) -> np.ndarray:
    old = PT.D
    PT.D = D
    try:
        R = PT.ClinicalRecords(rows, {}, train_cases, seed, fields=fields)
        return R.hv(cases)
    finally:
        PT.D = old


# ------------------------------------------------------------------ metrics
def ranking(dist: np.ndarray, valid: np.ndarray = None) -> np.ndarray:
    """Row-wise neighbour order (stable: ties by database index); invalid entries pushed to the end."""
    d = dist.astype(np.float64)
    if valid is not None:
        d = np.where(valid, d, np.inf)
    return np.argsort(d, axis=1, kind="stable")


def average_precision(rel_sorted: np.ndarray, n_valid: np.ndarray) -> np.ndarray:
    """rel_sorted (q, n) relevance in rank order (invalid at the end, rel 0) -> AP per query."""
    hits = np.cumsum(rel_sorted, 1)
    prec = hits / np.arange(1, rel_sorted.shape[1] + 1)
    tot = rel_sorted.sum(1)
    return np.where(tot > 0, (prec * rel_sorted).sum(1) / np.maximum(tot, 1), np.nan)


def auroc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def bootstrap(groups: np.ndarray, fn, n: int = 1000, seed: int = 0):
    """Patient-level bootstrap: resample the unique groups with replacement; fn(index array) -> value.
    Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(u))]
    point = fn(np.arange(len(groups)))
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(u), len(u))
        vals.append(fn(np.concatenate([members[i] for i in pick])))
    vals = np.array(vals, float)
    vals = vals[np.isfinite(vals)]
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
