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


def learn_projection(Xn: np.ndarray, Xl: np.ndarray, W0: np.ndarray, kind: str, seed: int, log=print,
                     n: int = 100_000, shrink: float = 0.1, steps: int = 400, lam: float = 1.0, batch: int = 1024,
                     lr: float = 3e-3) -> np.ndarray:
    """Idea A (label-guided projection, after ProViCNet's patch contrastive loss): replace the random projection
    W0 (in_dim x num_proj) of the thermometer encoder by a linear map learned on one split's training patches
    (prepared descriptors; Xn normal, Xl lesion), class-balanced sample of up to n / 2 each. The thermometer code,
    the memories and everything downstream stay unchanged.
      'lda'  within-class whitening: W = Sw^(-1/2) W0 with Sw the pooled within-class covariance (shrunk by
             `shrink` towards its mean eigenvalue), so Hamming distance approximates the Mahalanobis distance of
             LDA, in which the lesion-vs-normal direction is no longer drowned by high-variance normal directions.
      'nca'  W initialised at W0 and trained by a supervised-contrastive (NCA) loss on the projected descriptors:
             each patch's softmax over -||z_i - z_j||^2 / tau in a balanced batch should put its mass on patches of
             its own class; + lam ||W - W0||^2 / ||W0||^2 keeps the random-projection geometry of the normals."""
    rng = np.random.default_rng(seed + 777)
    k = min(n // 2, len(Xn), len(Xl))
    A = Xn[rng.choice(len(Xn), k, replace=False)].astype(np.float32)
    B = Xl[rng.choice(len(Xl), k, replace=len(Xl) < k)].astype(np.float32)
    if kind == "lda":
        Sw = 0.5 * (np.cov(A, rowvar=False) + np.cov(B, rowvar=False))
        ev, V = np.linalg.eigh(Sw)
        ev = (1 - shrink) * np.maximum(ev, 1e-8) + shrink * ev.mean()
        T = (V / np.sqrt(ev)) @ V.T
        log(f"projection lda: within-class whitening, eigenvalues {ev.min():.3g} .. {ev.max():.3g}")
        return (T @ W0).astype(np.float32)
    if kind != "nca":
        raise ValueError(kind)
    import torch
    torch.manual_seed(seed)
    W0t = torch.from_numpy(W0.astype(np.float32))
    W = torch.nn.Parameter(W0t.clone())
    opt = torch.optim.Adam([W], lr=lr)
    Xa, Xb = torch.from_numpy(A), torch.from_numpy(B)
    y = torch.cat([torch.zeros(batch // 2), torch.ones(batch // 2)])
    same = (y[:, None] == y[None, :]).float()
    eye = torch.eye(batch, dtype=torch.bool)
    with torch.no_grad():
        z = torch.cat([Xa[:batch // 2], Xb[:batch // 2]]) @ W0t
        tau = float(torch.cdist(z, z).pow(2).median())
    g = torch.Generator().manual_seed(seed)
    for t in range(steps):
        i = torch.randint(0, k, (batch // 2,), generator=g)
        j = torch.randint(0, k, (batch // 2,), generator=g)
        z = torch.cat([Xa[i], Xb[j]]) @ W
        logits = (-torch.cdist(z, z).pow(2) / tau).masked_fill(eye, -1e9)
        logp = logits - torch.logsumexp(logits, 1, keepdim=True)
        nca = -torch.logsumexp(logp.masked_fill(same == 0, -1e9).masked_fill(eye, -1e9), 1).mean()
        reg = lam * (W - W0t).pow(2).sum() / W0t.pow(2).sum()
        loss = nca + reg
        opt.zero_grad()
        loss.backward()
        opt.step()
        if t % 100 == 0 or t == steps - 1:
            log(f"projection nca step {t}: loss {nca.item():.4f} reg {reg.item():.4f}")
    return W.detach().numpy().astype(np.float32)
