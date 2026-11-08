#!/usr/bin/env python3
"""
hdc_classifier.py - supervised HDC building blocks for the PI-CAI lesion task.

Design A (patch level): binary hypervectors from Eric's BinaryThermometerHDC, one model per position bin.
  HDCPrototypeClassifier  class hypervectors (normal / lesion), several sub-prototypes per class found by
                          k-means in Hamming space, then perceptron-style HDC retraining (class-balanced):
                          a misclassified patch is added to the nearest sub-prototype of its true class and
                          subtracted from the nearest sub-prototype of the wrongly predicted class.
                          score = max similarity to a lesion sub-prototype - max similarity to a normal one.
  knn_vote_scores         design C: the labelled patch library in a faiss binary index; score = class-frequency
                          weighted share of lesion patches among the k nearest (Hamming) neighbours.

Design B (patient level): HDDB-style record hypervectors.
  LevelEncoder            order-preserving level codes for a numeric field: L levels, each step flips a fresh
                          block of bits, so Hamming distance grows linearly with the level gap (thermometer-like,
                          as HDDB encodes numeric columns); a dedicated random 'missing' hypervector per field.
  SymbolEncoder           random hypervector per category (+ 'unknown').
  bind = XOR, bundle = bitwise majority (ties broken by a fixed random hypervector).
  HDCBinaryClassifier     two class hypervectors (bit counts) + retraining epochs; score = sim(pos) - sim(neg).

Hypervectors are handled as {0,1} uint8 bit arrays (n, D); similarity uses bipolar dot products, which equal
D - 2 x Hamming distance.
"""
from __future__ import annotations

import numpy as np


def unpack_codes(codes_u8: np.ndarray, D: int) -> np.ndarray:
    """faiss-style packed bytes (little bit order, as hdc_codes produces) -> (n, D) {0,1} uint8."""
    return np.unpackbits(codes_u8, axis=1, bitorder="little")[:, :D]


def bipolar(bits: np.ndarray) -> np.ndarray:
    return bits.astype(np.float32) * 2.0 - 1.0


def majority(counts: np.ndarray) -> np.ndarray:
    """Bit counts in bipolar form (sum of +-1) -> bipolar prototype (ties -> +1)."""
    return np.where(counts >= 0, 1.0, -1.0).astype(np.float32)


# ------------------------------------------------------------------ design A: prototypes + retraining
class HDCPrototypeClassifier:
    def __init__(self, n_sub=(10, 5), epochs: int = 5, kmeans_iter: int = 5, batch: int = 2048, seed: int = 0):
        self.n_sub, self.epochs, self.kmeans_iter, self.batch = n_sub, epochs, kmeans_iter, batch
        self.rng = np.random.default_rng(seed)

    def _kmeans(self, Xb: np.ndarray, k: int) -> np.ndarray:
        k = max(1, min(k, len(Xb)))
        C = Xb[self.rng.choice(len(Xb), k, replace=False)].copy()
        for _ in range(self.kmeans_iter):
            a = (Xb @ majority(C).T).argmax(1)
            for j in range(k):
                m = a == j
                if m.any():
                    C[j] = Xb[m].sum(0)
        return C  # bipolar counts (n_sub, D)

    def fit(self, Xb: np.ndarray, y: np.ndarray):
        """Xb: (n, D) bipolar float32; y: 0 = normal, 1 = lesion."""
        self.counts = [self._kmeans(Xb[y == c], self.n_sub[c]) for c in (0, 1)]
        idx = [np.where(y == c)[0] for c in (0, 1)]
        n_bal = min(len(idx[0]), max(len(idx[1]), 1) * 1)  # class-balanced epochs
        for _ in range(self.epochs):
            pick = np.concatenate([self.rng.choice(idx[0], n_bal, replace=False),
                                   self.rng.choice(idx[1], n_bal, replace=len(idx[1]) < n_bal)])
            self.rng.shuffle(pick)
            for i0 in range(0, len(pick), self.batch):
                b = pick[i0:i0 + self.batch]
                X, t = Xb[b], y[b]
                P = [majority(c) for c in self.counts]
                S = [X @ p.T for p in P]  # (nb, n_sub_c)
                best = np.stack([s.max(1) for s in S], 1)
                pred = best.argmax(1)
                wrong = pred != t
                if not wrong.any():
                    continue
                for c in (0, 1):
                    # add to the true class's nearest sub-prototype
                    m = wrong & (t == c)
                    if m.any():
                        np.add.at(self.counts[c], S[c][m].argmax(1), X[m])
                    # subtract from the wrongly predicted class's nearest sub-prototype
                    m = wrong & (pred == c)
                    if m.any():
                        np.add.at(self.counts[c], S[c][m].argmax(1), -X[m])
        self.protos = [majority(c) for c in self.counts]
        return self

    def score(self, Xb: np.ndarray) -> np.ndarray:
        D = Xb.shape[1]
        return ((Xb @ self.protos[1].T).max(1) - (Xb @ self.protos[0].T).max(1)) / D


def knn_vote_scores(lib_codes: np.ndarray, lib_y: np.ndarray, q_codes: np.ndarray, bit_dim: int, k: int = 25):
    """Class-frequency weighted share of lesion patches among the k nearest library patches (Hamming)."""
    import faiss
    ix = faiss.IndexBinaryFlat(bit_dim)
    ix.add(np.ascontiguousarray(lib_codes))
    k = min(k, len(lib_codes))
    _, I = ix.search(np.ascontiguousarray(q_codes), k)
    n1, n0 = max(int(lib_y.sum()), 1), max(int((lib_y == 0).sum()), 1)
    w = np.where(lib_y[I] == 1, 1.0 / n1, 1.0 / n0)
    return (w * (lib_y[I] == 1)).sum(1) / w.sum(1)


# ------------------------------------------------------------------ design B: HDDB-style records
class LevelEncoder:
    """Order-preserving level hypervectors for one numeric field (quantile levels from training values)."""

    def __init__(self, D: int, levels: int, rng: np.random.Generator):
        self.D, self.L = D, levels
        base = rng.integers(0, 2, D, dtype=np.uint8)
        order = rng.permutation(D)
        step = D // (2 * (levels - 1))  # the two extreme levels differ in D/2 bits (orthogonal)
        self.hv = np.empty((levels, D), np.uint8)
        for l in range(levels):
            h = base.copy()
            flip = order[: l * step]
            h[flip] ^= 1
            self.hv[l] = h
        self.missing = rng.integers(0, 2, D, dtype=np.uint8)

    def fit(self, values: np.ndarray):
        v = values[~np.isnan(values)]
        self.edges = np.quantile(v, np.linspace(0, 1, self.L + 1)[1:-1]) if len(v) else np.zeros(self.L - 1)
        return self

    def encode(self, values: np.ndarray) -> np.ndarray:
        out = np.empty((len(values), self.D), np.uint8)
        ok = ~np.isnan(values)
        out[ok] = self.hv[np.searchsorted(self.edges, values[ok], side="right")]
        out[~ok] = self.missing
        return out


class SymbolEncoder:
    def __init__(self, D: int, rng: np.random.Generator):
        self.D, self.rng, self.table = D, rng, {}
        self.unknown = rng.integers(0, 2, D, dtype=np.uint8)

    def fit(self, values):
        for v in sorted(set(values)):
            self.table[v] = self.rng.integers(0, 2, self.D, dtype=np.uint8)
        return self

    def encode(self, values) -> np.ndarray:
        return np.stack([self.table.get(v, self.unknown) for v in values])


def bundle(hvs: list, tie: np.ndarray) -> np.ndarray:
    """Bitwise majority over a list of (n, D) {0,1} arrays; ties broken by the fixed tie hypervector."""
    s = np.sum([h.astype(np.int32) for h in hvs], axis=0) * 2 - len(hvs)
    return np.where(s > 0, 1, np.where(s < 0, 0, tie[None, :])).astype(np.uint8)


class HDCBinaryClassifier:
    """Two class hypervectors by bundling (bit counts), then perceptron-style retraining epochs."""

    def __init__(self, epochs: int = 20, lr: float = 1.0, seed: int = 0):
        self.epochs, self.lr = epochs, lr
        self.rng = np.random.default_rng(seed)

    def fit(self, H: np.ndarray, y: np.ndarray):
        X = bipolar(H)
        self.C = np.stack([X[y == c].sum(0) for c in (0, 1)]).astype(np.float32)
        for _ in range(self.epochs):
            for i in self.rng.permutation(len(X)):
                s = majority(self.C) @ X[i]
                p = int(s[1] > s[0])
                if p != y[i]:
                    self.C[y[i]] += self.lr * X[i]
                    self.C[p] -= self.lr * X[i]
        self.P = majority(self.C)
        return self

    def score(self, H: np.ndarray) -> np.ndarray:
        X = bipolar(H)
        s = X @ self.P.T
        return (s[:, 1] - s[:, 0]) / X.shape[1]
