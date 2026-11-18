"""Patient-level HDC scorer for the AUROC-only version.

Record encoding (HDDB style): every field f (a map descriptor of one map family, or a clinical field) is quantised
into L levels (quantile edges of the training records) and coded by a thermometer level hypervector (neighbouring
levels similar, the extremes orthogonal), bound to a random bipolar field key, and the fields are bundled by
summation (no binarisation, so no tie noise). A missing field contributes nothing.

Two associative-memory scorers over the training records:
  proto  two class hypervectors (csPCa yes / no) by bundling, refined by OnlineHD-style retraining epochs (a
         misclassified record is added to its class and subtracted from the other, scaled by 1 - similarity);
         score = cos(x, C_yes) - cos(x, C_no)
  knn    the vector-DB view: cosine k nearest training records, score = similarity-weighted share of csPCa among
         them, with class-balanced votes
Optional per-record training weights (e.g. ISUP 1 negatives as hard negatives); the label is always case csPCa.
"""
from __future__ import annotations

import numpy as np


class RecordEncoder:
    def __init__(self, names: list, D: int = 10000, levels: int = 16, seed: int = 0):
        self.names, self.D, self.L = list(names), D, levels
        rng = np.random.default_rng(seed)
        self.keys = rng.choice(np.array([-1, 1], np.int8), (len(names), D))
        # thermometer codes: level l flips the first l * step positions of a random order of a random base
        self.tables = []
        step = D // (2 * (levels - 1))
        for _ in names:
            base = rng.choice(np.array([-1, 1], np.int8), D)
            order = rng.permutation(D)
            t = np.repeat(base[None], levels, 0)
            for l in range(1, levels):
                t[l, order[: l * step]] *= -1
            self.tables.append(t)

    def fit(self, X: np.ndarray):
        self.edges = []
        for j in range(X.shape[1]):
            v = X[:, j][~np.isnan(X[:, j])]
            self.edges.append(np.unique(np.quantile(v, np.linspace(0, 1, self.L + 1)[1:-1])) if len(v) else
                              np.zeros(0))
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        H = np.zeros((len(X), self.D), np.float32)
        for j in range(X.shape[1]):
            ok = ~np.isnan(X[:, j])
            if not ok.any():
                continue
            lv = np.searchsorted(self.edges[j], X[ok, j], side="right")
            lv = np.minimum(lv * (self.L - 1) // max(len(self.edges[j]), 1), self.L - 1) if len(self.edges[j]) else lv * 0
            H[ok] += (self.tables[j][lv] * self.keys[j]).astype(np.float32)
        return H


def _unit(H):
    return H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)


class ProtoScorer:
    def __init__(self, epochs: int = 10, lr: float = 1.0, seed: int = 0):
        self.epochs, self.lr, self.rng = epochs, lr, np.random.default_rng(seed)

    def fit(self, H, y, w=None):
        X = _unit(H)
        w = np.ones(len(y)) if w is None else np.asarray(w, float)
        cb = np.array([1.0 / w[y == c].sum() for c in (0, 1)]) * w.sum() / 2  # class balance
        self.C = np.stack([(X[y == c] * w[y == c, None]).sum(0) * cb[c] for c in (0, 1)])
        for _ in range(self.epochs):
            for i in self.rng.permutation(len(X)):
                Cn = _unit(self.C)
                s = Cn @ X[i]
                p = int(s[1] > s[0])
                t = int(y[i])
                if p != t:
                    a = self.lr * w[i] * cb[t]
                    self.C[t] += a * (1 - s[t]) * X[i]
                    self.C[p] -= a * (1 - s[p]) * X[i]
        return self

    def score(self, H):
        s = _unit(H) @ _unit(self.C).T
        return s[:, 1] - s[:, 0]


class KNNScorer:
    def __init__(self, k: int = 50):
        self.k = k

    def fit(self, H, y, w=None):
        self.X, self.y = _unit(H), np.asarray(y)
        w = np.ones(len(y)) if w is None else np.asarray(w, float)
        cb = np.array([1.0 / w[self.y == c].sum() for c in (0, 1)])
        self.v = w * cb[self.y]
        return self

    def score(self, H):
        S = _unit(H) @ self.X.T
        k = min(self.k, S.shape[1])
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        s = np.take_along_axis(S, idx, 1)
        a = np.maximum(s, 0) * self.v[idx]
        return (a * (self.y[idx] == 1)).sum(1) / (a.sum(1) + 1e-12)
