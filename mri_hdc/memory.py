"""HDC associative memories and record encoders.

HDCPrototypeClassifier  patch memory of one position bin: several sub-prototypes per class (normal / lesion)
                        initialised by k-means in Hamming space, then refined by class-balanced perceptron-style
                        retraining: a misclassified patch is added to the nearest sub-prototype of its true class and
                        subtracted from the nearest sub-prototype of the predicted class. Prototypes are bit-count
                        accumulators read out by majority (sign). Patch score = soft-max similarity to the lesion
                        sub-prototypes - soft-max similarity to the normal ones.
HDCBinaryClassifier     two class hypervectors by bundling + retraining epochs (candidate re-scorer, clinical record).
LevelEncoder            order-preserving level codes for numeric fields (HDDB style; extremes are orthogonal),
                        with a dedicated random 'missing' hypervector.
SymbolEncoder           random hypervector per category (+ 'unknown').
bundle                  bitwise majority; bind = XOR.
"""
from __future__ import annotations

import numpy as np


def majority(counts: np.ndarray) -> np.ndarray:
    """Bipolar bit counts -> bipolar prototype (ties -> +1)."""
    return np.where(counts >= 0, 1.0, -1.0).astype(np.float32)


class HDCPrototypeClassifier:
    def __init__(self, n_sub=(20, 5), epochs: int = 20, kmeans_iter: int = 5, batch: int = 2048, seed: int = 0,
                 lr: float = 2.0, softmax_t: float = 0.02):
        self.n_sub, self.epochs, self.kmeans_iter, self.batch = n_sub, epochs, kmeans_iter, batch
        self.lr, self.T = lr, softmax_t
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
        return C

    def fit(self, Xb: np.ndarray, y: np.ndarray):
        """Xb: (n, D) bipolar float32; y: 0 = normal, 1 = lesion."""
        self.counts = [self._kmeans(Xb[y == c], self.n_sub[c]) for c in (0, 1)]
        idx = [np.where(y == c)[0] for c in (0, 1)]
        n_bal = min(len(idx[0]), max(len(idx[1]), 1))  # class-balanced epochs
        for _ in range(self.epochs):
            pick = np.concatenate([self.rng.choice(idx[0], n_bal, replace=False),
                                   self.rng.choice(idx[1], n_bal, replace=len(idx[1]) < n_bal)])
            self.rng.shuffle(pick)
            for i0 in range(0, len(pick), self.batch):
                b = pick[i0:i0 + self.batch]
                X, t = Xb[b], y[b]
                S = [X @ majority(c).T for c in self.counts]
                pred = np.stack([s.max(1) for s in S], 1).argmax(1)
                wrong = pred != t
                if not wrong.any():
                    continue
                for c in (0, 1):
                    m = wrong & (t == c)  # add to the true class's nearest sub-prototype
                    if m.any():
                        np.add.at(self.counts[c], S[c][m].argmax(1), self.lr * X[m])
                    m = wrong & (pred == c)  # subtract from the wrongly predicted class's nearest sub-prototype
                    if m.any():
                        np.add.at(self.counts[c], S[c][m].argmax(1), -self.lr * X[m])
        self.protos = [majority(c) for c in self.counts]
        return self

    def _softmax(self, S: np.ndarray) -> np.ndarray:
        m = S.max(1, keepdims=True)
        return (m + self.T * np.log(np.exp((S - m) / self.T).sum(1, keepdims=True)))[:, 0]

    def score(self, Xb: np.ndarray) -> np.ndarray:
        D = Xb.shape[1]
        return self._softmax(Xb @ self.protos[1].T / D) - self._softmax(Xb @ self.protos[0].T / D)


class LevelEncoder:
    def __init__(self, D: int, levels: int, rng: np.random.Generator):
        self.D, self.L = D, levels
        base = rng.integers(0, 2, D, dtype=np.uint8)
        order = rng.permutation(D)
        step = D // (2 * (levels - 1))  # the two extreme levels differ in D / 2 bits
        self.hv = np.empty((levels, D), np.uint8)
        for l in range(levels):
            h = base.copy()
            h[order[: l * step]] ^= 1
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
    """Bitwise majority over a list of (n, D) {0, 1} arrays; ties broken by the fixed tie hypervector."""
    s = np.sum([h.astype(np.int32) for h in hvs], axis=0) * 2 - len(hvs)
    return np.where(s > 0, 1, np.where(s < 0, 0, tie[None, :])).astype(np.uint8)


class HDCBinaryClassifier:
    def __init__(self, epochs: int = 20, lr: float = 1.0, seed: int = 0):
        self.epochs, self.lr = epochs, lr
        self.rng = np.random.default_rng(seed)

    def fit(self, H: np.ndarray, y: np.ndarray):
        X = H.astype(np.float32) * 2.0 - 1.0
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
        X = H.astype(np.float32) * 2.0 - 1.0
        s = X @ self.P.T
        return (s[:, 1] - s[:, 0]) / X.shape[1]
