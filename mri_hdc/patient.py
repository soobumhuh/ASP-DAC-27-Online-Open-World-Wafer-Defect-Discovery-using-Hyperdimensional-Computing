"""Patient level: clinical HDC record and its merge with the image score.

Clinical record (HDDB style, 2,048 bits): PSA density, PSA, prostate volume, age as 16-level codes (quantiles of the
split's training cases; a random 'missing' hypervector when absent), centre, scanner model and HBV b-value as
symbols; each bound (XOR) to a field key, bundled by majority. Class hypervectors csPCa yes / no by bundling +
retraining (epochs chosen on the split's validation cases from {5, 20, 50}). Only acquisition / pre-biopsy fields are
used; ISUP, Gleason, PI-RADS, histopathology and the csPCa label are never inputs.

Patient score = z(image case score) + w x z(clinical record score); z uses the split's validation cases.
"""
from __future__ import annotations

import csv

import numpy as np
from sklearn.metrics import roc_auc_score

from .memory import HDCBinaryClassifier, LevelEncoder, SymbolEncoder, bundle

D = 2048
NUMERIC = ["psad", "psa", "prostate_volume", "patient_age"]
SYMBOL = ["center", "scanner", "bvalue"]
_UNUSED = ["pz_max", "tz_max", "pz_cnt", "tz_cnt", "img"]  # keys drawn (unused) to keep the published random streams


class ClinicalRecords:
    def __init__(self, rows: dict, scan: dict, train_cases, seed: int, levels: int = 16, fields=None):
        rng = np.random.default_rng(1000 + seed)
        self.rows, self.scan = rows, scan
        self.fields = list(fields) if fields else NUMERIC + SYMBOL
        self.keys = {f: rng.integers(0, 2, D, dtype=np.uint8) for f in NUMERIC + SYMBOL + _UNUSED}
        self.tie = rng.integers(0, 2, D, dtype=np.uint8)
        self.enc = {f: LevelEncoder(D, levels, rng).fit(np.array([self.val(c, f) for c in train_cases])) for f in NUMERIC}
        for f in SYMBOL:
            self.enc[f] = SymbolEncoder(D, rng).fit([self.sym(c, f) for c in train_cases])

    def val(self, c, f):
        v = (self.rows[c].get(f) or "").strip()
        return float(v) if v else np.nan

    def sym(self, c, f):
        return self.rows[c].get("center", "NA") if f == "center" else self.scan.get(c, ("NA", "NA"))[f != "scanner"]

    def hv(self, cases):
        parts = []
        for f in self.fields:
            x = self.enc[f].encode(np.array([self.val(c, f) for c in cases])) if f in NUMERIC else \
                self.enc[f].encode([self.sym(c, f) for c in cases])
            parts.append(x ^ self.keys[f])
        return bundle(parts, self.tie) if len(parts) > 1 else parts[0]


def read_marksheet(path: str) -> dict:
    return {int(r["study_id"]): r for r in csv.DictReader(open(path))}


class HDCOrdinalClassifier:
    """Idea E (grade-ordered score, after MRI-PTPCa): one class hypervector per grade group g = 0 (no csPCa:
    ISUP 0-1), 1 (ISUP 2), 2 (ISUP 3), 3 (ISUP 4-5), seeded with an ordered level code (LevelEncoder: neighbouring
    grades are similar, the extremes orthogonal; weight `prior` x class size) plus the bundle of the class's training
    records, then refined by multi-class perceptron retraining. Score = expected grade under a soft-max over the
    record's similarity to the class hypervectors. ISUP is used only as the training label."""

    def __init__(self, G: int = 4, epochs: int = 20, lr: float = 1.0, seed: int = 0, prior: float = 0.5, T: float = 0.05):
        self.G, self.epochs, self.lr, self.prior, self.T = G, epochs, lr, prior, T
        self.rng = np.random.default_rng(seed)

    def fit(self, H: np.ndarray, g: np.ndarray):
        X = H.astype(np.float32) * 2.0 - 1.0
        L = LevelEncoder(X.shape[1], self.G, self.rng).hv.astype(np.float32) * 2 - 1
        self.C = np.stack([X[g == c].sum(0) + self.prior * max((g == c).sum(), 1) * L[c] for c in range(self.G)])
        w = 1.0 / np.maximum(np.bincount(g, minlength=self.G), 1)  # class-balanced updates
        for _ in range(self.epochs):
            for i in self.rng.permutation(len(X)):
                p = int(np.argmax(np.where(self.C >= 0, 1.0, -1.0) @ X[i]))
                if p != g[i]:
                    self.C[g[i]] += self.lr * w[g[i]] * len(X) / self.G * X[i]
                    self.C[p] -= self.lr * w[g[i]] * len(X) / self.G * X[i]
        self.P = np.where(self.C >= 0, 1.0, -1.0).astype(np.float32)
        return self

    def score(self, H: np.ndarray) -> np.ndarray:
        X = H.astype(np.float32) * 2.0 - 1.0
        s = X @ self.P.T / X.shape[1] / self.T
        p = np.exp(s - s.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        return p @ np.arange(self.G)


def grade_group(rows, c) -> int:
    v = (rows[c].get("case_ISUP") or "0").strip()
    i = int(float(v)) if v else 0
    return 0 if i <= 1 else min(i - 1, 3)


def clinical_scores(rows, scan, cases: dict, y: dict, seed: int, epochs=(5, 20, 50), fields=None,
                    kind: str = "binary") -> dict:
    """cases / y: {'train', 'val', 'test'} -> study ids / labels. Returns {'val': scores, 'test': scores, 'epochs'}.
    kind 'ordinal' (idea E): grade-ordered classifier trained on the case ISUP grade group of the training cases."""
    best = None
    for ep in epochs:
        R = ClinicalRecords(rows, scan, cases["train"], seed, fields=fields)
        if kind == "ordinal":
            clf = HDCOrdinalClassifier(epochs=ep, seed=seed).fit(R.hv(cases["train"]),
                                                                 np.array([grade_group(rows, c) for c in cases["train"]]))
        else:
            clf = HDCBinaryClassifier(epochs=ep, seed=seed).fit(R.hv(cases["train"]), np.asarray(y["train"]))
        v = roc_auc_score(y["val"], clf.score(R.hv(cases["val"])))
        if best is None or v > best[0]:
            best = (v, ep, R, clf)
    _, ep, R, clf = best
    return {"val": clf.score(R.hv(cases["val"])), "test": clf.score(R.hv(cases["test"])), "epochs": ep}


def zs(ref, x):
    ref = np.asarray(ref, float)
    return (np.asarray(x, float) - ref.mean()) / (ref.std() + 1e-9)


def merge(img_val, img_test, clin_val, clin_test, w: float):
    """z(image) + w x z(clinical), both standardised with the validation cases."""
    return zs(img_val, img_test) + w * zs(clin_val, clin_test)
