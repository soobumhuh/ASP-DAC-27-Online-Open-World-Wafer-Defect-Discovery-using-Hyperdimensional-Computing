"""HDC candidate re-scorer: re-rank lesion candidates with an HDDB-style record hypervector.

Record of a candidate (2,048 bits): each numeric field as a 16-level code (quantiles of the training candidates) bound
(XOR) to a random field key, the zone at its peak as a symbol, all bundled by majority. Fields: log size, peak, mean
probability, number of slices, slice level, inside-vs-ring contrast of T2W / ADC / HBV, compactness.
Two class hypervectors (true positive = IoU >= 0.1 with an expert lesion, false positive otherwise) by bundling and
retraining. New confidence = conf ** (1 - alpha) x rank ** alpha, rank = the HDC score's rank among the training
candidates. Training candidates are always out-of-sample: for test fold f, the CV test candidates of the other folds
(each extracted from a model that never saw them).
"""
from __future__ import annotations

import numpy as np

from .memory import HDCBinaryClassifier, LevelEncoder, SymbolEncoder, bundle

D = 2048
NUM = ["logsize", "peak", "mean", "nslices", "level", "c_t2w", "c_adc", "c_hbv", "compact"]


SYM = ["s_t2w", "s_adc", "s_hbv", "s_peak"]  # left-right symmetry fields (candidate minus mirror region)


class Rescorer:
    def __init__(self, seed: int = 0, levels: int = 16, epochs: int = 50, fields: list = None):
        rng = np.random.default_rng(seed)
        self.num = list(fields) if fields else NUM
        self.keys = {f: rng.integers(0, 2, D, dtype=np.uint8) for f in NUM + ["zone"]}  # published random streams
        self.tie = rng.integers(0, 2, D, dtype=np.uint8)
        self.lv = {f: LevelEncoder(D, levels, rng) for f in NUM}
        for f in self.num:  # extra fields get their own keys / level codes after the published ones
            if f not in self.keys:
                self.keys[f] = rng.integers(0, 2, D, dtype=np.uint8)
                self.lv[f] = LevelEncoder(D, levels, rng)
        self.zone = SymbolEncoder(D, rng).fit([0, 1, 2])
        self.epochs, self.seed = epochs, seed

    def encode(self, cands):
        parts = [self.lv[f].encode(np.array([c["feats"][f] for c in cands])) ^ self.keys[f] for f in self.num]
        parts.append(self.zone.encode([c["feats"]["zone"] for c in cands]) ^ self.keys["zone"])
        return bundle(parts, self.tie)

    def fit(self, cands):
        for f in self.num:
            self.lv[f].fit(np.array([c["feats"][f] for c in cands]))
        y = np.array([int(c["tp"]) for c in cands])
        H = self.encode(cands)
        self.clf = HDCBinaryClassifier(epochs=self.epochs, seed=self.seed).fit(H, y)
        self.train_scores = np.sort(self.clf.score(H))
        return self

    def rank(self, cands):
        s = self.clf.score(self.encode(cands))
        return (np.searchsorted(self.train_scores, s) + 0.5) / (len(self.train_scores) + 1)

    def confidences(self, ext: dict, alpha: float) -> dict:
        """{case: re-scored confidences} = conf ** (1 - alpha) x rank ** alpha."""
        out = {}
        for k, (cands, _) in ext.items():
            if cands:
                conf = np.array([c["conf"] for c in cands])
                out[k] = list(conf ** (1 - alpha) * self.rank(cands) ** alpha) if alpha > 0 else list(conf)
            else:
                out[k] = []
        return out


class ZoneRescorer:
    """Separate re-scorers for transition-zone candidates (zone 2 at the peak) and all others; each candidate is ranked
    against the training candidates of its own group. Falls back to one shared re-scorer when a group has < min_tp
    true positives."""

    def __init__(self, min_tp: int = 20, **kw):
        self.kw, self.min_tp = kw, min_tp

    def fit(self, cands):
        tz = [c for c in cands if c["feats"]["zone"] == 2]
        other = [c for c in cands if c["feats"]["zone"] != 2]
        self.all = Rescorer(**self.kw).fit(cands)
        self.parts = {g: (Rescorer(**self.kw).fit(cs) if sum(c["tp"] for c in cs) >= self.min_tp else self.all)
                      for g, cs in (("tz", tz), ("other", other))}
        return self

    def rank(self, cands):
        out = np.zeros(len(cands))
        for g, sel in (("tz", [c["feats"]["zone"] == 2 for c in cands]), ("other", [c["feats"]["zone"] != 2 for c in cands])):
            idx = np.where(sel)[0]
            if len(idx):
                out[idx] = self.parts[g].rank([cands[i] for i in idx])
        return out

    confidences = Rescorer.confidences


def make(rc: dict, epochs: int):
    """Re-scorer from the config's 'rescorer' section: extra fields (rescorer.fields: NUM + SYM) and / or zone split."""
    fields = (NUM + (SYM if rc.get("symmetry") else []) + list(rc.get("extra_fields", []))) \
        if (rc.get("symmetry") or rc.get("extra_fields")) else None
    kw = dict(seed=rc.get("seed", 0), epochs=epochs, fields=fields)
    return ZoneRescorer(**kw) if rc.get("zone_split") else Rescorer(**kw)
