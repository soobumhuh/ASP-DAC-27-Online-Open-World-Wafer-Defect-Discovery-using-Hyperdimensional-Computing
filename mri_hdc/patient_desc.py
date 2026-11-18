"""Patient-level descriptors of one case's z map (AUROC-only version).

The lesion-detection pipeline reduces a case to the maximum of at most five extracted candidates. For a patient-level
decision this throws away most of the map, so here every case is described by the whole map: global and zone-wise
quantiles and top-k means of the z map, suspicious volume above several z levels, 3D-smoothed peak, size of the
largest suspicious component, and the absolute ADC / relative HBV under the most suspicious voxels.

Inputs are only the model's map, the AI zone mask and the MRI itself; the expert lesion mask stored next to the map
is never read.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

THR = (1.0, 2.0, 3.0, 4.0)
TOPK = (10, 50, 200, 1000)


def _topmean(v: np.ndarray, k: int) -> float:
    if len(v) == 0:
        return np.nan
    k = min(k, len(v))
    return float(np.partition(v, len(v) - k)[len(v) - k:].mean())


def describe(c: dict, adc: np.ndarray = None, hbv: np.ndarray = None) -> dict:
    """c: one case of a map pickle ({'map', 'zone', ...}); adc: absolute ADC (Z, S, S) on the same grid; hbv: the
    case-normalised HBV channel (Z, S, S). Returns {name: value}."""
    m = c["map"].astype(np.float32)
    g = ~np.isnan(m)
    v = m[g]
    out = {}
    if len(v) == 0:
        return out
    for q in (50, 90, 95, 99, 99.9):
        out[f"q{q}"] = float(np.percentile(v, q))
    out["max"] = float(v.max())
    for k in TOPK:
        out[f"top{k}"] = _topmean(v, k)
    for t in THR:
        out[f"vol{t:g}"] = float(np.log1p((v > t).sum()))
        out[f"frac{t:g}"] = float((v > t).mean())
    # zones of the AI zone mask (1, 2); max / q99 / top-50 mean / suspicious volume per zone
    for zc in (1, 2):
        vz = m[g & (c["zone"] == zc)]
        out[f"z{zc}_max"] = float(vz.max()) if len(vz) else np.nan
        out[f"z{zc}_q99"] = float(np.percentile(vz, 99)) if len(vz) else np.nan
        out[f"z{zc}_top50"] = _topmean(vz, 50)
        out[f"z{zc}_vol2"] = float(np.log1p((vz > 2).sum()))
    # 3D-smoothed peak (normalised convolution inside the gland): blob-level evidence
    w = g.astype(np.float32)
    num = ndimage.gaussian_filter(np.nan_to_num(m), sigma=(1, 2, 2))
    den = ndimage.gaussian_filter(w, sigma=(1, 2, 2))
    sm = np.where(g, num / np.maximum(den, 1e-6), -np.inf)
    out["smax"] = float(sm.max())
    # largest connected suspicious component (z > 2, 3D 6-connectivity) and number of components above 3
    lab, n = ndimage.label(np.nan_to_num(m, nan=-1e3) > 2)
    out["cc2_max"] = float(np.log1p(np.bincount(lab.ravel())[1:].max())) if n else 0.0
    _, n3 = ndimage.label(np.nan_to_num(m, nan=-1e3) > 3)
    out["cc3_n"] = float(n3)
    out["slices3"] = float((np.nanmax(np.where(g, m, -1e3), axis=(1, 2)) > 3).sum())
    # tissue under the 50 most suspicious voxels
    flat = np.where(g, m, -np.inf).ravel()
    top = np.argpartition(flat, -min(50, len(v)))[-min(50, len(v)):]
    if adc is not None:
        a = adc.astype(np.float32).ravel()
        ag = adc[g].astype(np.float32)
        out["adc_top"] = float(a[top].mean())
        out["adc_p5"] = float(np.percentile(ag, 5))
        out["adc_p10"] = float(np.percentile(ag, 10))
        out["adc_rel"] = float(a[top].mean() - np.median(ag))
    if hbv is not None:
        h = hbv.astype(np.float32).ravel()
        out["hbv_rel"] = float(h[top].mean() - np.median(hbv[g]))
        out["hbv_p99"] = float(np.percentile(hbv[g], 99))
    return out
