"""
picai_bundle.py - MRI version of prepare_normal_only_phased_data().

Builds the same WaferDataBundle that Eric's pipeline expects, from the .npz
written by picai_slices.py. Everything downstream (HDC encoder, Stage 1 gate,
Stage 2 clusterer, metrics) runs unchanged.

Two tasks:
  lesion  : normal = slices from csPCa-negative patients
            unknown = slices with an expert csPCa lesion, one class per case ISUP grade
            (slices from positive patients with no lesion are dropped: they are neither clean normals nor lesions)
            With keep_pos_nonlesion=True, val/test keep those slices as an extra "pos_nonlesion" group:
            it is left out of known_ids/unknown_ids (so out of slice metrics and the Stage 2 stream),
            but its slices still count toward the patient score. Train normals stay negative-patient only.
  quality : normal = clean slices from csPCa-negative patients
            unknown = the same kind of slices with a synthetic artifact (noise, blur, ghosting, bias field)

Splits are by PATIENT, never by slice, so no patient appears in both train and test.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

from mri_artifacts import ARTIFACTS
from mri_features import extract
from wafer_data_prep_hdc_stage2 import WaferDataBundle, build_fixed_normal_only_phased_stream
from wafer_hdc_encoding import apply_clip, apply_standardize, fit_clip_stats, fit_standardize


def split_patients(pids: np.ndarray, pos: np.ndarray, val_frac: float, test_frac: float,
                   rng: np.random.Generator) -> Dict[int, str]:
    """Assign each patient to train / val / test, keeping the positive rate similar."""
    out = {}
    for flag in (0, 1):
        group = np.unique(pids[pos == flag])
        rng.shuffle(group)
        n_test = int(round(test_frac * len(group)))
        n_val = int(round(val_frac * len(group)))
        for i, p in enumerate(group):
            out[int(p)] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return out


POS_NONLESION = "pos_nonlesion"


def _lesion_labels(d, min_lesion_pixels: int, isup_classes: bool, keep_pos_nonlesion: bool):
    lesion = d["lesion_pixels"] >= min_lesion_pixels
    normal = d["case_cspca"] == 0
    keep = lesion | normal
    split_keep = keep.copy()  # patient split is always drawn from these slices, so the flag never changes it
    if isup_classes:
        names = ["normal", "lesion_ISUP2", "lesion_ISUP3", "lesion_ISUP4-5"]
        isup = np.clip(d["case_isup"], 2, 4) - 1  # 2->1, 3->2, 4 or 5->3
        y = np.where(lesion, isup, 0)
    else:
        names = ["normal", "csPCa_lesion"]
        y = lesion.astype(np.int64)
    if keep_pos_nonlesion:
        pos_nonlesion = (d["case_cspca"] == 1) & ~lesion
        y = np.where(pos_nonlesion, len(names), y)
        names = names + [POS_NONLESION]
        keep = keep | pos_nonlesion
    return keep, split_keep, y.astype(np.int64), names


def prepare_picai_bundle(
    slices_npz: str,
    task: str = "lesion",
    feature_set: str = "stats",
    seed: int = 0,
    val_frac: float = 0.15,
    test_frac: float = 0.25,
    min_lesion_pixels: int = 10,
    isup_classes: bool = True,
    keep_pos_nonlesion: Optional[bool] = None,  # None -> True for task=lesion
    artifact_frac: float = 0.5,
    clip_low_q: float = 1.0,
    clip_high_q: float = 99.0,
    cache_dir: str = "./mri_feature_cache",
) -> Tuple[WaferDataBundle, Dict[str, Dict[str, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    d = dict(np.load(slices_npz))
    images = d["images"].astype(np.float32)

    if keep_pos_nonlesion is None:
        keep_pos_nonlesion = task == "lesion"
    if task == "lesion":
        keep, split_keep, y_all, label_names = _lesion_labels(d, min_lesion_pixels, isup_classes, keep_pos_nonlesion)
    elif task == "quality":
        keep = d["case_cspca"] == 0
        split_keep = keep
        y_all = np.zeros(len(images), dtype=np.int64)
        label_names = ["clean"] + list(ARTIFACTS)
    else:
        raise ValueError("task must be 'lesion' or 'quality'")

    pids = d["patient_id"]
    split_of = split_patients(pids[split_keep], d["case_cspca"][split_keep], val_frac, test_frac, rng)
    unsplit = sorted({int(p) for p in pids[keep]} - set(split_of))
    if unsplit:  # positive patients with no lesion slice at all: not in the flag-off split, so leave them out
        print(f"note: {len(unsplit)} positive patient(s) with no lesion slice >= {min_lesion_pixels} px left out: {unsplit}")
        keep = keep & np.isin(pids, list(split_of))

    pnl_id = label_names.index(POS_NONLESION) if POS_NONLESION in label_names else -1
    parts: Dict[str, Dict[str, list]] = {s: {"img": [], "y": [], "pid": [], "z": [], "cspca": []}
                                         for s in ("train", "val", "test")}
    for i in np.where(keep)[0]:
        s = split_of[int(pids[i])]
        if s == "train" and y_all[i] == pnl_id:
            continue  # train normals come from negative patients only
        parts[s]["img"].append(images[i]); parts[s]["y"].append(y_all[i])
        parts[s]["pid"].append(pids[i]); parts[s]["z"].append(d["slice_z"][i])
        parts[s]["cspca"].append(d["case_cspca"][i])
        # quality task: val/test also get a corrupted copy of some slices
        if task == "quality" and s != "train" and rng.random() < artifact_frac:
            k = int(rng.integers(len(ARTIFACTS)))
            bad = list(ARTIFACTS.values())[k](images[i].copy(), rng).astype(np.float32)
            parts[s]["img"].append(bad); parts[s]["y"].append(k + 1)
            parts[s]["pid"].append(pids[i]); parts[s]["z"].append(d["slice_z"][i])
            parts[s]["cspca"].append(d["case_cspca"][i])

    meta = {s: {k: np.asarray(v) for k, v in p.items()} for s, p in parts.items()}
    for s in meta:
        if len(meta[s]["img"]) == 0:
            raise RuntimeError(f"split '{s}' is empty: use more patients")

    # features (cached, because texture features on thousands of slices take a while)
    os.makedirs(cache_dir, exist_ok=True)
    tag = f"{os.path.basename(slices_npz)}_{task}_{feature_set}_seed{seed}_v{val_frac}_t{test_frac}_a{artifact_frac}_m{min_lesion_pixels}"
    if keep_pos_nonlesion and task == "lesion":
        tag += "_pnl"
    cache = os.path.join(cache_dir, tag + ".npz")
    if os.path.exists(cache):
        X = dict(np.load(cache))
    else:
        X = {s: extract(meta[s]["img"], feature_set) for s in meta}
        np.savez_compressed(cache, **X)
    X_train, X_val, X_test = X["train"], X["val"], X["test"]
    y_train, y_val, y_test = (meta[s]["y"].astype(np.int64) for s in ("train", "val", "test"))

    normal_label = 0
    known_ids = [0]
    unknown_ids = [c for c in range(1, len(label_names)) if c != pnl_id]
    train_known = y_train == 0
    val_known = y_val == 0

    # same normal-only preprocessing as Eric's code
    lo, hi = fit_clip_stats(X_train[train_known], clip_low_q, clip_high_q)
    X_train, X_val, X_test = (apply_clip(x, lo, hi) for x in (X_train, X_val, X_test))
    mu, sigma = fit_standardize(X_train[train_known])
    X_train, X_val, X_test = (apply_standardize(x, mu, sigma) for x in (X_train, X_val, X_test))

    stream_sources, stream_indices, y_stream = build_fixed_normal_only_phased_stream(
        y_train=y_train, y_val=y_val, y_test=y_test, known_ids=known_ids,
        unknown_ids=unknown_ids, label_names=label_names, rng=rng,
    )

    bundle = WaferDataBundle(
        label_names=label_names, normal_label=normal_label,
        known_ids=known_ids, unknown_ids=unknown_ids,
        known_names=[label_names[0]], unknown_names=[label_names[c] for c in unknown_ids],
        W_train=meta["train"]["img"], W_val=meta["val"]["img"], W_test=meta["test"]["img"],
        X_aux_train=X_train, X_aux_val=X_val, X_aux_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        feature_names=[f"{feature_set}_{i}" for i in range(X_train.shape[1])],
        train_known_mask=train_known, val_known_mask=val_known,
        test_static_mask=y_test != pnl_id,
        stream_sources=stream_sources, stream_indices=stream_indices, y_stream_true=y_stream,
    )
    for s in ("train", "val", "test"):
        counts = {label_names[c]: int(n) for c, n in zip(*np.unique(meta[s]["y"], return_counts=True))}
        print(f"{s:5s}: {len(np.unique(meta[s]['pid']))} patients, slices {counts}")
    print("feature dim:", X_train.shape[1])
    return bundle, meta
