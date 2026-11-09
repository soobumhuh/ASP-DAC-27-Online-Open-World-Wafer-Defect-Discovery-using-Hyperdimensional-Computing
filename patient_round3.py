#!/usr/bin/env python3
"""
patient_round3.py - design B patient score, round 3: does clinical data add to the strong #110 image score?

Inputs: the cross-fitted per-case image dumps from patient_hdc.py (train cases scored by models that did not see
them; val / test by the model trained on all training cases) for the 5 CV splits, plus the marksheet / scanner.
Every hyperparameter is chosen per split on that split's internal validation patients (15% of the training folds);
the test fold is scored once. Fold 0's test = the held-out fold-0 test used throughout.

  a  all-HDC record: clinical fields (16-level codes, 'missing' HVs) + per-zone image fields + the image max score as a
     fine level code (L levels, linear Hamming similarity) repeated r times in the bundle (weighted bundling);
     tuned: L in {64, 256, 1024}, r in {1, 3, 5, 9}, retraining epochs in {5, 20, 50}
  b  plain-number merge (approved variant): score = z(image) + w * z(clinical HDC record), and a rank average
     (rank(image) + w * rank(clinical)) / (1 + w); tuned: w in {0, .1, .25, .5, 1, 2}, record epochs in {5, 20, 50}
  c  references: image max score alone, clinical-only HDC record
PI-CAI = (patient AUROC + design-A lesion AP of the same fold) / 2: picai_eval's lesion AP does not depend on the
case-level score.

  uv run python patient_round3.py --dumps results/lesion_ap2/B_t110 --ap_json results/lesion_ap2/final_t110.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from hdc_classifier import HDCBinaryClassifier, LevelEncoder, SymbolEncoder, bundle
from patient_hdc import NUMERIC, SYMBOL, scanner_info

D = 2048
ZONES = ["pz_max", "tz_max", "pz_cnt", "tz_cnt"]


def load_split(dump_dir, name):
    img = {}
    for tag, split in (("base", "val"), ("base", "test"), ("cf0", "test"), ("cf1", "test"), ("cf2", "test")):
        dmp = dict(np.load(os.path.join(dump_dir, name, f"cases_{tag}.npz")))
        for i, c in enumerate(dmp[f"{split}_study"]):
            r = {"img": float(dmp[f"{split}_max"][i])}
            for zn in ("pz", "tz"):
                r[f"{zn}_max"] = float(dmp[f"{split}_max_{zn}"][i])
                r[f"{zn}_cnt"] = float(np.log1p(dmp[f"{split}_cnt_{zn}"][i]))
            img[int(c)] = r
    return img


class Records:
    def __init__(self, rows, scan, img, train_cases, seed, levels=16, img_levels=256):
        rng = np.random.default_rng(1000 + seed)
        self.rows, self.scan, self.img = rows, scan, img
        self.keys = {f: rng.integers(0, 2, D, dtype=np.uint8) for f in NUMERIC + SYMBOL + ZONES + ["img"]}
        self.tie = rng.integers(0, 2, D, dtype=np.uint8)
        self.enc = {}
        for f in NUMERIC:
            self.enc[f] = LevelEncoder(D, levels, rng).fit(np.array([self.val(c, f) for c in train_cases]))
        for f in SYMBOL:
            self.enc[f] = SymbolEncoder(D, rng).fit([self.sym(c, f) for c in train_cases])
        for f in ZONES:
            self.enc[f] = LevelEncoder(D, 64, rng).fit(np.array([img[c][f] for c in train_cases]))
        self.enc["img"] = LevelEncoder(D, img_levels, rng).fit(np.array([img[c]["img"] for c in train_cases]))

    def val(self, c, f):
        v = (self.rows[c].get(f) or "").strip()
        return float(v) if v else np.nan

    def sym(self, c, f):
        return self.rows[c].get("center", "NA") if f == "center" else self.scan.get(c, ("NA", "NA"))[f != "scanner"]

    def field(self, f, cs):
        if f in NUMERIC:
            return self.enc[f].encode(np.array([self.val(c, f) for c in cs]))
        if f in SYMBOL:
            return self.enc[f].encode([self.sym(c, f) for c in cs])
        return self.enc[f].encode(np.array([self.img[c][f] for c in cs]))

    def hv(self, fields, cs, img_repeat=0):
        parts = [self.field(f, cs) ^ self.keys[f] for f in fields]
        if img_repeat:
            parts += [self.field("img", cs) ^ self.keys["img"]] * img_repeat
        return bundle(parts, self.tie)


def auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="results/lesion_ap2/B_t110")
    ap.add_argument("--splits", default="results/cv/splits.json")
    ap.add_argument("--slices_npz", default="/data/picai_all_masks.npz")
    ap.add_argument("--ap_json", default="results/lesion_ap2/final_t110.json")
    ap.add_argument("--marksheet", default="/data/picai_labels/clinical_information/marksheet.csv")
    ap.add_argument("--out", default="results/round3/patient")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.slices_npz, mmap_mode="r")
    pid, sid, cs = np.asarray(d["patient_id"]), np.asarray(d["study_id"]), np.asarray(d["case_cspca"])
    case_pid = {int(s): int(p) for s, p in zip(sid, pid)}
    case_y = {int(s): int(c) for s, c in zip(sid, cs)}
    rows = {int(r["study_id"]): r for r in csv.DictReader(open(args.marksheet))}
    scan = scanner_info("/data/picai_images")
    A = json.load(open(args.ap_json))["tuned"]["per_fold"]
    ap_of = {f"fold{p['fold']}": p["lesion_ap"] for p in A}
    a_case = {f"fold{p['fold']}": p["patient_auroc"] for p in A}

    results, all_scores = {}, {}
    for si, sp in enumerate(json.load(open(args.splits))):
        name = sp["name"]
        img = load_split(args.dumps, name)
        groups = {k: set(int(p) for p in sp[k]) for k in ("train", "val", "test")}
        cases = {k: sorted(c for c in img if case_pid[c] in groups[k]) for k in groups}
        y = {k: np.array([case_y[c] for c in cases[k]]) for k in cases}
        psad_ok = {k: np.array([bool((rows[c].get("psad") or "").strip()) for c in cases[k]]) for k in cases}
        ims = {k: np.array([img[c]["img"] for c in cases[k]]) for k in cases}
        res = {"img_alone": {"test": auc(y["test"], ims["test"]), "val": auc(y["val"], ims["val"])}}

        # clinical-only HDC record (shared by c and b), epochs tuned on val
        best = None
        for ep in (5, 20, 50):
            R = Records(rows, scan, img, cases["train"], si)
            clf = HDCBinaryClassifier(epochs=ep, seed=si).fit(R.hv(NUMERIC + SYMBOL, cases["train"]), y["train"])
            v = auc(y["val"], clf.score(R.hv(NUMERIC + SYMBOL, cases["val"])))
            if best is None or v > best[0]:
                best = (v, ep, R, clf)
        _, ep_c, Rc, clf_c = best
        clin = {k: clf_c.score(Rc.hv(NUMERIC + SYMBOL, cases[k])) for k in ("train", "val", "test")}
        res["clinical_record"] = {"test": auc(y["test"], clin["test"]), "val": best[0], "epochs": ep_c}

        # b: plain-number merges, w tuned on val
        mu = {"i": ims["train"].mean(), "c": clin["train"].mean()}
        sd = {"i": ims["train"].std() + 1e-9, "c": clin["train"].std() + 1e-9}
        z = lambda k, w: (ims[k] - mu["i"]) / sd["i"] + w * (clin[k] - mu["c"]) / sd["c"]
        r = lambda k, w: (rankdata(ims[k]) + w * rankdata(clin[k])) / (1 + w)
        case_scores = {k: {"cases": cases[k], "y": y[k].tolist(), "img": ims[k].tolist()} for k in ("val", "test")}
        for label, f in (("merge_z", z), ("merge_rank", r)):
            ws = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
            vals = {w: auc(y["val"], f("val", w)) for w in ws}
            w = max(ws, key=lambda w: (vals[w], -w))
            st = f("test", w)
            for k in ("val", "test"):
                case_scores[k][label] = np.asarray(f(k, w), dtype=float).tolist()
            res[label] = {"test": auc(y["test"], st), "val": vals[w], "w": w,
                          "test_psad_present": auc(y["test"][psad_ok["test"]], st[psad_ok["test"]]),
                          "test_psad_missing": auc(y["test"][~psad_ok["test"]], st[~psad_ok["test"]])}

        # a: all-HDC record with a fine, weighted image field
        best = None
        for L in (64, 256, 1024):
            R = Records(rows, scan, img, cases["train"], si, img_levels=L)
            for rep in (1, 3, 5, 9):
                H = {k: R.hv(NUMERIC + SYMBOL + ZONES, cases[k], img_repeat=rep) for k in ("train", "val", "test")}
                for ep in (5, 20, 50):
                    clf = HDCBinaryClassifier(epochs=ep, seed=si).fit(H["train"], y["train"])
                    v = auc(y["val"], clf.score(H["val"]))
                    if best is None or v > best[0]:
                        best = (v, L, rep, ep, clf.score(H["test"]))
        v, L, rep, ep, st = best
        res["all_hdc_record"] = {"test": auc(y["test"], st), "val": v, "img_levels": L, "img_repeat": rep, "epochs": ep,
                                 "test_psad_present": auc(y["test"][psad_ok["test"]], st[psad_ok["test"]]),
                                 "test_psad_missing": auc(y["test"][~psad_ok["test"]], st[~psad_ok["test"]])}
        for k in ("img_alone", "clinical_record"):
            sc = ims["test"] if k == "img_alone" else clin["test"]
            res[k]["test_psad_present"] = auc(y["test"][psad_ok["test"]], sc[psad_ok["test"]])
            res[k]["test_psad_missing"] = auc(y["test"][~psad_ok["test"]], sc[~psad_ok["test"]])
        res["A_case_score"] = {"test": a_case[name]}
        for k in res:
            res[k]["picai"] = (res[k]["test"] + ap_of[name]) / 2
        res["n_test"], res["n_test_psad"] = int(len(y["test"])), int(psad_ok["test"].sum())
        results[name] = res
        all_scores[name] = case_scores
        print(name, {k: round(v["test"], 3) for k, v in res.items() if isinstance(v, dict)}, flush=True)

    keys = [k for k in results["fold0"] if isinstance(results["fold0"][k], dict)]
    lines = ["| patient score | fold-0 patient AUROC | fold-0 PI-CAI | CV patient AUROC | CV PI-CAI | CV AUROC, PSAD present / missing | chosen (per fold) |",
             "|---|---|---|---|---|---|---|"]
    for k in keys:
        t = [results[f][k]["test"] for f in results]
        p = [results[f][k]["picai"] for f in results]
        pp = [results[f][k].get("test_psad_present", np.nan) for f in results]
        pm = [results[f][k].get("test_psad_missing", np.nan) for f in results]
        ch = "; ".join(f"{f}: " + ", ".join(f"{a}={results[f][k][a]}" for a in ("w", "img_levels", "img_repeat", "epochs")
                                             if a in results[f][k]) for f in results) if k not in ("img_alone", "A_case_score") else ""
        lines.append(f"| {k} | {results['fold0'][k]['test']:.3f} | {results['fold0'][k]['picai']:.3f} | "
                     f"{np.mean(t):.3f} ± {np.std(t):.3f} | {np.mean(p):.3f} ± {np.std(p):.3f} | "
                     f"{np.nanmean(pp):.3f} / {np.nanmean(pm):.3f} | {ch} |")
    md = "\n".join(lines)
    open(os.path.join(args.out, "table.md"), "w").write(md + "\n")
    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=2)
    json.dump(all_scores, open(os.path.join(args.out, "case_scores.json"), "w"))
    print(md)


if __name__ == "__main__":
    main()
