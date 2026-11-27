#!/usr/bin/env python3
"""Hypervector database, step 2: hypervectors + exact Hamming index + retrieval tests T1-T5 and the T7 scorer.

Everything is out-of-fold: a query from outer fold f searches only the database side built for f (exams of the other
folds, candidates / maps of the nested models nest<f><g>); same-patient items are always masked; encoders (level
edges, standardisation of the content descriptor, clinical level edges) are fitted on the database side only. Labels
are payload only. Results are pooled over the five folds; 95% CIs = 1000 patient-level bootstrap resamples.

  T1 lesion grade        ground-truth lesion queries vs database GT lesions + candidates; relevant = csPCa (GT or
                         TP candidate); precision@1/10/40, mAP, Spearman(query ISUP, mean ISUP of the 10 neighbours)
  T2 lesion vs normal    GT + sampled normal queries vs database GT + normals, same zone; 10-NN vote AUROC
  T3 patient retrieval   patient HVs; precision@10 for csPCa among csPCa queries, label / grade-group agreement@10,
                         kNN-vote AUROC (k from {5,10,20,40} by leave-one-database-fold-out); grid D x k regions
  T4 shortcut            same-centre / same-scanner rate among the 10 neighbours vs the base rate; T1 / T3 with the
                         database restricted to other centres (leave-one-centre-out)
  T5 binarising cost     T1 / T3 with the float descriptors (cosine) and with random bits; recall@10 of HV vs float
  T7 retrieval scorer    every candidate scored only by its database neighbours' TP labels (zone-filtered kNN vote,
                         k by nested LOFO); writes cases pickles for scripts/native_eval.py (official picai_eval)

  python scripts/hvdb_eval.py --items /data/hvdb/items --out /data/hvdb/eval
"""
import argparse
import json
import os
import pickle
import time

import numpy as np

import _path  # noqa: F401
from mri_hdc import evaluate as E
from mri_hdc import hvdb as H
from mri_hdc import patient as PT
from mri_hdc import rescorer as RS
from mri_hdc.pipeline import load_config, splits

TYPES = ["record", "content", "both"]
KS = (1, 10, 40)


def load(items_dir):
    meta = pickle.load(open(os.path.join(items_dir, "fixed.pkl"), "rb"))
    folds = {f: pickle.load(open(os.path.join(items_dir, f"fold{f}.pkl"), "rb")) for f in range(5)}
    return meta, folds


class Fold:
    """All arrays of one outer fold."""

    def __init__(self, f, data, meta, fields):
        self.f = f
        it = data["items"]
        self.items = it
        ex = meta["exams"]
        self.role = np.array([x["role"] for x in it])
        self.kind = np.array([x["kind"] for x in it])
        self.study = np.array([x["study"] for x in it])
        self.patient = np.array([ex[s]["patient"] for s in self.study])
        self.centre = np.array([ex[s]["centre"] for s in self.study])
        self.scanner = np.array([ex[s]["scanner"] for s in self.study])
        self.zone = np.array([x["zone"] for x in it])
        self.tp = np.array([bool(x["tp"]) for x in it])
        self.isup = np.array([x["isup"] for x in it])
        self.conf = np.array([x["conf"] for x in it])
        self.cfeat = data["content"]
        self.feats = [x["feats"] for x in it]
        self.fields = fields
        self.meta_ex = ex
        self.db = self.role == "db"
        self.dbc = self.db & (self.kind == "cand")
        F = np.array([[x[k] for k in fields] for x in self.feats], np.float64)
        mu, sd = F[self.dbc].mean(0), F[self.dbc].std(0) + 1e-9
        Z = np.eye(3)[np.clip(self.zone, 0, 2)]
        self.rec_float = np.concatenate([(F - mu) / sd, Z], 1).astype(np.float32)

    def encode(self, D):
        """Packed HVs per item type at dimension D (+ float descriptors and random bits for T5)."""
        ce = H.ContentEncoder(768, D).fit(self.cfeat[self.db, :768])
        cj = H.ContentEncoder(256, D, seed=H.RP_SEED + 1).fit(self.cfeat[self.db, 768:])
        re = H.record_encoder(D, self.fields, [self.feats[i] for i in np.where(self.dbc)[0]])
        bits = {"content": ce.encode(self.cfeat[:, :768]), "record": H.encode_record(re, self.feats),
                "content_joint": cj.encode(self.cfeat[:, 768:])}
        self.ce, self.re = ce, re
        P = {k: H.pack(v) for k, v in bits.items()}
        P["both"] = np.concatenate([P["content"], P["record"]], 1)
        self.bits, self.P = bits, P
        cz = ce.z(self.cfeat[:, :768])
        nz = lambda A: A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        self.float = {"content": cz, "record": self.rec_float,
                      "both": np.concatenate([nz(cz), nz(self.rec_float)], 1),
                      "content_joint": cj.z(self.cfeat[:, 768:])}
        rng = np.random.default_rng(99 + self.f)
        self.P["random"] = H.pack(rng.integers(0, 2, (len(self.items), D), dtype=np.uint8))
        return self


def dist_hv(P, qi, di):
    return H.HammingIndex(P[di]).dist(P[qi]).astype(np.float64)


def dist_float(Fm, qi, di):
    a = Fm[qi] / (np.linalg.norm(Fm[qi], axis=1, keepdims=True) + 1e-9)
    b = Fm[di] / (np.linalg.norm(Fm[di], axis=1, keepdims=True) + 1e-9)
    return -(a @ b.T).astype(np.float64)


# ------------------------------------------------------------------ T1 / T4 (regions)
def t1(F: Fold, dist, qi, di, valid=None):
    v = F.patient[qi][:, None] != F.patient[di][None, :]
    if valid is not None:
        v &= valid
    order = H.ranking(dist, v)
    nv = v.sum(1)
    rel = F.tp[di][order] & (np.arange(len(di))[None, :] < nv[:, None])
    out = {f"p@{k}": rel[:, :k].mean(1) for k in KS}
    out["ap"] = H.average_precision(rel.astype(np.float64), nv)
    top = order[:, :10]
    out["nbr_isup"] = F.isup[di][top].mean(1)
    out["same_centre"] = (F.centre[di][top] == F.centre[qi][:, None]).mean(1)
    out["base_centre"] = np.array([(F.centre[di][v[j]] == F.centre[q]).mean() for j, q in enumerate(qi)])
    out["same_scanner"] = (F.scanner[di][top] == F.scanner[qi][:, None]).mean(1)
    out["base_scanner"] = np.array([(F.scanner[di][v[j]] == F.scanner[q]).mean() for j, q in enumerate(qi)])
    out["top10"] = di[top]
    return out


def t2(F: Fold, dist, qi, di, k=10):
    v = F.patient[qi][:, None] != F.patient[di][None, :]
    zq, zd = F.zone[qi], F.zone[di]
    same = zq[:, None] == zd[None, :]
    ok = same.sum(1) > k
    v &= np.where(ok[:, None], same, True)
    order = H.ranking(dist, v)[:, :k]
    return (F.kind[di][order] == "gt").mean(1)


# ------------------------------------------------------------------ patients
def patient_items(F: Fold, meta, rows, fields, D, kreg, roles, clin_cache):
    """{role: (studies, {type: (n, bits) uint8})} for roles query / db / val."""
    out = {}
    ce = F.ce
    db_studies = sorted(set(F.study[F.db]))
    key = (F.f, D)
    if key not in clin_cache:
        allst = sorted(set(F.study))
        clin_cache[key] = dict(zip(allst, H.clinical_hv(rows, db_studies, allst, F.f, fields, D)))
    clin = clin_cache[key]
    for role in ("query", "db", "val"):
        studies = sorted(set(F.study[F.role == role]))
        gl = ce.encode(np.stack([meta["gland"][s][:768] for s in studies]))
        hv = {t: [] for t in ("record", "content")}
        sel_all = np.where((F.role == role) & (F.kind == "cand"))[0]
        by = {}
        for i in sel_all:
            by.setdefault(F.study[i], []).append(i)
        for j, s in enumerate(studies):
            c = sorted(by.get(s, []), key=lambda i: -F.conf[i])[:kreg]
            hv["content"].append(roles.patient(F.bits["content"][c], gl[j], clin[s]))
            hv["record"].append(roles.patient(F.bits["record"][c], None, clin[s]))
        hv = {t: np.stack(v) for t, v in hv.items()}
        hv["both"] = np.concatenate([hv["content"], hv["record"]], 1)
        out[role] = (np.array(studies), hv)
    return out


def patient_float(F: Fold, meta, rows, kreg, studies, role):
    nz = lambda A: A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    by = {}
    for i in np.where((F.role == role) & (F.kind == "cand"))[0]:
        by.setdefault(F.study[i], []).append(i)
    cz = F.float["content"]
    reg = np.stack([cz[sorted(by[s], key=lambda i: -F.conf[i])[:kreg]].mean(0) if s in by else np.zeros(cz.shape[1])
                    for s in studies])
    gl = F.ce.z(np.stack([meta["gland"][s][:768] for s in studies]))
    cl = np.array([[meta["exams"][s][k] for k in ("psad", "psa", "volume")] for s in studies], float)
    return reg, gl, cl


def t3(meta, qst, dst, dist, k=10, valid=None):
    ex = meta["exams"]
    y = lambda S: np.array([ex[s]["cspca"] for s in S])
    gg = lambda S: np.array([ex[s]["gg"] for s in S])
    pid = lambda S: np.array([ex[s]["patient"] for s in S])
    v = pid(qst)[:, None] != pid(dst)[None, :]
    if valid is not None:
        v &= valid
    order = H.ranking(dist, v)
    yq, yd, gq, gd = y(qst), y(dst), gg(qst), gg(dst)
    top = order[:, :k]
    cen = lambda S: np.array([ex[s]["centre"] for s in S])
    scn = lambda S: np.array([ex[s]["scanner"] for s in S])
    cq, cd, sq, sd = cen(qst), cen(dst), scn(qst), scn(dst)
    return {"order": order, "p@10": yd[top].mean(1), "agree@10": (yd[top] == yq[:, None]).mean(1),
            "gg_agree@10": (gd[top] == gq[:, None]).mean(1), "y": yq,
            "same_centre": (cd[top] == cq[:, None]).mean(1), "base_centre": np.array([(cd == c).mean() for c in cq]),
            "same_scanner": (sd[top] == sq[:, None]).mean(1), "base_scanner": np.array([(sd == c).mean() for c in sq]),
            "yd": yd}


def vote(order, yd, k):
    return yd[order[:, :k]].mean(1)


def pick_k(meta, dst, dist_dd, grid=(5, 10, 20, 40)):
    """k for the patient kNN vote: leave-one-database-fold-out inside the database side."""
    ex = meta["exams"]
    ef = np.array([ex[s]["efold"] for s in dst])
    yd = np.array([ex[s]["cspca"] for s in dst])
    sc = {k: [] for k in grid}
    for g in np.unique(ef):
        q, d = np.where(ef == g)[0], np.where(ef != g)[0]
        order = H.ranking(dist_dd[np.ix_(q, d)])
        for k in grid:
            sc[k].append(H.auroc(yd[q], vote(order, yd[d], k)))
    return max(grid, key=lambda k: (np.mean(sc[k]), -k)), {k: float(np.mean(v)) for k, v in sc.items()}


# ------------------------------------------------------------------ T7
def knn_scores(dist, yd, zq, zd, k, prior):
    out = np.empty(len(zq))
    for z in np.unique(zq):
        q = np.where(zq == z)[0]
        s = dist[q]
        if (zd == z).sum() > k:
            s = np.where((zd == z)[None, :], s, np.inf)
        nn = np.argpartition(s, k, axis=1)[:, :k]
        out[q] = (yd[nn].sum(1) + prior) / (k + 1)
    return out


def summarise(groups, arrs, fns):
    return {name: H.bootstrap(groups, fn, n=ARGS.boot) for name, fn in fns.items()}


def main():
    t0 = time.time()
    cfg = load_config(ARGS.config)
    fields = RS.NUM + list(cfg["rescorer"].get("extra_fields", []))
    pfields = cfg["patient"].get("fields")
    rows = PT.read_marksheet(os.path.join(cfg["paths"]["labels_dir"], "clinical_information", "marksheet.csv"))
    meta, data = load(ARGS.items)
    os.makedirs(ARGS.out, exist_ok=True)
    folds = {f: Fold(f, data[f], meta, fields) for f in range(5)}
    res = {"n_items": {}, "sizes": {}}
    for f, F in folds.items():
        res["n_items"][f] = {f"{r}/{k}": int(((F.role == r) & (F.kind == k)).sum())
                             for r in ("query", "db", "val") for k in ("gt", "cand", "normal")}
    exact_isup = np.mean([x.get("isup_exact", True) for F in folds.values() for x in F.items
                          if x["kind"] == "gt" and x["role"] == "query"])
    res["gt_isup_exact_fraction"] = float(exact_isup)
    region = {}
    for D in ARGS.dims:
        for F in folds.values():
            F.encode(D)
        res["sizes"][D] = {t: int(folds[0].P[t].shape[1] * 8) for t in ("record", "content", "both")}
        # ---------------- T1 / T4 / T5 / T2 for each item type
        for t in TYPES + ["content_joint", "random"] + [f"float_{x}" for x in ("record", "content", "both")]:
            if t.startswith("float_") and D != ARGS.dims[0]:
                continue
            acc, accl, acc2, recall = {}, {}, {"s": [], "y": [], "g": []}, []
            for f, F in folds.items():
                qi = np.where((F.role == "query") & (F.kind == "gt"))[0]
                di = np.where(F.db & np.isin(F.kind, ["gt", "cand"]))[0]
                dist = dist_float(F.float[t[6:]], qi, di) if t.startswith("float_") else dist_hv(F.P[t], qi, di)
                r = t1(F, dist, qi, di)
                r["isup_q"], r["pid"] = F.isup[qi], F.patient[qi]
                if t in ("record", "content", "both") and D == ARGS.dims[0]:
                    rf = t1(F, dist_float(F.float[t], qi, di), qi, di)
                    recall.append(np.array([len(set(a) & set(b)) / 10 for a, b in zip(r["top10"], rf["top10"])]))
                    rl = t1(F, dist, qi, di, valid=F.centre[qi][:, None] != F.centre[di][None, :])
                    for k in ("p@10", "ap"):
                        accl.setdefault(k, []).append(rl[k])
                for k, v in r.items():
                    if k != "top10":
                        acc.setdefault(k, []).append(v)
                q2 = np.where((F.role == "query") & np.isin(F.kind, ["gt", "normal"]))[0]
                d2 = np.where(F.db & np.isin(F.kind, ["gt", "normal"]))[0]
                dist2 = dist_float(F.float[t[6:]], q2, d2) if t.startswith("float_") else dist_hv(F.P[t], q2, d2)
                acc2["s"].append(t2(F, dist2, q2, d2))
                acc2["y"].append(F.kind[q2] == "gt")
                acc2["g"].append(F.patient[q2])
            A = {k: np.concatenate(v) for k, v in acc.items()}
            g = A["pid"]
            fns = {f"p@{k}": (lambda kk: lambda i: np.nanmean(A[kk][i]))(f"p@{k}") for k in KS}
            fns["mAP"] = lambda i: np.nanmean(A["ap"][i])
            fns["spearman_isup"] = lambda i: H.spearman(A["isup_q"][i], A["nbr_isup"][i])
            for k in ("same_centre", "base_centre", "same_scanner", "base_scanner"):
                fns[k] = (lambda kk: lambda i: np.mean(A[kk][i]))(k)
            out = summarise(g, A, fns)
            if accl:
                L = {k: np.concatenate(v) for k, v in accl.items()}
                out["loco_p@10"] = H.bootstrap(g, lambda i: np.mean(L["p@10"][i]), n=ARGS.boot)
                out["loco_mAP"] = H.bootstrap(g, lambda i: np.nanmean(L["ap"][i]), n=ARGS.boot)
            if recall:
                rc = np.concatenate(recall)
                out["recall@10_vs_float"] = H.bootstrap(g, lambda i: np.mean(rc[i]), n=ARGS.boot)
            s2, y2, g2 = (np.concatenate(acc2[k]) for k in ("s", "y", "g"))
            out["T2_auroc"] = H.bootstrap(g2, lambda i: H.auroc(y2[i], s2[i]), n=ARGS.boot)
            out["n_queries"] = int(len(g))
            region[f"{t}|D{D}"] = out
            print(f"[{time.time() - t0:.0f}s] regions {t} D{D}: " + ", ".join(
                f"{k} {v[0]:.3f}" for k, v in out.items() if isinstance(v, tuple)), flush=True)
        res["regions"] = region
        json.dump(res, open(os.path.join(ARGS.out, "hvdb_results.json"), "w"), indent=1, default=float)
        # ---------------- T3 / T4 patients, grid over k regions
        roles = H.Roles(D)
        clin_cache = {}
        for kreg in ARGS.kreg:
            for F in folds.values():
                F.pitems = patient_items(F, meta, rows, pfields, D, kreg, roles, clin_cache)
            for t in TYPES + (["random", "float"] if kreg == 3 and D == ARGS.dims[0] else []):
                acc, scores = {}, []
                for f, F in folds.items():
                    qst, qh = F.pitems["query"]
                    dst, dh = F.pitems["db"]
                    if t == "random":
                        rng = np.random.default_rng(5 + f)
                        qp = H.pack(rng.integers(0, 2, (len(qst), D), dtype=np.uint8))
                        dp = H.pack(rng.integers(0, 2, (len(dst), D), dtype=np.uint8))
                    elif t != "float":
                        qp, dp = H.pack(qh[t]), H.pack(dh[t])
                    if t == "float":
                        nz = lambda A: A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
                        blocks = []
                        for S, role in ((qst, "query"), (dst, "db")):
                            reg, gl, cl = patient_float(F, meta, rows, kreg, S, role)
                            blocks.append((reg, gl, cl))
                        dcl = blocks[1][2]
                        med, sd = np.nanmedian(dcl, 0), np.nanstd(dcl, 0) + 1e-9
                        mk = lambda b: np.concatenate([nz(b[0]), nz(b[1]), nz(np.nan_to_num((b[2] - med) / sd))], 1)
                        Qf, Df = mk(blocks[0]), mk(blocks[1])
                        dist = dist_float(np.concatenate([Qf, Df]), np.arange(len(Qf)), len(Qf) + np.arange(len(Df)))
                        dist_dd = dist_float(Df, np.arange(len(Df)), np.arange(len(Df)))
                    else:
                        dist = H.HammingIndex(dp).dist(qp).astype(np.float64)
                        dist_dd = H.HammingIndex(dp).dist(dp).astype(np.float64)
                        np.fill_diagonal(dist_dd, np.inf)
                    if t == "float":
                        np.fill_diagonal(dist_dd, np.inf)
                    kk, ksc = pick_k(meta, dst, dist_dd)
                    r = t3(meta, qst, dst, dist)
                    r["vote"] = vote(r["order"], r["yd"], kk)
                    cq = np.array([meta["exams"][s]["centre"] for s in qst])
                    cd = np.array([meta["exams"][s]["centre"] for s in dst])
                    rl = t3(meta, qst, dst, dist, valid=cq[:, None] != cd[None, :])
                    r["loco_p@10"] = rl["p@10"]
                    r["loco_vote"] = np.where(np.isfinite(dist).all(1), vote(rl["order"], rl["yd"], kk), np.nan)
                    r["study"] = qst
                    r["pid"] = np.array([meta["exams"][s]["patient"] for s in qst])
                    r["k"] = np.full(len(qst), kk)
                    if t in TYPES and D == ARGS.dims[0] and kreg == 3:  # recall@10 of HV vs float patient neighbours
                        F.pnbr = getattr(F, "pnbr", {})
                        F.pnbr[t] = r["order"][:, :10]
                    if t == "float":
                        for tt, top in getattr(F, "pnbr", {}).items():
                            r[f"recall_{tt}"] = np.array([len(set(a) & set(b)) / 10 for a, b in zip(top, r["order"][:, :10])])
                    for k, v in r.items():
                        if k not in ("order", "yd"):
                            acc.setdefault(k, []).append(v)
                A = {k: np.concatenate(v) for k, v in acc.items()}
                pos = A["y"] == 1
                fns = {"p@10_cspca": lambda i: np.mean(A["p@10"][i][pos[i]]),
                       "agree@10": lambda i: np.mean(A["agree@10"][i]),
                       "gg_agree@10": lambda i: np.mean(A["gg_agree@10"][i]),
                       "vote_auroc": lambda i: H.auroc(A["y"][i], A["vote"][i]),
                       "loco_p@10_cspca": lambda i: np.mean(A["loco_p@10"][i][pos[i]]),
                       "loco_vote_auroc": lambda i: H.auroc(A["y"][i], np.nan_to_num(A["loco_vote"][i], nan=0.5)),
                       "same_centre": lambda i: np.mean(A["same_centre"][i]),
                       "base_centre": lambda i: np.mean(A["base_centre"][i]),
                       "same_scanner": lambda i: np.mean(A["same_scanner"][i]),
                       "base_scanner": lambda i: np.mean(A["base_scanner"][i])}
                for tt in TYPES:
                    if f"recall_{tt}" in A:
                        fns[f"recall@10_{tt}_vs_float"] = (lambda key: lambda i: np.mean(A[key][i]))(f"recall_{tt}")
                out = summarise(A["pid"], A, fns)
                out["k_vote_per_fold"] = [int(v[0]) for v in acc["k"]]
                res.setdefault("patients", {})[f"{t}|D{D}|k{kreg}"] = out
                if t in TYPES:
                    pickle.dump({"study": A["study"], "vote": A["vote"], "y": A["y"]},
                                open(os.path.join(ARGS.out, f"patient_vote_{t}_D{D}_k{kreg}.pkl"), "wb"))
                print(f"[{time.time() - t0:.0f}s] patients {t} D{D} k{kreg}: " + ", ".join(
                    f"{k} {v[0]:.3f}" for k, v in out.items() if isinstance(v, tuple)), flush=True)
            json.dump(res, open(os.path.join(ARGS.out, "hvdb_results.json"), "w"), indent=1, default=float)
        if D == ARGS.dims[0]:  # packed HVs for the Qdrant step (D = first dim, k regions = 3 kept separately)
            for f, F in folds.items():
                F.pitems = patient_items(F, meta, rows, pfields, D, 3, roles, clin_cache)
                np.savez(os.path.join(ARGS.out, f"hv_f{f}_D{D}.npz"),
                         **{f"region_{t}": np.packbits(F.bits[t], 1) for t in ("record", "content")},
                         **{f"patient_{r}_{t}": np.packbits(F.pitems[r][1][t], 1) for r in ("query", "db")
                            for t in ("record", "content")},
                         **{f"patient_{r}_study": F.pitems[r][0] for r in ("query", "db")})
            if ARGS.t7:
                t7(folds, meta, cfg, rows, D)
    json.dump(res, open(os.path.join(ARGS.out, "hvdb_results.json"), "w"), indent=1, default=float)
    print(f"done {time.time() - t0:.0f}s")


def t7(folds, meta, cfg, rows, D):
    """Candidates scored only from their database neighbours' TP labels; crop-grid picai + cases pickles."""
    import SimpleITK  # noqa: F401  (picai_eval import order)
    from sklearn.metrics import roc_auc_score
    sp = splits(cfg)
    ex = meta["exams"]
    d = np.load(cfg["paths"]["slices"], mmap_mode="r")
    sid = np.asarray(d["study_id"])
    order = np.lexsort((np.asarray(d["slice_z"]), sid))
    rows_of = {}
    for i in order:
        rows_of.setdefault(int(sid[i]), []).append(int(i))
    lesion = d["lesion_mask"]
    out = {}
    for t in TYPES:
        K, per = {}, []
        for f, F in folds.items():
            P = F.P[t]
            dbc = np.where(F.dbc)[0]
            yd = F.tp[dbc].astype(float)
            prior = float(yd.mean())
            # k by leave-one-database-fold-out (crop-grid PI-CAI score on the held-out database fold's exams)
            efd = np.array([ex[s]["efold"] for s in F.study[dbc]])
            grid = ARGS.t7_k
            sc = {k: [] for k in grid}
            for g in np.unique(efd):
                qg, rest = dbc[efd == g], dbc[efd != g]
                dist = dist_hv(P, qg, rest)
                for k in grid:
                    s = knn_scores(dist, F.tp[rest].astype(float), F.zone[qg], F.zone[rest], k, float(F.tp[rest].mean()))
                    sc[k].append(picai_items(F, qg, s, rows_of, lesion)["picai_score"])
            k = max(grid, key=lambda x: (np.mean(sc[x]), -x))
            res_f = {}
            for role in ("query", "val"):
                qi = np.where((F.role == role) & (F.kind == "cand"))[0]
                s = knn_scores(dist_hv(P, qi, dbc), yd, F.zone[qi], F.zone[dbc], k, prior)
                res_f[role] = (qi, s)
            qi, s = res_f["query"]
            keep = {}
            r = picai_items(F, qi, s, rows_of, lesion, keep=keep, all_studies=sorted(set(F.study[F.role == "query"])))
            cases = {c: sorted(x for x in ex if ex[x]["patient"] in set(sp[f"fold{f}"][c])) for c in ("train", "val", "test")}
            y = {c: [ex[x]["cspca"] for x in cases[c]] for c in cases}
            clin = PT.clinical_scores(rows, {}, cases, y, seed=f, fields=cfg["patient"].get("fields"))
            img = lambda role, studies: [max([sv for i, sv in zip(*res_f[role]) if F.study[i] == x], default=0.0)
                                         for x in studies]
            img_val, img_test = img("val", cases["val"]), img("query", cases["test"])
            merged = PT.merge(img_val, img_test, clin["val"], clin["test"], 0.25)
            K[("rescore", f)] = keep
            K[("image_only", f)] = {**keep, "case_pred": dict(zip(cases["test"], map(float, img_test)))}
            K[("merged", f)] = {**keep, "case_pred": dict(zip(cases["test"], map(float, merged)))}
            cands = {}
            for x in sorted(set(F.study[F.role == "query"])):
                cands[x] = ((len(rows_of[x]), 96, 96), [])
            for i, sv in zip(qi, s):
                cands[F.study[i]][1].append((F.items[i]["mask"], float(sv)))
            K[("candidates", f)] = cands
            per.append({"fold": f, "k": int(k), "k_lofo": {str(a): float(np.mean(b)) for a, b in sc.items()},
                        "crop_lesion_ap": r["lesion_ap"],
                        "crop_auroc_image_only": float(roc_auc_score(y["test"], img_test)),
                        "crop_auroc_merged": float(roc_auc_score(y["test"], merged))})
            print(f"T7 {t} fold {f}: {per[-1]}", flush=True)
        pickle.dump(K, open(os.path.join(ARGS.out, f"t7_{t}_cases.pkl"), "wb"))
        out[t] = per
    json.dump(out, open(os.path.join(ARGS.out, "t7_crop.json"), "w"), indent=1)


def picai_items(F, idx, scores, rows_of, lesion, keep=None, all_studies=None):
    studies = all_studies or sorted(set(F.study[idx]))
    det, maps = {}, {}
    for x in studies:
        sh = (len(rows_of[x]), 96, 96)
        det[x] = np.zeros(int(np.prod(sh)), np.float32)
        maps[x] = {"lesion": (np.asarray(lesion[rows_of[x]]) > 0).astype(np.int32),
                   "cspca": int(F.meta_ex[x]["cspca"])}
    for i, sv in zip(idx, scores):
        det[F.study[i]][F.items[i]["mask"]] = sv
    det = {x: v.reshape(maps[x]["lesion"].shape) for x, v in det.items()}
    return E.picai(det, maps, workers=ARGS.workers, keep=keep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sota/best_sota.yaml")
    ap.add_argument("--items", default="/data/hvdb/items")
    ap.add_argument("--out", default="/data/hvdb/eval")
    ap.add_argument("--dims", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument("--kreg", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--t7", action="store_true")
    ap.add_argument("--t7-k", type=int, nargs="+", default=[10, 20, 40, 80])
    ap.add_argument("--workers", type=int, default=16)
    ARGS = ap.parse_args()
    main()
