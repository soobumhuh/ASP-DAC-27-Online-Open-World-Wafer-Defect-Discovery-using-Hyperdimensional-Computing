#!/usr/bin/env python3
"""Hypervector database, step 3: the same items in Qdrant; top-k recall vs the exact numpy Hamming index + latency.

Collections (only new ones, prefixed sohuh_hvdb_; recreated by this script, nothing else is touched):
  sohuh_hvdb_regions_bq    named vectors 'content' / 'record' (D bits as +-1 float16, Dot = D - 2 x Hamming),
                           binary quantization (always in RAM): search on the 1-bit codes (= Hamming)
  sohuh_hvdb_regions_u8    the same bits as uint8 0 / 1 with Manhattan distance (= Hamming exactly), HNSW
  sohuh_hvdb_patients_bq / _u8   patient HVs (k regions = 3), same two layouts
Every database item of every outer fold is stored once per fold with payload outer_fold, patient, study, kind, zone,
centre, scanner, size, detector score, ISUP, csPCa, grade group, PSA, PSAD; payload indexes on zone / centre /
outer_fold / kind / patient. Queries: fold f's ground-truth lesions (regions) / exams (patients) filtered to
outer_fold == f and patient != query patient (regions: kind in {gt, cand}, as in T1).
Recall@k is tie-aware: a returned item counts as correct if its exact Hamming distance is <= the exact k-th distance.

  PYTHONPATH=/data/hvdb/pylib python scripts/hvdb_qdrant.py --eval /data/hvdb/eval --items /data/hvdb/items
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models as qm

import _path  # noqa: F401

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://172.17.0.1:6333")
ap.add_argument("--eval", default="/data/hvdb/eval")
ap.add_argument("--items", default="/data/hvdb/items")
ap.add_argument("--D", type=int, default=2048)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--max-queries", type=int, default=200, help="per fold, for the latency / recall test")
ap.add_argument("--keep", action="store_true", help="keep the collections after the test")
a = ap.parse_args()
PFX = "sohuh_hvdb_"
cl = QdrantClient(url=a.url, timeout=120, check_compatibility=False)
meta = pickle.load(open(os.path.join(a.items, "fixed.pkl"), "rb"))
ex = meta["exams"]


def payload_region(it, f):
    e = ex[it["study"]]
    return {"outer_fold": f, "patient": e["patient"], "study": it["study"], "kind": it["kind"], "zone": int(it["zone"]),
            "centre": e["centre"], "scanner": e["scanner"], "size": int(it["size"]), "score": float(it["conf"]),
            "isup": int(it["isup"]), "cspca_region": bool(it["tp"]), "case_cspca": e["cspca"], "grade_group": e["gg"],
            "psa": None if np.isnan(e["psa"]) else e["psa"], "psad": None if np.isnan(e["psad"]) else e["psad"]}


def payload_patient(s, f):
    e = ex[s]
    return {"outer_fold": f, "patient": e["patient"], "study": int(s), "centre": e["centre"], "scanner": e["scanner"],
            "case_cspca": e["cspca"], "case_isup": e["case_isup"], "grade_group": e["gg"],
            "psa": None if np.isnan(e["psa"]) else e["psa"], "psad": None if np.isnan(e["psad"]) else e["psad"]}


def make(name, names, layout):
    if cl.collection_exists(name):
        cl.delete_collection(name)
    if layout == "bq":
        vc = {n: qm.VectorParams(size=a.D, distance=qm.Distance.DOT, datatype=qm.Datatype.FLOAT16, on_disk=True)
              for n in names}
        cl.create_collection(name, vectors_config=vc, quantization_config=qm.BinaryQuantization(
            binary=qm.BinaryQuantizationConfig(always_ram=True)))
    else:
        vc = {n: qm.VectorParams(size=a.D, distance=qm.Distance.MANHATTAN, datatype=qm.Datatype.UINT8) for n in names}
        cl.create_collection(name, vectors_config=vc)
    for field, sch in (("zone", qm.PayloadSchemaType.INTEGER), ("centre", qm.PayloadSchemaType.KEYWORD),
                       ("outer_fold", qm.PayloadSchemaType.INTEGER), ("kind", qm.PayloadSchemaType.KEYWORD),
                       ("patient", qm.PayloadSchemaType.INTEGER)):
        cl.create_payload_index(name, field, field_schema=sch)


def vec(bits, layout):
    return (bits.astype(np.float32) * 2 - 1).tolist() if layout == "bq" else bits.astype(np.uint8).tolist()


def upload(name, layout, rows):
    """rows: list of (id, {name: bits}, payload)."""
    t0 = time.time()
    for i in range(0, len(rows), 128):
        cl.upsert(name, points=[qm.PointStruct(id=pid, vector={n: vec(b, layout) for n, b in v.items()}, payload=p)
                                for pid, v, p in rows[i:i + 128]], wait=False)
    while cl.get_collection(name).status != qm.CollectionStatus.GREEN:
        time.sleep(2)
    return time.time() - t0


def search(name, layout, vname, qbits, flt, mode):
    params = None
    if mode == "exact":
        params = qm.SearchParams(exact=True)
    elif mode == "bq_norescore":
        params = qm.SearchParams(quantization=qm.QuantizationSearchParams(ignore=False, rescore=False))
    elif mode == "bq_rescore":
        params = qm.SearchParams(quantization=qm.QuantizationSearchParams(ignore=False, rescore=True, oversampling=4.0))
    elif mode == "hnsw":
        params = qm.SearchParams(hnsw_ef=128)
    t0 = time.perf_counter()
    r = cl.query_points(name, query=vec(qbits, layout), using=vname, query_filter=flt, limit=a.k, search_params=params,
                        with_payload=False)
    return [p.id for p in r.points], (time.perf_counter() - t0) * 1000


def main():
    res = {"D": a.D, "k": a.k, "collections": {}}
    hv = {f: np.load(os.path.join(a.eval, f"hv_f{f}_D{a.D}.npz")) for f in range(5)}
    items = {f: pickle.load(open(os.path.join(a.items, f"fold{f}.pkl"), "rb"))["items"] for f in range(5)}
    unpack = lambda x: np.unpackbits(x, axis=1)[:, :a.D]
    for level in ("regions", "patients"):
        names = ["content", "record"]
        rows, queries, exact_bits = [], [], {}
        pid = 0
        for f in range(5):
            if level == "regions":
                it = items[f]
                bits = {n: unpack(hv[f][f"region_{n}"]) for n in names}
                db = [i for i, x in enumerate(it) if x["role"] == "db"]
                ids = {}
                for i in db:
                    ids[i] = pid
                    rows.append((pid, {n: bits[n][i] for n in names}, payload_region(it[i], f)))
                    pid += 1
                q = [i for i, x in enumerate(it) if x["role"] == "query" and x["kind"] == "gt"][:a.max_queries]
                dbs = [i for i in db if it[i]["kind"] in ("gt", "cand")]
                for i in q:
                    queries.append((f, ex[it[i]["study"]]["patient"], {n: bits[n][i] for n in names},
                                    {n: bits[n][dbs] for n in names}, np.array([ids[j] for j in dbs]),
                                    np.array([ex[it[j]["study"]]["patient"] for j in dbs])))
            else:
                bits = {n: unpack(hv[f][f"patient_db_{n}"]) for n in names}
                qb = {n: unpack(hv[f][f"patient_query_{n}"]) for n in names}
                dst, qst = hv[f]["patient_db_study"], hv[f]["patient_query_study"]
                start = pid
                for j, s in enumerate(dst):
                    rows.append((pid, {n: bits[n][j] for n in names}, payload_patient(s, f)))
                    pid += 1
                for j, s in list(enumerate(qst))[:a.max_queries]:
                    queries.append((f, ex[s]["patient"], {n: qb[n][j] for n in names}, bits,
                                    np.arange(start, pid), np.array([ex[x]["patient"] for x in dst])))
        for layout in ("bq", "u8"):
            name = f"{PFX}{level}_{layout}"
            make(name, names, layout)
            t_up = upload(name, layout, rows)
            info = cl.get_collection(name)
            out = {"points": info.points_count, "upload_s": t_up,
                   "bytes_per_item_vectors": {"bq": {"float16_raw": 2 * a.D, "bq_ram": a.D // 8},
                                              "u8": {"uint8": a.D}}[layout] | {"numpy_packed": a.D // 8},
                   "payload_bytes_mean": float(np.mean([len(json.dumps(r[2])) for r in rows[:2000]]))}
            modes = ["exact", "bq_norescore", "bq_rescore", "hnsw"] if layout == "bq" else ["exact", "hnsw"]
            for vname in names:
                for mode in modes:
                    rec, lat = [], []
                    for f, pat, qb, dbits, dids, dpat in queries:
                        flt = qm.Filter(must=[qm.FieldCondition(key="outer_fold", match=qm.MatchValue(value=f))]
                                        + ([qm.FieldCondition(key="kind", match=qm.MatchAny(any=["gt", "cand"]))]
                                           if level == "regions" else []),
                                        must_not=[qm.FieldCondition(key="patient", match=qm.MatchValue(value=int(pat)))])
                        got, ms = search(name, layout, vname, qb[vname], flt, mode)
                        ham = np.count_nonzero(dbits[vname] != qb[vname][None], axis=1)
                        ham = np.where(dpat != pat, ham, 10 ** 9)
                        kth = np.sort(ham)[a.k - 1]
                        dmap = dict(zip(dids.tolist(), ham.tolist()))
                        rec.append(np.mean([dmap.get(g, 10 ** 9) <= kth for g in got]) if got else 0.0)
                        lat.append(ms)
                    out[f"{vname}/{mode}"] = {"recall@k": float(np.mean(rec)), "latency_ms_median": float(np.median(lat)),
                                              "latency_ms_p95": float(np.percentile(lat, 95)), "n_queries": len(rec)}
                    print(level, layout, vname, mode, out[f"{vname}/{mode}"], flush=True)
            # numpy exact index latency (same queries, one at a time)
            from mri_hdc.hvdb import HammingIndex, pack
            lat = []
            for f, pat, qb, dbits, dids, dpat in queries[:200]:
                idx = HammingIndex(pack(dbits["content"]))
                t0 = time.perf_counter()
                dd = idx.dist(pack(qb["content"][None]))[0]
                dd = np.where(dpat != pat, dd, 10 ** 9)
                np.argpartition(dd, a.k)[:a.k]
                lat.append((time.perf_counter() - t0) * 1000)
            out["numpy_exact_latency_ms_median"] = float(np.median(lat))
            res["collections"][name] = out
            if not a.keep:
                cl.delete_collection(name)
    json.dump(res, open(os.path.join(a.eval, "qdrant.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
