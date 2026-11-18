"""Blocking v2 sweep on validation S1: recall / volume / perfect-matcher ceiling.

    python -m scripts.audit_blocking_v2 --n 10000 --caps 1000 --save
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src.blocking.candidates import PAIR_SHIFT, SimCandidateGenerator
from src.config import load_config
from src.evaluation.metrics import evaluate
from src.train import true_pairs_from_gt
from scripts.audit_blocking import OUT, split_rows

ALL_KEYS = ["name_skel_pair", "name_first", "name_prefix5", "num_street", "state_name_skel", "addr_tokens",
            "name_compact", "name_prefix8", "addr_pair", "name_addr", "num_street1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--caps", default="1000", help="one cap for all keys, or name=cap,...")
    ap.add_argument("--keys", default=",".join(ALL_KEYS))
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    P, T = cfg["paths"], cfg["train"]
    names = a.keys.split(",")
    if "=" in a.caps:
        caps = {k: int(v) for k, v in (x.split("=") for x in a.caps.split(","))}
    else:
        caps = {k: int(a.caps) for k in names}
    gen = SimCandidateGenerator(P["processed_dir"], "train", caps, top_k=10**6)
    s1_ids = gen.records.s1_ids()
    _, _, rows_val = split_rows(len(s1_ids), T)
    rows = np.sort(rows_val[:a.n])
    ids = s1_ids[rows]
    true_all = true_pairs_from_gt(P["processed_dir"])
    trace = pd.read_parquet(os.path.join(OUT, "blocking_trace.parquet"))
    trace = trace[trace.s1_row.isin(rows)]
    tcodes = (trace.s1_row.to_numpy() << PAIR_SHIFT) | trace.cand.to_numpy()

    parts, t0 = [], time.time()
    t_stage1 = t_sim = 0.0
    for i in range(0, len(rows), a.chunk):
        t1 = time.time()
        df = gen.key_candidates(rows[i:i + a.chunk])
        t2 = time.time()
        df = gen.cheap_features(df)
        t3 = time.time()
        t_stage1 += t2 - t1
        t_sim += t3 - t2
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    code = (df.s1_row.to_numpy() << PAIR_SHIFT) | df.cand.to_numpy()
    df["y"] = np.isin(code, tcodes)
    per = df.groupby("s1_row").size()
    n_true = len(trace)
    res = {"caps": caps, "n_s1": len(rows), "true_pairs": n_true,
           "stage1_pairs_per_s1": {"mean": float(per.mean()), "p50": float(per.median()),
                                   "p99": float(per.quantile(.99)), "max": int(per.max())},
           "stage1_recall": float(df.y.sum() / n_true),
           "sec_per_1k_s1": {"stage1": 1000 * t_stage1 / len(rows), "cheap_sim": 1000 * t_sim / len(rows)}}
    d = df.sort_values(["s1_row", "pre_score"], ascending=[True, False], kind="stable")
    d["rank"] = d.groupby("s1_row").cumcount() + 1
    res["recall_at_k"] = {k: float(d.y[d["rank"] <= k].sum() / n_true) for k in (20, 40, 60, 80, 100, 150, 200)}
    true = true_all[true_all.s1_id.isin(set(ids))]
    s1_of = pd.Series(ids, index=rows)
    for k in (40, 60, 100):
        hit = d[(d["rank"] <= k) & d.y]
        hp = trace.set_index(["s1_row", "cand"]).loc[list(zip(hit.s1_row, hit.cand))].reset_index()
        res[f"ceiling_top{k}"] = evaluate(hp[["s1_id", "other_id"]], true, ids)["macro_f05"]
    res["ceiling_stage1"] = evaluate(
        trace.set_index(["s1_row", "cand"]).loc[list(zip(df.s1_row[df.y], df.cand[df.y]))]
        .reset_index()[["s1_id", "other_id"]], true, ids)["macro_f05"]
    # unique contribution of each key among true pairs found
    yt = df[df.y]
    res["unique_true_by_key"] = {n: int(((yt.kbits & (1 << i)) > 0).where(yt.n_keys == 1, False).sum())
                                 for i, n in enumerate(gen.keys)}
    print(json.dumps(res, indent=1))
    if a.save:
        df.to_parquet(os.path.join(OUT, "stage1_v2_sample.parquet"), index=False)
    del s1_of


if __name__ == "__main__":
    main()
