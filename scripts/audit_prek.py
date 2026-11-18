"""Audit: why does the pre_k cap lose true pairs, and what ranking would keep them?

For a sample of validation S1 it materialises ALL stage-1 (key-union) candidates
with per-key flags and block sizes, then compares candidate-ranking rules by the
recall they keep at several budgets.

    python -m scripts.audit_prek [--n 10000]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.blocking.candidates import PAIR_SHIFT, CandidateGenerator, _expand
from src.config import load_config
from src.features.pair_features import fast_similarity
from scripts.audit_blocking import OUT, split_rows


def stage1_frame(gen, rows):
    wanted = pd.Index(rows)
    frames = []
    for name in gen.keys:
        r_all, k_all = gen.s1[name]
        m = wanted.get_indexer(r_all) >= 0
        r, k = r_all[m], k_all[m]
        order = np.argsort(k, kind="stable")
        r, k = r[order], k[order]
        okeys, ocands = gen.others[name]
        lo = np.searchsorted(okeys, k, "left")
        hi = np.searchsorted(okeys, k, "right")
        bsize = (hi - lo)
        sr, cand = _expand(r, k, okeys, ocands)
        bs = np.repeat(bsize[bsize > 0], bsize[bsize > 0])
        frames.append(pd.DataFrame({"code": (sr << PAIR_SHIFT) | cand, "key": name,
                                    "w": 1.0 / np.log2(1 + bs)}))
    df = pd.concat(frames, ignore_index=True)
    g = df.groupby("code")
    out = pd.DataFrame({"n_keys": g.size(), "idf": g.w.sum()})
    for name in gen.keys:
        out[f"k_{name}"] = df[df.key == name].groupby("code").size().reindex(out.index, fill_value=0) > 0
    out = out.reset_index()
    out["s1_row"] = out.code.to_numpy() >> PAIR_SHIFT
    out["cand"] = out.code.to_numpy() & ((1 << PAIR_SHIFT) - 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()
    cfg = load_config()
    P, B, T = cfg["paths"], cfg["blocking"], cfg["train"]
    gen = CandidateGenerator(P["processed_dir"], "train", B["keys"], B["pre_k"], B["top_k"])
    s1_ids = gen.records.s1_ids()
    _, _, rows_val = split_rows(len(s1_ids), T)
    rows = np.sort(rows_val[:a.n])
    trace = pd.read_parquet(os.path.join(OUT, "blocking_trace.parquet"))
    trace = trace[trace.s1_row.isin(rows)]
    true_codes = set(((trace.s1_row.to_numpy() << PAIR_SHIFT) | trace.cand.to_numpy()).tolist())

    df = stage1_frame(gen, rows)
    df["y"] = df.code.isin(true_codes)
    per = df.groupby("s1_row").size()
    print(f"S1 sample {len(rows):,}; stage-1 pairs {len(df):,}; per S1 mean {per.mean():.0f} "
          f"median {per.median():.0f} p90 {per.quantile(.9):.0f} p99 {per.quantile(.99):.0f} max {per.max()}")
    print(f"true pairs {len(trace):,}; in stage 1: {df.y.sum():,}")

    s1 = gen.records.take_s1(df.s1_row.to_numpy(), ["name_core", "addr_core"])
    ot = gen.records.take_cand(df.cand.to_numpy(), ["name_core", "addr_core"])
    ns, as_ = fast_similarity(s1["name_core"], ot["name_core"], s1["addr_core"], ot["addr_core"])
    df["name_s"], df["addr_s"] = ns, as_
    df["pre_score"] = np.maximum(ns, as_) + 0.5 * np.minimum(ns, as_) + 0.15 * np.minimum(df.n_keys, 4)
    df["src"] = (df.cand.to_numpy() >> 23) + 2

    n_true = len(trace)
    rules = {"n_keys (current, tie=cand id)": ["n_keys", "cand_neg"],
             "idf-weighted keys": ["idf"],
             "pre_score on ALL stage-1": ["pre_score"]}
    df["cand_neg"] = -df.cand
    print("\nrecall of true pairs kept per S1 budget (fraction of ALL true pairs)")
    print("rule".ljust(34), *(f"k={k:<5}" for k in (20, 40, 60, 100, 200, 400)))
    for label, cols in rules.items():
        d = df.sort_values(["s1_row"] + cols, ascending=[True] + [False] * len(cols), kind="stable")
        rank = d.groupby("s1_row").cumcount() + 1
        vals = [d.y[rank <= k].sum() / n_true for k in (20, 40, 60, 100, 200, 400)]
        print(label.ljust(34), *(f"{v:.4f}" for v in vals))
    # current 2-stage: pre_k 100 by n_keys then top 40 by pre_score
    d = df.sort_values(["s1_row", "n_keys", "cand_neg"], ascending=[True, False, False], kind="stable")
    d = d[d.groupby("s1_row").cumcount() < 100]
    d = d.sort_values(["s1_row", "pre_score"], ascending=[True, False], kind="stable")
    d = d[d.groupby("s1_row").cumcount() < 40]
    print(f"current pipeline (pre_k=100 n_keys -> top 40 pre_score): {d.y.sum() / n_true:.4f}")
    lost = df[df.y & ~df.code.isin(d.code)]
    print("\nlost-by-caps true pairs: n_keys distribution\n", lost.n_keys.value_counts().sort_index().to_dict())
    print("lost-by-caps by source:", lost.src.value_counts().to_dict())
    print("stage-1 size of S1 with lost pairs (median):", per.reindex(lost.s1_row.unique()).median())
    df.drop(columns=["cand_neg"]).to_parquet(os.path.join(OUT, "stage1_sample.parquet"), index=False)


if __name__ == "__main__":
    main()
