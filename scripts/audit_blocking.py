"""Audit: where are true pairs lost in candidate generation? (validation S1 of src.train)

Reproduces the exact validation split of src.train (same seed / permutation) and,
for every true (S1, S2/S3) pair, records
  * which blocking keys contain it (stage 1, before any cap)
  * whether it survives the pre_k cap (stage 2) and the top_k re-rank (stage 3)
Writes experiments/audit/blocking_trace.parquet and prints a summary.

    python -m scripts.audit_blocking [--n-valid 60000]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.blocking.candidates import PAIR_SHIFT, CandidateGenerator, _expand, encode
from src.config import load_config
from src.evaluation.metrics import evaluate
from src.train import true_pairs_from_gt

OUT = os.path.join("experiments", "audit")


def id_to_cand(gen):
    """entity_id -> encoded candidate id for all S2/S3 records."""
    parts = []
    for k in (2, 3):
        ids = gen.records.tables[k].column("entity_id").to_numpy(zero_copy_only=False)
        parts.append(pd.Series(encode(np.full(len(ids), k), np.arange(len(ids))), index=ids))
    return pd.concat(parts)


def split_rows(n_s1, T, n_valid=None):
    rng = np.random.default_rng(T["seed"])
    perm = rng.permutation(n_s1)
    n_train = T["n_train_s1"]
    n_es = max(1000, n_train // 10)
    n_valid = n_valid or T["n_valid_s1"]
    return perm[:n_train], perm[n_train:n_train + n_es], perm[n_train + n_es:n_train + n_es + n_valid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join("configs", "pipeline.yaml"))
    ap.add_argument("--n-valid", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    P, B, T = cfg["paths"], cfg["blocking"], cfg["train"]
    os.makedirs(OUT, exist_ok=True)

    gen = CandidateGenerator(P["processed_dir"], "train", B["keys"], B["pre_k"], B["top_k"])
    s1_ids = gen.records.s1_ids()
    _, _, rows_val = split_rows(len(s1_ids), T, a.n_valid)
    rows_val = np.sort(rows_val)
    val_ids = s1_ids[rows_val]

    tp = true_pairs_from_gt(P["processed_dir"])
    tp = tp[tp.s1_id.isin(set(val_ids))].copy()
    row_of = pd.Series(np.arange(len(s1_ids)), index=s1_ids)
    c_of = id_to_cand(gen)
    tp["s1_row"] = row_of.loc[tp.s1_id].to_numpy()
    tp["cand"] = c_of.loc[tp.other_id].to_numpy()
    tp["src"] = tp.other_id.str[1].astype(int)
    tp["code"] = (tp.s1_row.to_numpy() << PAIR_SHIFT) | tp.cand.to_numpy()
    print(f"validation S1: {len(val_ids):,}  true pairs: {len(tp):,}", flush=True)

    # ---------------- stage 1: per key membership (no cap)
    wanted = pd.Index(rows_val)
    per_key_sizes = {}
    for name in gen.keys:
        rows, keys = gen.s1[name]
        m = wanted.get_indexer(rows) >= 0
        r, k = rows[m], keys[m]
        order = np.argsort(k, kind="stable")
        sr, cand = _expand(r[order], k[order], *gen.others[name])
        codes = np.unique((sr << PAIR_SHIFT) | cand)
        per_key_sizes[name] = len(codes)
        tp[f"k_{name}"] = np.isin(tp.code.to_numpy(), codes)
        print(f"  key {name}: {len(codes):,} pairs, recall {tp[f'k_{name}'].mean():.4f}", flush=True)
    kcols = [f"k_{n}" for n in gen.keys]
    tp["stage1"] = tp[kcols].any(axis=1)
    tp["n_keys_true"] = tp[kcols].sum(axis=1)

    # ---------------- stage 2 + 3 (chunked, exactly as src.train does)
    s2_codes, s3_codes, s3_rank = [], [], []
    cand_sizes = []
    for i in range(0, len(rows_val), B["s1_chunk"]):
        part = rows_val[i:i + B["s1_chunk"]]
        kc = gen.key_candidates(part)
        s2_codes.append((kc.s1_row.to_numpy() << PAIR_SHIFT) | kc.cand.to_numpy())
        fc = gen.generate(part)
        fc["rank"] = fc.groupby("s1_row").cumcount() + 1
        s3_codes.append((fc.s1_row.to_numpy() << PAIR_SHIFT) | fc.cand.to_numpy())
        s3_rank.append(fc["rank"].to_numpy())
        cand_sizes.append(kc.groupby("s1_row").size())
        print(f"  stages 2/3: {i + len(part):,}/{len(rows_val):,}", flush=True)
    s2 = np.concatenate(s2_codes)
    s3 = pd.Series(np.concatenate(s3_rank), index=np.concatenate(s3_codes))
    tp["stage2"] = np.isin(tp.code.to_numpy(), s2)
    tp["stage3"] = np.isin(tp.code.to_numpy(), s3.index.to_numpy())
    tp["pre_rank"] = s3.reindex(tp.code.to_numpy()).to_numpy()

    # ---------------- summary
    n = len(tp)
    summ = {"true_pairs": n,
            "stage1_key_union": float(tp.stage1.mean()),
            "stage2_pre_k": float(tp.stage2.mean()),
            "stage3_top_k": float(tp.stage3.mean()),
            "lost_no_key": int((~tp.stage1).sum()),
            "lost_pre_k": int((tp.stage1 & ~tp.stage2).sum()),
            "lost_top_k": int((tp.stage2 & ~tp.stage3).sum()),
            "per_key_recall": {n_: float(tp[f"k_{n_}"].mean()) for n_ in gen.keys},
            "per_key_pairs": per_key_sizes}
    # marginal recall of each key (pairs found ONLY by that key)
    summ["unique_to_key"] = {n_: int((tp[f"k_{n_}"] & (tp.n_keys_true == 1)).sum()) for n_ in gen.keys}
    summ["recall_by_source"] = tp.groupby("src").stage3.mean().to_dict()
    summ["stage1_by_source"] = tp.groupby("src").stage1.mean().to_dict()

    for stage in ("stage1", "stage2", "stage3"):
        ceil = evaluate(tp[tp[stage]][["s1_id", "other_id"]], true_pairs_from_gt(P["processed_dir"]), val_ids)
        summ[f"ceiling_{stage}"] = ceil["macro_f05"]
    tp.drop(columns=["code"]).to_parquet(os.path.join(OUT, "blocking_trace.parquet"), index=False)
    pd.Series(val_ids).to_frame("s1_id").to_parquet(os.path.join(OUT, "valid_ids.parquet"), index=False)
    with open(os.path.join(OUT, "blocking_summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
