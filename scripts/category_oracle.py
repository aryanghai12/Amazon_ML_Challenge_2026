"""Per-error-category oracle gain of an experiment (validation S1, same decision rule).

For each residual category: F0.5 if ONLY that category's false merges were removed and its
in-candidate misses recovered. Categories are assigned from the pair's own features.

    python -m scripts.category_oracle experiments/results/E5_rounds [--t 0.65 --r 0.7]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from scripts._amb import ambiguous_pairs
from scripts.error_budget import per_entity
from src.matching.aggregation import decide_pairs

CACHE = "experiments/cache/cc96b7c254"


def categorize(v):
    f = pd.read_parquet(f"{CACHE}/val.parquet", columns=["s1_id", "other_id", "name_tset", "addr_tset",
                                                         "addr_missing_o", "o_translit", "nums_conflict",
                                                         "compact_partial"])
    d = pd.read_parquet(f"{CACHE}/val_diff.parquet", columns=["nm_sub_o", "nm_sub_s1", "nu_unm_o"])
    v = v.merge(pd.concat([f, d], axis=1), on=["s1_id", "other_id"], how="left")
    c = np.select([v.amb, v.addr_missing_o == 1, (v.name_tset < 0.5) & (v.compact_partial < 0.6),
                   v.o_translit == 1, (v.nu_unm_o > 0) | (v.nums_conflict == 1),
                   (v.nm_sub_o > 0) | (v.nm_sub_s1 > 0)],
                  ["0 indistinguishable", "1 empty other address", "2 name replaced (address link)",
                   "4 native script", "3 house number", "4 name word substituted"], "5 other")
    return v.assign(cat=c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("res")
    ap.add_argument("--t", type=float, default=None)
    ap.add_argument("--r", type=float, default=None)
    a = ap.parse_args()
    rep = json.load(open(f"{a.res}/report.json"))
    t = a.t if a.t is not None else rep["best_decision"]["threshold"]
    r = a.r if a.r is not None else rep["best_decision"]["relative"]
    v = pd.read_parquet(f"{a.res}/valid_scored.parquet")
    tp = ambiguous_pairs()
    ids = pd.read_parquet("experiments/audit/valid_ids.parquet").s1_id.to_numpy()
    true = tp[tp.s1_id.isin(set(ids))]
    ambset = set(zip(true.s1_id[true.amb], true.other_id[true.amb]))
    v["amb"] = [(x, y) in ambset for x, y in zip(v.s1_id, v.other_id)]
    v = categorize(v)
    kept = decide_pairs(v[v.p >= 0.02][["s1_id", "other_id", "p"]], t, r, True).assign(_k=1)
    m = v.merge(kept[["s1_id", "other_id", "_k"]], on=["s1_id", "other_id"], how="left")
    pred = m[m._k == 1]
    f0 = per_entity(pred, true[["s1_id", "other_id"]], ids).mean()
    rows = []
    for c in sorted(m.cat.unique()):
        fp = (m._k == 1) & (m._y == 0) & (m.cat == c)
        mi = m._k.isna() & (m._y == 1) & (m.cat == c)
        fixed = pd.concat([m[(m._k == 1) & ~fp], m[mi]])
        rows.append({"category": c, "false_merges": int(fp.sum()), "misses": int(mi.sum()),
                     "oracle_gain": per_entity(fixed, true[["s1_id", "other_id"]], ids).mean() - f0})
    tab = pd.DataFrame(rows)
    sfp = int(((m._k == 1) & (m._y == 0) & ~m.s1_id.isin(set(true.s1_id))).sum())
    print(f"macro F0.5 {f0:.4f} (t={t}, r={r}); singleton false merges {sfp}")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
