"""Error budget of an experiment from its validation scores (sequential counterfactuals):

  actual -> remove false merges (singletons, then matched S1) -> add in-candidate misses -> add
  blocking misses (= 1.0). Also ranking quality and slices.

    python -m scripts.error_budget experiments/results/<exp> [--rule thr|ef] [--t 0.6 --r 0.7 | --alpha 1 --m 0]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.evaluation.metrics import f05
from src.matching.aggregation import decide_pairs
from src.matching.expected_f import decide_expected_f
from src.train import true_pairs_from_gt


def per_entity(pred, true, ids):
    ids = pd.Index(ids)
    pred = pred[["s1_id", "other_id"]].drop_duplicates()
    tp = pred.merge(true, on=["s1_id", "other_id"])
    g = lambda d: d.groupby("s1_id").size().reindex(ids, fill_value=0).to_numpy()  # noqa: E731
    return f05(g(tp), g(pred), g(true))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("res")
    ap.add_argument("--rule", default="thr")
    ap.add_argument("--t", type=float, default=None)
    ap.add_argument("--r", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--m", type=float, default=0.0)
    a = ap.parse_args()
    v = pd.read_parquet(os.path.join(a.res, "valid_scored.parquet"))
    rep = json.load(open(os.path.join(a.res, "report.json")))
    t = a.t if a.t is not None else rep["best_decision"]["threshold"]
    r = a.r if a.r is not None else rep["best_decision"]["relative"]
    ids = pd.read_parquet("experiments/audit/valid_ids.parquet").s1_id.to_numpy()
    true = true_pairs_from_gt("data/processed")
    true = true[true.s1_id.isin(set(ids))].drop_duplicates()
    s = v[["s1_id", "other_id", "p"]]
    kept = decide_pairs(s[s.p >= 0.02], t, r, True) if a.rule == "thr" else decide_expected_f(s, a.alpha, a.m)
    kept = kept.merge(true.assign(_y=1), on=["s1_id", "other_id"], how="left")
    kept["_y"] = kept._y.fillna(0).astype(int)
    tpp = kept[kept._y == 1][["s1_id", "other_id"]]
    ntrue = true.groupby("s1_id").size().reindex(ids, fill_value=0)
    single = set(ntrue[ntrue == 0].index)
    fp = kept[kept._y == 0]
    f0 = per_entity(kept, true, ids).mean()
    f1a = per_entity(kept.drop(fp[fp.s1_id.isin(single)].index), true, ids).mean()
    f1 = per_entity(tpp, true, ids).mean()
    cand_true = v[v._y == 1][["s1_id", "other_id"]]
    f2 = per_entity(pd.concat([tpp, cand_true]), true, ids).mean()
    out = {"rule": a.rule, "params": {"t": t, "r": r} if a.rule == "thr" else {"alpha": a.alpha, "m": a.m},
           "macro_f05": f0,
           "loss_false_merges_singletons": f1a - f0, "loss_false_merges_matched": f1 - f1a,
           "loss_matcher_misses_in_candidates": f2 - f1, "loss_blocking_misses": 1 - f2,
           "candidate_recall": len(cand_true) / len(true), "ceiling_perfect_matcher": f2,
           "n_fp": len(fp), "n_fp_singletons": int(fp.s1_id.isin(single).sum()),
           "n_missed_in_cand": len(cand_true) - len(tpp), "n_blocking_missed": len(true) - len(cand_true)}
    d = v.sort_values(["s1_id", "p"], ascending=[True, False])
    d["rank"] = d.groupby("s1_id").cumcount() + 1
    out["true_rank_topk"] = {k: round(float((d[d._y == 1]["rank"] <= k).mean()), 4) for k in (1, 3, 5, 10)}
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
