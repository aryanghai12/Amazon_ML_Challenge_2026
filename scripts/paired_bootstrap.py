"""Paired bootstrap (over validation S1) of the macro-F0.5 difference between two experiments.

    python -m scripts.paired_bootstrap experiments/results/E5_rounds experiments/results/E8_competition
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from scripts.error_budget import per_entity
from src.matching.aggregation import decide_pairs
from src.train import true_pairs_from_gt


def entity_f(res, ids, true):
    rep = json.load(open(f"{res}/report.json"))
    t, r = rep["best_decision"]["threshold"], rep["best_decision"]["relative"]
    v = pd.read_parquet(f"{res}/valid_scored.parquet", columns=["s1_id", "other_id", "p"])
    return per_entity(decide_pairs(v[v.p >= 0.02], t, r, True), true, ids)


def main():
    a, b = sys.argv[1], sys.argv[2]
    ids = pd.read_parquet("experiments/audit/valid_ids.parquet").s1_id.to_numpy()
    true = true_pairs_from_gt("data/processed")
    true = true[true.s1_id.isin(set(ids))]
    d = entity_f(b, ids, true) - entity_f(a, ids, true)
    rng = np.random.default_rng(0)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    print(f"{b} - {a}: delta {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  "
          f"S1 better {int((d > 0).sum())}, worse {int((d < 0).sum())}")


if __name__ == "__main__":
    main()
