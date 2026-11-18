"""Compare decision rules on an experiment's validation scores, honestly:
parameters are tuned on half A of the validation S1 and reported on half B (and vice versa).

    python -m scripts.eval_decision experiments/results/E1_blockv2
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate
from src.matching.aggregation import decide_pairs
from src.matching.expected_f import decide_expected_f
from src.train import true_pairs_from_gt


def main():
    res = sys.argv[1]
    v = pd.read_parquet(res if res.endswith(".parquet") else f"{res}/valid_scored.parquet",
                        columns=["s1_id", "other_id", "p"])
    ids = pd.read_parquet("experiments/audit/valid_ids.parquet").s1_id.to_numpy()
    true = true_pairs_from_gt("data/processed")
    true = true[true.s1_id.isin(set(ids))]
    rng = np.random.default_rng(1)
    half = rng.random(len(ids)) < 0.5
    A, B = ids[half], ids[~half]

    def ev(kept, s):
        return evaluate(kept[kept.s1_id.isin(set(s))], true, s)["macro_f05"]

    grids = {"threshold+relative": [dict(t=t, r=r) for t, r in itertools.product(
                 [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8], [0.0, 0.3, 0.5, 0.7])],
             "expected_f": [dict(alpha=a, m=m) for a, m in itertools.product(
                 [0.8, 1.0, 1.2, 1.5, 2.0], [0.0, 0.1, 0.25, 0.5])]}
    rule = {"threshold+relative": lambda s, t, r: decide_pairs(s, t, r, True),
            "expected_f": lambda s, alpha, m: decide_expected_f(s, alpha, m)}
    out = {}
    for name, grid in grids.items():
        scores = [(g, ev(rule[name](v, **g), A), ev(rule[name](v, **g), B)) for g in grid]
        bestA = max(scores, key=lambda x: x[1])
        bestB = max(scores, key=lambda x: x[2])
        full_best = max(scores, key=lambda x: x[1] + x[2])
        out[name] = {"tuned_on_A_eval_B": bestA[2], "tuned_on_B_eval_A": bestB[1],
                     "cross_fitted_mean": (bestA[2] + bestB[1]) / 2, "params_A": bestA[0], "params_B": bestB[0],
                     "full_valid_best": ev(rule[name](v, **full_best[0]), ids), "params_full": full_best[0]}
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
