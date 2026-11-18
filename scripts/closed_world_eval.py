"""Closed-world validation: the validation S1 scored TOGETHER with every other train S1.

On the hidden test every S1 is scored, so one-to-one resolution (each S2/S3 record goes
to at most one S1) sees all competitors. The open-world validation of src.train only
scores 60k S1, so records owned by other S1 are never taken away. This script uses the
scores of `python -m src.predict --split train` (all 2.2M train S1) and evaluates the
validation S1 both ways.

    python -m scripts.closed_world_eval [--scored data/interim/scored_train.parquet]
"""
from __future__ import annotations

import argparse
import itertools
import json

import pandas as pd

from src.evaluation.metrics import evaluate
from src.matching.aggregation import decide_pairs
from src.train import true_pairs_from_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="data/interim/scored_train.parquet")
    a = ap.parse_args()
    sc = pd.read_parquet(a.scored, columns=["s1_id", "other_id", "p"])
    ids = pd.read_parquet("experiments/audit/valid_ids.parquet").s1_id.to_numpy()
    vset = set(ids)
    true = true_pairs_from_gt("data/processed")
    true = true[true.s1_id.isin(vset)]
    open_sc = sc[sc.s1_id.isin(vset)]
    rows = []
    for t, r in itertools.product([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], [0.0, 0.5, 0.7]):
        eo = evaluate(decide_pairs(open_sc, t, r, True), true, ids)
        kc = decide_pairs(sc, t, r, True)
        ec = evaluate(kc[kc.s1_id.isin(vset)], true, ids)
        rows.append({"threshold": t, "relative": r, "open_world": eo["macro_f05"], "closed_world": ec["macro_f05"],
                     "closed_P": ec["micro_precision"], "closed_R": ec["micro_recall"]})
    tab = pd.DataFrame(rows).sort_values("closed_world", ascending=False)
    print(tab.head(12).to_string(index=False))
    print(json.dumps({"best_open": tab.sort_values("open_world").iloc[-1].to_dict(),
                      "best_closed": tab.iloc[0].to_dict()}, indent=1))


if __name__ == "__main__":
    main()
