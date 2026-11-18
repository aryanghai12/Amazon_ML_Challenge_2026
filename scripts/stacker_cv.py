"""E6: entity-aware stacker, cross-fitted on the validation S1 (by S1, K folds).

Every validation pair gets an OUT-OF-FOLD stacker score; the decision threshold is then
tuned cross-fitted (half A -> half B and back, as scripts/eval_decision). Reports the
same metrics as the other experiments and saves the OOF scores as a valid_scored.parquet.

    python -m scripts.stacker_cv experiments/results/E5_rounds --out experiments/results/E6_stacker
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate
from src.matching import stacker
from src.matching.aggregation import decide_pairs
from src.train import true_pairs_from_gt

CACHE = "experiments/cache/cc96b7c254"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("res")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--no-raw", action="store_true")
    a = ap.parse_args()
    v = pd.read_parquet(f"{a.res}/valid_scored.parquet")
    v = v[v.p >= 0.02].reset_index(drop=True)      # same rows as stored for test (min_keep_prob)
    if not a.no_raw:
        base = pd.read_parquet(f"{CACHE}/val.parquet", columns=["s1_id", "other_id"] + [
            c for c in stacker.RAW if c not in ("nm_sub_o", "nm_sub_s1", "nu_unm_o", "nu_trunc", "nu_edit1",
                                                "src3")])
        diff = pd.read_parquet(f"{CACHE}/val_diff.parquet", columns=["nm_sub_o", "nm_sub_s1", "nu_unm_o",
                                                                     "nu_trunc", "nu_edit1"])
        v = v.merge(pd.concat([base, diff], axis=1), on=["s1_id", "other_id"], how="left")
    X = stacker.stack_features(v)
    ids = pd.read_parquet("experiments/audit/valid_ids.parquet").s1_id.to_numpy()
    rng = np.random.default_rng(7)
    fold_of = pd.Series(rng.integers(0, a.folds, len(ids)), index=ids)
    f = fold_of.reindex(v.s1_id).to_numpy()
    oof = np.zeros(len(v))
    for k in range(a.folds):
        tr = f != k
        m = stacker.fit(X[tr], v._y.to_numpy()[tr])
        oof[~tr] = m.predict(X[~tr])
        print(f"  fold {k}: trained on {tr.sum():,} pairs", flush=True)
    s = v[["s1_id", "other_id", "_y"]].assign(p=oof)
    os.makedirs(a.out, exist_ok=True)
    s.to_parquet(f"{a.out}/valid_scored.parquet", index=False)

    true = true_pairs_from_gt("data/processed")
    true = true[true.s1_id.isin(set(ids))]
    half = np.random.default_rng(1).random(len(ids)) < 0.5
    A, B = ids[half], ids[~half]
    grid = list(itertools.product([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75], [0.0, 0.3, 0.5]))
    rows = []
    for t, r in grid:
        kept = decide_pairs(s[s.p >= 0.02][["s1_id", "other_id", "p"]], t, r, True)
        rows.append({"threshold": t, "relative": r,
                     "A": evaluate(kept[kept.s1_id.isin(set(A))], true, A)["macro_f05"],
                     "B": evaluate(kept[kept.s1_id.isin(set(B))], true, B)["macro_f05"],
                     "full": evaluate(kept, true, ids)["macro_f05"]})
    tab = pd.DataFrame(rows)
    bA, bB = tab.loc[tab.A.idxmax()], tab.loc[tab.B.idxmax()]
    best = tab.loc[tab.full.idxmax()]
    res = {"macro_f05_full_tuned": float(best.full), "best": {"threshold": best.threshold, "relative": best.relative},
           "cross_fitted": float((bA.B + bB.A) / 2), "params_A": [bA.threshold, bA.relative],
           "params_B": [bB.threshold, bB.relative]}
    json.dump({"best_decision": res["best"], **res}, open(f"{a.out}/report.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
