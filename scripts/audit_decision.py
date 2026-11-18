"""Entity-level decision audit of an experiment's validation scores.

Separates RANKING quality from the per-S1 CUT-OFF choice:
  oracle_prefix     : per S1, the best prefix of its candidates sorted by p (k chosen with labels)
  oracle_prefix_121 : same after global one-to-one resolution
  oracle_count      : per S1, keep exactly n_true(in candidates) top-p candidates
and splits the missed / false pairs of the actual rule by mechanism.

    python -m scripts.audit_decision experiments/results/E5_rounds
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.evaluation.metrics import f05
from src.matching.aggregation import decide_pairs
from src.train import true_pairs_from_gt


def best_prefix(d, ids, ntrue_all):
    """d: s1_id, p, _y sorted by s1_id, p desc. -> macro F0.5 of the oracle prefix choice."""
    g = d.groupby("s1_id", sort=False)
    k = g.cumcount().to_numpy() + 1
    tp = g["_y"].cumsum().to_numpy()
    nt = ntrue_all.reindex(d.s1_id).to_numpy()
    f = f05(tp, k, nt)
    best = pd.Series(f).groupby(d.s1_id.to_numpy(), sort=False).max()
    ent = ntrue_all.reindex(ids)
    empty_f = (ent == 0).astype(float)                   # predicting nothing
    fb = best.reindex(ids).fillna(0).to_numpy()
    return np.maximum(fb, empty_f.to_numpy()).mean()


def main():
    res = sys.argv[1]
    t = float(sys.argv[2]) if len(sys.argv) > 2 else None
    r = float(sys.argv[3]) if len(sys.argv) > 3 else None
    rep = json.load(open(f"{res}/report.json"))
    t = t if t is not None else rep["best_decision"]["threshold"]
    r = r if r is not None else rep["best_decision"]["relative"]
    v = pd.read_parquet(f"{res}/valid_scored.parquet")
    ids = pd.read_parquet("experiments/audit/valid_ids.parquet").s1_id.to_numpy()
    true = true_pairs_from_gt("data/processed")
    true = true[true.s1_id.isin(set(ids))]
    ntrue = true.groupby("s1_id").size().reindex(ids, fill_value=0)

    d = v.sort_values(["s1_id", "p"], ascending=[True, False], kind="stable")
    out = {"oracle_prefix": best_prefix(d, ids, ntrue)}
    d121 = v.sort_values("p", ascending=False, kind="stable").drop_duplicates("other_id")
    d121 = d121.sort_values(["s1_id", "p"], ascending=[True, False], kind="stable")
    out["oracle_prefix_after_121"] = best_prefix(d121, ids, ntrue)
    kin = d.groupby("s1_id")._y.transform("sum").to_numpy()
    rank = d.groupby("s1_id").cumcount().to_numpy() + 1
    oc = d[rank <= kin]
    g = lambda x: x.groupby("s1_id").size().reindex(ids, fill_value=0).to_numpy()  # noqa: E731
    out["oracle_count_topn"] = float(f05(g(oc[oc._y == 1]), g(oc), ntrue.to_numpy()).mean())

    kept = decide_pairs(v[v.p >= 0.02][["s1_id", "other_id", "p"]], t, r, True).assign(_k=1)
    m = v.merge(kept[["s1_id", "other_id", "_k"]], on=["s1_id", "other_id"], how="left")
    miss = m[(m._y == 1) & m._k.isna()]
    above = v[v.p >= t]
    owner = above.sort_values("p", ascending=False).drop_duplicates("other_id").set_index("other_id").s1_id
    mech = np.where(miss.p < t, "below_threshold",
                    np.where(owner.reindex(miss.other_id).to_numpy() != miss.s1_id.to_numpy(), "one_to_one",
                             "relative_rule"))
    out["missed_by_mechanism"] = pd.Series(mech).value_counts().to_dict()
    out["missed_p_quantiles"] = np.quantile(miss.p, [.1, .25, .5, .75, .9]).round(3).tolist()
    mm = m.sort_values(["s1_id", "p"], ascending=[True, False])
    mm["rank"] = mm.groupby("s1_id").cumcount() + 1
    mr = mm[(mm._y == 1) & mm._k.isna()]
    out["missed_rank_within_s1"] = mr["rank"].clip(upper=8).value_counts().sort_index().to_dict()
    # do missed pairs sit ABOVE a false pair of the same S1? (pure ranking error) or below all positives
    fp = m[(m._y == 0) & (m._k == 1)]
    out["n_fp"], out["n_missed"] = len(fp), len(miss)
    top_false = m[m._y == 0].groupby("s1_id").p.max()
    out["missed_ranked_above_every_false_of_its_s1"] = float(
        (miss.p.to_numpy() > top_false.reindex(miss.s1_id).fillna(0).to_numpy()).mean())
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
