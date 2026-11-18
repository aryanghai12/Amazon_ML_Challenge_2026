"""Information-theoretic ceiling from indistinguishable records (validation S1).

A true record o is INDISTINGUISHABLE when its address is empty and its owner's raw
name (case-insensitive) is shared by >= 2 S1 records of the same country: nothing in
the provided fields separates the tied S1 entities for that record. An oracle that
knows every other true pair can then only
  A) leave such records out, or
  B) give each to one tied S1 chosen uniformly (closed world: one owner per record).
max(A, B) bounds the macro F0.5 of ANY compliant system (it ignores all other noise).

    python -m scripts.ambiguity_bound
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.evaluation.metrics import evaluate
from src.train import true_pairs_from_gt


def main():
    ids = pd.read_parquet("experiments/audit/valid_ids.parquet").s1_id.to_numpy()
    vset = set(ids)
    tp = true_pairs_from_gt("data/processed")
    s1 = pq.read_table("data/processed/train/source1.parquet",
                       columns=["entity_id", "business_name", "country"]).to_pandas()
    s1["nm"] = s1.country + "#" + s1.business_name.str.lower().str.strip()
    grp = s1.groupby("nm").entity_id.agg(list)
    s1 = s1.set_index("entity_id")
    miss = []
    for k in (2, 3):
        t = pq.read_table(f"data/processed/train/source{k}.parquet", columns=["entity_id", "addr_missing"]).to_pandas()
        miss.append(t[t.addr_missing == 1].entity_id)
    empty = set(pd.concat(miss))
    tp["nm"] = s1.nm.reindex(tp.s1_id).to_numpy()
    tp["k"] = grp.str.len().reindex(tp.nm).to_numpy()
    amb = tp[tp.other_id.isin(empty) & (tp.k >= 2)]
    # only ambiguous records whose tie set touches the validation S1 matter
    amb = amb[[bool(set(grp[n]) & vset) for n in amb.nm]]
    clean = tp[tp.s1_id.isin(vset)]
    clean = clean[~clean.other_id.isin(set(amb.other_id))]
    truev = tp[tp.s1_id.isin(vset)]
    fA = evaluate(clean, truev, ids)["macro_f05"]
    rng = np.random.default_rng(0)
    fB = []
    for _ in range(5):
        pick = [grp[n][rng.integers(len(grp[n]))] for n in amb.nm]
        extra = pd.DataFrame({"s1_id": pick, "other_id": amb.other_id.to_numpy()})
        fB.append(evaluate(pd.concat([clean[["s1_id", "other_id"]], extra]), truev, ids)["macro_f05"])
    n_amb_v = int(amb.s1_id.isin(vset).sum())
    print(f"validation true pairs: {len(truev):,}; indistinguishable (owned by a validation S1): {n_amb_v:,} "
          f"({n_amb_v / len(truev):.2%}); tie-set size median {amb.k.median():.0f}")
    print(f"ceiling A (omit ambiguous records)      : {fA:.4f}")
    print(f"ceiling B (assign to random tied S1)     : {np.mean(fB):.4f} (+/- {np.std(fB):.4f})")
    print(f"=> information ceiling from this effect alone: {max(fA, np.mean(fB)):.4f}")


if __name__ == "__main__":
    main()
