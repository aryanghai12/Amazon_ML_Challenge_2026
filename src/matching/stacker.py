"""Entity-aware second stage (decision v3).

The pair matcher scores every candidate in isolation. The audit of E5 showed that
the ranking inside an S1 is already good (oracle per-S1 cut-off on E5 scores: 0.9945
vs 0.9807) but a single global threshold cuts true matches off the tail of multi-match
entities (74% of the missed true pairs rank above every false candidate of their S1).

The stacker re-scores each candidate from the SHAPE of its S1's score list - p of the
neighbours, the gap to the next candidate (the cliff between matches and non-matches),
how many candidates are confidently matched, per-source counts - plus a few raw pair
features. It is trained on scores the matcher produced for S1 it never saw (out of
sample), so its inputs have the same distribution as on test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW = ["name_tset", "addr_tset", "addr_missing_o", "addr_missing_s1", "compact_ratio", "o_translit",
       "nums_conflict", "nm_sub_o", "nm_sub_s1", "nu_unm_o", "nu_trunc", "nu_edit1", "src3"]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stack_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: s1_id, other_id, p (+ RAW columns if present), ANY row order. Returns features aligned to df."""
    d = df[["s1_id", "p"]].copy()
    d["src3"] = df["other_id"].str.startswith("S3").astype(np.int8).to_numpy()
    d["_i"] = np.arange(len(d))
    d = d.sort_values(["s1_id", "p"], ascending=[True, False], kind="stable")
    g = d.groupby("s1_id", sort=False)
    lp = _logit(d["p"].to_numpy())
    d["lp"] = lp
    d["rank"] = g.cumcount().to_numpy() + 1
    d["p_prev"] = g["p"].shift(1).fillna(1.0).to_numpy()
    d["p_next"] = g["p"].shift(-1).fillna(0.0).to_numpy()
    d["lgap_next"] = lp - _logit(d["p_next"].to_numpy())
    d["lgap_prev"] = _logit(d["p_prev"].to_numpy()) - lp
    d["p_max"] = g["p"].transform("max").to_numpy()
    d["lgap_max"] = _logit(d["p_max"].to_numpy()) - lp
    d["p_sum"] = g["p"].transform("sum").to_numpy()
    d["p_sum_above"] = (g["p"].cumsum() - d["p"]).to_numpy()
    for c in (0.9, 0.5, 0.1):
        d[f"n_above_{int(c * 100)}"] = (d["p"] >= c).astype(np.int64).groupby(d["s1_id"], sort=False) \
            .transform("sum").to_numpy()
    # largest logit gap between consecutive candidates in the S1 and where it sits relative to me
    gaps = d["lgap_next"].where(d["rank"] <= 12, 0.0)
    d["max_gap"] = gaps.groupby(d["s1_id"], sort=False).transform("max").to_numpy()
    cliff_rank = d.assign(_g=gaps).sort_values(["s1_id", "_g"], ascending=[True, False]).groupby(
        "s1_id", sort=False)["rank"].first()
    d["above_cliff"] = (d["rank"].to_numpy() <= cliff_rank.reindex(d["s1_id"]).to_numpy()).astype(np.int8)
    gs = d.groupby(["s1_id", "src3"], sort=False)
    d["rank_in_src"] = gs.cumcount().to_numpy() + 1
    d["n_src_above_50"] = (d["p"] >= 0.5).astype(np.int64).groupby([d["s1_id"], d["src3"]], sort=False) \
        .transform("sum").to_numpy()
    d = d.sort_values("_i")
    X = d.drop(columns=["s1_id", "_i"]).reset_index(drop=True)
    for c in RAW:
        if c in df.columns and c not in X.columns:
            X[c] = df[c].to_numpy()
    return X


PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "verbose": -1, "num_threads": 0}


def fit(X, y, rounds=400):
    import lightgbm as lgb
    return lgb.train(PARAMS, lgb.Dataset(X, np.asarray(y, dtype=np.int8)), num_boost_round=rounds)
