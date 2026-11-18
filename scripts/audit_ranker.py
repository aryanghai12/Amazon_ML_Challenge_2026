"""Can a learned stage-2 ranker keep more true pairs than the heuristic pre_score?
Exploration on the saved v2 stage-1 sample: fit on half of the S1, measure on the other half.

    python -m scripts.audit_ranker
"""
from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.audit_blocking import OUT
from src.blocking.ranker import ranker_features


def recall_at(d, score, ks, n_true):
    d = d.assign(_s=score).sort_values(["s1_row", "_s"], ascending=[True, False], kind="stable")
    r = d.groupby("s1_row").cumcount() + 1
    return {k: round(float(d.y[r <= k].sum() / n_true), 4) for k in ks}


def main():
    df = pd.read_parquet(os.path.join(OUT, "stage1_v2_sample.parquet"))
    trace = pd.read_parquet(os.path.join(OUT, "blocking_trace.parquet"))
    s1 = np.sort(df.s1_row.unique())
    rng = np.random.default_rng(0)
    fit_rows = set(rng.choice(s1, len(s1) // 2, replace=False).tolist())
    fit = df[df.s1_row.isin(fit_rows)]
    ev = df[~df.s1_row.isin(fit_rows)]
    n_true_ev = int(trace.s1_row.isin(set(ev.s1_row.unique())).sum())
    n_keys = 11
    Xf, Xe = ranker_features(fit.copy(), n_keys), ranker_features(ev.copy(), n_keys)
    ks = (10, 20, 40, 60, 100)
    print("true pairs in eval half:", n_true_ev, " stage-1 recall:", round(ev.y.sum() / n_true_ev, 4))
    print("heuristic pre_score :", recall_at(ev, ev.pre_score.to_numpy(), ks, n_true_ev))
    print("idf only            :", recall_at(ev, ev.idf.to_numpy(), (100, 200, 500, 1000), n_true_ev))
    params = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 31, "min_data_in_leaf": 200,
              "feature_fraction": 0.9, "verbose": -1, "num_threads": 0}
    m = lgb.train(params, lgb.Dataset(Xf, fit.y.astype(int)), num_boost_round=200)
    print("learned (binary)    :", recall_at(ev, m.predict(Xe), ks, n_true_ev))
    # cheap-only ranker (no string similarity) for a first cut before the sims
    cheap = [c for c in Xf.columns if c not in ("name_s", "addr_s", "name_s_gap", "addr_s_gap", "mx_s")]
    mc = lgb.train(params, lgb.Dataset(Xf[cheap], fit.y.astype(int)), num_boost_round=200)
    print("learned, no sims    :", recall_at(ev, mc.predict(Xe[cheap]), (100, 200, 300, 500, 1000), n_true_ev))
    imp = pd.Series(m.feature_importance("gain"), index=Xf.columns).sort_values(ascending=False)
    print(imp.head(15).round(0).to_dict())


if __name__ == "__main__":
    main()
