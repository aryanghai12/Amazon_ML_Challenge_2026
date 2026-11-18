"""Learned stage-2 candidate ranker (blocking v2).

A small LightGBM model scores every stage-1 (key-union) pair from cheap signals
only - which keys fired, their IDF weight, and the name / address token-set
similarity - so the top_k kept per S1 are chosen by estimated match probability
instead of a hand-weighted formula. Trained on the fit S1 of the train split.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


def ranker_features(df: pd.DataFrame, n_keys: int) -> pd.DataFrame:
    """df: s1_row, cand, n_keys, idf, kbits, name_s, addr_s (one row per stage-1 pair)."""
    X = pd.DataFrame({"name_s": df["name_s"].to_numpy(), "addr_s": df["addr_s"].to_numpy(),
                      "n_keys": df["n_keys"].to_numpy(), "idf": df["idf"].to_numpy()})
    bits = df["kbits"].to_numpy()
    for i in range(n_keys):
        X[f"k{i}"] = ((bits >> i) & 1).astype(np.int8)
    X["src3"] = ((df["cand"].to_numpy() >> 23) == 1).astype(np.int8)
    g = df.groupby("s1_row", sort=False)
    X["n_stage1"] = g["cand"].transform("size").to_numpy().astype(np.float32)
    X["name_s_gap"] = (g["name_s"].transform("max") - df["name_s"]).to_numpy()
    X["addr_s_gap"] = (g["addr_s"].transform("max") - df["addr_s"]).to_numpy()
    X["idf_gap"] = (g["idf"].transform("max") - df["idf"]).to_numpy()
    X["mx_s"] = np.maximum(X["name_s"], X["addr_s"])
    return X


class StageRanker:
    def __init__(self, booster=None, n_keys=0, keys=None):
        self.booster, self.n_keys, self.keys = booster, n_keys, keys or []

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(ranker_features(df, self.n_keys), num_threads=0)

    @classmethod
    def fit(cls, df: pd.DataFrame, y, keys, rounds=200):
        import lightgbm as lgb
        # ~0.1% positives: without max_delta_step / L2 the Newton steps overshoot and
        # training DIVERGES (train logloss rises, AUC 0.92 < a single raw feature's 0.94)
        params = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 31, "min_data_in_leaf": 200,
                  "feature_fraction": 0.9, "max_delta_step": 1.0, "lambda_l2": 10.0, "verbose": -1,
                  "num_threads": 0}
        X = ranker_features(df, len(keys))
        b = lgb.train(params, lgb.Dataset(X, np.asarray(y, dtype=np.int8)), num_boost_round=rounds)
        return cls(b, len(keys), list(keys))

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.booster.save_model(os.path.join(directory, "stage_ranker.txt"))
        with open(os.path.join(directory, "stage_ranker.json"), "w") as f:
            json.dump({"keys": self.keys}, f)

    @classmethod
    def load(cls, directory):
        import lightgbm as lgb
        with open(os.path.join(directory, "stage_ranker.json")) as f:
            keys = json.load(f)["keys"]
        return cls(lgb.Booster(model_file=os.path.join(directory, "stage_ranker.txt")), len(keys), keys)
