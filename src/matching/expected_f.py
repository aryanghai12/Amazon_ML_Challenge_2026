"""Expected-F0.5 decision rule (decision v2).

The metric is macro F0.5 per Source-1 entity, so instead of one global probability
threshold we choose, for every S1 separately, the prediction set that maximises
the EXPECTED F0.5 under the model's (calibrated) match probabilities:

  candidates sorted by p (after one-to-one resolution): p1 >= p2 >= ...
  k = 0 :  E[F] = P(no true match)            ~ prod(1 - p_i)
  k >= 1:  E[F] ~ 1.25 * sum_{i<=k} p_i / (0.25 * (sum_i p_i + m) + k)

`alpha` (p <- p**alpha) corrects calibration and `m` is the expected number of true
matches that are not among the candidates. Both are tuned on validation data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def one_to_one(scored: pd.DataFrame) -> pd.DataFrame:
    """Each S2/S3 record is kept only for its highest-scoring S1."""
    return scored.sort_values("p", ascending=False, kind="stable").drop_duplicates("other_id", keep="first")


def decide_expected_f(scored: pd.DataFrame, alpha=1.0, m=0.0, min_p=0.02, beta2=0.25) -> pd.DataFrame:
    """scored: s1_id, other_id, p -> kept pairs."""
    d = one_to_one(scored[scored["p"] >= min_p])
    d = d.assign(q=np.clip(d["p"].to_numpy(np.float64), 1e-9, 1 - 1e-9) ** alpha)
    d = d.sort_values(["s1_id", "q"], ascending=[True, False], kind="stable").reset_index(drop=True)
    g = d.groupby("s1_id", sort=False)
    k = g.cumcount().to_numpy() + 1
    cum = g["q"].cumsum().to_numpy()
    tot = g["q"].transform("sum").to_numpy()
    ef = (1 + beta2) * cum / (beta2 * (tot + m) + k)
    # expected F of predicting nothing: P(no match among candidates) (and none outside)
    lq = pd.Series(np.log1p(-d["q"].to_numpy()), index=d.index)
    ef0 = np.exp(lq.groupby(d["s1_id"].to_numpy(), sort=False).transform("sum").to_numpy()) * np.exp(-m)
    best_pos = pd.Series(ef).groupby(d["s1_id"].to_numpy(), sort=False).transform("idxmax").to_numpy()
    keep = (k <= k[best_pos]) & (ef[best_pos] > ef0)
    return d.loc[keep, ["s1_id", "other_id", "p"]].reset_index(drop=True)
