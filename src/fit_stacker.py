"""Fit the entity-aware stacker (decision v3) for a trained matcher.

The matcher's scores on the VALIDATION S1 are out of sample (those S1 were never used to
fit it), so they have the same distribution as the test scores. The stacker is trained on
all of them; its threshold / relative rule comes from a cross-fitted evaluation (K folds by
S1, then tune on half A / report on half B and back) - see scripts/stacker_cv.py.

    python -m src.fit_stacker --config configs/pipeline_final.yaml --exp E10_comp2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import pandas as pd

from src.config import load_config
from src.matching import stacker

CACHE = "experiments/cache/cc96b7c254"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    res = os.path.join("experiments", "results", a.exp)
    out = os.path.join("experiments", "results", a.exp + "_stacker")
    # 1) honest estimate + decision rule (cross-fitted)
    subprocess.run([sys.executable, "-m", "scripts.stacker_cv", res, "--out", out], check=True)
    rep = json.load(open(os.path.join(out, "report.json")))
    # 2) final stacker on all validation S1
    v = pd.read_parquet(os.path.join(res, "valid_scored.parquet"))
    v = v[v.p >= cfg["decision"]["min_keep_prob"]].reset_index(drop=True)   # same rows as on test
    base = pd.read_parquet(f"{CACHE}/val.parquet", columns=["s1_id", "other_id"] + [
        c for c in stacker.RAW if c not in ("nm_sub_o", "nm_sub_s1", "nu_unm_o", "nu_trunc", "nu_edit1", "src3")])
    diff = pd.read_parquet(f"{CACHE}/val_diff.parquet", columns=["nm_sub_o", "nm_sub_s1", "nu_unm_o", "nu_trunc",
                                                                 "nu_edit1"])
    v = v.merge(pd.concat([base, diff], axis=1), on=["s1_id", "other_id"], how="left")
    X = stacker.stack_features(v)
    m = stacker.fit(X, v["_y"])
    md = cfg["paths"]["models_dir"]
    m.save_model(os.path.join(md, "stacker.txt"))
    dec_path = os.path.join(md, "decision.json")
    dec = json.load(open(dec_path))
    dec["stacker"] = {"features": list(X.columns), "threshold": rep["best"]["threshold"],
                      "relative": rep["best"]["relative"], "valid_macro_f05_cross_fitted": rep["cross_fitted"]}
    json.dump(dec, open(dec_path, "w"), indent=1)
    print(f"stacker saved to {md}; decision t={rep['best']['threshold']} r={rep['best']['relative']}, "
          f"cross-fitted valid macro F0.5 {rep['cross_fitted']:.5f}")


if __name__ == "__main__":
    main()
