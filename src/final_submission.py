"""Assemble the final submission from a time-boxed v2 (E9) test run + the baseline for the rest.

The v2 pipeline (blocking v2 + E8 matcher + entity-aware stacker) costs ~4 h on the full test
set on the team laptop. `python -m src.predict --deadline-min N --tag v2test` scores as many
test S1 as fit in the time box; this script then writes, for EVERY test S1:
  * v2-processed S1: v2 candidates; matches = stacker re-score -> cross-fitted threshold rule,
    one-to-one over all v2 S1
  * remaining S1   : the validated baseline (M20260927_1454) candidate and match rows, minus any
    record already matched by a v2 S1 (one-to-one across the two parts)
and runs our checks + the official validator.

    python -m src.final_submission --config configs/pipeline_final.yaml --tag v2test
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import lightgbm as lgb
import pandas as pd

from src.config import load_config
from src.io.output_writer import (CAND_HEADER, MATCH_HEADER, check_submission, run_official_validator,
                                  write_id_lists)
from src.matching.aggregation import decide_pairs, pairs_to_lists
from src.matching.stacker import stack_features

BASE_CAND = os.path.join("data", "interim", "candidate_pairs_baseline_M20260927_1454.tsv")
BASE_MATCH = os.path.join("data", "interim", "matching_results_baseline_M20260927_1454.tsv")


def read_raw(path):
    """{s1_id: raw comma-separated id string} (no splitting: the candidate files hold ~70M ids)."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))


def read_lists(path):
    return {k: (v.split(",") if v else []) for k, v in read_raw(path).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="v2test")
    a = ap.parse_args()
    cfg = load_config(a.config)
    P = cfg["paths"]
    dec = json.load(open(os.path.join(P["models_dir"], "decision.json")))
    st = dec["stacker"]
    s1_ids = pd.read_parquet(os.path.join(P["processed_dir"], "test", "source1.parquet"),
                             columns=["entity_id"])["entity_id"].tolist()
    done = set(pd.read_parquet(os.path.join("data", "interim", f"{a.tag}_done.parquet")).s1_id)

    # ---- v2 part
    scored = pd.read_parquet(os.path.join("data", "interim", f"scored_{a.tag}.parquet"))
    booster = lgb.Booster(model_file=os.path.join(P["models_dir"], "stacker.txt"))
    scored["p"] = booster.predict(stack_features(scored)[st["features"]])
    kept = decide_pairs(scored[["s1_id", "other_id", "p"]], st["threshold"], st["relative"], True)
    v2_match = pairs_to_lists(kept, sorted(done))
    v2_cand = read_raw(os.path.join("data", "interim", f"candidate_pairs_{a.tag}.tsv"))
    taken = set(kept.other_id)

    # ---- baseline part (remaining S1)
    base_match = read_lists(BASE_MATCH)
    base_cand = read_raw(BASE_CAND)
    match, cand = {}, {}
    n_dropped = 0
    for s in s1_ids:
        if s in done:
            match[s], cand[s] = v2_match.get(s, []), v2_cand.get(s, "")
        else:
            m = [x for x in base_match.get(s, []) if x not in taken]
            n_dropped += len(base_match.get(s, [])) - len(m)
            match[s], cand[s] = m, base_cand.get(s, "")
    out_m = os.path.join(P["output_dir"], "matching_results.tsv")
    out_c = os.path.join(P["output_dir"], "candidate_pairs.tsv")
    with open(out_c, "w", encoding="utf-8", newline="\n") as f:        # raw candidate strings, as read
        f.write("\t".join(CAND_HEADER) + "\n")
        for s in s1_ids:
            f.write(f"{s}\t{cand[s]}\n")
    write_id_lists(out_m, MATCH_HEADER, s1_ids, match)
    n_match = sum(len(v) for v in match.values())
    print(f"S1 rows {len(s1_ids):,}: v2 (E9) {len(done):,} ({len(done) / len(s1_ids):.1%}), baseline "
          f"{len(s1_ids) - len(done):,}; matches {n_match:,}; baseline matches dropped by one-to-one {n_dropped:,}")
    problems = check_submission(out_m, out_c, s1_ids)
    print("own checks:", "PASS" if not problems else problems)
    code = run_official_validator(P.get("validator"), out_m, out_c, os.path.join(P["dataset_dir"], "test"))
    print("official validator exit code:", code)
    json.dump({"v2_s1": len(done), "total_s1": len(s1_ids), "matches": n_match, "own_checks": problems,
               "validator_exit": code, "stacker": {k: st[k] for k in ("threshold", "relative",
                                                                      "valid_macro_f05_cross_fitted")}},
              open(os.path.join("data", "interim", "final_submission.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
