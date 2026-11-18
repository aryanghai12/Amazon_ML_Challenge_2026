"""Audit: error budget of the baseline (blocking + matcher + decision) on the validation S1.

Re-creates the validation candidates exactly as src.train does, scores them with
the saved model, and decomposes 1 - macro F0.5 by sequential counterfactuals:

  actual -> remove false merges -> add missed in-candidate true pairs -> add blocking misses (= 1.0)

Each step's gain is the F0.5 lost to that error source (sequential attribution).

    python -m scripts.audit_matcher [--models models/baseline_M20260927_1454]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.blocking.candidates import CandidateGenerator
from src.config import load_config
from src.evaluation.metrics import evaluate, f05
from src.matching.aggregation import decide_pairs
from src.matching.model import LGBMMatcher
from src.train import collect, true_pairs_from_gt
from scripts.audit_blocking import OUT, split_rows


def per_entity(pred, true, ids):
    ids = pd.Index(ids)
    pred = pred[["s1_id", "other_id"]].drop_duplicates()
    tp = pred.merge(true, on=["s1_id", "other_id"])
    npred = pred.groupby("s1_id").size().reindex(ids, fill_value=0).to_numpy()
    ntrue = true.groupby("s1_id").size().reindex(ids, fill_value=0).to_numpy()
    ntp = tp.groupby("s1_id").size().reindex(ids, fill_value=0).to_numpy()
    return f05(ntp, npred, ntrue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join("models", "baseline_M20260927_1454"))
    ap.add_argument("--rescore", action="store_true", help="recompute candidates + features")
    a = ap.parse_args()
    cfg = load_config(os.path.join(a.models, "pipeline.yaml"))
    P, B, T = cfg["paths"], cfg["blocking"], cfg["train"]
    with open(os.path.join(a.models, "decision.json")) as f:
        dec = json.load(f)
    thr, rel = dec["best"]["threshold"], dec["best"]["relative"]
    true_all = true_pairs_from_gt(P["processed_dir"])

    cache = os.path.join(OUT, "valid_scored_baseline.parquet")
    gen = CandidateGenerator(P["processed_dir"], "train", B["keys"], B["pre_k"], B["top_k"])
    s1_ids = gen.records.s1_ids()
    _, _, rows_val = split_rows(len(s1_ids), T)
    val_ids = s1_ids[np.sort(rows_val)]
    if a.rescore or not os.path.exists(cache):
        meta, X = collect(gen, rows_val, B["s1_chunk"], true_all, tag="valid")
        model = LGBMMatcher.load(a.models)
        meta["p"] = model.predict(X).astype(np.float32)
        out = pd.concat([meta.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        out.to_parquet(cache, index=False)
    df = pd.read_parquet(cache)
    true = true_all[true_all.s1_id.isin(set(val_ids))].drop_duplicates()
    trace = pd.read_parquet(os.path.join(OUT, "blocking_trace.parquet"))

    scored = df[df.p >= dec["min_keep_prob"]][["s1_id", "other_id", "p"]]
    kept = decide_pairs(scored, thr, rel, dec["one_to_one"])
    ev = evaluate(kept, true, val_ids)
    print(f"reproduced baseline valid macro F0.5 = {ev['macro_f05']:.10f}")

    kept = kept.merge(true.assign(_y=1), on=["s1_id", "other_id"], how="left")
    kept["_y"] = kept._y.fillna(0).astype(int)
    tp_pred = kept[kept._y == 1][["s1_id", "other_id"]]
    n_true = true.groupby("s1_id").size().reindex(val_ids, fill_value=0)
    single_ids = set(n_true[n_true == 0].index)

    # ---------- sequential counterfactuals
    f0 = per_entity(kept, true, val_ids).mean()
    fp = kept[kept._y == 0]
    fp_single = fp[fp.s1_id.isin(single_ids)]
    fp_multi = fp[~fp.s1_id.isin(single_ids)]
    f1a = per_entity(kept.drop(fp_single.index), true, val_ids).mean()
    f1 = per_entity(tp_pred, true, val_ids).mean()
    cand_true = df[df._y == 1][["s1_id", "other_id", "p"]]
    missed = cand_true.merge(tp_pred.assign(_k=1), on=["s1_id", "other_id"], how="left")
    missed = missed[missed._k.isna()].drop(columns="_k")
    # why was each in-candidate true pair not predicted?
    above = scored[scored.p >= thr]
    owner = above.sort_values("p", ascending=False).drop_duplicates("other_id")[["other_id", "s1_id"]]
    missed = missed.merge(owner.rename(columns={"s1_id": "owner"}), on="other_id", how="left")
    missed["reason"] = np.where(missed.p < thr, "below_threshold",
                        np.where(missed.owner != missed.s1_id, "one_to_one_lost", "relative_rule"))
    f2 = per_entity(pd.concat([tp_pred, cand_true[["s1_id", "other_id"]]]), true, val_ids).mean()
    budget = {"actual": f0,
              "false_merges_on_singletons": f1a - f0,
              "false_merges_on_matched_s1": f1 - f1a,
              "matcher_misses_in_candidates": f2 - f1,
              "blocking_misses": 1.0 - f2}
    # split matcher misses by reason (marginal: add only that reason's pairs to f1 state)
    for r, g in missed.groupby("reason"):
        budget[f"  miss:{r}"] = per_entity(pd.concat([tp_pred, g[["s1_id", "other_id"]]]), true, val_ids).mean() - f1
    # split blocking misses by stage (marginal on top of f2 state)
    tr = trace
    base_pairs = pd.concat([tp_pred, cand_true[["s1_id", "other_id"]]])
    for lab, mask in (("no_key", ~tr.stage1), ("pre_k_cap", tr.stage1 & ~tr.stage2),
                      ("top_k_cap", tr.stage2 & ~tr.stage3)):
        add = tr[mask][["s1_id", "other_id"]]
        budget[f"  block:{lab}"] = per_entity(pd.concat([base_pairs, add]), true, val_ids).mean() - f2
    counts = {"pred_pairs": len(kept), "tp": int(kept._y.sum()), "fp": len(fp),
              "fp_on_singletons": len(fp_single), "fp_on_matched": len(fp_multi),
              "singleton_s1_with_fp": fp_single.s1_id.nunique(), "n_singletons": len(single_ids),
              "true_pairs": len(true), "true_in_candidates": len(cand_true),
              "missed_in_candidates": len(missed), **missed.reason.value_counts().to_dict(),
              "blocking_missed": int((~tr.stage3).sum())}
    print(json.dumps({"budget": budget, "counts": counts}, indent=1))

    # ---------- ranking quality within candidates
    d = df.sort_values(["s1_id", "p"], ascending=[True, False])
    d["rank"] = d.groupby("s1_id").cumcount() + 1
    tr_r = d[d._y == 1]
    k_true = tr_r.groupby("s1_id").size()
    print("\nrank of in-candidate true pairs by model p:",
          {k: round(float((tr_r["rank"] <= k).mean()), 4) for k in (1, 2, 3, 5, 10, 20, 40)})
    # "perfect top-n": for each S1 the n_true(in cand) highest-scored candidates are exactly the true ones?
    d = d.merge(k_true.rename("kt"), left_on="s1_id", right_index=True, how="left")
    d["kt"] = d.kt.fillna(0)
    ok = d[d["rank"] <= d.kt].groupby("s1_id")._y.mean()
    print(f"S1 whose top-(n true in cand) by p are all true: {(ok == 1).mean():.4f} of {len(ok):,}")
    print("p of true pairs  quantiles:", np.round(np.quantile(df.p[df._y == 1], [.01, .05, .1, .25, .5]), 3))
    print("p of false pairs quantiles:", np.round(np.quantile(df.p[df._y == 0], [.5, .9, .99, .999, .9999]), 3))

    # ---------- slices
    s1c = gen.records.tables[1].select(["entity_id"]).to_pandas()
    ctry = pd.read_parquet(os.path.join(P["processed_dir"], "train", "source1.parquet"), columns=["entity_id", "country"])
    ctry = ctry[ctry.entity_id.isin(set(val_ids))].set_index("entity_id").country
    fe = pd.Series(per_entity(kept, true, val_ids), index=val_ids)
    fc = pd.Series(per_entity(base_pairs, true, val_ids), index=val_ids)
    sl = pd.DataFrame({"f": fe, "ceil_cand": fc, "country": ctry.reindex(val_ids).to_numpy(),
                       "n_true": n_true.to_numpy()})
    print("\nby country:\n", sl.groupby("country")[["f", "ceil_cand"]].mean())
    sl["n_true_b"] = sl.n_true.clip(upper=8)
    print("\nby number of true matches:\n", sl.groupby("n_true_b").agg(n=("f", "size"), f=("f", "mean"),
                                                                     ceil=("ceil_cand", "mean")))
    missed.to_parquet(os.path.join(OUT, "missed_in_cand_baseline.parquet"), index=False)
    fp.to_parquet(os.path.join(OUT, "fp_baseline.parquet"), index=False)
    del s1c


if __name__ == "__main__":
    main()
