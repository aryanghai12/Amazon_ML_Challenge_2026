"""Score every TEST Source 1 entity once and write output/candidate_pairs.tsv.

  python -m src.predict                     # uses configs/pipeline.yaml + models/

Writes
  output/candidate_pairs.tsv               exactly the pairs the model scored (top_k per S1)
  data/interim/scored_test.parquet         s1_id, other_id, p   (p >= min_keep_prob)
Then run  python -m src.make_submission --variant 1  (seconds) to create
output/matching_results.tsv from the stored scores.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src.config import load_config
from src.io.output_writer import CAND_HEADER, StreamingIdListWriter
from src.matching.model import LGBMMatcher
from src.scoring import build_pairs, chunks, make_diff, make_generator, make_rarity

SCORED_PATH = os.path.join("data", "interim", "scored_test.parquet")


def cand_tag(B, models_dir):
    """Short hash of the blocking config + stage ranker: saved candidates are only reused when identical."""
    import hashlib
    h = hashlib.md5(json.dumps(B, sort_keys=True).encode())
    rk = os.path.join(models_dir, "stage_ranker.txt")
    if os.path.exists(rk):
        with open(rk, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:8]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join("configs", "pipeline.yaml"))
    ap.add_argument("--split", default="test", help="test (default) or train for a dry run")
    ap.add_argument("--limit", type=int, default=None, help="only the first N S1 rows (smoke test)")
    ap.add_argument("--deadline-min", type=float, default=None,
                    help="stop after this many minutes; the S1 processed so far are listed in <tag>_done.parquet")
    ap.add_argument("--tag", default=None, help="write candidates / scores under data/interim/<tag>_* (not output/)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    P, B = cfg["paths"], cfg["blocking"]
    with open(os.path.join(P["models_dir"], "decision.json")) as f:
        decision = json.load(f)
    if decision["blocking"] != B:
        print("WARNING: blocking config differs from the one used in training:", decision["blocking"])
    model = LGBMMatcher.load(P["models_dir"])
    t0 = time.time()
    ranker = None
    if B.get("version", 1) == 2:
        from src.blocking.ranker import StageRanker
        ranker = StageRanker.load(P["models_dir"])
    gen = make_generator(P, B, a.split, ranker=ranker)
    diff = make_diff(P, cfg.get("features"), a.split)
    rarity = make_rarity(P, B, cfg.get("features"), gen)
    extra = []
    if (cfg.get("features") or {}).get("brand"):
        from src.features.brand import BrandFeatures, BrandScorer
        extra.append(BrandFeatures(BrandScorer.load(P["models_dir"]), gen.records))
    if (cfg.get("features") or {}).get("competition"):
        from src.features.competition import Competition
        extra.append(Competition(P["processed_dir"], a.split, gen.records, B.get("index_name", "index_v2")))
    for spec in (cfg.get("features") or {}).get("competition_v2", []) or []:
        from src.features.competition import CompetitionV2
        extra.append(CompetitionV2(P["processed_dir"], a.split, gen.records, spec["key"], spec["prefix"],
                                   B.get("index_name", "index_v2"), cap=spec.get("cap", 48)))
    s1_ids = gen.records.s1_ids()
    rows = np.arange(len(s1_ids)) if a.limit is None else np.arange(min(a.limit, len(s1_ids)))

    # only the TEST candidates go to output/ (a train dry run must never overwrite the submission file)
    cand_path = (os.path.join(P["output_dir"], "candidate_pairs.tsv") if a.split == "test" and not a.tag
                 else os.path.join("data", "interim", f"candidate_pairs_{a.tag or a.split}.tsv"))
    done_ids = []
    writer = StreamingIdListWriter(cand_path, CAND_HEADER)
    scored_parts, n_pairs = [], 0
    # candidates are saved per chunk so another model can rescore without regenerating them
    cdir = os.path.join("data", "interim", f"candidates_{a.split}_{cand_tag(B, P['models_dir'])}")
    os.makedirs(cdir, exist_ok=True)
    for part in chunks(rows, B["s1_chunk"]):
        cpath = os.path.join(cdir, f"rows_{part[0]:08d}_{part[-1]:08d}.parquet")
        cands = pd.read_parquet(cpath) if os.path.exists(cpath) else None
        if cands is None:
            cands = gen.generate(part)
            cands.to_parquet(cpath, index=False)
        meta, X = build_pairs(gen, part, diff, rarity, cands=cands, extra=extra)
        lists = {}
        if X is not None:
            p = model.predict(X)
            meta["p"] = p.astype(np.float32)
            lists = meta.groupby("s1_id", sort=False)["other_id"].agg(list).to_dict()
            keep_m = meta["p"].to_numpy() >= decision["min_keep_prob"]
            keep = meta[keep_m][["s1_id", "other_id", "p"]]
            # raw pair columns used by the entity-aware stacker (src/matching/stacker.py)
            from src.matching.stacker import RAW
            for c in RAW:
                if c in X.columns:
                    keep[c] = X[c].to_numpy()[keep_m]
            scored_parts.append(keep)
            n_pairs += len(meta)
        writer.write(s1_ids[part], lists)
        done_ids.append(s1_ids[part])
        if a.deadline_min and time.time() - t0 > a.deadline_min * 60:
            print(f"deadline reached after {len(np.concatenate(done_ids)):,} S1", flush=True)
            break
        print(f"  {part[-1] + 1:,}/{len(rows):,} S1 scored, {n_pairs:,} candidate pairs "
              f"({time.time() - t0:.0f}s)", flush=True)
    writer.close()
    scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else \
        pd.DataFrame(columns=["s1_id", "other_id", "p"])
    os.makedirs(os.path.dirname(SCORED_PATH), exist_ok=True)
    out_scored = (os.path.join("data", "interim", f"scored_{a.tag}.parquet") if a.tag else
                  SCORED_PATH if a.split == "test" else SCORED_PATH.replace("test", a.split))
    scored.to_parquet(out_scored, index=False)
    if a.tag:
        pd.DataFrame({"s1_id": np.concatenate(done_ids)}).to_parquet(
            os.path.join("data", "interim", f"{a.tag}_done.parquet"), index=False)
    meta_out = {"split": a.split, "n_s1": int(len(rows)), "candidate_pairs": int(n_pairs),
                "scored_kept": int(len(scored)), "seconds": round(time.time() - t0),
                "model_exp_id": decision["exp_id"]}
    with open(os.path.join("data", "interim", f"predict_{a.split}.json"), "w") as f:
        json.dump(meta_out, f, indent=1)
    print(json.dumps(meta_out, indent=1))
    print(f"wrote {cand_path}; next: python -m src.make_submission --variant 1")


if __name__ == "__main__":
    main()
