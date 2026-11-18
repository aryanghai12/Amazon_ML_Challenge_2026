"""Shared step used by training and prediction: candidates -> features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.blocking.candidates import decode
from src.features.pair_features import O_COLS, S1_COLS, add_group_features, pair_features


def build_pairs(gen, s1_rows, diff=None, rarity=None, cands=None, extra=()):
    """Returns (meta, X): meta has s1_row, cand, s1_id, other_id; X the features.
    diff: optional DiffFeaturizer (pair_features v2 token-difference features).
    cands: previously generated candidates of these S1 (skips candidate generation)."""
    if cands is None:
        cands = gen.generate(s1_rows)
    if cands.empty:
        return (pd.DataFrame(columns=["s1_row", "cand", "s1_id", "other_id"]), None)
    s1 = gen.records.take_s1(cands["s1_row"].to_numpy(), S1_COLS + ["entity_id"])
    o = gen.records.take_cand(cands["cand"].to_numpy(), O_COLS + ["entity_id"])
    src, _ = decode(cands["cand"].to_numpy())
    X = pair_features(s1, o, cands["n_keys"].to_numpy(), cands["pre_score"].to_numpy(), src)
    if "kbits" in cands:                         # blocking v2: which keys fired + their IDF weight
        X["idf"] = cands["idf"].to_numpy()
        bits = cands["kbits"].to_numpy()
        for i, name in enumerate(gen.keys):
            X[f"key_{name}"] = ((bits >> i) & 1).astype(np.int8)
    if diff is not None:
        X = pd.concat([X, diff(s1, o)], axis=1)
    for fx in ([rarity] if rarity is not None else []) + list(extra):
        for k, v in fx.features(cands["s1_row"].to_numpy(), cands["cand"].to_numpy()).items():
            X[k] = v
    X["s1_row"] = cands["s1_row"].to_numpy()
    X = add_group_features(X).drop(columns=["s1_row"])
    meta = pd.DataFrame({"s1_row": cands["s1_row"].to_numpy(), "cand": cands["cand"].to_numpy(),
                         "s1_id": s1["entity_id"], "other_id": o["entity_id"]})
    return meta, X


def chunks(rows, size):
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def make_generator(P, B, split, ranker=None):
    """blocking.version 1 (baseline: n_keys pre-cap) or 2 (key union + learned ranker)."""
    from src.blocking.candidates import CandidateGenerator, SimCandidateGenerator
    if B.get("version", 1) == 2:
        return SimCandidateGenerator(P["processed_dir"], split, B["keys"], B["top_k"],
                                     B.get("index_name", "index_v2"), B.get("max_stage1", 4000), ranker=ranker)
    return CandidateGenerator(P["processed_dir"], split, B["keys"], B["pre_k"], B["top_k"])


def true_codes(gen, true_pairs):
    """(s1_row << PAIR_SHIFT | cand) codes of the ground-truth pairs of this split."""
    from src.blocking.candidates import PAIR_SHIFT, encode
    s1 = pd.Index(gen.records.s1_ids())
    rows = s1.get_indexer(true_pairs["s1_id"])
    cand = np.full(len(true_pairs), -1, dtype=np.int64)
    for k in (2, 3):
        ids = pd.Index(gen.records.tables[k].column("entity_id").to_numpy(zero_copy_only=False))
        pos = ids.get_indexer(true_pairs["other_id"])
        m = pos >= 0
        cand[m] = encode(np.full(m.sum(), k), pos[m])
    ok = (rows >= 0) & (cand >= 0)
    return np.unique((rows[ok].astype(np.int64) << PAIR_SHIFT) | cand[ok])


def fit_stage_ranker(gen, rows, codes, s1_chunk=5000, rounds=200):
    """Train the v2 stage-2 ranker on ALL stage-1 pairs of the given (fit) S1 rows."""
    from src.blocking.candidates import PAIR_SHIFT
    from src.blocking.ranker import StageRanker
    parts = []
    for part in chunks(rows, s1_chunk):
        parts.append(gen.cheap_features(gen.key_candidates(part)))
    df = pd.concat(parts, ignore_index=True)
    y = np.isin((df["s1_row"].to_numpy() << PAIR_SHIFT) | df["cand"].to_numpy(), codes)
    print(f"  stage ranker: {len(df):,} stage-1 pairs of {len(rows):,} S1, {int(y.sum()):,} positive", flush=True)
    return StageRanker.fit(df, y, gen.keys, rounds)


def make_diff(P, F, split):
    """DiffFeaturizer for a split when features.diff is on (token ranks cached per split)."""
    if not (F or {}).get("diff"):
        return None
    import json
    import os
    from src.features.diff_features import DiffFeaturizer, token_rank
    path = os.path.join(P["processed_dir"], split, "token_rank.json")
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(token_rank(P["processed_dir"], split), f)
    with open(path) as f:
        # worker count is configurable: each worker holds the token-rank table (~90 MB)
        return DiffFeaturizer(json.load(f), workers=int(os.environ.get("DIFF_WORKERS", 0)) or None)


def diff_for_meta(gen, meta, diff, chunk=500_000):
    """Diff features for already generated pairs (meta: s1_row, cand) - used to extend a cache."""
    out = []
    for i in range(0, len(meta), chunk):
        m = meta.iloc[i:i + chunk]
        s1 = gen.records.take_s1(m["s1_row"].to_numpy(), S1_COLS)
        o = gen.records.take_cand(m["cand"].to_numpy(), O_COLS)
        out.append(diff(s1, o))
    return pd.concat(out, ignore_index=True)


def make_rarity(P, B, F, gen):
    if not (F or {}).get("rarity"):
        return None
    from src.features.rarity import Rarity
    return Rarity(P["processed_dir"], gen.split, B.get("index_name", "index_v2"), n=gen.records.n,
                  dup=bool(F.get("s1dup")))
