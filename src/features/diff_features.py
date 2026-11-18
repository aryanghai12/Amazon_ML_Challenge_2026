"""Token-level difference features (pair_features v2).

The audit of the baseline errors showed that distractor records are near-copies
of a real business with ONE deliberate change, while true-match noise is a
corruption of the same content:

  distractor                                  true-match noise
  house number changed (3317 -> 3319)         house number truncated (136 -> 13, 302 -> 02)
  one name word substituted (Papers->Overseas) one name word misspelt (Private -> Presaive)
  legal form swapped (Private -> Public, +LLP) legal form dropped / abbreviated

Set-similarity scores (token_set_ratio, Jaccard) cannot tell these apart, so for
every pair we align the tokens and count how the leftovers relate:
  exact | fuzzy (typo: Levenshtein sim >= 0.7, same skeleton, or prefix) | substituted
and for numbers: exact | truncated (prefix / suffix) | one substitution | unmatched.
`common` is the frequency rank of the substituted words (generic 'services' vs rare).
"""
from __future__ import annotations

import math
import os
from collections import Counter
from multiprocessing import get_context

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from src.preprocessing.normalize import skeleton

LEGAL_FLAGS = ["pvt", "ltd", "llc", "inc", "llp", "corp", "co", "pc", "lp", "pllc"]
DIFF_NAMES = (
    ["nm_exact", "nm_fuzzy", "nm_sub_s1", "nm_sub_o", "nm_sub_rank_s1", "nm_sub_rank_o", "nm_order_same",
     "ad_exact", "ad_fuzzy", "ad_sub_s1", "ad_sub_o", "ad_sub_rank_s1", "ad_sub_rank_o",
     "nu_exact", "nu_trunc", "nu_edit1", "nu_unm_s1", "nu_unm_o", "nu_mindiff", "nu_first_trunc",
     "pub_s1", "pub_o"]
    + [f"lg_s1_{x}" for x in LEGAL_FLAGS] + [f"lg_o_{x}" for x in LEGAL_FLAGS])

_RANK: dict = {}
_RANK_MISS = 50_000


def _fuzzy(a, b):
    if Levenshtein.normalized_similarity(a, b) >= 0.7:
        return True
    if len(a) >= 3 and len(b) >= 3 and (a.startswith(b) or b.startswith(a)):
        return True
    return len(a) >= 3 and skeleton(a) == skeleton(b)


def _align(ta, tb):
    """-> exact, fuzzy, sub_a list, sub_b list"""
    ca, cb = Counter(ta), Counter(tb)
    exact = sum((ca & cb).values())
    ra = list((ca - cb).elements())
    rb = list((cb - ca).elements())
    fz = 0
    for x in list(ra):
        for y in rb:
            if _fuzzy(x, y):
                fz += 1
                ra.remove(x)
                rb.remove(y)
                break
    return exact, fz, ra, rb


def _rank(tokens):
    return math.log1p(min((_RANK.get(t, _RANK_MISS) for t in tokens), default=_RANK_MISS * 2))


def _nums(a, b):
    A, B = a.split(), b.split()
    sa, sb = set(A), set(B)
    exact = len(sa & sb)
    ua, ub = [x for x in A if x not in sb], [x for x in B if x not in sa]
    trunc = edit1 = 0
    for y in list(ub):
        for x in ua:
            if len(x) != len(y) and (x.startswith(y) or x.endswith(y) or y.startswith(x) or y.endswith(x)):
                trunc += 1
                ua.remove(x)
                ub.remove(y)
                break
    for y in list(ub):
        for x in ua:
            if len(x) == len(y) and Levenshtein.distance(x, y) == 1:
                edit1 += 1
                ua.remove(x)
                ub.remove(y)
                break
    md = min((abs(int(x) - int(y)) for x in ua for y in ub if len(x) <= 7 and len(y) <= 7), default=-1)
    ft = 0
    if A and B and A[0] != B[0]:
        x, y = A[-1], B[-1]  # largest (usually the house number: sorted by length then value)
        ft = int(x.startswith(y) or x.endswith(y) or y.startswith(x) or y.endswith(x))
    return exact, trunc, edit1, len(ua), len(ub), (math.log1p(md) if md >= 0 else -1.0), ft


def _one(n1, n2, a1, a2, u1, u2, l1, l2):
    t1, t2 = n1.split(), n2.split()
    ne, nf, ns1, ns2 = _align(t1, t2)
    common = [t for t in t1 if t in set(t2)]
    order_same = int(common == [t for t in t2 if t in set(t1)]) if common else -1
    ae, af, as1, as2 = _align(a1.split(), a2.split())
    lg1, lg2 = set(l1.split()), set(l2.split())
    return ([ne, nf, len(ns1), len(ns2), _rank(ns1) if ns1 else -1.0, _rank(ns2) if ns2 else -1.0, order_same,
             ae, af, len(as1), len(as2), _rank(as1) if as1 else -1.0, _rank(as2) if as2 else -1.0]
            + list(_nums(u1, u2)) + [int("public" in t1), int("public" in t2)]
            + [int(x in lg1) for x in LEGAL_FLAGS] + [int(x in lg2) for x in LEGAL_FLAGS])


def _init(rank):
    global _RANK
    _RANK = rank


def _work(args):
    return [_one(*r) for r in zip(*args)]


def token_rank(processed, split, top=_RANK_MISS) -> dict:
    """Frequency rank of name + address tokens over this split's S2/S3 (inputs only, no labels)."""
    import pyarrow.parquet as pq
    c = Counter()
    for k in (2, 3):
        t = pq.read_table(os.path.join(processed, split, f"source{k}.parquet"), columns=["name_core", "addr_core"])
        for col in ("name_core", "addr_core"):
            for s in t.column(col).to_pylist():
                c.update(s.split())
    return {w: i for i, (w, _) in enumerate(c.most_common(top))}


class DiffFeaturizer:
    def __init__(self, rank: dict, workers=None):
        self.rank = rank
        self.workers = workers or max(1, (os.cpu_count() or 2) - 1)
        self.pool = None

    def __call__(self, s1: dict, o: dict) -> pd.DataFrame:
        cols = [s1["name_core"], o["name_core"], s1["addr_core"], o["addr_core"], s1["addr_nums"], o["addr_nums"],
                s1["name_legal"], o["name_legal"]]
        cols = [list(c) for c in cols]
        n = len(cols[0])
        if n == 0:
            return pd.DataFrame(columns=DIFF_NAMES)
        if self.pool is None:
            self.pool = get_context("spawn").Pool(self.workers, initializer=_init, initargs=(self.rank,))
        step = max(2000, -(-n // (self.workers * 4)))
        parts = self.pool.map(_work, [[c[i:i + step] for c in cols] for i in range(0, n, step)])
        arr = np.asarray([r for p in parts for r in p], dtype=np.float32)
        return pd.DataFrame(arr, columns=DIFF_NAMES)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool = None
