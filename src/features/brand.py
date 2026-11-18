"""Pseudo-brand likelihood (pair_features v4).

When a record's name has been replaced, the link to its S1 is the address alone. The
residual audit (E5) showed that true copies carry synthetic pseudo-brands built from a
small syllable inventory ('Kelozetacalo', 'Tavoflux', 'Umbracalohalo') while the other
names at the same address are real businesses in the same building ('Downtown Tattoo
Corp'). This scorer is a character-trigram log-likelihood ratio between

  brand model      : names of FIT-S1 true matches whose name shares nothing with the S1
  background model : a sample of all S2/S3 names of the split

learned from the training split only; score = mean LLR per trigram of name_compact.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter

import numpy as np

N = 3


def _grams(s):
    s = "^" + "".join(ch for ch in s if ch.isalpha()) + "$"
    return [s[i:i + N] for i in range(len(s) - N + 1)]


class BrandScorer:
    def __init__(self, llr=None, default=0.0):
        self.llr, self.default = llr or {}, default

    @classmethod
    def fit(cls, brand_names, background_names, min_count=3):
        b, g = Counter(), Counter()
        for s in brand_names:
            b.update(_grams(s))
        for s in background_names:
            g.update(_grams(s))
        nb, ng = sum(b.values()), sum(g.values())
        vocab = {k for k in set(b) | set(g) if b[k] + g[k] >= min_count}
        V = len(vocab) + 1
        llr = {k: math.log((b[k] + 1) / (nb + V)) - math.log((g[k] + 1) / (ng + V)) for k in vocab}
        return cls(llr, default=math.log(1 / (nb + V)) - math.log(1 / (ng + V)))

    def score(self, s):
        gs = _grams(s)
        if len(gs) < 2:
            return 0.0
        return sum(self.llr.get(x, self.default) for x in gs) / len(gs)

    def score_many(self, names):
        return np.fromiter((self.score(s) for s in names), dtype=np.float32, count=len(names))

    def save(self, directory):
        with open(os.path.join(directory, "brand_llr.json"), "w") as f:
            json.dump({"llr": self.llr, "default": self.default}, f)

    @classmethod
    def load(cls, directory):
        with open(os.path.join(directory, "brand_llr.json")) as f:
            d = json.load(f)
        return cls(d["llr"], d["default"])


class BrandFeatures:
    """Per-record brand score for a split (computed once), looked up per pair."""

    def __init__(self, scorer: BrandScorer, records):
        self.s1 = scorer.score_many(records.tables[1].column("name_compact").to_pylist())
        self.o = {k: scorer.score_many(records.tables[k].column("name_compact").to_pylist()) for k in (2, 3)}

    def features(self, s1_rows, cand):
        from src.blocking.candidates import decode
        src, row = decode(cand)
        o = np.where(src == 2, self.o[2][np.minimum(row, len(self.o[2]) - 1)],
                     self.o[3][np.minimum(row, len(self.o[3]) - 1)])
        return {"brand_o": o, "brand_s1": self.s1[s1_rows]}
