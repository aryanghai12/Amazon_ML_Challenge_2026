"""Record rarity features (pair_features v3).

For a key type (e.g. name_compact) the rarity of a record is the size of its
smallest block among the S2/S3 records of the split: an exact name shared by 1
record is strong evidence, one shared by 500 ('home services') is not. Computed
from the blocking indexes (split inputs only, no labels):

  exact compact name + empty other address:  P(match) 0.97 at 1 record, 0.012 above 50
"""
from __future__ import annotations

import os

import numpy as np
import pyarrow.parquet as pq

RARITY_KEYS = ["name_compact", "name_first", "name_skel_pair", "addr_pair"]
NONE = 1_000_000                       # record has no key of this type


class Rarity:
    def __init__(self, processed, split, index_name="index_v2", keys=RARITY_KEYS, n=None, dup=False):
        idx = os.path.join(processed, split, index_name)
        self.keys = list(keys)
        self.dup = dup
        self.s1, self.o, self.s1dup, self.odup = {}, {}, {}, {}
        for name in self.keys:
            t = pq.read_table(os.path.join(idx, f"{name}__others.parquet"))
            k = t.column("key").to_numpy()
            u, cnt = np.unique(k, return_counts=True)
            bs = np.repeat(cnt, cnt).astype(np.int32)
            arrs = {}
            for src in (2, 3):
                m = t.column("src").to_numpy() == src
                a = np.full(n[src], NONE, dtype=np.int32)
                np.minimum.at(a, t.column("row").to_numpy()[m], bs[m])
                arrs[src] = a
            self.o[name] = arrs
            s = pq.read_table(os.path.join(idx, f"{name}__s1.parquet"))
            sk, sr = s.column("key").to_numpy(), s.column("row").to_numpy()
            pos = np.minimum(np.searchsorted(u, sk), len(u) - 1)
            size = np.where(u[pos] == sk, cnt[pos], NONE).astype(np.int32)
            a = np.full(n[1], NONE, dtype=np.int32)
            np.minimum.at(a, sr, size)
            self.s1[name] = a
            if dup:
                # how many S1 (reference) records share the key: for the S1 itself, and for the
                # candidate = how many reference entities compete for that record
                order = np.argsort(sk, kind="stable")
                su, scnt = np.unique(sk, return_counts=True)
                d1 = np.full(n[1], NONE, dtype=np.int32)
                np.minimum.at(d1, sr[order], np.repeat(scnt, scnt).astype(np.int32))
                self.s1dup[name] = np.where(d1 == NONE, 0, d1)
                pos = np.minimum(np.searchsorted(su, k), len(su) - 1)
                dk = np.where(su[pos] == k, scnt[pos], NONE).astype(np.int32)
                arrs = {}
                for src in (2, 3):
                    m = t.column("src").to_numpy() == src
                    a = np.full(n[src], NONE, dtype=np.int32)
                    np.minimum.at(a, t.column("row").to_numpy()[m], dk[m])
                    arrs[src] = np.where(a == NONE, 0, a)
                self.odup[name] = arrs
            del t, s

    def features(self, s1_rows, cand) -> dict:
        from src.blocking.candidates import decode
        src, row = decode(cand)
        out = {}
        for name in self.keys:
            o = np.where(src == 2, self.o[name][2][np.minimum(row, len(self.o[name][2]) - 1)],
                         self.o[name][3][np.minimum(row, len(self.o[name][3]) - 1)])
            out[f"rar_{name}_o"] = np.log1p(o).astype(np.float32)
            out[f"rar_{name}_s1"] = np.log1p(self.s1[name][s1_rows]).astype(np.float32)
            if self.dup:
                od = np.where(src == 2, self.odup[name][2][np.minimum(row, len(self.odup[name][2]) - 1)],
                              self.odup[name][3][np.minimum(row, len(self.odup[name][3]) - 1)])
                out[f"s1dup_{name}_o"] = np.log1p(od).astype(np.float32)
                out[f"s1dup_{name}_s1"] = np.log1p(self.s1dup[name][s1_rows]).astype(np.float32)
        return out
