"""Competing-S1 features (pair_features v5).

50.7% of S1 records share their exact name with another S1 of the same country, and the
residual audit showed false merges where the record really belongs to a same-name S1 at a
neighbouring address (3317 -> 3319) while true copies with a corrupted number were missed.
The pair model only sees (our S1, record). Here, for candidate record o, we look at the
OTHER S1 records with exactly o's compact name (the reference entities competing for o)
and measure how well the best competitor's address fits o compared with ours:

  cmp_best_addr   best address token-set similarity of o to a competing S1 (-1: none)
  cmp_best_nums   best number overlap (jaccard) of o with a competing S1   (-1: none)
  cmp_margin      our address similarity - cmp_best_addr
  cmp_in_group    1 if our S1 itself has o's exact compact name

Inputs only (S1 records + blocking index), no labels.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist


def _nums_jacc(a, b):
    out = np.full(len(a), -1.0, dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        if x and y:
            A, B = set(x.split()), set(y.split())
            out[i] = len(A & B) / len(A | B)
        elif x or y:
            out[i] = 0.0
    return out


class Competition:
    def __init__(self, processed, split, records, index_name="index_v2", cap=64):
        idx = os.path.join(processed, split, index_name)
        s = pq.read_table(os.path.join(idx, "name_compact__s1.parquet"))
        sk, sr = s.column("key").to_numpy(), s.column("row").to_numpy().astype(np.int64)
        order = np.argsort(sk, kind="stable")
        self.sk, self.sr = sk[order], sr[order]
        o = pq.read_table(os.path.join(idx, "name_compact__others.parquet"))
        ok, osrc, orow = o.column("key").to_numpy(), o.column("src").to_numpy(), o.column("row").to_numpy()
        self.okey = {}
        for k in (2, 3):
            a = np.zeros(records.n[k], dtype=np.uint64)
            m = osrc == k
            a[orow[m]] = ok[m]
            self.okey[k] = a
        self.rec, self.cap = records, cap
        self.s1_addr = np.asarray(records.tables[1].column("addr_core").to_pylist(), dtype=object)
        self.s1_nums = np.asarray(records.tables[1].column("addr_nums").to_pylist(), dtype=object)

    def features(self, s1_rows, cand):
        from src.blocking.candidates import decode
        s1_rows = np.asarray(s1_rows, dtype=np.int64)
        cand = np.asarray(cand, dtype=np.int64)
        uc, inv = np.unique(cand, return_inverse=True)
        src, row = decode(uc)
        key = np.where(src == 2, self.okey[2][np.minimum(row, len(self.okey[2]) - 1)],
                       self.okey[3][np.minimum(row, len(self.okey[3]) - 1)])
        lo = np.searchsorted(self.sk, key, "left")
        n = np.minimum(np.searchsorted(self.sk, key, "right") - lo, self.cap)
        n[key == 0] = 0
        tot = int(n.sum())
        ci = np.repeat(np.arange(len(uc)), n)
        pos = np.repeat(lo - np.concatenate(([0], np.cumsum(n)[:-1])), n) + np.arange(tot)
        grp_s1 = self.sr[pos] if tot else np.empty(0, np.int64)
        o = self.rec.take_cand(uc, ["addr_core", "addr_nums"])
        a_sim = (cpdist(self.s1_addr[grp_s1].tolist(), o["addr_core"][ci].tolist(), scorer=fuzz.token_set_ratio,
                        workers=-1, dtype=np.float32) / 100.0) if tot else np.empty(0, np.float32)
        n_sim = _nums_jacc(self.s1_nums[grp_s1], o["addr_nums"][ci]) if tot else np.empty(0, np.float32)
        g = pd.DataFrame({"ci": ci, "s1": grp_s1, "a": a_sim, "n": n_sim})
        # best and second best competitor per record (by address, then numbers)
        g["sc"] = g["a"] + 0.01 * g["n"]
        g = g.sort_values(["ci", "sc"], ascending=[True, False], kind="stable")
        first = g.groupby("ci").nth(0).set_index("ci")
        second = g.groupby("ci").nth(1).set_index("ci")
        pair_ci = inv
        b1 = first.reindex(pair_ci)
        b2 = second.reindex(pair_ci)
        own = b1["s1"].to_numpy() == s1_rows
        best_a = np.where(own, b2["a"].to_numpy(), b1["a"].to_numpy())
        best_n = np.where(own, b2["n"].to_numpy(), b1["n"].to_numpy())
        best_a = np.nan_to_num(best_a, nan=-1.0).astype(np.float32)
        best_n = np.nan_to_num(best_n, nan=-1.0).astype(np.float32)
        # our own address similarity to o (for the margin)
        own_a = cpdist(self.s1_addr[s1_rows].tolist(), o["addr_core"][inv].tolist(), scorer=fuzz.token_set_ratio,
                       workers=-1, dtype=np.float32) / 100.0
        in_grp = pd.Series(1, index=pd.MultiIndex.from_arrays([g["ci"].to_numpy(), g["s1"].to_numpy()]))
        in_g = in_grp.reindex(pd.MultiIndex.from_arrays([inv, s1_rows])).notna().to_numpy()
        return {"cmp_best_addr": best_a, "cmp_best_nums": best_n,
                "cmp_margin": np.where(best_a >= 0, own_a - best_a, 1.0).astype(np.float32),
                "cmp_in_group": in_g.astype(np.int8)}


class CompetitionV2:
    """Competition over a key type that may give a record SEVERAL keys (e.g. addr_pair), comparing
    both the address and the full name + legal form of the record with every competing S1.

    Features (prefix p_):  {p}_n       competing S1 records in the group (excluding ours)
                           {p}_addr    best competitor address token-set similarity (-1: none)
                           {p}_name    best competitor name+legal similarity        (-1: none)
                           {p}_amargin our address similarity - {p}_addr
                           {p}_nmargin our name+legal similarity - {p}_name
    """

    def __init__(self, processed, split, records, key="name_compact", prefix="cmpn", index_name="index_v2",
                 cap=48):
        idx = os.path.join(processed, split, index_name)
        s = pq.read_table(os.path.join(idx, f"{key}__s1.parquet"))
        sk, sr = s.column("key").to_numpy(), s.column("row").to_numpy().astype(np.int64)
        order = np.argsort(sk, kind="stable")
        self.sk, self.sr = sk[order], sr[order]
        o = pq.read_table(os.path.join(idx, f"{key}__others.parquet"))
        enc = ((o.column("src").to_numpy().astype(np.int64) - 2) << 23) | o.column("row").to_numpy().astype(np.int64)
        order = np.argsort(enc, kind="stable")
        self.ok_enc, self.ok_key = enc[order], o.column("key").to_numpy()[order]
        self.rec, self.cap, self.prefix = records, cap, prefix
        t1 = records.tables[1]
        self.s1_addr = np.asarray(t1.column("addr_core").to_pylist(), dtype=object)
        self.s1_name = np.asarray([f"{a} {b}".strip() for a, b in zip(t1.column("name_core").to_pylist(),
                                                                     t1.column("name_legal").to_pylist())],
                                  dtype=object)

    def features(self, s1_rows, cand):
        s1_rows = np.asarray(s1_rows, dtype=np.int64)
        uc, inv = np.unique(np.asarray(cand, dtype=np.int64), return_inverse=True)
        # keys of every unique record
        lo = np.searchsorted(self.ok_enc, uc, "left")
        n = np.searchsorted(self.ok_enc, uc, "right") - lo
        rec_i = np.repeat(np.arange(len(uc)), n)
        keys = self.ok_key[np.repeat(lo - np.concatenate(([0], np.cumsum(n)[:-1])), n) + np.arange(int(n.sum()))]
        # S1 rows sharing each key (capped per key), deduplicated per record
        klo = np.searchsorted(self.sk, keys, "left")
        kn = np.minimum(np.searchsorted(self.sk, keys, "right") - klo, self.cap)
        tot = int(kn.sum())
        ci = np.repeat(rec_i, kn)
        grp = self.sr[np.repeat(klo - np.concatenate(([0], np.cumsum(kn)[:-1])), kn) + np.arange(tot)] if tot \
            else np.empty(0, np.int64)
        g = pd.DataFrame({"ci": ci, "s1": grp}).drop_duplicates()
        o = self.rec.take_cand(uc, ["addr_core", "name_core", "name_legal"])
        o_name = np.asarray([f"{a} {b}".strip() for a, b in zip(o["name_core"], o["name_legal"])], dtype=object)
        cis, s1s = g["ci"].to_numpy(), g["s1"].to_numpy()

        def sim(a, b):
            if len(a) == 0:
                return np.empty(0, np.float32)
            return cpdist(list(a), list(b), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100.0
        g["a"] = sim(self.s1_addr[s1s], o["addr_core"][cis])
        g["nm"] = sim(self.s1_name[s1s], o_name[cis])
        # exclude OUR S1 from its own competitor set: best and second best per record
        g["sc"] = g["a"] + g["nm"]
        g = g.sort_values(["ci", "sc"], ascending=[True, False], kind="stable")
        cnt = g.groupby("ci").size().reindex(np.arange(len(uc)), fill_value=0).to_numpy()
        first = g.groupby("ci").nth(0).set_index("ci").reindex(inv)
        second = g.groupby("ci").nth(1).set_index("ci").reindex(inv)
        own = first["s1"].to_numpy() == s1_rows
        pick = lambda c: np.nan_to_num(np.where(own, second[c].to_numpy(), first[c].to_numpy()),  # noqa: E731
                                       nan=-1.0).astype(np.float32)
        best_a, best_n = pick("a"), pick("nm")
        own_a = sim(self.s1_addr[s1_rows], o["addr_core"][inv])
        own_n = sim(self.s1_name[s1_rows], o_name[inv])
        in_grp = pd.Series(1, index=pd.MultiIndex.from_arrays([g["ci"].to_numpy(), g["s1"].to_numpy()]))
        mine = in_grp.reindex(pd.MultiIndex.from_arrays([inv, s1_rows])).notna().to_numpy()
        p = self.prefix
        return {f"{p}_n": np.log1p(cnt[inv] - mine).astype(np.float32), f"{p}_addr": best_a, f"{p}_name": best_n,
                f"{p}_amargin": np.where(best_a >= 0, own_a - best_a, 1.0).astype(np.float32),
                f"{p}_nmargin": np.where(best_n >= 0, own_n - best_n, 1.0).astype(np.float32)}
