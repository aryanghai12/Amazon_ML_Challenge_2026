"""Print raw examples of error categories (for manual inspection during the audit).

    python -m scripts.audit_examples missed|fp|nokey [n]
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scripts.audit_blocking import OUT

COLS = ["entity_id", "business_name", "business_address", "name_core", "addr_core", "addr_nums"]


def lookup(ids):
    out = []
    for k in (1, 2, 3):
        t = pq.read_table(os.path.join("data", "processed", "train", f"source{k}.parquet"), columns=COLS)
        t = t.filter(pc.is_in(t.column("entity_id"), value_set=__import__("pyarrow").array(list(ids))))
        out.append(t.to_pandas())
    return pd.concat(out).set_index("entity_id")


def main():
    kind = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    if kind == "missed":
        df = pd.read_parquet(os.path.join(OUT, "missed_in_cand_baseline.parquet")).sample(n, random_state=1)
    elif kind == "fp":
        df = pd.read_parquet(os.path.join(OUT, "fp_baseline.parquet")).sample(n, random_state=1)
    elif kind == "nokey":
        tr = pd.read_parquet(os.path.join(OUT, "blocking_trace.parquet"))
        df = tr[~tr.stage1].sample(n, random_state=1)
    else:
        tr = pd.read_parquet(os.path.join(OUT, "blocking_trace.parquet"))
        df = tr[tr.stage1 & ~tr.stage2].sample(n, random_state=1)
    rec = lookup(set(df.s1_id) | set(df.other_id))
    pd.set_option("display.width", 250)
    for r in df.itertuples():
        a, b = rec.loc[r.s1_id], rec.loc[r.other_id]
        p = f"p={r.p:.3f}" if hasattr(r, "p") else ""
        print(f"--- {r.s1_id} vs {r.other_id} {p} {getattr(r, 'reason', '')}")
        print(f"  S1: {a.business_name!r:60} | {a.business_address!r}")
        print(f"  S{r.other_id[1]}: {b.business_name!r:60} | {b.business_address!r}")


if __name__ == "__main__":
    main()
