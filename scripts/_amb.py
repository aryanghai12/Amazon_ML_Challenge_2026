"""Shared helper: flag indistinguishable records (empty address, owner's raw name shared by >=2 S1)."""
import pandas as pd, pyarrow.parquet as pq
from src.train import true_pairs_from_gt

def ambiguous_pairs():
    tp = true_pairs_from_gt("data/processed")
    s1 = pq.read_table("data/processed/train/source1.parquet", columns=["entity_id", "business_name", "country"]).to_pandas()
    s1["nm"] = s1.country + "#" + s1.business_name.str.lower().str.strip()
    s1["k"] = s1.groupby("nm").entity_id.transform("size")
    k = s1.set_index("entity_id").k
    empty = set()
    for src in (2, 3):
        t = pq.read_table(f"data/processed/train/source{src}.parquet", columns=["entity_id", "addr_missing"]).to_pandas()
        empty |= set(t.entity_id[t.addr_missing == 1])
    tp["amb"] = tp.other_id.isin(empty) & (k.reindex(tp.s1_id).to_numpy() >= 2)
    return tp
