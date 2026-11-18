"""Tests for the v2 components: diff features, expected-F decision, stage-ranker features."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.blocking.ranker import ranker_features  # noqa: E402
from src.features import diff_features as dfeat  # noqa: E402
from src.matching.expected_f import decide_expected_f  # noqa: E402


def one(n1, n2, a1="", a2="", u1="", u2="", l1="", l2=""):
    return dict(zip(dfeat.DIFF_NAMES, dfeat._one(n1, n2, a1, a2, u1, u2, l1, l2)))


def test_number_truncation_vs_change():
    # true-match noise truncates the house number, a distractor changes it
    assert one("x", "x", u1="136", u2="13")["nu_trunc"] == 1
    assert one("x", "x", u1="3317", u2="3319")["nu_edit1"] == 1
    changed = one("x", "x", u1="565", u2="570")
    assert changed["nu_unm_o"] == 1 and changed["nu_mindiff"] == pytest.approx(np.log1p(5))


def test_name_typo_vs_substitution():
    typo = one("surat communicationsprivate", "surat communicatiesprivate")
    assert typo["nm_fuzzy"] == 1 and typo["nm_sub_o"] == 0
    sub = one("go security", "go services")
    assert sub["nm_sub_s1"] == 1 and sub["nm_sub_o"] == 1


def test_public_and_legal_flags():
    f = one("xm ethanol", "xm ethanol public", l1="ltd pvt", l2="ltd")
    assert f["pub_o"] == 1 and f["pub_s1"] == 0
    assert f["lg_s1_pvt"] == 1 and f["lg_o_pvt"] == 0


def test_expected_f_keeps_confident_and_drops_doubtful():
    scored = pd.DataFrame({"s1_id": ["A", "A", "A", "B"],
                           "other_id": ["S2-1", "S3-1", "S2-9", "S2-2"],
                           "p": [0.99, 0.95, 0.05, 0.03]})
    kept = decide_expected_f(scored)
    assert set(kept.other_id) == {"S2-1", "S3-1"}          # B predicted singleton, S2-9 dropped


def test_expected_f_one_to_one():
    scored = pd.DataFrame({"s1_id": ["A", "B"], "other_id": ["S2-1", "S2-1"], "p": [0.9, 0.8]})
    kept = decide_expected_f(scored)
    assert kept.s1_id.tolist() == ["A"]


def test_ranker_features_shape_and_bits():
    df = pd.DataFrame({"s1_row": [0, 0, 1], "cand": [5, 1 << 23, 7], "n_keys": [2, 1, 1],
                       "idf": [0.5, 0.2, 0.1], "kbits": [0b101, 0b010, 0b001],
                       "name_s": [0.9, 0.2, 0.5], "addr_s": [0.8, 0.1, 0.4]})
    X = ranker_features(df, 3)
    assert X[["k0", "k1", "k2"]].to_numpy().tolist() == [[1, 0, 1], [0, 1, 0], [1, 0, 0]]
    assert X.src3.tolist() == [0, 1, 0]
    assert X.name_s_gap.tolist() == pytest.approx([0.0, 0.7, 0.0])
