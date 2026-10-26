"""
tests/test_generator.py - invariants of src/synth/generate.py.

These encode the HARD RULES of the project: the generator must never emit
real-looking contact data, never invent tactic tags outside the taxonomy,
and always record the template_family the splitter depends on.
"""
from __future__ import annotations

import re

import pandas as pd
import pytest

from src.config import CHANNELS, SCRIPT_STYLES, TACTIC_TAGS, VERDICTS
from src.synth.generate import build_synthetic_frame

# A run of 7+ digits is almost certainly a real phone/acct number.
LONG_DIGIT_RUN = re.compile(r"\d{7,}")


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    """One small corpus, built once for the whole module (fast)."""
    return build_synthetic_frame(n_scam=80, n_suspicious=25, n_safe=80, seed=42)


def test_no_long_digit_runs(corpus):
    bad = [t for t in corpus["text"] if LONG_DIGIT_RUN.search(str(t))]
    assert not bad, f"real-looking digit runs: {bad[:3]}"


def test_no_real_urls_or_phone_numbers(corpus):
    # Only placeholders are allowed; a real link/number would be a privacy
    # problem AND a shortcut for the model.
    bad = [t for t in corpus["text"]
           if re.search(r"https?://|www\.\w|\b\d{10}\b", str(t))]
    assert not bad, f"real-looking URLs/phones: {bad[:3]}"


def test_all_enums_valid(corpus):
    assert set(corpus["verdict"]) <= set(VERDICTS)
    assert set(corpus["channel"]) <= set(CHANNELS)
    assert set(corpus["script_style"]) <= set(SCRIPT_STYLES)


def test_tactics_subset_of_taxonomy(corpus):
    for tags in corpus["tactics"]:
        assert set(tags) <= set(TACTIC_TAGS), f"unknown tags: {set(tags) - set(TACTIC_TAGS)}"


def test_every_scam_has_at_least_one_tactic(corpus):
    scams = corpus[corpus["verdict"] == "scam"]
    empty = scams[[not t for t in scams["tactics"]]]
    assert empty.empty, f"scam rows without tactics: {empty['id'].tolist()[:5]}"


def test_every_row_has_a_template_family(corpus):
    blank = corpus[corpus["template_family"].astype(str).str.strip() == ""]
    assert blank.empty, "rows without template_family break the splitter"


def test_all_thirteen_tactics_covered():
    big = build_synthetic_frame(n_scam=600, n_suspicious=0, n_safe=0, seed=42)
    covered = {t for tags in big["tactics"] for t in tags}
    missing = set(TACTIC_TAGS) - covered
    assert not missing, f"tactics with no example: {missing}"


def test_rows_are_marked_synthetic(corpus):
    assert (corpus["source"] == "synthetic").all()
    assert (corpus["labeler"] == "synth_gen_v1").all()


def test_hard_negative_share(corpus):
    safe = corpus[corpus["verdict"] == "safe"]
    hard = safe[safe["notes"].astype(str).str.contains("hard_negative")]
    assert len(hard) / len(safe) >= 0.50, "safe rows must be >=50% hard negatives"


def test_seeded_generation_is_reproducible():
    a = build_synthetic_frame(30, 10, 30, seed=42)
    b = build_synthetic_frame(30, 10, 30, seed=42)
    assert a["text"].tolist() == b["text"].tolist()
