"""
tests/test_phase4.py - the three promises Phase 4 makes, as tests.

  1. NOTHING IS FITTED ON TEST. Temperature and thresholds are validation-only,
     and a test split reaching the fitter raises instead of quietly fitting.
  2. PROBES NEVER ENTER TRAINING. Every probe row carries source=probe_* and no
     probe row may coincide with a TRAIN row.
  3. THE BAND BEHAVES. t_high / t_low produce scam / suspicious / safe exactly
     as documented, and an empty band says so instead of inventing coverage.

Plus the mechanics: attacks keep placeholders, the Hinglish rewrite keeps
placeholders and digits, and equal-cell weighting really is equal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import calibrate
from src.phase4_common import (
    apply_band,
    apply_temperature,
    assert_val_only,
    band_summary,
    claim_flag,
    ece,
    equal_cell_weights,
    fit_temperature,
    fit_thresholds,
    logit,
    sigmoid,
    too_small,
)
from src.robustness import (
    assert_probe_isolation,
    attack_all_caps,
    attack_emoji,
    attack_spaces_inside_words,
    attack_typos,
    attack_url_obfuscation,
    build_safe_chat_probe,
    make_substituter,
    substitution_pairs,
)

SEED = 42


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def val_frame(n_syn_scam=200, n_syn_safe=200, n_real_scam=60, n_real_safe=60) -> pd.DataFrame:
    """A fake validation split with a controllable cell mix."""
    rows = []
    for i in range(n_syn_scam):
        rows.append({"split": "val", "verdict": "scam", "source": "synthetic",
                     "prob_raw": 0.90, "text": f"syn scam {i}"})
    for i in range(n_syn_safe):
        rows.append({"split": "val", "verdict": "safe", "source": "synthetic",
                     "prob_raw": 0.05, "text": f"syn safe {i}"})
    for i in range(n_real_scam):
        rows.append({"split": "val", "verdict": "scam", "source": "external_reported",
                     "prob_raw": 0.97, "text": f"real scam {i}"})
    for i in range(n_real_safe):
        rows.append({"split": "val", "verdict": "safe", "source": "public_ham_eval",
                     "prob_raw": 0.12, "text": f"real safe {i}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. validation-only fitting
# ---------------------------------------------------------------------------
def test_assert_val_only_accepts_val():
    assert_val_only(val_frame(), "anything")          # must not raise


@pytest.mark.parametrize("split", ["test_a", "test_b", "test_real",
                                   "suspicious_holdout", "probe", "train"])
def test_assert_val_only_rejects_everything_else(split):
    frame = pd.DataFrame({"split": [split, "val"]})
    with pytest.raises(AssertionError, match="validation-only"):
        assert_val_only(frame, "temperature")


def test_calibration_refuses_to_fit_on_a_test_split(monkeypatch):
    """If a test frame ever reached the fitter, it must raise - not fit."""
    test_rows = pd.DataFrame({
        "split": ["test_b"] * 4,
        "verdict": ["scam", "safe", "scam", "safe"],
        "source": ["synthetic"] * 4,
        "prob_raw": [0.9, 0.1, 0.8, 0.2],
        "text": ["a", "b", "c", "d"],
    })
    monkeypatch.setattr(calibrate, "load_predictor", lambda name: object())
    monkeypatch.setattr(calibrate, "predictions_for_split",
                        lambda predictor, name, split: test_rows)
    with pytest.raises(AssertionError, match="validation"):
        calibrate.fit_on_validation("fake_model")


def test_coverage_sweep_is_also_validation_only(monkeypatch):
    monkeypatch.setattr(calibrate, "load_predictor", lambda name: object())
    monkeypatch.setattr(calibrate, "predictions_for_split",
                        lambda predictor, name, split: pd.DataFrame({
                            "split": ["test_real"], "verdict": ["scam"],
                            "source": ["external_reported"], "prob_raw": [0.9],
                            "text": ["x"]}))
    with pytest.raises(AssertionError, match="validation"):
        calibrate.coverage_sweep("fake_model", {"temperature": 1.0})


def test_fit_on_validation_uses_the_reweighted_cells():
    """The reported cell counts must be the four (verdict x real) cells."""
    frame = val_frame()
    weights, counts = equal_cell_weights(frame)
    assert counts == {"synthetic/scam": 200, "synthetic/safe": 200,
                      "real/scam": 60, "real/safe": 60}
    # Every cell carries the same TOTAL weight, and the four sum to 1.
    is_real = ~frame["source"].isin(["synthetic", "synthetic_llm"])
    is_scam = frame["verdict"] == "scam"
    for real in (True, False):
        for scam in (True, False):
            mask = (is_real == real) & (is_scam == scam)
            assert weights[mask].sum() == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 2. probes never enter training
# ---------------------------------------------------------------------------
def test_probe_isolation_rejects_a_train_text(monkeypatch):
    train_text = "this message is in the training split"
    monkeypatch.setattr("src.robustness.load_messages", lambda: pd.DataFrame({
        "id": ["t1"], "text": [train_text], "split": ["train"]}))
    probes = pd.DataFrame({
        "id": ["probe_1"],
        "text": [train_text],
        "source": ["probe_safe_chat"],
        "split": ["probe"],
    })
    with pytest.raises(AssertionError, match="never be training data"):
        assert_probe_isolation(probes)


def test_probe_isolation_rejects_a_non_probe_source(monkeypatch):
    monkeypatch.setattr("src.robustness.load_messages",
                        lambda: pd.DataFrame({"id": [], "text": [], "split": []}))
    probes = pd.DataFrame({"id": ["x"], "text": ["hello"],
                           "source": ["synthetic"], "split": ["probe"]})
    with pytest.raises(AssertionError, match="probe_"):
        assert_probe_isolation(probes)


def test_probe_isolation_allows_a_test_row_copy(monkeypatch):
    """An attack that changes nothing may reproduce a TEST message."""
    monkeypatch.setattr("src.robustness.load_messages", lambda: pd.DataFrame({
        "id": ["test_1"], "text": ["held out text"], "split": ["test_b"]}))
    probes = pd.DataFrame({"id": ["probe_1"], "text": ["held out text"],
                           "source": ["probe_attack_emoji"], "split": ["probe"]})
    assert_probe_isolation(probes)                     # must not raise


def test_safe_chat_probe_is_fresh_and_balanced():
    rng = np.random.default_rng(SEED)
    probe = build_safe_chat_probe(40, rng)
    assert len(probe) == 40
    assert probe["text"].nunique() == 40               # no duplicates
    assert set(probe["source"]) == {"probe_safe_chat"}
    assert set(probe["verdict"]) == {"safe"}
    styles = set(probe["script_style"])
    assert {"english", "roman_hinglish"} <= styles     # both styles present
    assert probe["tactics"].map(lambda t: isinstance(t, list)).all()


# ---------------------------------------------------------------------------
# 3. calibration + band maths
# ---------------------------------------------------------------------------
def test_temperature_one_is_identity():
    p = np.array([0.01, 0.2, 0.5, 0.9])
    assert np.allclose(apply_temperature(p, 1.0), p)


def test_logit_and_sigmoid_roundtrip():
    p = np.array([0.05, 0.4, 0.75])
    assert np.allclose(sigmoid(logit(p)), p)


def test_fit_temperature_recovers_scale_direction():
    # Over-confident scores (too close to 0/1) need T > 1 to soften them.
    rng = np.random.default_rng(SEED)
    y = rng.integers(0, 2, size=500).astype(float)
    p = np.where(y == 1, 0.98, 0.02)
    p = np.where(rng.random(500) < 0.1, 1 - p, p)      # 10% label noise
    temp = fit_temperature(y, p)
    assert temp > 1.0


def test_fit_thresholds_respects_the_precision_target():
    rng = np.random.default_rng(SEED)
    y = np.r_[np.ones(400), np.zeros(400)]
    p = np.r_[rng.uniform(0.5, 1.0, 400), rng.uniform(0.0, 0.5, 400)]
    out = fit_thresholds(y, p, None, 0.95, 0.95)
    assert out["precision_at_t_high"] >= 0.95
    assert out["npv_at_t_low"] >= 0.95
    # Every threshold must come with its evidence count.
    assert out["n_safe_behind_t_low"] > 0
    assert out["n_scam_behind_t_high"] > 0


def test_empty_band_is_reported_not_invented():
    """Bimodal scores: one cut satisfies both targets, so say so."""
    y = np.r_[np.ones(200), np.zeros(200)]
    p = np.r_[np.full(200, 0.99), np.full(200, 0.01)]
    out = fit_thresholds(y, p, None, 0.95, 0.95)
    assert out["n_rows_in_band"] == 0
    assert out["abstention_available"] is False
    assert "no validation rows" in out["band_reason"]
    assert out["t_low"] == out["t_high"]


def test_band_assigns_three_verdicts():
    p = np.array([0.99, 0.60, 0.05])
    labels = apply_band(p, t_high=0.9, t_low=0.1)
    assert list(labels) == ["scam", "suspicious", "safe"]


def test_a_row_exactly_on_a_shared_cut_is_scam():
    """With t_high == t_low the tie must go to the conservative verdict."""
    labels = apply_band(np.array([0.346]), t_high=0.346, t_low=0.346)
    assert labels[0] == "scam"


def test_band_summary_counts_coverage():
    p = np.array([0.99, 0.60, 0.05, 0.95])
    truth = np.array(["scam", "scam", "safe", "scam"], dtype=object)
    summary = band_summary(pd.DataFrame({"text": ["a"] * 4}), p, 0.9, 0.1, truth)
    assert summary["n_scam"] == 2
    assert summary["n_safe"] == 1
    assert summary["n_suspicious"] == 1
    assert summary["coverage"] == pytest.approx(0.75)
    assert summary["accuracy_on_covered"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. the n-gate
# ---------------------------------------------------------------------------
def test_too_small_thresholds():
    assert too_small(29) is True
    assert too_small(30) is False
    assert "too small for claims" in claim_flag(5)
    assert claim_flag(224) == ""


def test_ece_is_zero_for_perfect_predictions():
    y = np.r_[np.ones(100), np.zeros(100)]
    p = np.r_[np.ones(100), np.zeros(100)]
    assert ece(y, np.clip(p, 1e-6, 1 - 1e-6), None) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 5. attacks keep what must be kept
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("attack", [attack_typos, attack_spaces_inside_words,
                                    attack_all_caps, attack_emoji])
def test_attacks_keep_placeholders(attack):
    text = "Your account will be blocked, verify at <URL> or call <PHONE> now"
    out = attack(text, np.random.default_rng(SEED))
    for token in ("<URL>", "<PHONE>"):
        assert token in out


def test_url_obfuscation_rewrites_a_real_link():
    out = attack_url_obfuscation("pay at https://kyc-verify.info/claim now",
                                 np.random.default_rng(SEED))
    assert "hxxps://" in out and "[.]" in out
    assert "https://" not in out


def test_url_obfuscation_handles_a_scrubbed_placeholder():
    out = attack_url_obfuscation("verify here <URL>", np.random.default_rng(SEED))
    assert "<URL>" not in out and "[.]" in out


# ---------------------------------------------------------------------------
# 6. the Hinglish rewrite
# ---------------------------------------------------------------------------
def test_substitution_pairs_are_verified_against_the_pools():
    from src.robustness import _component_pools

    pool = " ".join(c["text"]["roman_hinglish"] for c in _component_pools()).lower()
    pairs, dropped = substitution_pairs()
    assert len(pairs) > 100
    for en, hi in pairs:
        assert hi.lower() in pool, f"{en!r} -> {hi!r} is not in the pools"
    for en, hi in dropped:
        assert hi.lower() not in pool


def test_hinglish_rewrite_keeps_placeholders_and_digits():
    pairs, _ = substitution_pairs()
    sub = make_substituter(pairs)
    text = "Dear user, your account 1234 will be blocked, verify <URL> now"
    out = sub(text)
    assert "<URL>" in out
    assert "1234" in out
    assert "your account" not in out.lower()            # it was rewritten
    assert "aapka account" in out.lower()


def test_hinglish_rewrite_is_deterministic_and_does_not_cascade():
    pairs, _ = substitution_pairs()
    sub = make_substituter(pairs)
    text = "Please click here to update your details"
    assert sub(text) == sub(text)
    assert sub(text).count("aapka") == 1
