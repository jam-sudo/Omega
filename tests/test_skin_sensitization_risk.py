"""Tests for skin sensitization risk assessment."""

import pytest

from omega_pbpk.risk.skin_sensitization_risk import (
    SkinSensitizationResult,
    assess_skin_sensitization,
    estimate_ec3_value,
    screen_skin_sensitization,
)

VALID_CLASSES = {"non_sensitizer", "low", "moderate", "strong", "extreme"}
VALID_DPRA = {"non_reactive", "low_moderate", "moderate_high", "high_extreme"}


# ---------------------------------------------------------------------------
# assess_skin_sensitization
# ---------------------------------------------------------------------------


def test_returns_skin_sensitization_result():
    result = assess_skin_sensitization("TestCompound", logP=2.0, mw=300.0)
    assert isinstance(result, SkinSensitizationResult)


def test_no_alerts_zero_score():
    result = assess_skin_sensitization("Inert", logP=5.0, mw=300.0)
    assert result.skin_sensitization_score == 0.0


def test_no_alerts_non_sensitizer():
    result = assess_skin_sensitization("Inert", logP=5.0, mw=300.0)
    assert result.sensitization_class == "non_sensitizer"


def test_isocyanate_high_score():
    """Single isocyanate with logP 2 (1-4 range) → score = 40 * 1.2 = 48."""
    result = assess_skin_sensitization(
        "MDI",
        logP=2.0,
        mw=250.0,
        smiles_features={"n_isocyanates": 1},
    )
    assert result.skin_sensitization_score >= 40.0


def test_isocyanate_occupational_hazard():
    result = assess_skin_sensitization(
        "TDI", logP=2.0, mw=174.0, smiles_features={"n_isocyanates": 1}
    )
    assert result.occupational_hazard is True


def test_multiple_alerts_accumulate():
    result = assess_skin_sensitization(
        "MultiAlert",
        logP=2.0,
        mw=300.0,
        smiles_features={
            "n_michael_acceptors": 2,
            "n_epoxides_skin": 1,
        },
    )
    # 2*20 + 1*25 = 65, * 1.2 = 78
    assert result.skin_sensitization_score >= 50.0


def test_score_capped_at_100():
    result = assess_skin_sensitization(
        "MultiAlerts",
        logP=2.0,
        mw=300.0,
        smiles_features={
            "n_isocyanates": 2,
            "n_michael_acceptors": 3,
            "n_epoxides_skin": 2,
        },
    )
    assert result.skin_sensitization_score <= 100.0


def test_sensitization_class_valid():
    result = assess_skin_sensitization("Compound", logP=2.0, mw=300.0)
    assert result.sensitization_class in VALID_CLASSES


def test_dpra_category_valid():
    result = assess_skin_sensitization("Compound", logP=2.0, mw=300.0)
    assert result.dpra_category in VALID_DPRA


def test_requires_llna_when_score_above_20():
    result = assess_skin_sensitization(
        "Sensitizer",
        logP=2.0,
        mw=300.0,
        smiles_features={"n_michael_acceptors": 2},  # 40 * 1.2 = 48
    )
    assert result.skin_sensitization_score > 20.0
    assert result.requires_llna_testing is True


def test_no_llna_when_score_low():
    result = assess_skin_sensitization("Inert", logP=5.0, mw=300.0)
    assert result.requires_llna_testing is False


def test_high_mw_reduces_score():
    feats = {"n_michael_acceptors": 1}
    result_normal = assess_skin_sensitization("Small", logP=2.0, mw=300.0, smiles_features=feats)
    result_large = assess_skin_sensitization("Large", logP=2.0, mw=1500.0, smiles_features=feats)
    assert result_large.skin_sensitization_score < result_normal.skin_sensitization_score


def test_optimal_logp_increases_score():
    """logP 2 (optimal 1-4) → higher score than logP 7 (poor penetration)."""
    feats = {"n_michael_acceptors": 1}
    result_opt = assess_skin_sensitization("Opt", logP=2.0, mw=300.0, smiles_features=feats)
    result_poor = assess_skin_sensitization("Poor", logP=7.0, mw=300.0, smiles_features=feats)
    assert result_opt.skin_sensitization_score > result_poor.skin_sensitization_score


def test_notes_nonempty():
    result = assess_skin_sensitization("Compound", logP=2.0, mw=300.0)
    assert len(result.notes) > 0


def test_validation_mw_zero_raises():
    with pytest.raises(ValueError, match="mw must be > 0"):
        assess_skin_sensitization("X", logP=2.0, mw=0.0)


def test_validation_n_isocyanates_negative_raises():
    with pytest.raises(ValueError, match="n_isocyanates must be >= 0"):
        assess_skin_sensitization("X", logP=2.0, mw=300.0, smiles_features={"n_isocyanates": -1})


def test_occupational_hazard_high_score():
    """Score > 60 without isocyanates still triggers occupational hazard."""
    result = assess_skin_sensitization(
        "Sensitizer",
        logP=2.0,
        mw=300.0,
        smiles_features={"n_michael_acceptors": 3},  # 60 * 1.2 = 72 > 60
    )
    assert result.occupational_hazard is True


def test_n_structural_alerts_counts_correctly():
    result = assess_skin_sensitization(
        "Multi",
        logP=2.0,
        mw=300.0,
        smiles_features={"n_michael_acceptors": 2, "n_epoxides_skin": 1},
    )
    assert result.n_structural_alerts == 3


# ---------------------------------------------------------------------------
# screen_skin_sensitization
# ---------------------------------------------------------------------------


def test_screen_sorted_descending():
    compounds = [
        {"compound_name": "Low", "logP": 2.0, "mw": 300.0, "smiles_features": {}},
        {
            "compound_name": "High",
            "logP": 2.0,
            "mw": 300.0,
            "smiles_features": {"n_isocyanates": 1},
        },
        {
            "compound_name": "Mid",
            "logP": 2.0,
            "mw": 300.0,
            "smiles_features": {"n_michael_acceptors": 1},
        },
    ]
    results = screen_skin_sensitization(compounds)
    scores = [r.skin_sensitization_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_screen_returns_all_results():
    compounds = [{"compound_name": f"C{i}", "logP": 2.0, "mw": 300.0} for i in range(5)]
    results = screen_skin_sensitization(compounds)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# estimate_ec3_value
# ---------------------------------------------------------------------------


def test_ec3_high_score_low_value():
    """Higher sensitization score → lower EC3 (more potent)."""
    ec3_high = estimate_ec3_value(sensitization_score=80.0, logP=2.0)
    ec3_low = estimate_ec3_value(sensitization_score=10.0, logP=2.0)
    assert ec3_high < ec3_low


def test_ec3_within_bounds():
    for score in [0.0, 50.0, 100.0]:
        ec3 = estimate_ec3_value(sensitization_score=score, logP=2.0)
        assert 0.001 <= ec3 <= 100.0


def test_ec3_positive():
    ec3 = estimate_ec3_value(sensitization_score=0.0, logP=0.0)
    assert ec3 > 0.0
