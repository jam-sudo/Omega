"""Tests for Phase 774 — Drug Anti-Target Avoidance Scorer."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.anti_target_scorer import (
    AntiTargetScoreResult,
    score_anti_target_avoidance,
    screen_anti_target,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_VALID_RISK_LEVELS = {"low", "moderate", "high"}


# ---------------------------------------------------------------------------
# Basic return type tests
# ---------------------------------------------------------------------------


def test_returns_anti_target_score_result():
    result = score_anti_target_avoidance("CompoundA")
    assert isinstance(result, AntiTargetScoreResult)


def test_all_risk_scores_between_0_and_100():
    result = score_anti_target_avoidance(
        "CompoundA",
        logp=3.0,
        mw_da=350.0,
        psa=60.0,
        pka_basic=9.0,
    )
    assert 0.0 <= result.herg_risk_score <= 100.0
    assert 0.0 <= result.mao_risk_score <= 100.0
    assert 0.0 <= result.cox1_risk_score <= 100.0
    assert 0.0 <= result.m1_risk_score <= 100.0
    assert 0.0 <= result.overall_anti_target_risk <= 100.0


def test_has_hydrazine_elevates_mao_risk():
    result = score_anti_target_avoidance("HydrazineCompound", has_hydrazine=True)
    assert result.mao_risk_score >= 50.0
    assert result.mao_risk_level in ("moderate", "high")


def test_has_propargylamine_elevates_mao_risk():
    result = score_anti_target_avoidance("PropargylamineCompound", has_propargylamine=True)
    assert result.mao_risk_score >= 40.0


def test_has_carboxylic_acid_elevates_cox1_risk():
    result = score_anti_target_avoidance("AcidCompound", has_carboxylic_acid=True)
    assert result.cox1_risk_score >= 25.0


def test_aryl_acetic_acid_high_cox1_risk():
    result = score_anti_target_avoidance(
        "NSAIDCompound",
        has_carboxylic_acid=True,
        has_aryl_acetic_acid=True,
    )
    assert result.cox1_risk_score >= 60.0
    assert result.cox1_risk_level in ("moderate", "high")


def test_high_logp_high_basic_pka_elevates_herg_risk():
    result = score_anti_target_avoidance(
        "hERGCompound",
        logp=5.0,
        psa=30.0,
        pka_basic=10.0,
    )
    assert result.herg_risk_score >= 30.0


def test_has_tertiary_amine_elevates_m1_risk():
    result = score_anti_target_avoidance(
        "AnticholinergicCompound",
        has_tertiary_amine=True,
        logp=3.0,
        psa=20.0,
    )
    assert result.m1_risk_score >= 30.0


def test_selectivity_score_equals_100_minus_overall_risk():
    result = score_anti_target_avoidance(
        "AnyCompound",
        logp=2.0,
        psa=80.0,
        pka_basic=4.0,
    )
    assert abs(result.selectivity_score - (100.0 - result.overall_anti_target_risk)) < 1e-9


def test_no_structural_alerts_low_overall_risk():
    result = score_anti_target_avoidance(
        "CleanCompound",
        logp=1.0,
        mw_da=250.0,
        psa=100.0,
        pka_basic=None,
        has_hydrazine=False,
        has_propargylamine=False,
        has_carboxylic_acid=False,
        has_aryl_acetic_acid=False,
        has_aryl_propionic_acid=False,
        has_tertiary_amine=False,
        aromatic_amine_count=0,
    )
    assert result.overall_anti_target_risk < 30.0
    assert result.overall_risk_level == "low"


def test_screen_anti_target_returns_sorted_list():
    compounds = [
        {"compound_name": "A", "logp": 5.0, "pka_basic": 10.0, "psa": 20.0},
        {"compound_name": "B", "logp": 1.0, "psa": 90.0},
        {"compound_name": "C", "has_hydrazine": True},
    ]
    results = screen_anti_target(compounds)
    assert isinstance(results, list)
    assert len(results) == 3
    scores = [r.selectivity_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_screen_anti_target_sorted_by_selectivity_descending():
    compounds = [
        {
            "compound_name": "Risky",
            "logp": 6.0,
            "pka_basic": 10.0,
            "psa": 10.0,
            "has_hydrazine": True,
        },
        {"compound_name": "Safe", "logp": 1.0, "psa": 120.0},
    ]
    results = screen_anti_target(compounds)
    assert results[0].selectivity_score >= results[1].selectivity_score


def test_risk_level_one_of_valid_values():
    result = score_anti_target_avoidance("TestCompound", logp=4.0, pka_basic=9.0, psa=40.0)
    assert result.herg_risk_level in _VALID_RISK_LEVELS
    assert result.mao_risk_level in _VALID_RISK_LEVELS
    assert result.cox1_risk_level in _VALID_RISK_LEVELS
    assert result.m1_risk_level in _VALID_RISK_LEVELS
    assert result.overall_risk_level in _VALID_RISK_LEVELS


def test_validation_mw_le_zero_raises():
    with pytest.raises(ValueError, match="mw_da"):
        score_anti_target_avoidance("Bad", mw_da=0.0)


def test_validation_negative_mw_raises():
    with pytest.raises(ValueError, match="mw_da"):
        score_anti_target_avoidance("Bad", mw_da=-100.0)


def test_validation_negative_psa_raises():
    with pytest.raises(ValueError, match="psa"):
        score_anti_target_avoidance("Bad", psa=-1.0)


def test_validation_negative_aromatic_amine_count_raises():
    with pytest.raises(ValueError, match="aromatic_amine_count"):
        score_anti_target_avoidance("Bad", aromatic_amine_count=-1)


def test_notes_nonempty():
    result = score_anti_target_avoidance("Compound")
    assert len(result.notes) > 0


def test_recommendations_nonempty():
    result = score_anti_target_avoidance("Compound")
    assert len(result.recommendations) > 0


def test_large_mw_reduces_herg_risk():
    """Molecules > 600 Da get reduced hERG score."""
    result_small = score_anti_target_avoidance(
        "SmallMol", logp=4.0, mw_da=400.0, psa=30.0, pka_basic=9.0
    )
    result_large = score_anti_target_avoidance(
        "LargeMol", logp=4.0, mw_da=650.0, psa=30.0, pka_basic=9.0
    )
    assert result_large.herg_risk_score <= result_small.herg_risk_score


def test_aromatic_amine_count_increases_mao_risk():
    r0 = score_anti_target_avoidance("NoAmine", aromatic_amine_count=0)
    r2 = score_anti_target_avoidance("TwoAmines", aromatic_amine_count=2)
    assert r2.mao_risk_score > r0.mao_risk_score


def test_overall_risk_is_max_of_all_risks():
    result = score_anti_target_avoidance(
        "MaxTest",
        logp=5.0,
        pka_basic=10.0,
        psa=20.0,
        has_hydrazine=True,
        has_carboxylic_acid=True,
        has_tertiary_amine=True,
    )
    expected_max = max(
        result.herg_risk_score,
        result.mao_risk_score,
        result.cox1_risk_score,
        result.m1_risk_score,
    )
    assert abs(result.overall_anti_target_risk - expected_max) < 1e-9


def test_result_is_frozen():
    result = score_anti_target_avoidance("FrozenTest")
    with pytest.raises(Exception):
        result.herg_risk_score = 999.0  # type: ignore[misc]
