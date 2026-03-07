"""Tests for Cosolvent Solubility Enhancement Predictor — Phase 818."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.cosolvent_solubility import (
    CosolventSolubilityResult,
    optimize_cosolvent_fraction,
    predict_cosolvent_solubility,
)

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_cosolvent_solubility_result():
    result = predict_cosolvent_solubility(
        "DrugA",
        logp=2.0,
        aqueous_solubility_mg_mL=0.1,
        cosolvent_name="PEG400",
        cosolvent_fraction=0.2,
        target_mg_mL=1.0,
    )
    assert isinstance(result, CosolventSolubilityResult)


def test_drug_name_preserved():
    result = predict_cosolvent_solubility(
        "Ibuprofen",
        logp=4.0,
        aqueous_solubility_mg_mL=0.05,
        cosolvent_name="Ethanol",
        cosolvent_fraction=0.05,
        target_mg_mL=0.1,
    )
    assert result.drug_name == "Ibuprofen"


def test_cosolvent_name_preserved():
    result = predict_cosolvent_solubility(
        "DrugA",
        logp=2.0,
        aqueous_solubility_mg_mL=0.1,
        cosolvent_name="DMSO",
        cosolvent_fraction=0.05,
        target_mg_mL=1.0,
    )
    assert result.cosolvent_name == "DMSO"


# ---------------------------------------------------------------------------
# Enhancement fold >= 1.0 for positive logP drug
# ---------------------------------------------------------------------------


def test_enhancement_fold_ge_1_positive_logp():
    result = predict_cosolvent_solubility(
        "DrugA",
        logp=3.0,
        aqueous_solubility_mg_mL=0.1,
        cosolvent_name="PEG400",
        cosolvent_fraction=0.3,
        target_mg_mL=0.5,
    )
    assert result.enhancement_fold >= 1.0


def test_enhancement_fold_equals_1_at_zero_fraction():
    """At f=0, S_mix == S_water → fold == 1."""
    result = predict_cosolvent_solubility(
        "DrugA",
        logp=3.0,
        aqueous_solubility_mg_mL=0.1,
        cosolvent_name="PEG400",
        cosolvent_fraction=0.0,
        target_mg_mL=5.0,
    )
    assert result.enhancement_fold == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Higher fraction → higher enhancement
# ---------------------------------------------------------------------------


def test_higher_fraction_higher_enhancement():
    s_water = 0.05
    low = predict_cosolvent_solubility(
        "DrugA",
        logp=3.0,
        aqueous_solubility_mg_mL=s_water,
        cosolvent_name="PEG400",
        cosolvent_fraction=0.1,
        target_mg_mL=1.0,
    )
    high = predict_cosolvent_solubility(
        "DrugA",
        logp=3.0,
        aqueous_solubility_mg_mL=s_water,
        cosolvent_name="PEG400",
        cosolvent_fraction=0.4,
        target_mg_mL=1.0,
    )
    assert high.enhanced_solubility_mg_mL > low.enhanced_solubility_mg_mL


def test_higher_fraction_higher_fold():
    s_water = 0.05
    low = predict_cosolvent_solubility(
        "DrugA",
        logp=3.0,
        aqueous_solubility_mg_mL=s_water,
        cosolvent_name="Ethanol",
        cosolvent_fraction=0.05,
        target_mg_mL=1.0,
    )
    high = predict_cosolvent_solubility(
        "DrugA",
        logp=3.0,
        aqueous_solubility_mg_mL=s_water,
        cosolvent_name="Ethanol",
        cosolvent_fraction=0.1,
        target_mg_mL=1.0,
    )
    assert high.enhancement_fold > low.enhancement_fold


# ---------------------------------------------------------------------------
# Log-linear relationship
# ---------------------------------------------------------------------------


def test_log_linear_relationship():
    """For PEG400, logP=2 → sigma=1.8; doubling fraction doubles log-enhancement."""
    s_water = 0.1
    f1, f2 = 0.1, 0.2
    r1 = predict_cosolvent_solubility(
        "DrugA",
        logp=2.0,
        aqueous_solubility_mg_mL=s_water,
        cosolvent_name="PEG400",
        cosolvent_fraction=f1,
        target_mg_mL=10.0,
    )
    r2 = predict_cosolvent_solubility(
        "DrugA",
        logp=2.0,
        aqueous_solubility_mg_mL=s_water,
        cosolvent_name="PEG400",
        cosolvent_fraction=f2,
        target_mg_mL=10.0,
    )
    # log10(fold2) / log10(fold1) should equal f2/f1 = 2.0
    log_fold1 = math.log10(r1.enhancement_fold)
    log_fold2 = math.log10(r2.enhancement_fold)
    assert log_fold2 == pytest.approx(log_fold1 * (f2 / f1), rel=1e-6)


def test_sigma_reported_correctly_peg400():
    """PEG400 sigma = 0.9 * logP clamped to [0,4]."""
    result = predict_cosolvent_solubility(
        "DrugA",
        logp=3.0,
        aqueous_solubility_mg_mL=0.1,
        cosolvent_name="PEG400",
        cosolvent_fraction=0.3,
        target_mg_mL=1.0,
    )
    expected_sigma = min(0.9 * 3.0, 4.0)
    assert result.sigma == pytest.approx(expected_sigma, rel=1e-6)


def test_sigma_clamped_at_max_for_high_logp():
    """For logP=10, Ethanol sigma should be clamped at 5."""
    result = predict_cosolvent_solubility(
        "DrugX",
        logp=10.0,
        aqueous_solubility_mg_mL=0.001,
        cosolvent_name="Ethanol",
        cosolvent_fraction=0.05,
        target_mg_mL=1.0,
    )
    assert result.sigma == pytest.approx(5.0)


def test_sigma_zero_for_zero_logp():
    """For logP=0, sigma = 0 for any cosolvent."""
    result = predict_cosolvent_solubility(
        "DrugZ",
        logp=0.0,
        aqueous_solubility_mg_mL=5.0,
        cosolvent_name="PEG400",
        cosolvent_fraction=0.3,
        target_mg_mL=1.0,
    )
    assert result.sigma == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_fraction_below_zero_raises():
    with pytest.raises(ValueError):
        predict_cosolvent_solubility(
            "DrugA",
            logp=2.0,
            aqueous_solubility_mg_mL=0.1,
            cosolvent_name="PEG400",
            cosolvent_fraction=-0.01,
            target_mg_mL=1.0,
        )


def test_fraction_above_one_raises():
    with pytest.raises(ValueError):
        predict_cosolvent_solubility(
            "DrugA",
            logp=2.0,
            aqueous_solubility_mg_mL=0.1,
            cosolvent_name="PEG400",
            cosolvent_fraction=1.01,
            target_mg_mL=1.0,
        )


def test_zero_aqueous_solubility_raises():
    with pytest.raises(ValueError):
        predict_cosolvent_solubility(
            "DrugA",
            logp=2.0,
            aqueous_solubility_mg_mL=0.0,
            cosolvent_name="PEG400",
            cosolvent_fraction=0.3,
            target_mg_mL=1.0,
        )


def test_negative_aqueous_solubility_raises():
    with pytest.raises(ValueError):
        predict_cosolvent_solubility(
            "DrugA",
            logp=2.0,
            aqueous_solubility_mg_mL=-0.1,
            cosolvent_name="PEG400",
            cosolvent_fraction=0.3,
            target_mg_mL=1.0,
        )


# ---------------------------------------------------------------------------
# formulation_feasible
# ---------------------------------------------------------------------------


def test_formulation_feasible_true_when_enhanced_ge_target():
    """Low target → feasible."""
    result = predict_cosolvent_solubility(
        "DrugA",
        logp=3.0,
        aqueous_solubility_mg_mL=1.0,
        cosolvent_name="PEG400",
        cosolvent_fraction=0.3,
        target_mg_mL=0.5,
    )
    assert result.formulation_feasible is True
    assert result.enhanced_solubility_mg_mL >= 0.5


def test_formulation_feasible_false_when_enhanced_lt_target():
    """Very high target → not feasible."""
    result = predict_cosolvent_solubility(
        "DrugA",
        logp=1.0,
        aqueous_solubility_mg_mL=0.001,
        cosolvent_name="Glycerol",
        cosolvent_fraction=0.1,
        target_mg_mL=1000.0,
    )
    assert result.formulation_feasible is False


# ---------------------------------------------------------------------------
# max_safe_fraction and max_enhancement_fold
# ---------------------------------------------------------------------------


def test_max_safe_fraction_peg400():
    result = predict_cosolvent_solubility(
        "DrugA",
        logp=2.0,
        aqueous_solubility_mg_mL=0.1,
        cosolvent_name="PEG400",
        cosolvent_fraction=0.3,
        target_mg_mL=1.0,
    )
    assert result.max_safe_fraction == pytest.approx(0.5)


def test_max_safe_fraction_ethanol():
    result = predict_cosolvent_solubility(
        "DrugA",
        logp=2.0,
        aqueous_solubility_mg_mL=0.1,
        cosolvent_name="Ethanol",
        cosolvent_fraction=0.05,
        target_mg_mL=1.0,
    )
    assert result.max_safe_fraction == pytest.approx(0.1)


def test_max_enhancement_fold_gt_fold_at_lower_fraction():
    result = predict_cosolvent_solubility(
        "DrugA",
        logp=3.0,
        aqueous_solubility_mg_mL=0.1,
        cosolvent_name="PEG400",
        cosolvent_fraction=0.2,
        target_mg_mL=1.0,
    )
    # max_safe is 0.5 > 0.2, so max_fold should be >= fold
    assert result.max_enhancement_fold >= result.enhancement_fold


# ---------------------------------------------------------------------------
# optimize_cosolvent_fraction
# ---------------------------------------------------------------------------


def test_optimize_returns_result():
    result = optimize_cosolvent_fraction(
        "DrugA", logp=3.0, aqueous_solubility_mg_mL=0.05, cosolvent_name="PEG400", target_mg_mL=0.5
    )
    assert isinstance(result, CosolventSolubilityResult)


def test_optimize_feasible_when_target_achievable():
    """High-logP drug in PEG400 should easily reach target."""
    result = optimize_cosolvent_fraction(
        "DrugA", logp=4.0, aqueous_solubility_mg_mL=0.01, cosolvent_name="PEG400", target_mg_mL=0.05
    )
    assert result.formulation_feasible is True


def test_optimize_uses_minimum_fraction():
    """Fraction returned should be minimal sufficient fraction."""
    target = 0.02
    result = optimize_cosolvent_fraction(
        "DrugA",
        logp=3.0,
        aqueous_solubility_mg_mL=0.01,
        cosolvent_name="PEG400",
        target_mg_mL=target,
    )
    if result.formulation_feasible:
        # Stepping back by one grid step should not achieve the target (or f=0)
        prev_frac = max(0.0, result.cosolvent_fraction - 0.05)
        prev_result = predict_cosolvent_solubility(
            "DrugA",
            logp=3.0,
            aqueous_solubility_mg_mL=0.01,
            cosolvent_name="PEG400",
            cosolvent_fraction=prev_frac,
            target_mg_mL=target,
        )
        assert not prev_result.formulation_feasible or prev_frac == 0.0


def test_optimize_at_max_safe_when_target_not_achievable():
    """If target cannot be met, result should be at max_safe_fraction."""
    result = optimize_cosolvent_fraction(
        "DrugA",
        logp=0.5,
        aqueous_solubility_mg_mL=0.001,
        cosolvent_name="Glycerol",
        target_mg_mL=10000.0,
    )
    assert result.max_safe_fraction == pytest.approx(0.6)
    assert not result.formulation_feasible


# ---------------------------------------------------------------------------
# Unknown cosolvent
# ---------------------------------------------------------------------------


def test_unknown_cosolvent_returns_result():
    result = predict_cosolvent_solubility(
        "DrugA",
        logp=2.0,
        aqueous_solubility_mg_mL=0.1,
        cosolvent_name="SpecialMix",
        cosolvent_fraction=0.2,
        target_mg_mL=1.0,
    )
    assert isinstance(result, CosolventSolubilityResult)


def test_unknown_cosolvent_max_safe_fraction():
    result = predict_cosolvent_solubility(
        "DrugA",
        logp=2.0,
        aqueous_solubility_mg_mL=0.1,
        cosolvent_name="SpecialMix",
        cosolvent_fraction=0.2,
        target_mg_mL=1.0,
    )
    assert result.max_safe_fraction == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Notes non-empty
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = predict_cosolvent_solubility(
        "DrugA",
        logp=2.0,
        aqueous_solubility_mg_mL=0.1,
        cosolvent_name="PEG400",
        cosolvent_fraction=0.3,
        target_mg_mL=1.0,
    )
    assert len(result.notes) > 0
