"""Tests for Phase 760: Drug Purity and Impurity Impact Predictor."""

import pytest

from omega_pbpk.prediction.impurity_impact_predictor import (
    ImpurityImpactResult,
    assess_impurity_impact,
    screen_impurities,
)

# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_returns_dataclass():
    result = assess_impurity_impact(
        drug_name="DrugA",
        impurity_name="ImpX",
        daily_dose_drug_mg=500.0,
        impurity_pct=0.02,
    )
    assert isinstance(result, ImpurityImpactResult)


# ---------------------------------------------------------------------------
# daily_dose_impurity_mg calculation
# ---------------------------------------------------------------------------


def test_daily_dose_impurity_calculation():
    daily_dose = 1000.0
    pct = 0.5
    result = assess_impurity_impact(
        drug_name="D", impurity_name="I", daily_dose_drug_mg=daily_dose, impurity_pct=pct
    )
    expected = daily_dose * pct / 100.0
    assert abs(result.daily_dose_impurity_mg - expected) < 1e-9


# ---------------------------------------------------------------------------
# Below reporting threshold
# ---------------------------------------------------------------------------


def test_below_reporting_threshold():
    """impurity_pct=0.01% and tiny absolute dose → below_reporting."""
    result = assess_impurity_impact(
        drug_name="D", impurity_name="I", daily_dose_drug_mg=100.0, impurity_pct=0.01
    )
    assert result.ich_classification == "below_reporting"
    assert not result.reporting_threshold_exceeded


def test_below_reporting_action_none():
    result = assess_impurity_impact(
        drug_name="D", impurity_name="I", daily_dose_drug_mg=100.0, impurity_pct=0.01
    )
    assert result.regulatory_action_required == "none"


# ---------------------------------------------------------------------------
# Above 0.05% but below 0.10% → reporting
# ---------------------------------------------------------------------------


def test_reporting_threshold_exceeded():
    result = assess_impurity_impact(
        drug_name="D", impurity_name="I", daily_dose_drug_mg=100.0, impurity_pct=0.07
    )
    assert result.reporting_threshold_exceeded
    assert result.ich_classification == "reporting"
    assert result.regulatory_action_required == "report_to_authority"


# ---------------------------------------------------------------------------
# Above 0.10% → qualification
# ---------------------------------------------------------------------------


def test_qualification_threshold_exceeded():
    result = assess_impurity_impact(
        drug_name="D", impurity_name="I", daily_dose_drug_mg=100.0, impurity_pct=0.5
    )
    assert result.qualification_threshold_exceeded
    assert result.ich_classification == "qualification"
    assert result.regulatory_action_required == "qualification_study_required"


# ---------------------------------------------------------------------------
# Genotoxic impurity exceeds TTC
# ---------------------------------------------------------------------------


def test_genotoxic_ttc_exceeded():
    """Large dose of genotoxic impurity exceeds TTC of 1.5 µg/day."""
    result = assess_impurity_impact(
        drug_name="D",
        impurity_name="GenoImp",
        daily_dose_drug_mg=1000.0,
        impurity_pct=0.001,  # 0.01 mg → 10 µg > 1.5 µg
        is_genotoxic=True,
    )
    assert result.genotoxic_risk == "ttc_exceeded"
    assert result.regulatory_action_required == "immediate_action_genotoxic"


def test_genotoxic_below_ttc():
    """Genotoxic but below TTC → risk is 'none'."""
    result = assess_impurity_impact(
        drug_name="D",
        impurity_name="GenoLow",
        daily_dose_drug_mg=10.0,
        impurity_pct=0.001,  # 0.0001 mg → 0.1 µg < 1.5 µg
        is_genotoxic=True,
    )
    assert result.genotoxic_risk == "none"


def test_non_genotoxic_risk_not_applicable():
    result = assess_impurity_impact(
        drug_name="D",
        impurity_name="I",
        daily_dose_drug_mg=500.0,
        impurity_pct=0.1,
        is_genotoxic=False,
    )
    assert result.genotoxic_risk == "not_applicable"


# ---------------------------------------------------------------------------
# Potency factor
# ---------------------------------------------------------------------------


def test_potency_factor_applied():
    daily_dose = 500.0
    pct = 0.2
    potency = 2.5
    result = assess_impurity_impact(
        drug_name="D",
        impurity_name="I",
        daily_dose_drug_mg=daily_dose,
        impurity_pct=pct,
        potency_relative_to_parent=potency,
    )
    expected_imp_dose = daily_dose * pct / 100.0
    expected_eff_dose = expected_imp_dose * potency
    assert abs(result.effective_pharmacological_dose_mg - expected_eff_dose) < 1e-9


def test_zero_potency_gives_zero_effective_dose():
    result = assess_impurity_impact(
        drug_name="D",
        impurity_name="I",
        daily_dose_drug_mg=500.0,
        impurity_pct=0.5,
        potency_relative_to_parent=0.0,
    )
    assert result.effective_pharmacological_dose_mg == 0.0


# ---------------------------------------------------------------------------
# screen_impurities
# ---------------------------------------------------------------------------


def test_screen_impurities_returns_list():
    impurities = [
        {"impurity_name": "A", "impurity_pct": 0.5},
        {"impurity_name": "B", "impurity_pct": 0.1},
        {"impurity_name": "C", "impurity_pct": 0.02},
    ]
    results = screen_impurities("Drug", 1000.0, impurities)
    assert isinstance(results, list)
    assert len(results) == 3


def test_screen_impurities_sorted_desc():
    impurities = [
        {"impurity_name": "Low", "impurity_pct": 0.02},
        {"impurity_name": "High", "impurity_pct": 0.8},
        {"impurity_name": "Mid", "impurity_pct": 0.15},
    ]
    results = screen_impurities("Drug", 1000.0, impurities)
    pcts = [r.impurity_pct for r in results]
    assert pcts == sorted(pcts, reverse=True)


def test_screen_impurities_optional_keys():
    """screen_impurities handles missing optional keys gracefully."""
    impurities = [
        {"impurity_name": "A", "impurity_pct": 0.05, "is_genotoxic": True, "potency": 3.0},
        {"impurity_name": "B", "impurity_pct": 0.01},
    ]
    results = screen_impurities("Drug", 500.0, impurities)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_daily_dose_zero_raises():
    with pytest.raises(ValueError):
        assess_impurity_impact(
            drug_name="D", impurity_name="I", daily_dose_drug_mg=0.0, impurity_pct=0.1
        )


def test_validation_daily_dose_negative_raises():
    with pytest.raises(ValueError):
        assess_impurity_impact(
            drug_name="D", impurity_name="I", daily_dose_drug_mg=-100.0, impurity_pct=0.1
        )


def test_validation_impurity_pct_over_100_raises():
    with pytest.raises(ValueError):
        assess_impurity_impact(
            drug_name="D", impurity_name="I", daily_dose_drug_mg=500.0, impurity_pct=101.0
        )


def test_validation_impurity_pct_negative_raises():
    with pytest.raises(ValueError):
        assess_impurity_impact(
            drug_name="D", impurity_name="I", daily_dose_drug_mg=500.0, impurity_pct=-1.0
        )


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = assess_impurity_impact(
        drug_name="D", impurity_name="I", daily_dose_drug_mg=500.0, impurity_pct=0.1
    )
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
