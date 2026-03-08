"""Tests for Phase 335 — IV Precipitation Risk Assessment."""

import pytest

from omega_pbpk.prediction.iv_precipitation_risk import (
    IVPrecipitationResult,
    assess_iv_precipitation_risk,
)

# --- Helpers ---


def make_basic(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        drug_conc_mg_mL=1.0,
        formulation_pH=4.0,
        solubility_mg_mL=0.5,
        pka=4.5,
        drug_type="acid",
        cosolvent_pct=0.0,
        osmolality_mOsm_kg=290.0,
    )
    defaults.update(kwargs)
    return assess_iv_precipitation_risk(**defaults)


# --- Type and dataclass checks ---


def test_returns_correct_type():
    result = make_basic()
    assert isinstance(result, IVPrecipitationResult)


def test_result_is_frozen():
    result = make_basic()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"  # type: ignore[misc]


# --- Risk score bounds ---


def test_risk_score_in_bounds_low():
    result = make_basic(drug_conc_mg_mL=0.001, solubility_mg_mL=100.0)
    assert 0.0 <= result.precipitation_risk_score <= 100.0


def test_risk_score_in_bounds_high():
    result = make_basic(drug_conc_mg_mL=1000.0, solubility_mg_mL=0.001)
    assert result.precipitation_risk_score == 100.0


def test_risk_score_capped_at_100():
    # Very high concentration relative to solubility must be capped
    result = make_basic(drug_conc_mg_mL=5000.0, solubility_mg_mL=0.001)
    assert result.precipitation_risk_score <= 100.0


# --- Risk class validity ---


def test_risk_class_low():
    # Low conc → low risk score → "low"
    result = make_basic(drug_conc_mg_mL=0.001, solubility_mg_mL=100.0)
    assert result.risk_class == "low"


def test_risk_class_high():
    result = make_basic(drug_conc_mg_mL=1000.0, solubility_mg_mL=0.001)
    assert result.risk_class == "high"


def test_risk_class_valid_set():
    for conc in [0.001, 1.0, 1000.0]:
        result = make_basic(drug_conc_mg_mL=conc)
        assert result.risk_class in {"low", "moderate", "high"}


# --- Precipitation risk True/False ---


def test_high_conc_precipitation_risk_true():
    # drug_conc 100 → after /10 = 10 >> solubility 0.1
    result = make_basic(
        drug_conc_mg_mL=100.0,
        solubility_mg_mL=0.1,
        drug_type="neutral",
    )
    assert result.precipitation_risk is True


def test_low_conc_no_precipitation_risk():
    # drug_conc 0.01 → after /10 = 0.001 << solubility 10.0
    result = make_basic(
        drug_conc_mg_mL=0.01,
        solubility_mg_mL=10.0,
        drug_type="neutral",
    )
    assert result.precipitation_risk is False


# --- Cosolvent effects ---


def test_cosolvent_increases_effective_solubility():
    r_no_cosolvent = make_basic(cosolvent_pct=0.0)
    r_with_cosolvent = make_basic(cosolvent_pct=50.0)
    assert (
        r_with_cosolvent.effective_solubility_in_formulation_mg_mL
        > r_no_cosolvent.effective_solubility_in_formulation_mg_mL
    )


def test_cosolvent_fold_positive():
    result = make_basic(cosolvent_pct=20.0)
    assert result.cosolvent_contribution_fold > 1.0


def test_zero_cosolvent_fold_is_one():
    result = make_basic(cosolvent_pct=0.0)
    assert abs(result.cosolvent_contribution_fold - 1.0) < 1e-9


# --- Hemolysis risk ---


def test_hemolysis_risk_true_high_osmolality():
    result = make_basic(osmolality_mOsm_kg=450.0)
    assert result.hemolysis_risk is True


def test_hemolysis_risk_true_low_osmolality():
    result = make_basic(osmolality_mOsm_kg=100.0)
    assert result.hemolysis_risk is True


def test_hemolysis_risk_false_normal_osmolality():
    result = make_basic(osmolality_mOsm_kg=290.0)
    assert result.hemolysis_risk is False


def test_hemolysis_risk_false_border():
    # |390 - 290| = 100, not > 100 → False
    result = make_basic(osmolality_mOsm_kg=390.0)
    assert result.hemolysis_risk is False


# --- Precipitation concentration correctness ---


def test_precipitation_conc_is_drug_conc_over_10():
    result = make_basic(drug_conc_mg_mL=5.0)
    assert abs(result.precipitation_conc_after_dilution_mg_mL - 0.5) < 1e-9


# --- Validation errors ---


def test_raises_on_zero_drug_conc():
    with pytest.raises(ValueError, match="drug_conc_mg_mL"):
        make_basic(drug_conc_mg_mL=0.0)


def test_raises_on_negative_drug_conc():
    with pytest.raises(ValueError, match="drug_conc_mg_mL"):
        make_basic(drug_conc_mg_mL=-1.0)


def test_raises_on_zero_solubility():
    with pytest.raises(ValueError, match="solubility_mg_mL"):
        make_basic(solubility_mg_mL=0.0)


def test_raises_on_negative_solubility():
    with pytest.raises(ValueError, match="solubility_mg_mL"):
        make_basic(solubility_mg_mL=-5.0)


def test_raises_on_negative_cosolvent():
    with pytest.raises(ValueError, match="cosolvent_pct"):
        make_basic(cosolvent_pct=-10.0)


def test_raises_on_invalid_drug_type():
    with pytest.raises(ValueError, match="drug_type"):
        make_basic(drug_type="zwitterion")


# --- drug_type neutrality ---


def test_neutral_drug_returns_physiological_solubility():
    result = make_basic(drug_type="neutral", solubility_mg_mL=2.0)
    assert abs(result.solubility_physiological_mg_mL - 2.0) < 1e-9


# --- String fields non-empty ---


def test_recommendations_non_empty():
    result = make_basic()
    assert isinstance(result.recommendations, str)
    assert len(result.recommendations) > 0


def test_notes_non_empty():
    result = make_basic()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
