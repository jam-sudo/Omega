"""Tests for Phase 333 — Drug Crystallization Risk Predictor."""

import math

import pytest

from omega_pbpk.prediction.crystallization_risk_predictor import (
    CrystallizationRiskResult,
    predict_crystallization_risk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_RISK_CLASSES = {"low", "moderate", "high"}
_VALID_FORMULATIONS = {
    "No special formulation required",
    "Consider solubilizing excipients or amorphous form",
    "Use ASD (amorphous solid dispersion) or nano-formulation",
}


def _make_result(
    drug_name="TestDrug",
    solubility_mg_mL=1.0,
    logP=2.0,
    pKa=7.0,
    drug_type="neutral",
    dose_mg=100.0,
    volume_of_distribution_mL=250.0,
):
    return predict_crystallization_risk(
        drug_name=drug_name,
        solubility_mg_mL=solubility_mg_mL,
        logP=logP,
        pKa=pKa,
        drug_type=drug_type,
        dose_mg=dose_mg,
        volume_of_distribution_mL=volume_of_distribution_mL,
    )


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _make_result()
    assert isinstance(result, CrystallizationRiskResult)


def test_result_is_frozen():
    result = _make_result()
    with pytest.raises((AttributeError, TypeError)):
        result.risk_score = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. risk_score bounds
# ---------------------------------------------------------------------------


def test_risk_score_non_negative():
    result = _make_result(dose_mg=1.0, volume_of_distribution_mL=1000.0)
    assert result.risk_score >= 0.0


def test_risk_score_capped_at_100():
    # Very high dose relative to solubility -> SR >> 5 -> risk_score capped
    result = _make_result(solubility_mg_mL=0.001, dose_mg=1000.0, volume_of_distribution_mL=10.0)
    assert result.risk_score <= 100.0


# ---------------------------------------------------------------------------
# 3. risk_class validity
# ---------------------------------------------------------------------------


def test_risk_class_valid_set():
    result = _make_result()
    assert result.risk_class in _VALID_RISK_CLASSES


# ---------------------------------------------------------------------------
# 4. SR < 1 → risk_class "low" and infinite t_cryst
# ---------------------------------------------------------------------------


def test_low_sr_risk_class_low():
    # dose=1 mg, vod=1000 mL → local=0.001 mg/mL; sol=1.0 mg/mL → SR=0.001
    result = _make_result(dose_mg=1.0, volume_of_distribution_mL=1000.0)
    assert result.risk_class == "low"


def test_low_sr_infinite_t_cryst():
    result = _make_result(dose_mg=1.0, volume_of_distribution_mL=1000.0)
    assert math.isinf(result.time_to_crystallization_h)


# ---------------------------------------------------------------------------
# 5. SR ≥ 3 → risk_class "high"
# ---------------------------------------------------------------------------


def test_high_sr_risk_class_high():
    # local = 200/100 = 2 mg/mL, sol_at_ph = 0.1 mg/mL → SR = 20
    result = _make_result(
        solubility_mg_mL=0.1,
        dose_mg=200.0,
        volume_of_distribution_mL=100.0,
        drug_type="neutral",
    )
    assert result.supersaturation_ratio >= 3.0
    assert result.risk_class == "high"


# ---------------------------------------------------------------------------
# 6. Higher dose → higher SR → higher risk_score
# ---------------------------------------------------------------------------


def test_higher_dose_higher_risk():
    low = _make_result(dose_mg=10.0)
    high = _make_result(dose_mg=500.0)
    assert high.supersaturation_ratio > low.supersaturation_ratio
    assert high.risk_score >= low.risk_score


# ---------------------------------------------------------------------------
# 7. High solubility → low SR → low risk
# ---------------------------------------------------------------------------


def test_high_solubility_low_risk():
    result = _make_result(solubility_mg_mL=100.0, dose_mg=10.0)
    assert result.risk_class == "low"


# ---------------------------------------------------------------------------
# 8. amorphous_factor ≥ 1.0 always
# ---------------------------------------------------------------------------


def test_amorphous_factor_min_one():
    # Hydrophilic drug, logP = -3 (10^(-3/4) < 1 → clipped to 1)
    result = _make_result(logP=-3.0)
    assert result.amorphous_factor >= 1.0


def test_amorphous_factor_grows_with_logp():
    r1 = _make_result(logP=1.0)
    r2 = _make_result(logP=4.0)
    assert r2.amorphous_factor >= r1.amorphous_factor


# ---------------------------------------------------------------------------
# 9. formulation_recommendation nonempty and valid
# ---------------------------------------------------------------------------


def test_formulation_recommendation_nonempty():
    result = _make_result()
    assert len(result.formulation_recommendation) > 0


def test_formulation_recommendation_valid():
    for drug_type in ("acid", "base", "neutral"):
        result = _make_result(drug_type=drug_type)
        assert result.formulation_recommendation in _VALID_FORMULATIONS


# ---------------------------------------------------------------------------
# 10. Neutral drug: solubility unchanged with pH
# ---------------------------------------------------------------------------


def test_neutral_drug_solubility_unchanged():
    sol = 2.5
    result = _make_result(solubility_mg_mL=sol, drug_type="neutral")
    assert result.solubility_at_ph_mg_mL == pytest.approx(sol, rel=1e-9)


# ---------------------------------------------------------------------------
# 11. Acid drug at pH 6.8 → higher solubility than intrinsic (when pKa < 6.8)
# ---------------------------------------------------------------------------


def test_acid_drug_higher_sol_at_ph():
    # pKa = 4.0: sol_at_ph = sol * (1 + 10^(6.8-4.0)) = sol * (1 + 631) >> sol
    result = _make_result(drug_type="acid", pKa=4.0, solubility_mg_mL=0.01)
    assert result.solubility_at_ph_mg_mL > result.solubility_mg_mL


def test_base_drug_lower_sol_at_ph_high_pka():
    # pKa = 9.0: sol_at_ph = sol * (1 + 10^(9-6.8)) = sol * (1 + 158) >> intrinsic
    # Bases with high pKa are positively charged at pH 6.8 → high sol
    result = _make_result(drug_type="base", pKa=9.0, solubility_mg_mL=0.01)
    assert result.solubility_at_ph_mg_mL > result.solubility_mg_mL


# ---------------------------------------------------------------------------
# 12. Validation errors
# ---------------------------------------------------------------------------


def test_error_solubility_zero():
    with pytest.raises(ValueError, match="solubility"):
        _make_result(solubility_mg_mL=0.0)


def test_error_solubility_negative():
    with pytest.raises(ValueError, match="solubility"):
        _make_result(solubility_mg_mL=-1.0)


def test_error_dose_zero():
    with pytest.raises(ValueError, match="dose"):
        _make_result(dose_mg=0.0)


def test_error_dose_negative():
    with pytest.raises(ValueError, match="dose"):
        _make_result(dose_mg=-50.0)


def test_error_vod_zero():
    with pytest.raises(ValueError, match="volume"):
        _make_result(volume_of_distribution_mL=0.0)


def test_error_vod_negative():
    with pytest.raises(ValueError, match="volume"):
        _make_result(volume_of_distribution_mL=-100.0)


def test_error_invalid_drug_type():
    with pytest.raises(ValueError, match="drug_type"):
        _make_result(drug_type="zwitterion")


# ---------------------------------------------------------------------------
# 13. Notes nonempty
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = _make_result()
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# 14. Crystallization rate positive when SR > 1
# ---------------------------------------------------------------------------


def test_crystallization_rate_positive_when_sr_gt1():
    result = _make_result(solubility_mg_mL=0.1, dose_mg=100.0, volume_of_distribution_mL=100.0)
    if result.supersaturation_ratio > 1.0:
        assert result.crystallization_rate_per_h > 0.0


def test_t_cryst_finite_when_sr_gt1():
    result = _make_result(solubility_mg_mL=0.1, dose_mg=100.0, volume_of_distribution_mL=100.0)
    if result.supersaturation_ratio > 1.0:
        assert math.isfinite(result.time_to_crystallization_h)
