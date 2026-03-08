"""Tests for Phase 371 — Cytokine Release Syndrome Risk Predictor."""

import pytest

from omega_pbpk.risk.cytokine_release_risk import (
    CRSRiskResult,
    predict_crs_risk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_GRADES = {"grade_0", "grade_1", "grade_2", "grade_3", "grade_4"}
VALID_MONITORING = {"standard", "enhanced", "intensive"}


def _default_kwargs(**overrides):
    base = dict(
        drug_name="TestDrug",
        drug_class="monoclonal_ab",
        target_antigen="cd20",
        dose_mg_per_kg=1.0,
        t_cell_engagement=False,
        cytokine_amplification_factor=2.0,
        infusion_rate_mg_per_h=0.5,
        premedication=False,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_crs_risk_result():
    result = predict_crs_risk(**_default_kwargs())
    assert isinstance(result, CRSRiskResult)


# ---------------------------------------------------------------------------
# Score bounds
# ---------------------------------------------------------------------------


def test_crs_risk_score_in_range():
    result = predict_crs_risk(**_default_kwargs())
    assert 0.0 <= result.crs_risk_score <= 100.0


def test_crs_risk_score_capped_at_100():
    result = predict_crs_risk(
        **_default_kwargs(
            drug_class="car_t",
            target_antigen="cd3",
            dose_mg_per_kg=10.0,
            t_cell_engagement=True,
            cytokine_amplification_factor=10.0,
            infusion_rate_mg_per_h=5.0,
            premedication=False,
        )
    )
    assert result.crs_risk_score == 100.0


def test_crs_risk_score_minimum_zero():
    result = predict_crs_risk(
        **_default_kwargs(
            drug_class="adc",
            target_antigen="her2",
            dose_mg_per_kg=0.01,
            t_cell_engagement=False,
            cytokine_amplification_factor=1.0,
            infusion_rate_mg_per_h=0.05,
            premedication=True,
        )
    )
    assert result.crs_risk_score >= 0.0


# ---------------------------------------------------------------------------
# Grade predictions
# ---------------------------------------------------------------------------


def test_predicted_crs_grade_valid():
    result = predict_crs_risk(**_default_kwargs())
    assert result.predicted_crs_grade in VALID_GRADES


def test_car_t_high_grade():
    result = predict_crs_risk(
        **_default_kwargs(
            drug_class="car_t",
            target_antigen="cd19",
            dose_mg_per_kg=5.0,
            t_cell_engagement=True,
            cytokine_amplification_factor=8.0,
            infusion_rate_mg_per_h=2.0,
        )
    )
    assert result.predicted_crs_grade in {"grade_3", "grade_4"}


def test_low_risk_scenario_grade_0_or_1():
    result = predict_crs_risk(
        **_default_kwargs(
            drug_class="adc",
            target_antigen="her2",
            dose_mg_per_kg=0.1,
            t_cell_engagement=False,
            cytokine_amplification_factor=1.0,
            infusion_rate_mg_per_h=0.05,
            premedication=True,
        )
    )
    assert result.predicted_crs_grade in {"grade_0", "grade_1"}


# ---------------------------------------------------------------------------
# Comparative tests
# ---------------------------------------------------------------------------


def test_car_t_higher_risk_than_monoclonal_ab():
    car_t = predict_crs_risk(**_default_kwargs(drug_class="car_t"))
    mab = predict_crs_risk(**_default_kwargs(drug_class="monoclonal_ab"))
    assert car_t.crs_risk_score > mab.crs_risk_score


def test_t_cell_engagement_increases_risk():
    with_tce = predict_crs_risk(**_default_kwargs(t_cell_engagement=True))
    without_tce = predict_crs_risk(**_default_kwargs(t_cell_engagement=False))
    assert with_tce.crs_risk_score > without_tce.crs_risk_score


def test_premedication_reduces_score():
    no_premed = predict_crs_risk(**_default_kwargs(premedication=False))
    with_premed = predict_crs_risk(**_default_kwargs(premedication=True))
    assert with_premed.crs_risk_score < no_premed.crs_risk_score


def test_higher_dose_higher_score():
    low = predict_crs_risk(**_default_kwargs(dose_mg_per_kg=0.1))
    high = predict_crs_risk(**_default_kwargs(dose_mg_per_kg=4.0))
    assert high.crs_risk_score > low.crs_risk_score


def test_cd3_higher_than_her2():
    cd3 = predict_crs_risk(**_default_kwargs(target_antigen="cd3"))
    her2 = predict_crs_risk(**_default_kwargs(target_antigen="her2"))
    assert cd3.crs_risk_score > her2.crs_risk_score


def test_fast_infusion_higher_than_slow():
    slow = predict_crs_risk(**_default_kwargs(infusion_rate_mg_per_h=0.05))
    fast = predict_crs_risk(**_default_kwargs(infusion_rate_mg_per_h=2.0))
    assert fast.crs_risk_score > slow.crs_risk_score


# ---------------------------------------------------------------------------
# Monitoring frequency
# ---------------------------------------------------------------------------


def test_monitoring_frequency_valid():
    result = predict_crs_risk(**_default_kwargs())
    assert result.monitoring_frequency in VALID_MONITORING


def test_grade_0_standard_monitoring():
    result = predict_crs_risk(
        **_default_kwargs(
            drug_class="adc",
            target_antigen="her2",
            dose_mg_per_kg=0.05,
            t_cell_engagement=False,
            cytokine_amplification_factor=1.0,
            infusion_rate_mg_per_h=0.05,
            premedication=True,
        )
    )
    if result.predicted_crs_grade == "grade_0":
        assert result.monitoring_frequency == "standard"


def test_high_grade_intensive_monitoring():
    result = predict_crs_risk(
        **_default_kwargs(
            drug_class="car_t",
            target_antigen="cd3",
            t_cell_engagement=True,
            dose_mg_per_kg=5.0,
            cytokine_amplification_factor=10.0,
            infusion_rate_mg_per_h=2.0,
        )
    )
    assert result.monitoring_frequency == "intensive"


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------


def test_risk_factors_is_list():
    result = predict_crs_risk(**_default_kwargs())
    assert isinstance(result.risk_factors, list)


def test_mitigation_strategies_is_list():
    result = predict_crs_risk(**_default_kwargs())
    assert isinstance(result.mitigation_strategies, list)


def test_score_components_is_dict():
    result = predict_crs_risk(**_default_kwargs())
    assert isinstance(result.score_components, dict)


def test_score_components_contain_base():
    result = predict_crs_risk(**_default_kwargs())
    assert "base_class_risk" in result.score_components


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_drug_class():
    with pytest.raises(ValueError, match="drug_class"):
        predict_crs_risk(**_default_kwargs(drug_class="unknown_class"))


def test_invalid_antigen():
    with pytest.raises(ValueError, match="target_antigen"):
        predict_crs_risk(**_default_kwargs(target_antigen="cd99"))


def test_dose_zero():
    with pytest.raises(ValueError, match="dose_mg_per_kg"):
        predict_crs_risk(**_default_kwargs(dose_mg_per_kg=0.0))


def test_dose_negative():
    with pytest.raises(ValueError, match="dose_mg_per_kg"):
        predict_crs_risk(**_default_kwargs(dose_mg_per_kg=-1.0))


def test_infusion_rate_zero():
    with pytest.raises(ValueError, match="infusion_rate_mg_per_h"):
        predict_crs_risk(**_default_kwargs(infusion_rate_mg_per_h=0.0))


def test_cytokine_amplification_below_range():
    with pytest.raises(ValueError, match="cytokine_amplification_factor"):
        predict_crs_risk(**_default_kwargs(cytokine_amplification_factor=0.5))


def test_cytokine_amplification_above_range():
    with pytest.raises(ValueError, match="cytokine_amplification_factor"):
        predict_crs_risk(**_default_kwargs(cytokine_amplification_factor=11.0))
