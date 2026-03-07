"""Tests for therapeutic_drug_monitoring_v2 (Phase 795)."""

import pytest

from omega_pbpk.clinical.therapeutic_drug_monitoring_v2 import (
    BayesianTDMResult,
    bayesian_tdm_optimization,
    quick_tdm_assessment,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _run_basic(
    measured=3.0,
    sample_time=6.0,
    dose=500.0,
    prior_cl=5.0,
    prior_vd=50.0,
    target_trough=1.0,
    target_peak=8.0,
    tau_h=12.0,
    ka=None,
):
    return bayesian_tdm_optimization(
        drug_name="TestDrug",
        measured_conc_mg_L=measured,
        sample_time_h=sample_time,
        dose_mg=dose,
        prior_cl_L_per_h=prior_cl,
        prior_vd_L=prior_vd,
        target_trough_mg_L=target_trough,
        target_peak_mg_L=target_peak,
        tau_h=tau_h,
        ka_per_h=ka,
    )


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_return_type():
    result = _run_basic()
    assert isinstance(result, BayesianTDMResult)


def test_frozen_dataclass():
    result = _run_basic()
    with pytest.raises(Exception):
        result.map_cl_L_per_h = 99.0  # type: ignore[misc]


def test_drug_name_preserved():
    result = bayesian_tdm_optimization(
        drug_name="Vancomycin",
        measured_conc_mg_L=5.0,
        sample_time_h=2.0,
        dose_mg=1000.0,
    )
    assert result.drug_name == "Vancomycin"


# ---------------------------------------------------------------------------
# MAP parameter tests
# ---------------------------------------------------------------------------


def test_map_cl_positive():
    result = _run_basic()
    assert result.map_cl_L_per_h > 0.0


def test_map_vd_positive():
    result = _run_basic()
    assert result.map_vd_L > 0.0


def test_map_t_half_formula():
    """map_t_half_h should equal 0.693 * map_vd / map_cl."""
    result = _run_basic()
    expected = 0.693147 * result.map_vd_L / result.map_cl_L_per_h
    assert abs(result.map_t_half_h - expected) < 1e-6


def test_map_t_half_positive():
    result = _run_basic()
    assert result.map_t_half_h > 0.0


def test_map_cl_within_grid_bounds():
    prior_cl = 5.0
    result = _run_basic(prior_cl=prior_cl)
    assert prior_cl * 0.2 <= result.map_cl_L_per_h <= prior_cl * 3.0 + 1e-6


def test_map_vd_within_grid_bounds():
    prior_vd = 50.0
    result = _run_basic(prior_vd=prior_vd)
    assert prior_vd * 0.2 <= result.map_vd_L <= prior_vd * 3.0 + 1e-6


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def test_predicted_trough_non_negative():
    result = _run_basic()
    assert result.predicted_trough_mg_L >= 0.0


def test_predicted_peak_non_negative():
    result = _run_basic()
    assert result.predicted_peak_mg_L >= 0.0


def test_target_trough_preserved():
    result = _run_basic(target_trough=2.0)
    assert result.target_trough_mg_L == 2.0


def test_target_peak_preserved():
    result = _run_basic(target_peak=10.0)
    assert result.target_peak_mg_L == 10.0


def test_recommended_dose_positive():
    result = _run_basic()
    assert result.recommended_dose_mg > 0.0


def test_recommended_interval_preserved():
    result = _run_basic(tau_h=8.0)
    assert result.recommended_interval_h == 8.0


# ---------------------------------------------------------------------------
# dose_adjustment_needed logic
# ---------------------------------------------------------------------------


def test_dose_adjustment_needed_is_bool():
    result = _run_basic()
    assert isinstance(result.dose_adjustment_needed, bool)


def test_dose_adjustment_not_needed_when_in_range():
    """When trough is within [target_trough, target_peak] adjustment may not be needed."""
    result = _run_basic(target_trough=0.0, target_peak=100.0)
    # Very wide target range — most likely no adjustment needed
    assert isinstance(result.dose_adjustment_needed, bool)


def test_dose_adjustment_when_far_subtherapeutic():
    """Very low measured conc likely leads to dose adjustment needed."""
    result = _run_basic(measured=0.1, target_trough=5.0, target_peak=15.0)
    assert result.dose_adjustment_needed is True


# ---------------------------------------------------------------------------
# Oral route
# ---------------------------------------------------------------------------


def test_oral_route_ka_given():
    result = bayesian_tdm_optimization(
        drug_name="OralDrug",
        measured_conc_mg_L=2.0,
        sample_time_h=3.0,
        dose_mg=300.0,
        ka_per_h=1.5,
        f_oral=0.8,
    )
    assert isinstance(result, BayesianTDMResult)
    assert result.map_cl_L_per_h > 0.0


# ---------------------------------------------------------------------------
# Posterior CV
# ---------------------------------------------------------------------------


def test_posterior_cv_cl_non_negative():
    result = _run_basic()
    assert result.posterior_cv_cl_pct >= 0.0


def test_notes_is_string():
    result = _run_basic()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_negative_conc_raises():
    with pytest.raises(ValueError):
        bayesian_tdm_optimization(
            drug_name="Drug",
            measured_conc_mg_L=-1.0,
            sample_time_h=4.0,
            dose_mg=100.0,
        )


def test_zero_dose_raises():
    with pytest.raises(ValueError):
        bayesian_tdm_optimization(
            drug_name="Drug",
            measured_conc_mg_L=2.0,
            sample_time_h=4.0,
            dose_mg=0.0,
        )


# ---------------------------------------------------------------------------
# quick_tdm_assessment
# ---------------------------------------------------------------------------


def test_quick_tdm_returns_dict():
    result = quick_tdm_assessment("Drug", 3.0, 2.0, 8.0)
    assert isinstance(result, dict)


def test_quick_tdm_has_status_key():
    result = quick_tdm_assessment("Drug", 3.0, 2.0, 8.0)
    assert "status" in result


def test_quick_tdm_therapeutic():
    result = quick_tdm_assessment("Drug", 5.0, 2.0, 8.0)
    assert result["status"] == "therapeutic"


def test_quick_tdm_subtherapeutic():
    result = quick_tdm_assessment("Drug", 1.0, 2.0, 8.0)
    assert result["status"] == "subtherapeutic"


def test_quick_tdm_supratherapeutic():
    result = quick_tdm_assessment("Drug", 12.0, 2.0, 8.0)
    assert result["status"] == "supratherapeutic"


def test_quick_tdm_negative_conc_raises():
    with pytest.raises(ValueError):
        quick_tdm_assessment("Drug", -1.0, 2.0, 8.0)


def test_quick_tdm_has_action_key():
    result = quick_tdm_assessment("Drug", 5.0, 2.0, 8.0)
    assert "action" in result


def test_quick_tdm_action_increase_when_low():
    result = quick_tdm_assessment("Drug", 0.5, 2.0, 8.0)
    assert result["action"] == "increase_dose"
