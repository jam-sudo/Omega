"""Tests for Phase 1007 — cardiac_output_pk module."""

import pytest

from omega_pbpk.core.cardiac_output_pk import (
    CardiacOutputPKResult,
    simulate_cardiac_output_pk,
)


VALID_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cardiac_output_L_per_min=5.0,
)


def test_returns_result_type():
    result = simulate_cardiac_output_pk(**VALID_KWARGS)
    assert isinstance(result, CardiacOutputPKResult)


def test_cmax_positive():
    result = simulate_cardiac_output_pk(**VALID_KWARGS)
    assert result.cmax_plasma_mg_L > 0


def test_auc_positive():
    result = simulate_cardiac_output_pk(**VALID_KWARGS)
    assert result.auc_plasma_mg_h_per_L > 0


def test_hepatic_extraction_range():
    result = simulate_cardiac_output_pk(**VALID_KWARGS)
    assert 0.0 <= result.hepatic_extraction <= 1.0


def test_renal_extraction_range():
    result = simulate_cardiac_output_pk(**VALID_KWARGS)
    assert 0.0 <= result.renal_extraction <= 1.0


def test_total_body_clearance_positive():
    result = simulate_cardiac_output_pk(**VALID_KWARGS)
    assert result.total_body_clearance_L_per_h > 0


def test_t_half_positive():
    result = simulate_cardiac_output_pk(**VALID_KWARGS)
    assert result.t_half_h > 0


def test_heart_failure_impact_valid_values():
    result = simulate_cardiac_output_pk(**VALID_KWARGS)
    assert result.heart_failure_impact in {"none", "mild", "moderate", "severe"}


def test_heart_failure_none_for_normal_co():
    result = simulate_cardiac_output_pk(drug_name="Drug", dose_mg=50.0, cardiac_output_L_per_min=5.5)
    assert result.heart_failure_impact == "none"


def test_heart_failure_mild():
    result = simulate_cardiac_output_pk(drug_name="Drug", dose_mg=50.0, cardiac_output_L_per_min=4.0)
    assert result.heart_failure_impact == "mild"


def test_heart_failure_moderate():
    result = simulate_cardiac_output_pk(drug_name="Drug", dose_mg=50.0, cardiac_output_L_per_min=3.0)
    assert result.heart_failure_impact == "moderate"


def test_heart_failure_severe_for_low_co():
    result = simulate_cardiac_output_pk(drug_name="Drug", dose_mg=50.0, cardiac_output_L_per_min=2.0)
    assert result.heart_failure_impact == "severe"


def test_low_co_higher_auc_than_normal():
    """Reduced CO → slower clearance → higher AUC."""
    normal = simulate_cardiac_output_pk(drug_name="Drug", dose_mg=100.0, cardiac_output_L_per_min=5.0)
    low_co = simulate_cardiac_output_pk(drug_name="Drug", dose_mg=100.0, cardiac_output_L_per_min=2.0)
    assert low_co.auc_plasma_mg_h_per_L > normal.auc_plasma_mg_h_per_L


def test_iv_initial_plasma_concentration_nonzero():
    result = simulate_cardiac_output_pk(drug_name="Drug", dose_mg=100.0, route="iv")
    assert result.c_plasma_mg_L[0] > 0


def test_oral_initial_plasma_concentration_zero():
    result = simulate_cardiac_output_pk(drug_name="Drug", dose_mg=100.0, route="oral")
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0)


def test_invalid_dose_raises_value_error():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_cardiac_output_pk(drug_name="Drug", dose_mg=-10.0)


def test_invalid_co_raises_value_error():
    with pytest.raises(ValueError, match="cardiac_output_L_per_min"):
        simulate_cardiac_output_pk(drug_name="Drug", dose_mg=100.0, cardiac_output_L_per_min=0.0)


def test_invalid_route_raises_value_error():
    with pytest.raises(ValueError, match="route"):
        simulate_cardiac_output_pk(drug_name="Drug", dose_mg=100.0, route="rectal")


def test_time_array_length_matches_concentration():
    result = simulate_cardiac_output_pk(drug_name="Drug", dose_mg=100.0, t_end_h=12.0, dt_h=0.5)
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) == len(result.c_liver_mg_L)
    assert len(result.times_h) == len(result.c_kidney_mg_L)
    assert len(result.times_h) == len(result.c_muscle_mg_L)
