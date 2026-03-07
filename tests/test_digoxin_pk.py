"""Tests for Phase 983 — Digoxin PK model."""

import pytest

from omega_pbpk.clinical.digoxin_pk import DigoxinPKResult, simulate_digoxin_pk


def test_returns_digoxin_pk_result():
    result = simulate_digoxin_pk()
    assert isinstance(result, DigoxinPKResult)


def test_drug_name():
    result = simulate_digoxin_pk()
    assert result.drug_name == "Digoxin"


def test_cmax_positive():
    result = simulate_digoxin_pk()
    assert result.cmax_ng_mL > 0


def test_c_at_6h_positive():
    result = simulate_digoxin_pk()
    assert result.c_at_6h_ng_mL > 0


def test_auc_positive():
    result = simulate_digoxin_pk()
    assert result.auc_ng_h_per_mL > 0


def test_within_therapeutic_range_is_bool():
    result = simulate_digoxin_pk()
    assert isinstance(result.within_therapeutic_range, bool)


def test_toxicity_risk_valid():
    result = simulate_digoxin_pk()
    assert result.toxicity_risk in {"safe", "borderline", "toxic"}


def test_dose_recommendation_valid():
    result = simulate_digoxin_pk()
    assert result.dose_recommendation in {"decrease", "maintain", "increase"}


def test_times_length_matches_concentrations():
    result = simulate_digoxin_pk()
    assert len(result.times_h) == len(result.c_plasma_ng_mL)
    assert len(result.times_h) == len(result.c_tissue_ng_mL)


def test_reduced_gfr_higher_c_at_6h():
    """Lower GFR means less renal elimination -> higher exposure."""
    normal = simulate_digoxin_pk(gfr_mL_min=80.0)
    reduced = simulate_digoxin_pk(gfr_mL_min=20.0)
    assert reduced.c_at_6h_ng_mL > normal.c_at_6h_ng_mL


def test_heavier_patient_lower_concentration():
    """Larger Vd in heavier patient -> lower concentration per mg."""
    light = simulate_digoxin_pk(patient_weight_kg=50.0)
    heavy = simulate_digoxin_pk(patient_weight_kg=100.0)
    assert heavy.c_at_6h_ng_mL < light.c_at_6h_ng_mL


def test_very_low_gfr_toxicity_or_out_of_range():
    """Very low GFR with standard dose should produce elevated levels."""
    result = simulate_digoxin_pk(gfr_mL_min=5.0, dose_mg=0.25)
    assert result.toxicity_risk == "toxic" or not result.within_therapeutic_range


def test_standard_dose_normal_gfr_therapeutic_range():
    """Standard dose (0.25 mg) with normal GFR should be in range."""
    result = simulate_digoxin_pk(dose_mg=0.25, gfr_mL_min=80.0)
    assert result.within_therapeutic_range is True


def test_iv_higher_initial_cmax_than_oral():
    """IV bolus bypasses absorption phase so initial Cmax is higher."""
    oral = simulate_digoxin_pk(route="oral")
    iv = simulate_digoxin_pk(route="iv")
    assert iv.cmax_ng_mL > oral.cmax_ng_mL


def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        simulate_digoxin_pk(dose_mg=-0.1)


def test_zero_dose_raises():
    with pytest.raises(ValueError):
        simulate_digoxin_pk(dose_mg=0.0)


def test_invalid_gfr_raises():
    with pytest.raises(ValueError):
        simulate_digoxin_pk(gfr_mL_min=-10.0)


def test_zero_gfr_raises():
    with pytest.raises(ValueError):
        simulate_digoxin_pk(gfr_mL_min=0.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError):
        simulate_digoxin_pk(route="intramuscular")


def test_tmax_within_simulation_time():
    result = simulate_digoxin_pk(t_end_h=72.0)
    assert 0.0 <= result.tmax_h <= 72.0


def test_high_dose_toxic():
    """Very high dose should produce toxic levels."""
    result = simulate_digoxin_pk(dose_mg=2.0)
    assert result.toxicity_risk == "toxic" or result.c_at_6h_ng_mL > 2.0


def test_very_low_dose_increase_recommended():
    """Very low dose should trigger increase recommendation."""
    result = simulate_digoxin_pk(dose_mg=0.01)
    assert result.dose_recommendation == "increase"
