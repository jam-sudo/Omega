"""Tests for bile acid enterohepatic circulation (EHC) PK model — Phase 1013."""

import math
import pytest

from omega_pbpk.core.bile_acid_enterohepatic_pk import (
    BileAcidEnterohepticResult,
    simulate_enterohepatic_circulation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_default(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        f_ehc=0.3,
        k_biliary_per_h=0.5,
        bile_lag_h=4.0,
        k_reabsorption_per_h=0.8,
        v_plasma_L=5.0,
        cl_plasma_L_per_h=3.0,
        route="oral",
        t_end_h=48.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_enterohepatic_circulation(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_returns_correct_type():
    result = _run_default()
    assert isinstance(result, BileAcidEnterohepticResult)


def test_drug_name_preserved():
    result = _run_default(drug_name="MyDrug")
    assert result.drug_name == "MyDrug"


def test_dose_preserved():
    result = _run_default(dose_mg=250.0)
    assert result.dose_mg == 250.0


# ---------------------------------------------------------------------------
# List fields shape
# ---------------------------------------------------------------------------

def test_list_lengths_consistent():
    result = _run_default(t_end_h=24.0, dt_h=0.1)
    n = len(result.times_h)
    assert len(result.c_plasma_mg_L) == n
    assert len(result.c_liver_mg_L) == n
    assert len(result.a_bile_mg) == n
    assert len(result.c_total_mg_L) == n


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def test_oral_initial_plasma_zero():
    result = _run_default(route="oral")
    assert result.c_plasma_mg_L[0] == 0.0


def test_iv_initial_plasma_positive():
    result = _run_default(route="iv")
    # C_plasma[0] = dose_mg / v_plasma_L
    expected = 100.0 / 5.0
    assert math.isclose(result.c_plasma_mg_L[0], expected, rel_tol=1e-9)


def test_bile_initial_zero():
    result = _run_default()
    assert result.a_bile_mg[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Peak metrics
# ---------------------------------------------------------------------------

def test_cmax_first_peak_positive():
    result = _run_default()
    assert result.cmax_first_peak_mg_L > 0.0


def test_cmax_second_peak_non_negative():
    result = _run_default()
    assert result.cmax_second_peak_mg_L >= 0.0


def test_tmax_first_positive():
    result = _run_default()
    assert result.tmax_first_h > 0.0


def test_tmax_second_after_first():
    result = _run_default()
    assert result.tmax_second_h > result.tmax_first_h


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------

def test_auc_positive():
    result = _run_default()
    assert result.auc_total_mg_h_per_L > 0.0


def test_ehc_second_peak_present_with_ehc():
    """With EHC enabled, a secondary peak should arise from bile reabsorption."""
    r_ehc = _run_default(f_ehc=0.5, bile_lag_h=4.0, t_end_h=48.0)
    # EHC creates a secondary peak; second tmax should be after first
    assert r_ehc.tmax_second_h > r_ehc.tmax_first_h


def test_ehc_auc_comparable_to_no_ehc():
    """EHC delays drug via bile but overall AUC over a long period is comparable (dose/CL).

    Both cases approach AUC = dose / CL = 100 / 3 ~ 33.3 mg*h/L over 48 h.
    """
    r_no_ehc = _run_default(f_ehc=0.0)
    r_ehc = _run_default(f_ehc=0.3)
    expected_auc = 100.0 / 3.0  # dose / CL
    # Both should be within 15% of theoretical AUC
    assert abs(r_no_ehc.auc_total_mg_h_per_L - expected_auc) / expected_auc < 0.15
    assert abs(r_ehc.auc_total_mg_h_per_L - expected_auc) / expected_auc < 0.15


# ---------------------------------------------------------------------------
# f_ehc attribute
# ---------------------------------------------------------------------------

def test_f_ehc_in_result():
    result = _run_default(f_ehc=0.25)
    assert result.f_ehc == pytest.approx(0.25)


def test_f_ehc_within_bounds():
    result = _run_default(f_ehc=0.5)
    assert 0.0 <= result.f_ehc <= 1.0


# ---------------------------------------------------------------------------
# EHC cycle count
# ---------------------------------------------------------------------------

def test_number_of_cycles_at_least_one():
    result = _run_default(bile_lag_h=4.0, t_end_h=24.0)
    assert result.number_of_cycles >= 1


def test_number_of_cycles_scales_with_time():
    r_short = _run_default(bile_lag_h=6.0, t_end_h=12.0)
    r_long = _run_default(bile_lag_h=6.0, t_end_h=48.0)
    assert r_long.number_of_cycles >= r_short.number_of_cycles


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_enterohepatic_circulation("X", dose_mg=0.0)


def test_invalid_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_enterohepatic_circulation("X", dose_mg=-10.0)


def test_invalid_f_ehc_above_one_raises():
    with pytest.raises(ValueError, match="f_ehc"):
        simulate_enterohepatic_circulation("X", dose_mg=100.0, f_ehc=1.5)


def test_invalid_f_ehc_negative_raises():
    with pytest.raises(ValueError, match="f_ehc"):
        simulate_enterohepatic_circulation("X", dose_mg=100.0, f_ehc=-0.1)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        simulate_enterohepatic_circulation("X", dose_mg=100.0, route="subcutaneous")
