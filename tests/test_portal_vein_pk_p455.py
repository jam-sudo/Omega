"""Tests for Phase 455 — Portal Vein PK Model (portal_vein_pk_p455)."""

import pytest

from omega_pbpk.core.portal_vein_pk_p455 import (
    PortalVeinPKResult,
    simulate_portal_vein_pk,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_return_type():
    result = simulate_portal_vein_pk("DrugA", dose_mg=100.0)
    assert isinstance(result, PortalVeinPKResult)


def test_result_fields_populated():
    result = simulate_portal_vein_pk("DrugA", dose_mg=100.0)
    assert result.drug_name == "DrugA"
    assert result.dose_mg == 100.0
    assert len(result.times_h) > 0
    assert len(result.c_portal_mg_L) == len(result.times_h)
    assert len(result.c_systemic_mg_L) == len(result.times_h)


def test_time_starts_at_zero():
    result = simulate_portal_vein_pk("DrugA", dose_mg=100.0)
    assert result.times_h[0] == pytest.approx(0.0)


def test_initial_concentrations_are_zero():
    result = simulate_portal_vein_pk("DrugA", dose_mg=100.0)
    # At t=0, nothing absorbed yet
    assert result.c_portal_mg_L[0] == pytest.approx(0.0)
    assert result.c_systemic_mg_L[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Physiological expectations
# ---------------------------------------------------------------------------


def test_portal_peaks_before_systemic():
    """Portal vein concentration should peak earlier than systemic."""
    result = simulate_portal_vein_pk("DrugA", dose_mg=100.0, ka_per_h=2.0, cl_hepatic_L_per_h=20.0)
    assert result.tmax_portal_h <= result.tmax_systemic_h


def test_portal_cmax_greater_than_systemic_cmax():
    """Portal first-pass concentration should exceed systemic concentration."""
    result = simulate_portal_vein_pk(
        "DrugA", dose_mg=100.0, ka_per_h=1.0, cl_hepatic_L_per_h=40.0, vd_portal_L=2.0
    )
    assert result.cmax_portal_mg_L > result.cmax_systemic_mg_L


def test_portal_to_systemic_ratio_greater_than_one():
    result = simulate_portal_vein_pk("DrugA", dose_mg=200.0, cl_hepatic_L_per_h=30.0)
    assert result.portal_to_systemic_cmax_ratio > 1.0


def test_systemic_auc_positive():
    result = simulate_portal_vein_pk("DrugA", dose_mg=100.0)
    assert result.auc_systemic_mg_h_per_L > 0.0


def test_portal_auc_positive():
    result = simulate_portal_vein_pk("DrugA", dose_mg=100.0)
    assert result.auc_portal_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


def test_dose_doubling_doubles_portal_auc():
    r1 = simulate_portal_vein_pk("DrugA", dose_mg=100.0)
    r2 = simulate_portal_vein_pk("DrugA", dose_mg=200.0)
    ratio = r2.auc_portal_mg_h_per_L / r1.auc_portal_mg_h_per_L
    assert ratio == pytest.approx(2.0, rel=0.05)


def test_dose_doubling_doubles_systemic_auc():
    r1 = simulate_portal_vein_pk("DrugA", dose_mg=100.0)
    r2 = simulate_portal_vein_pk("DrugA", dose_mg=200.0)
    ratio = r2.auc_systemic_mg_h_per_L / r1.auc_systemic_mg_h_per_L
    assert ratio == pytest.approx(2.0, rel=0.05)


def test_dose_doubling_doubles_portal_cmax():
    r1 = simulate_portal_vein_pk("DrugA", dose_mg=100.0)
    r2 = simulate_portal_vein_pk("DrugA", dose_mg=200.0)
    ratio = r2.cmax_portal_mg_L / r1.cmax_portal_mg_L
    assert ratio == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# Parameter effects
# ---------------------------------------------------------------------------


def test_higher_ka_earlier_tmax_portal():
    """Faster absorption => earlier portal peak."""
    r_slow = simulate_portal_vein_pk("DrugA", dose_mg=100.0, ka_per_h=0.5)
    r_fast = simulate_portal_vein_pk("DrugA", dose_mg=100.0, ka_per_h=3.0)
    assert r_fast.tmax_portal_h < r_slow.tmax_portal_h


def test_higher_cl_hepatic_lower_systemic_cmax():
    """Higher hepatic clearance => more first-pass extraction => lower systemic Cmax."""
    r_low = simulate_portal_vein_pk("DrugA", dose_mg=100.0, cl_hepatic_L_per_h=5.0)
    r_high = simulate_portal_vein_pk("DrugA", dose_mg=100.0, cl_hepatic_L_per_h=60.0)
    assert r_high.cmax_systemic_mg_L < r_low.cmax_systemic_mg_L


def test_higher_cl_hepatic_lower_systemic_auc():
    r_low = simulate_portal_vein_pk("DrugA", dose_mg=100.0, cl_hepatic_L_per_h=5.0)
    r_high = simulate_portal_vein_pk("DrugA", dose_mg=100.0, cl_hepatic_L_per_h=60.0)
    assert r_high.auc_systemic_mg_h_per_L < r_low.auc_systemic_mg_h_per_L


def test_f_hepatic_between_zero_and_one():
    result = simulate_portal_vein_pk("DrugA", dose_mg=100.0)
    assert 0.0 < result.f_hepatic < 1.0


def test_f_hepatic_decreases_with_higher_cl_hepatic():
    r_low = simulate_portal_vein_pk("DrugA", dose_mg=100.0, cl_hepatic_L_per_h=5.0)
    r_high = simulate_portal_vein_pk("DrugA", dose_mg=100.0, cl_hepatic_L_per_h=70.0)
    assert r_high.f_hepatic < r_low.f_hepatic


def test_lower_f_gut_lower_portal_cmax():
    r_full = simulate_portal_vein_pk("DrugA", dose_mg=100.0, f_gut=1.0)
    r_partial = simulate_portal_vein_pk("DrugA", dose_mg=100.0, f_gut=0.3)
    assert r_partial.cmax_portal_mg_L < r_full.cmax_portal_mg_L


# ---------------------------------------------------------------------------
# Cmax and tmax fields match arrays
# ---------------------------------------------------------------------------


def test_cmax_portal_equals_max_array():
    result = simulate_portal_vein_pk("DrugA", dose_mg=100.0)
    assert result.cmax_portal_mg_L == pytest.approx(max(result.c_portal_mg_L))


def test_cmax_systemic_equals_max_array():
    result = simulate_portal_vein_pk("DrugA", dose_mg=100.0)
    assert result.cmax_systemic_mg_L == pytest.approx(max(result.c_systemic_mg_L))


def test_notes_is_string():
    result = simulate_portal_vein_pk("DrugA", dose_mg=100.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_portal_vein_pk("DrugA", dose_mg=0.0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_portal_vein_pk("DrugA", dose_mg=-10.0)


def test_invalid_ka_zero():
    with pytest.raises(ValueError, match="ka_per_h"):
        simulate_portal_vein_pk("DrugA", dose_mg=100.0, ka_per_h=0.0)


def test_invalid_cl_hepatic():
    with pytest.raises(ValueError, match="cl_hepatic"):
        simulate_portal_vein_pk("DrugA", dose_mg=100.0, cl_hepatic_L_per_h=0.0)


def test_invalid_vd_portal():
    with pytest.raises(ValueError, match="vd_portal"):
        simulate_portal_vein_pk("DrugA", dose_mg=100.0, vd_portal_L=0.0)


def test_invalid_vd_sys():
    with pytest.raises(ValueError, match="vd_sys"):
        simulate_portal_vein_pk("DrugA", dose_mg=100.0, vd_sys_L=0.0)


def test_invalid_f_gut_zero():
    with pytest.raises(ValueError, match="f_gut"):
        simulate_portal_vein_pk("DrugA", dose_mg=100.0, f_gut=0.0)


def test_invalid_f_gut_greater_than_one():
    with pytest.raises(ValueError, match="f_gut"):
        simulate_portal_vein_pk("DrugA", dose_mg=100.0, f_gut=1.5)
