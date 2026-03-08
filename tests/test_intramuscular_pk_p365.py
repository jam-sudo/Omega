"""Tests for Phase 365: Intramuscular Drug Absorption Model (3-compartment)."""

import pytest

from omega_pbpk.core.intramuscular_pk_p365 import (
    IntramuscularPKResult365,
    simulate_im_pk_p365,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_result(**kwargs):
    params = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        release_model="first_order",
        k_release_per_h=0.5,
        k_abs_per_h=1.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        t_end_h=24.0,
        dt_h=0.1,
    )
    params.update(kwargs)
    return simulate_im_pk_p365(**params)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = default_result()
    assert isinstance(result, IntramuscularPKResult365)


# ---------------------------------------------------------------------------
# Array lengths
# ---------------------------------------------------------------------------


def test_array_lengths_equal():
    result = default_result()
    n = len(result.times_h)
    assert n > 0
    assert len(result.c_depot_mg_mL) == n
    assert len(result.c_muscle_mg_mL) == n
    assert len(result.c_systemic_mg_L) == n


def test_times_start_at_zero():
    result = default_result()
    assert result.times_h[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Depot: starts at dose/v_depot, decreases monotonically
# ---------------------------------------------------------------------------


def test_depot_initial_concentration():
    dose = 100.0
    v_depot = 2.0
    result = simulate_im_pk_p365(
        drug_name="X",
        dose_mg=dose,
        v_depot_mL=v_depot,
        t_end_h=24.0,
        dt_h=0.1,
    )
    assert result.c_depot_mg_mL[0] == pytest.approx(dose / v_depot)


def test_depot_decreases_over_time():
    result = default_result()
    # Depot should be non-increasing
    for i in range(1, len(result.c_depot_mg_mL)):
        assert result.c_depot_mg_mL[i] <= result.c_depot_mg_mL[i - 1] + 1e-10


# ---------------------------------------------------------------------------
# Systemic: starts at 0, rises to positive Cmax
# ---------------------------------------------------------------------------


def test_systemic_starts_at_zero():
    result = default_result()
    assert result.c_systemic_mg_L[0] == pytest.approx(0.0)


def test_cmax_positive():
    result = default_result()
    assert result.cmax_systemic_mg_L > 0.0


def test_tmax_positive():
    result = default_result()
    assert result.tmax_systemic_h > 0.0


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------


def test_auc_positive():
    result = default_result()
    assert result.auc_systemic_mg_h_per_L > 0.0


def test_auc_increases_with_dose():
    r1 = default_result(dose_mg=100.0)
    r2 = default_result(dose_mg=200.0)
    assert r2.auc_systemic_mg_h_per_L > r1.auc_systemic_mg_h_per_L


# ---------------------------------------------------------------------------
# flip_flop_pk is bool
# ---------------------------------------------------------------------------


def test_flip_flop_is_bool():
    result = default_result()
    assert isinstance(result.flip_flop_pk, bool)


def test_flip_flop_true_for_slow_release():
    # Very slow release → absorption rate-limited → flip-flop
    result = simulate_im_pk_p365(
        drug_name="SlowDrug",
        dose_mg=100.0,
        release_model="first_order",
        k_release_per_h=0.02,  # very slow release
        k_abs_per_h=10.0,  # fast absorption
        cl_sys_L_per_h=5.0,
        vd_sys_L=10.0,
        t_end_h=120.0,
        dt_h=0.1,
    )
    # t_half_sys = 0.693*10/5 ≈ 1.4 h; slow release → tmax >> 2.8 h
    assert result.flip_flop_pk is True


# ---------------------------------------------------------------------------
# Zero-order: sustained systemic concentration
# ---------------------------------------------------------------------------


def test_zero_order_release_model():
    result = simulate_im_pk_p365(
        drug_name="OilDrug",
        dose_mg=100.0,
        release_model="zero_order",
        release_rate_mg_per_h=5.0,
        k_abs_per_h=1.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        t_end_h=24.0,
        dt_h=0.1,
    )
    assert result.release_model == "zero_order"
    assert result.cmax_systemic_mg_L > 0.0
    assert result.auc_systemic_mg_h_per_L > 0.0


def test_zero_order_sustained_vs_first_order():
    """Zero-order should produce a later or similarly timed Cmax than first_order."""
    fo = simulate_im_pk_p365(
        drug_name="Drug",
        dose_mg=100.0,
        release_model="first_order",
        k_release_per_h=2.0,
        k_abs_per_h=2.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        t_end_h=24.0,
        dt_h=0.05,
    )
    zo = simulate_im_pk_p365(
        drug_name="Drug",
        dose_mg=100.0,
        release_model="zero_order",
        release_rate_mg_per_h=2.0,
        k_abs_per_h=2.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        t_end_h=24.0,
        dt_h=0.05,
    )
    # Zero-order should spread out the AUC and have a later tmax
    assert zo.tmax_systemic_h >= fo.tmax_systemic_h


# ---------------------------------------------------------------------------
# First-order: faster initial release than zero-order at same nominal rate
# ---------------------------------------------------------------------------


def test_first_order_faster_initial_rise():
    """First-order with high k_release has rapid early rise."""
    fo = simulate_im_pk_p365(
        drug_name="Drug",
        dose_mg=100.0,
        release_model="first_order",
        k_release_per_h=5.0,
        k_abs_per_h=5.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        t_end_h=6.0,
        dt_h=0.05,
    )
    zo = simulate_im_pk_p365(
        drug_name="Drug",
        dose_mg=100.0,
        release_model="zero_order",
        release_rate_mg_per_h=1.0,  # slow steady rate
        k_abs_per_h=5.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        t_end_h=6.0,
        dt_h=0.05,
    )
    # First-order with high k should produce an earlier tmax
    assert fo.tmax_systemic_h <= zo.tmax_systemic_h


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


def test_dose_linearity_auc():
    r1 = default_result(dose_mg=50.0)
    r2 = default_result(dose_mg=100.0)
    ratio = r2.auc_systemic_mg_h_per_L / r1.auc_systemic_mg_h_per_L
    assert ratio == pytest.approx(2.0, rel=0.05)


def test_dose_linearity_cmax():
    r1 = default_result(dose_mg=50.0)
    r2 = default_result(dose_mg=100.0)
    ratio = r2.cmax_systemic_mg_L / r1.cmax_systemic_mg_L
    assert ratio == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# Bioavailability
# ---------------------------------------------------------------------------


def test_bioavailability_in_range():
    result = default_result(t_end_h=48.0)
    assert 0.0 <= result.bioavailability_pct <= 100.0


def test_bioavailability_approaches_100_long_simulation():
    result = simulate_im_pk_p365(
        drug_name="Drug",
        dose_mg=100.0,
        release_model="first_order",
        k_release_per_h=1.0,
        k_abs_per_h=2.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        t_end_h=120.0,
        dt_h=0.1,
    )
    assert result.bioavailability_pct > 80.0


# ---------------------------------------------------------------------------
# release_model field stored in result
# ---------------------------------------------------------------------------


def test_release_model_stored_first_order():
    result = default_result(release_model="first_order")
    assert result.release_model == "first_order"


def test_release_model_stored_zero_order():
    result = default_result(release_model="zero_order")
    assert result.release_model == "zero_order"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_error_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_im_pk_p365(drug_name="X", dose_mg=0.0)


def test_error_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_im_pk_p365(drug_name="X", dose_mg=-10.0)


def test_error_cl_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_im_pk_p365(drug_name="X", dose_mg=100.0, cl_sys_L_per_h=0.0)


def test_error_vd_zero():
    with pytest.raises(ValueError, match="vd_sys_L"):
        simulate_im_pk_p365(drug_name="X", dose_mg=100.0, vd_sys_L=0.0)


def test_error_invalid_release_model():
    with pytest.raises(ValueError, match="release_model"):
        simulate_im_pk_p365(drug_name="X", dose_mg=100.0, release_model="invalid_mode")
