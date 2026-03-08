"""Tests for Phase 518 — Drug Distribution in Pleural Fluid."""

import pytest

from omega_pbpk.core.pleural_fluid_pk import (
    PleuralFluidPKResult,
    compare_pleural_ps,
    simulate_pleural_fluid_pk,
)

# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------

DEFAULT_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cl_L_per_h=5.0,
    vd_plasma_L=20.0,
)


def default_result(**overrides):
    kw = dict(DEFAULT_KWARGS)
    kw.update(overrides)
    return simulate_pleural_fluid_pk(**kw)


# ---------------------------------------------------------------------------
# Return type and field presence
# ---------------------------------------------------------------------------


def test_return_type():
    r = default_result()
    assert isinstance(r, PleuralFluidPKResult)


def test_fields_present():
    r = default_result()
    assert hasattr(r, "drug_name")
    assert hasattr(r, "dose_mg")
    assert hasattr(r, "ps_pleural_L_per_h")
    assert hasattr(r, "times_h")
    assert hasattr(r, "c_plasma_mg_L")
    assert hasattr(r, "c_pleural_mg_L")
    assert hasattr(r, "cmax_plasma")
    assert hasattr(r, "tmax_plasma_h")
    assert hasattr(r, "auc_plasma")
    assert hasattr(r, "cmax_pleural")
    assert hasattr(r, "tmax_pleural_h")
    assert hasattr(r, "auc_pleural")
    assert hasattr(r, "pleural_plasma_auc_ratio")
    assert hasattr(r, "pleural_penetration_pct")
    assert hasattr(r, "t_half_plasma_h")
    assert hasattr(r, "notes")


def test_drug_name_stored():
    r = default_result(drug_name="Meropenem")
    assert r.drug_name == "Meropenem"


def test_dose_stored():
    r = default_result(dose_mg=500.0)
    assert r.dose_mg == 500.0


def test_ps_stored():
    r = default_result(ps_pleural_L_per_h=0.01)
    assert r.ps_pleural_L_per_h == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_plasma_starts_at_dose_over_vd():
    r = default_result(dose_mg=100.0, vd_plasma_L=20.0)
    expected = 100.0 / 20.0
    assert r.c_plasma_mg_L[0] == pytest.approx(expected, rel=1e-6)


def test_pleural_starts_at_zero():
    r = default_result()
    assert r.c_pleural_mg_L[0] == 0.0


def test_times_start_at_zero():
    r = default_result()
    assert r.times_h[0] == 0.0


# ---------------------------------------------------------------------------
# Pleural dynamics
# ---------------------------------------------------------------------------


def test_pleural_rises_after_start():
    r = default_result()
    # Pleural should be higher at some mid-point than at t=0
    assert max(r.c_pleural_mg_L) > 0.0


def test_pleural_eventually_declines():
    r = default_result(t_end_h=48.0)
    concs = r.c_pleural_mg_L
    peak_idx = concs.index(max(concs))
    # After peak there should be lower values (decline)
    if peak_idx < len(concs) - 1:
        assert concs[-1] < concs[peak_idx]


def test_plasma_declines_monotonically_approx():
    # After IV bolus, plasma should generally decline
    r = default_result()
    assert r.c_plasma_mg_L[-1] < r.c_plasma_mg_L[0]


# ---------------------------------------------------------------------------
# AUC and PK metrics
# ---------------------------------------------------------------------------


def test_auc_plasma_positive():
    r = default_result()
    assert r.auc_plasma > 0.0


def test_auc_pleural_positive():
    r = default_result()
    assert r.auc_pleural > 0.0


def test_cmax_plasma_equals_initial():
    # For IV bolus, Cmax_plasma is the initial concentration
    r = default_result(dose_mg=100.0, vd_plasma_L=20.0)
    assert r.cmax_plasma == pytest.approx(100.0 / 20.0, rel=1e-6)


def test_tmax_plasma_at_zero():
    r = default_result()
    assert r.tmax_plasma_h == pytest.approx(0.0)


def test_cmax_pleural_positive():
    r = default_result()
    assert r.cmax_pleural > 0.0


def test_t_half_plasma_positive():
    r = default_result()
    assert r.t_half_plasma_h > 0.0


def test_times_lists_same_length():
    r = default_result()
    assert len(r.times_h) == len(r.c_plasma_mg_L) == len(r.c_pleural_mg_L)


# ---------------------------------------------------------------------------
# Penetration metrics
# ---------------------------------------------------------------------------


def test_pleural_penetration_pct_in_range():
    r = default_result()
    assert 0.0 <= r.pleural_penetration_pct <= 100.0


def test_pleural_plasma_auc_ratio_positive():
    r = default_result()
    assert r.pleural_plasma_auc_ratio > 0.0


def test_higher_ps_higher_pleural_penetration():
    r_low = default_result(ps_pleural_L_per_h=0.001)
    r_high = default_result(ps_pleural_L_per_h=0.1)
    assert r_high.pleural_penetration_pct > r_low.pleural_penetration_pct


def test_higher_ps_higher_auc_ratio():
    r_low = default_result(ps_pleural_L_per_h=0.001)
    r_high = default_result(ps_pleural_L_per_h=0.1)
    assert r_high.pleural_plasma_auc_ratio > r_low.pleural_plasma_auc_ratio


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


def test_dose_linearity_auc_plasma():
    r1 = default_result(dose_mg=100.0)
    r2 = default_result(dose_mg=200.0)
    assert r2.auc_plasma == pytest.approx(2.0 * r1.auc_plasma, rel=0.01)


def test_dose_linearity_auc_pleural():
    r1 = default_result(dose_mg=100.0)
    r2 = default_result(dose_mg=200.0)
    assert r2.auc_pleural == pytest.approx(2.0 * r1.auc_pleural, rel=0.01)


# ---------------------------------------------------------------------------
# compare_pleural_ps
# ---------------------------------------------------------------------------


def test_compare_pleural_ps_returns_list():
    results = compare_pleural_ps("Drug", 100.0, 5.0, 20.0, ps_values=[0.001, 0.01, 0.1])
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_pleural_ps_sorted_descending():
    results = compare_pleural_ps("Drug", 100.0, 5.0, 20.0, ps_values=[0.1, 0.001, 0.01])
    ratios = [r.pleural_plasma_auc_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_compare_pleural_ps_single_value():
    results = compare_pleural_ps("Drug", 100.0, 5.0, 20.0, ps_values=[0.005])
    assert len(results) == 1
    assert isinstance(results[0], PleuralFluidPKResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_pleural_fluid_pk("Drug", dose_mg=0.0, cl_L_per_h=5.0, vd_plasma_L=20.0)


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_pleural_fluid_pk("Drug", dose_mg=-10.0, cl_L_per_h=5.0, vd_plasma_L=20.0)


def test_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_pleural_fluid_pk("Drug", dose_mg=100.0, cl_L_per_h=0.0, vd_plasma_L=20.0)


def test_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_plasma"):
        simulate_pleural_fluid_pk("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_plasma_L=0.0)


def test_ps_zero_raises():
    with pytest.raises(ValueError, match="ps_pleural"):
        simulate_pleural_fluid_pk(
            "Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_plasma_L=20.0, ps_pleural_L_per_h=0.0
        )
