"""Tests for peritoneal dialysis PK model — Phase 1135."""

import math
import pytest

from omega_pbpk.core.peritoneal_dialysis_pk import (
    PeritonealDialysisPKResult,
    simulate_peritoneal_dialysis_pk,
    _koa,
    _mw_class,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=500.0,
        mw_Da=250.0,
        fu_plasma=0.5,
        cl_residual_L_per_h=1.0,
        vd_L=30.0,
        dwell_time_h=4.0,
        n_exchanges=4,
        t_end_h=24.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_peritoneal_dialysis_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type and basic fields
# ---------------------------------------------------------------------------

def test_returns_correct_type():
    result = _run()
    assert isinstance(result, PeritonealDialysisPKResult)


def test_drug_name_preserved():
    result = _run(drug_name="Vancomycin")
    assert result.drug_name == "Vancomycin"


def test_dose_preserved():
    result = _run(dose_mg=1000.0)
    assert result.dose_mg == pytest.approx(1000.0)


def test_mw_preserved():
    result = _run(mw_Da=300.0)
    assert result.mw_Da == pytest.approx(300.0)


def test_fu_preserved():
    result = _run(fu_plasma=0.8)
    assert result.fu_plasma == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Time series properties
# ---------------------------------------------------------------------------

def test_times_start_at_zero():
    result = _run()
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_end_near_t_end():
    result = _run(t_end_h=24.0, dt_h=0.1)
    assert result.times_h[-1] == pytest.approx(24.0, abs=0.2)


def test_c_plasma_starts_at_dose_over_vd():
    result = _run(dose_mg=500.0, vd_L=30.0)
    expected = 500.0 / 30.0
    assert result.c_plasma_mg_L[0] == pytest.approx(expected)


def test_c_dialysate_starts_at_zero():
    result = _run()
    assert result.c_dialysate_mg_L[0] == pytest.approx(0.0)


def test_c_plasma_decreases_over_time():
    result = _run()
    # Compare first vs last
    assert result.c_plasma_mg_L[-1] < result.c_plasma_mg_L[0]


def test_c_dialysate_initially_rises():
    result = _run()
    # Dialysate should be higher at some point than at t=0
    assert max(result.c_dialysate_mg_L) > 0.0


def test_list_lengths_equal():
    result = _run()
    assert len(result.times_h) == len(result.c_plasma_mg_L) == len(result.c_dialysate_mg_L)


# ---------------------------------------------------------------------------
# Derived PK metrics
# ---------------------------------------------------------------------------

def test_cmax_equals_initial_concentration():
    # Bolus IV: peak is at t=0
    result = _run(dose_mg=500.0, vd_L=30.0)
    assert result.cmax_plasma_mg_L == pytest.approx(500.0 / 30.0, rel=0.01)


def test_auc_positive():
    result = _run()
    assert result.auc_plasma_mg_h_per_L > 0.0


def test_fraction_removed_in_unit_interval():
    result = _run()
    assert 0.0 <= result.fraction_removed_by_pd <= 1.0


def test_pd_clearance_positive():
    result = _run()
    assert result.pd_clearance_L_per_h >= 0.0


# ---------------------------------------------------------------------------
# MW effects
# ---------------------------------------------------------------------------

def test_small_mw_higher_koa_than_large():
    koa_small = _koa(100.0)
    koa_large = _koa(50000.0)
    assert koa_small > koa_large


def test_small_mw_more_removal():
    r_small = _run(mw_Da=100.0, fu_plasma=0.9)
    r_large = _run(mw_Da=50000.0, fu_plasma=0.9)
    assert r_small.fraction_removed_by_pd > r_large.fraction_removed_by_pd


def test_mw_class_small():
    result = _run(mw_Da=200.0)
    assert result.mw_class == "small"


def test_mw_class_medium():
    result = _run(mw_Da=1000.0)
    assert result.mw_class == "medium"


def test_mw_class_large():
    result = _run(mw_Da=10000.0)
    assert result.mw_class == "large"


# ---------------------------------------------------------------------------
# fu_plasma effects
# ---------------------------------------------------------------------------

def test_higher_fu_more_removal():
    r_low = _run(fu_plasma=0.1, mw_Da=150.0)
    r_high = _run(fu_plasma=0.9, mw_Da=150.0)
    assert r_high.fraction_removed_by_pd > r_low.fraction_removed_by_pd


# ---------------------------------------------------------------------------
# Dose supplement flag
# ---------------------------------------------------------------------------

def test_dose_supplement_needed_when_high_removal():
    # High fu, small MW → high removal
    result = _run(mw_Da=100.0, fu_plasma=0.99, cl_residual_L_per_h=0.1)
    if result.fraction_removed_by_pd > 0.25:
        assert result.dose_supplement_needed is True


def test_dose_supplement_not_needed_when_low_removal():
    # Large MW, low fu → minimal PD removal
    result = _run(mw_Da=100000.0, fu_plasma=0.01)
    if result.fraction_removed_by_pd <= 0.25:
        assert result.dose_supplement_needed is False


def test_dose_supplement_flag_consistent_with_fraction():
    result = _run()
    if result.fraction_removed_by_pd > 0.25:
        assert result.dose_supplement_needed is True
    else:
        assert result.dose_supplement_needed is False


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=0.0)


def test_invalid_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=-100.0)


def test_invalid_cl_residual_negative_raises():
    with pytest.raises(ValueError, match="cl_residual"):
        _run(cl_residual_L_per_h=-1.0)


def test_invalid_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_L"):
        _run(vd_L=0.0)


def test_invalid_mw_zero_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        _run(mw_Da=0.0)


def test_invalid_fu_zero_raises():
    with pytest.raises(ValueError, match="fu_plasma"):
        _run(fu_plasma=0.0)


def test_invalid_fu_above_one_raises():
    with pytest.raises(ValueError, match="fu_plasma"):
        _run(fu_plasma=1.5)


def test_invalid_dwell_time_zero_raises():
    with pytest.raises(ValueError, match="dwell_time_h"):
        _run(dwell_time_h=0.0)
