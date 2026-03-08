"""Tests for Phase 620 — Drug Synovial Fluid Distribution."""

import pytest

from omega_pbpk.core.synovial_pk_p620 import (
    SynovialPKResult,
    simulate_synovial_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_KWARGS = dict(
    drug_name="Methotrexate",
    dose_mg=20.0,
    logP=0.5,
    mw_Da=400.0,
    cl_sys_L_per_h=3.0,
    vd_sys_L=30.0,
    fu_plasma=0.5,
    t_end_h=24.0,
    dt_h=0.05,
)


def run_base(**overrides):
    kwargs = {**BASE_KWARGS, **overrides}
    return simulate_synovial_pk(**kwargs)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = run_base()
    assert isinstance(result, SynovialPKResult)


def test_list_fields_are_lists():
    result = run_base()
    assert isinstance(result.times_h, list)
    assert isinstance(result.c_plasma_mg_L, list)
    assert isinstance(result.c_synovial_mg_L, list)
    assert isinstance(result.synovial_plasma_ratio, list)


def test_list_lengths_match():
    result = run_base()
    n = len(result.times_h)
    assert n == len(result.c_plasma_mg_L)
    assert n == len(result.c_synovial_mg_L)
    assert n == len(result.synovial_plasma_ratio)


def test_expected_step_count():
    result = run_base(t_end_h=10.0, dt_h=0.05)
    # n_steps = int(10.0/0.05) + 1 = 201
    assert len(result.times_h) == 201


def test_times_start_at_zero():
    result = run_base()
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_end_at_t_end():
    result = run_base(t_end_h=24.0, dt_h=0.05)
    assert result.times_h[-1] == pytest.approx(24.0, abs=0.1)


def test_drug_name_preserved():
    result = run_base(drug_name="Naproxen")
    assert result.drug_name == "Naproxen"


def test_dose_preserved():
    result = run_base(dose_mg=50.0)
    assert result.dose_mg == 50.0


# ---------------------------------------------------------------------------
# Concentration positivity and plausibility
# ---------------------------------------------------------------------------


def test_plasma_concentrations_non_negative():
    result = run_base()
    assert all(c >= 0.0 for c in result.c_plasma_mg_L)


def test_synovial_concentrations_non_negative():
    result = run_base()
    assert all(c >= 0.0 for c in result.c_synovial_mg_L)


def test_cmax_plasma_positive():
    result = run_base()
    assert result.cmax_plasma_mg_L > 0.0


def test_cmax_synovial_positive():
    result = run_base()
    assert result.cmax_synovial_mg_L > 0.0


def test_cmax_plasma_matches_list_max():
    result = run_base()
    assert result.cmax_plasma_mg_L == pytest.approx(max(result.c_plasma_mg_L))


def test_cmax_synovial_matches_list_max():
    result = run_base()
    assert result.cmax_synovial_mg_L == pytest.approx(max(result.c_synovial_mg_L))


def test_tmax_synovial_valid():
    result = run_base()
    assert 0.0 <= result.tmax_synovial_h <= BASE_KWARGS["t_end_h"]


# ---------------------------------------------------------------------------
# MW effect on synovial penetration
# ---------------------------------------------------------------------------


def test_smaller_mw_increases_k_in():
    """Smaller MW → higher permeability → more synovial penetration."""
    r_small = run_base(mw_Da=200.0)
    r_large = run_base(mw_Da=1000.0)
    assert r_small.auc_synovial_mg_h_per_L >= r_large.auc_synovial_mg_h_per_L


def test_smaller_mw_reduces_lag():
    r_small = run_base(mw_Da=200.0)
    r_large = run_base(mw_Da=1000.0)
    assert r_small.lag_to_synovial_h <= r_large.lag_to_synovial_h


def test_larger_mw_reduces_synovial_auc_ratio():
    r_small = run_base(mw_Da=200.0)
    r_large = run_base(mw_Da=1000.0)
    assert r_small.synovial_auc_ratio >= r_large.synovial_auc_ratio


# ---------------------------------------------------------------------------
# fu_plasma effect
# ---------------------------------------------------------------------------


def test_higher_fu_increases_synovial_penetration():
    r_low = run_base(fu_plasma=0.05)
    r_high = run_base(fu_plasma=0.8)
    assert r_high.auc_synovial_mg_h_per_L >= r_low.auc_synovial_mg_h_per_L


def test_fu_plasma_at_unity_allowed():
    result = run_base(fu_plasma=1.0)
    assert result.cmax_synovial_mg_L > 0.0


# ---------------------------------------------------------------------------
# AUC and ratios
# ---------------------------------------------------------------------------


def test_auc_plasma_positive():
    result = run_base()
    assert result.auc_plasma_mg_h_per_L > 0.0


def test_auc_synovial_positive():
    result = run_base()
    assert result.auc_synovial_mg_h_per_L > 0.0


def test_synovial_auc_ratio_formula():
    result = run_base()
    expected = result.auc_synovial_mg_h_per_L / result.auc_plasma_mg_h_per_L
    assert result.synovial_auc_ratio == pytest.approx(expected, rel=1e-6)


def test_higher_dose_scales_auc_linearly():
    r1 = run_base(dose_mg=20.0)
    r2 = run_base(dose_mg=40.0)
    ratio = r2.auc_plasma_mg_h_per_L / r1.auc_plasma_mg_h_per_L
    assert ratio == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# Lag time and efficacy coverage
# ---------------------------------------------------------------------------


def test_lag_to_synovial_non_negative():
    result = run_base()
    assert result.lag_to_synovial_h >= 0.0


def test_efficacy_coverage_non_negative():
    result = run_base()
    assert result.efficacy_coverage_h >= 0.0


def test_efficacy_coverage_bounded_by_t_end():
    result = run_base()
    assert result.efficacy_coverage_h <= BASE_KWARGS["t_end_h"] + BASE_KWARGS["dt_h"]


def test_high_dose_increases_efficacy_coverage():
    r_low = run_base(dose_mg=5.0)
    r_high = run_base(dose_mg=100.0)
    assert r_high.efficacy_coverage_h >= r_low.efficacy_coverage_h


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        run_base(dose_mg=-1.0)


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        run_base(dose_mg=0.0)


def test_zero_cl_raises():
    with pytest.raises(ValueError, match="cl_sys"):
        run_base(cl_sys_L_per_h=0.0)


def test_zero_vd_raises():
    with pytest.raises(ValueError, match="vd_sys"):
        run_base(vd_sys_L=0.0)


def test_zero_fu_raises():
    with pytest.raises(ValueError, match="fu_plasma"):
        run_base(fu_plasma=0.0)


def test_fu_above_one_raises():
    with pytest.raises(ValueError, match="fu_plasma"):
        run_base(fu_plasma=1.5)


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_is_string():
    result = run_base()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


def test_synovial_plasma_ratio_at_t0_near_zero():
    """At t=0 both concentrations are 0; ratio is clamped by 1e-10 denom."""
    result = run_base()
    # The first ratio C_syn[0]/max(1e-10, C_plasma[0]) should be near 0
    assert result.synovial_plasma_ratio[0] == pytest.approx(0.0, abs=1e-6)
