"""Tests for clinical/continuous_infusion_pk.py — Phase 629."""

import pytest

from omega_pbpk.clinical.continuous_infusion_pk import (
    InfusionResult,
    calculate_infusion_rate,
    calculate_loading_dose,
    optimize_infusion_regimen,
    simulate_continuous_infusion,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_sim(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        infusion_rate_mg_per_h=10.0,
        cl_L_per_h=2.0,
        vd_L=20.0,
        t_end_h=24.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_continuous_infusion(**defaults)


# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_returns_result():
    result = _default_sim()
    assert isinstance(result, InfusionResult)


def test_lengths_match():
    result = _default_sim()
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) > 0


# ---------------------------------------------------------------------------
# CSS accuracy
# ---------------------------------------------------------------------------


def test_css_analytical():
    """After long infusion, C should approach analytical CSS = R/CL."""
    rate = 10.0
    cl = 2.0
    result = simulate_continuous_infusion(
        drug_name="Drug",
        infusion_rate_mg_per_h=rate,
        cl_L_per_h=cl,
        vd_L=20.0,
        t_end_h=48.0,
        dt_h=0.05,
    )
    css_analytical = rate / cl  # 5.0 mg/L
    assert abs(result.css_achieved_mg_L - css_analytical) / css_analytical < 0.02  # within 2%


def test_plasma_near_css_at_end_long_simulation():
    rate = 6.0
    cl = 3.0
    css_expected = 2.0
    result = simulate_continuous_infusion(
        drug_name="X",
        infusion_rate_mg_per_h=rate,
        cl_L_per_h=cl,
        vd_L=10.0,
        t_end_h=30.0,
        dt_h=0.05,
    )
    assert abs(result.css_achieved_mg_L - css_expected) < 0.1


# ---------------------------------------------------------------------------
# Loading dose effects
# ---------------------------------------------------------------------------


def test_loading_dose_accelerates_ss():
    """With loading dose, concentration starts closer to CSS."""
    cl = 2.0
    vd = 20.0
    rate = 10.0
    css = rate / cl  # 5.0

    # With loading dose = CSS * Vd
    with_ld = simulate_continuous_infusion(
        "D", rate, cl, vd, loading_dose_mg=css * vd, t_end_h=5.0, dt_h=0.1
    )
    # Without loading dose
    without_ld = simulate_continuous_infusion(
        "D", rate, cl, vd, loading_dose_mg=0.0, t_end_h=5.0, dt_h=0.1
    )

    # At early time point (e.g., t=1h), concentration should be higher with loading dose
    c_with_at_1h = with_ld.c_plasma_mg_L[10]  # index 10 ~ t=1h
    c_without_at_1h = without_ld.c_plasma_mg_L[10]
    assert c_with_at_1h > c_without_at_1h


def test_zero_loading_dose_slower():
    """Without loading dose, takes longer to reach 95% SS."""
    rate, cl, vd = 10.0, 2.0, 20.0
    # time_to_95_pct_ss_h is the analytical value (same for both)
    # But actual concentration without LD is below target initially
    result_no_ld = simulate_continuous_infusion("D", rate, cl, vd, loading_dose_mg=0.0, t_end_h=5.0)
    result_ld = simulate_continuous_infusion(
        "D", rate, cl, vd, loading_dose_mg=rate / cl * vd, t_end_h=5.0
    )
    # C at t=0: no LD → 0; LD → css
    assert result_no_ld.c_plasma_mg_L[0] == pytest.approx(0.0)
    assert result_ld.c_plasma_mg_L[0] > 0.0


# ---------------------------------------------------------------------------
# calculate_infusion_rate
# ---------------------------------------------------------------------------


def test_calculate_infusion_rate():
    rate = calculate_infusion_rate(target_css_mg_L=5.0, cl_L_per_h=2.0)
    assert rate == pytest.approx(10.0)


def test_infusion_rate_proportional_to_target():
    rate1 = calculate_infusion_rate(2.0, 3.0)
    rate2 = calculate_infusion_rate(4.0, 3.0)
    assert rate2 == pytest.approx(2 * rate1)


# ---------------------------------------------------------------------------
# calculate_loading_dose
# ---------------------------------------------------------------------------


def test_calculate_loading_dose():
    dose = calculate_loading_dose(target_css_mg_L=5.0, vd_L=20.0)
    assert dose == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Plasma profile shape
# ---------------------------------------------------------------------------


def test_plasma_rises_monotonically_no_loading():
    result = simulate_continuous_infusion(
        "D",
        infusion_rate_mg_per_h=5.0,
        cl_L_per_h=1.0,
        vd_L=10.0,
        loading_dose_mg=0.0,
        t_end_h=10.0,
        dt_h=0.1,
    )
    concs = result.c_plasma_mg_L
    # Should be monotonically non-decreasing
    for i in range(1, len(concs)):
        assert concs[i] >= concs[i - 1] - 1e-9


def test_time_to_ss_positive():
    result = _default_sim()
    assert result.time_to_95_pct_ss_h > 0


def test_fluctuation_zero():
    result = _default_sim()
    assert result.fluctuation_pct == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


def test_dose_proportionality_css():
    """2x infusion rate → 2x CSS."""
    cl = 2.0
    vd = 20.0
    r1 = simulate_continuous_infusion("D", 5.0, cl, vd, t_end_h=48.0, dt_h=0.1)
    r2 = simulate_continuous_infusion("D", 10.0, cl, vd, t_end_h=48.0, dt_h=0.1)
    assert abs(r2.css_achieved_mg_L / r1.css_achieved_mg_L - 2.0) < 0.05


def test_higher_cl_lower_css_for_same_rate():
    """Higher CL → lower CSS for same infusion rate."""
    vd = 20.0
    rate = 10.0
    r_low_cl = simulate_continuous_infusion("D", rate, 1.0, vd, t_end_h=60.0, dt_h=0.1)
    r_high_cl = simulate_continuous_infusion("D", rate, 5.0, vd, t_end_h=60.0, dt_h=0.1)
    assert r_high_cl.css_achieved_mg_L < r_low_cl.css_achieved_mg_L


# ---------------------------------------------------------------------------
# optimize_infusion_regimen
# ---------------------------------------------------------------------------


def test_optimize_returns_result():
    result = optimize_infusion_regimen("Drug", target_css_mg_L=5.0, cl_L_per_h=2.0, vd_L=20.0)
    assert isinstance(result, InfusionResult)


def test_optimize_loading_dose_equals_css_vd():
    target = 5.0
    vd = 20.0
    result = optimize_infusion_regimen("Drug", target, cl_L_per_h=2.0, vd_L=vd)
    assert result.loading_dose_mg == pytest.approx(target * vd)


def test_optimize_infusion_rate_equals_css_cl():
    target = 5.0
    cl = 2.0
    result = optimize_infusion_regimen("Drug", target, cl_L_per_h=cl, vd_L=20.0)
    assert result.infusion_rate_mg_per_h == pytest.approx(target * cl)


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------


def test_invalid_rate_raises():
    with pytest.raises(ValueError):
        simulate_continuous_infusion("D", infusion_rate_mg_per_h=-1.0, cl_L_per_h=2.0, vd_L=20.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError):
        simulate_continuous_infusion("D", infusion_rate_mg_per_h=5.0, cl_L_per_h=0.0, vd_L=20.0)


def test_invalid_vd_raises():
    with pytest.raises(ValueError):
        simulate_continuous_infusion("D", infusion_rate_mg_per_h=5.0, cl_L_per_h=2.0, vd_L=-5.0)


def test_invalid_target_raises():
    with pytest.raises(ValueError):
        calculate_infusion_rate(target_css_mg_L=0.0, cl_L_per_h=2.0)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = _default_sim()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
