"""Tests for Phase 264: recirculatory pharmacokinetic model."""

from __future__ import annotations

import pytest

from omega_pbpk.core.recirculatory_pk import (
    RecirculatoryPKResult,
    simulate_recirculatory_pk,
)

# ---------------------------------------------------------------------------
# Default parameter set for convenience
# ---------------------------------------------------------------------------

_BASE_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cl_central_L_per_h=5.0,
    v_central_L=10.0,
    q_fast_L_per_h=20.0,
    v_fast_L=15.0,
    q_slow_L_per_h=5.0,
    v_slow_L=30.0,
    ke0_per_h=1.0,
    v_effect_L=0.5,
    t_end_h=12.0,
    dt_h=0.05,
)


def _run(**overrides):
    kw = dict(_BASE_KWARGS)
    kw.update(overrides)
    return simulate_recirculatory_pk(**kw)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "param",
    [
        "dose_mg",
        "cl_central_L_per_h",
        "v_central_L",
        "q_fast_L_per_h",
        "v_fast_L",
        "q_slow_L_per_h",
        "v_slow_L",
        "ke0_per_h",
        "v_effect_L",
    ],
)
def test_invalid_param_raises(param):
    with pytest.raises(ValueError, match=param):
        _run(**{param: 0.0})


def test_negative_param_raises():
    with pytest.raises(ValueError):
        _run(dose_mg=-50.0)


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------


def test_c_central_initial_equals_dose_over_volume():
    result = _run(dose_mg=100.0, v_central_L=10.0)
    assert abs(result.c_central_mg_L[0] - 10.0) < 1e-6


# ---------------------------------------------------------------------------
# Basic positivity / structure
# ---------------------------------------------------------------------------


def test_cmax_central_positive():
    result = _run()
    assert result.cmax_central > 0.0


def test_auc_central_positive():
    result = _run()
    assert result.auc_central > 0.0


def test_effect_cmax_positive():
    result = _run()
    assert result.effect_cmax > 0.0


def test_effect_tmax_positive():
    result = _run()
    assert result.effect_tmax_h > 0.0


def test_t_half_positive():
    result = _run()
    assert result.t_half_central_h > 0.0


# ---------------------------------------------------------------------------
# Array lengths
# ---------------------------------------------------------------------------


def test_array_lengths_consistent():
    result = _run()
    n = len(result.times_h)
    assert len(result.c_central_mg_L) == n
    assert len(result.c_fast_mg_L) == n
    assert len(result.c_slow_mg_L) == n
    assert len(result.c_effect_mg_L) == n


# ---------------------------------------------------------------------------
# Linearity
# ---------------------------------------------------------------------------


def test_double_dose_doubles_cmax():
    r1 = _run(dose_mg=100.0)
    r2 = _run(dose_mg=200.0)
    assert abs(r2.cmax_central / r1.cmax_central - 2.0) < 0.01


def test_double_dose_doubles_auc():
    r1 = _run(dose_mg=100.0)
    r2 = _run(dose_mg=200.0)
    assert abs(r2.auc_central / r1.auc_central - 2.0) < 0.01


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------


def test_fast_compartment_has_drug():
    result = _run()
    assert max(result.c_fast_mg_L) > 0.0


def test_slow_compartment_has_drug():
    result = _run()
    assert max(result.c_slow_mg_L) > 0.0


# ---------------------------------------------------------------------------
# Drug name stored
# ---------------------------------------------------------------------------


def test_drug_name_stored():
    result = _run(drug_name="Propofol")
    assert result.drug_name == "Propofol"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = _run()
    assert isinstance(result, RecirculatoryPKResult)


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_contains_drug_name():
    result = _run(drug_name="Fentanyl")
    assert "Fentanyl" in result.notes


# ===========================================================================
# Phase 622 — physiological recirculatory model tests
# ===========================================================================

from omega_pbpk.core.recirculatory_pk import (  # noqa: E402
    RecirculatoryResult,
    simulate_recirculatory,
)


def _default_sim(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cardiac_output_L_per_h=390.0,
        vp_arterial_L=0.75,
        vp_venous_L=1.5,
        vp_lung_L=0.5,
        vt_peripheral_L=20.0,
        kp_peripheral=1.0,
        cl_hepatic_L_per_h=10.0,
        cl_lung_L_per_h=0.0,
        f_peripheral=0.8,
        t_end_h=4.0,
        dt_h=0.002,
    )
    defaults.update(kwargs)
    return simulate_recirculatory(**defaults)


def test_622_returns_result():
    result = _default_sim()
    assert isinstance(result, RecirculatoryResult)


def test_622_arterial_conc_positive():
    result = _default_sim()
    assert result.cmax_arterial > 0


def test_622_venous_conc_positive():
    result = _default_sim()
    assert any(c > 0 for c in result.c_venous_mg_L)


def test_622_lung_conc_positive():
    result = _default_sim()
    assert any(c > 0 for c in result.c_lung_mg_L)


def test_622_peripheral_conc_positive():
    result = _default_sim()
    assert any(c > 0 for c in result.c_peripheral_mg_L)


def test_622_auc_arterial_positive():
    result = _default_sim()
    assert result.auc_arterial > 0


def test_622_lengths_match():
    result = _default_sim()
    n = len(result.times_h)
    assert len(result.c_arterial_mg_L) == n
    assert len(result.c_venous_mg_L) == n
    assert len(result.c_lung_mg_L) == n
    assert len(result.c_peripheral_mg_L) == n


def test_622_dose_proportionality_cmax():
    r1 = _default_sim(dose_mg=100.0)
    r2 = _default_sim(dose_mg=200.0)
    ratio = r2.cmax_arterial / r1.cmax_arterial
    assert abs(ratio - 2.0) < 0.05, f"Expected ~2.0, got {ratio:.3f}"


def test_622_dose_proportionality_auc():
    r1 = _default_sim(dose_mg=100.0)
    r2 = _default_sim(dose_mg=200.0)
    ratio = r2.auc_arterial / r1.auc_arterial
    assert abs(ratio - 2.0) < 0.05, f"Expected ~2.0, got {ratio:.3f}"


def test_622_higher_cl_lower_auc():
    r_low = _default_sim(cl_hepatic_L_per_h=5.0)
    r_high = _default_sim(cl_hepatic_L_per_h=50.0)
    assert r_high.auc_arterial < r_low.auc_arterial


def test_622_mean_transit_time_positive():
    result = _default_sim()
    assert result.mean_transit_time_h > 0


def test_622_extraction_ratio_in_0_1():
    result = _default_sim(cl_lung_L_per_h=0.0)
    assert 0.0 <= result.extraction_ratio_lung <= 1.0


def test_622_zero_lung_clearance_er_zero():
    result = _default_sim(cl_lung_L_per_h=0.0)
    assert result.extraction_ratio_lung == 0.0


def test_622_higher_lung_cl_lower_arterial_conc():
    # Use high lung CL relative to CO to see a meaningful AUC difference
    r_no_lung = _default_sim(cl_lung_L_per_h=0.0)
    r_lung = _default_sim(cl_lung_L_per_h=195.0)  # 50% of CO
    assert r_lung.auc_arterial < r_no_lung.auc_arterial


def test_622_peripheral_rises_after_arterial():
    """Peripheral tissue concentration should peak at or after arterial."""
    result = _default_sim(t_end_h=6.0)
    c_per = result.c_peripheral_mg_L
    times = result.times_h
    tmax_per_idx = c_per.index(max(c_per))
    tmax_per = times[tmax_per_idx]
    assert tmax_per >= result.tmax_arterial_h


def test_622_high_peripheral_kp_higher_tissue():
    """Higher kp_peripheral should lead to higher tissue AUC."""
    r_low = _default_sim(kp_peripheral=1.0, t_end_h=4.0)
    r_high = _default_sim(kp_peripheral=3.0, t_end_h=4.0)
    auc_per_low = sum(
        (r_low.c_peripheral_mg_L[i] + r_low.c_peripheral_mg_L[i - 1])
        / 2.0
        * (r_low.times_h[i] - r_low.times_h[i - 1])
        for i in range(1, len(r_low.times_h))
    )
    auc_per_high = sum(
        (r_high.c_peripheral_mg_L[i] + r_high.c_peripheral_mg_L[i - 1])
        / 2.0
        * (r_high.times_h[i] - r_high.times_h[i - 1])
        for i in range(1, len(r_high.times_h))
    )
    assert auc_per_high > auc_per_low


def test_622_invalid_dose_raises():
    with pytest.raises(ValueError):
        _default_sim(dose_mg=-10.0)


def test_622_invalid_co_raises():
    with pytest.raises(ValueError):
        _default_sim(cardiac_output_L_per_h=0.0)


def test_622_invalid_volume_raises():
    with pytest.raises(ValueError):
        _default_sim(vp_arterial_L=0.0)


def test_622_invalid_f_peripheral_raises():
    with pytest.raises(ValueError):
        _default_sim(f_peripheral=1.0)


def test_622_notes_nonempty():
    result = _default_sim()
    assert len(result.notes) > 0
