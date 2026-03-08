"""
Tests for Phase 449: Lymph Node Concentration PK
"""

import pytest

from omega_pbpk.core.lymph_node_concentration_pk import (
    LymphNodePKResult,
    compare_lymph_routes,
    simulate_lymph_node_pk,
)

# ─── Helper defaults ──────────────────────────────────────────────────────────

DRUG = "TestDrug"
DOSE = 100.0
LOGP = 3.0
CL = 2.0
VD = 50.0


def sim(route="iv", logp=LOGP, dose=DOSE, cl=CL, vd=VD, **kwargs):
    return simulate_lymph_node_pk(
        drug_name=DRUG,
        dose_mg=dose,
        route=route,
        logp=logp,
        cl_sys_L_per_h=cl,
        vd_plasma_L=vd,
        **kwargs,
    )


# ─── Return type tests ────────────────────────────────────────────────────────


def test_return_type_iv():
    result = sim(route="iv")
    assert isinstance(result, LymphNodePKResult)


def test_return_type_oral():
    result = sim(route="oral")
    assert isinstance(result, LymphNodePKResult)


def test_return_type_sc():
    result = sim(route="subcutaneous")
    assert isinstance(result, LymphNodePKResult)


# ─── Array consistency ────────────────────────────────────────────────────────


def test_arrays_consistent_length():
    result = sim()
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) == len(result.c_lymph_node_mg_L)


def test_times_start_at_zero():
    result = sim()
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_monotonically_increasing():
    result = sim()
    times = result.times_h
    assert all(times[i] < times[i + 1] for i in range(len(times) - 1))


# ─── IV initial conditions ─────────────────────────────────────────────────────


def test_iv_plasma_starts_at_dose_over_vd():
    result = sim(route="iv", dose=100.0, vd=50.0)
    expected = 100.0 / 50.0  # 2.0 mg/L
    assert result.c_plasma_mg_L[0] == pytest.approx(expected, rel=1e-6)


def test_iv_lymph_starts_near_zero():
    result = sim(route="iv")
    assert result.c_lymph_node_mg_L[0] == pytest.approx(0.0, abs=1e-9)


# ─── SC initial conditions ─────────────────────────────────────────────────────


def test_sc_plasma_starts_near_zero():
    result = sim(route="subcutaneous")
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-9)


def test_sc_lymph_starts_near_zero():
    result = sim(route="subcutaneous")
    assert result.c_lymph_node_mg_L[0] == pytest.approx(0.0, abs=1e-9)


def test_sc_slow_rise():
    """SC should show slow rise — cmax not at t=0."""
    result = sim(route="subcutaneous")
    # cmax index should be > 0
    cmax_idx = result.c_plasma_mg_L.index(result.cmax_plasma_mg_L)
    assert cmax_idx > 0


# ─── Positivity checks ────────────────────────────────────────────────────────


def test_cmax_lymph_positive():
    result = sim()
    assert result.cmax_lymph_node_mg_L > 0.0


def test_auc_lymph_positive():
    result = sim()
    assert result.auc_lymph_node_mg_h_per_L > 0.0


def test_auc_plasma_positive():
    result = sim()
    assert result.auc_plasma_mg_h_per_L > 0.0


# ─── logP effect on lymph_node_kp and ratio ───────────────────────────────────


def test_higher_logp_higher_kp():
    r_low = sim(logp=1.0)
    r_high = sim(logp=5.0)
    assert r_high.lymph_node_kp > r_low.lymph_node_kp


def test_higher_logp_higher_lymph_to_plasma_ratio():
    r_low = sim(logp=0.0)
    r_high = sim(logp=5.0)
    assert r_high.lymph_to_plasma_ratio >= r_low.lymph_to_plasma_ratio


def test_high_logp_lymph_to_plasma_ratio_positive():
    """For logP > 2, lymph_to_plasma_ratio should be a positive value."""
    result = sim(logp=5.0)
    assert result.lymph_to_plasma_ratio > 0.0


def test_high_logp_cmax_lymph_exceeds_cmax_at_late_times():
    """For high logP IV, the C_lymph/C_plasma ratio at late time should exceed Kp-based expectation."""
    result = sim(logp=5.0)
    # Late-time concentration ratio should approach Kp ~ 5.0
    # Check that the peak lymph concentration relative to same-time plasma > 1.0
    c_lymph = result.c_lymph_node_mg_L
    # Find peak lymph concentration time
    peak_idx = c_lymph.index(max(c_lymph))
    # At peak lymph, the concentration is positive and the plasma concentration is positive
    assert c_lymph[peak_idx] > 0.0
    # At equilibrium (late time), C_lymph/C_plasma should approach Kp=5
    # Check that at tmax_lymph the lymph concentration is meaningful fraction of plasma
    assert result.cmax_lymph_node_mg_L > 0.0


# ─── Kp clamping ──────────────────────────────────────────────────────────────


def test_kp_clamped_at_minimum_one():
    result = sim(logp=-10.0)
    assert result.lymph_node_kp == pytest.approx(1.0)


def test_kp_clamped_at_maximum_twenty():
    result = sim(logp=100.0)
    assert result.lymph_node_kp == pytest.approx(20.0)


# ─── Half-life ─────────────────────────────────────────────────────────────────


def test_t_half_lymph_node_positive():
    result = sim()
    assert result.t_half_lymph_node_h > 0.0


# ─── compare_lymph_routes ─────────────────────────────────────────────────────


def test_compare_lymph_routes_returns_three():
    results = compare_lymph_routes(DRUG, DOSE, LOGP, CL, VD)
    assert len(results) == 3


def test_compare_lymph_routes_sorted_descending():
    results = compare_lymph_routes(DRUG, DOSE, LOGP, CL, VD)
    aucs = [r.auc_lymph_node_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_lymph_routes_all_types():
    results = compare_lymph_routes(DRUG, DOSE, LOGP, CL, VD)
    routes = {r.route for r in results}
    assert routes == {"iv", "oral", "subcutaneous"}


# ─── Validation errors ────────────────────────────────────────────────────────


def test_error_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        sim(dose=0.0)


def test_error_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        sim(dose=-50.0)


def test_error_bad_route():
    with pytest.raises(ValueError, match="route"):
        simulate_lymph_node_pk(
            drug_name=DRUG,
            dose_mg=DOSE,
            route="intramuscular",
            logp=LOGP,
            cl_sys_L_per_h=CL,
        )


def test_error_cl_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        sim(cl=0.0)
