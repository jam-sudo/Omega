"""
Tests for Phase 433 — sublingual_absorption_model.py
~23 tests
"""

import pytest

from omega_pbpk.core.sublingual_absorption_model import (
    SublingualAbsorptionResult,
    compare_routes_sublingual,
    simulate_sublingual_absorption,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_sim(
    route="sublingual",
    logp=2.0,
    dose_mg=10.0,
    ionization_type="neutral",
    drug_pka=7.0,
    cl_sys=5.0,
    vd_sys=10.0,
    t_end_min=30.0,
):
    return simulate_sublingual_absorption(
        drug_name="TestDrug",
        dose_mg=dose_mg,
        route=route,
        drug_pka=drug_pka,
        logp=logp,
        ionization_type=ionization_type,
        cl_sys_L_per_h=cl_sys,
        vd_sys_L=vd_sys,
        t_end_min=t_end_min,
    )


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_return_type():
    r = default_sim()
    assert isinstance(r, SublingualAbsorptionResult)


def test_times_start_at_zero():
    r = default_sim()
    assert r.times_min[0] == pytest.approx(0.0)


def test_arrays_consistent_length():
    r = default_sim()
    n = len(r.times_min)
    assert len(r.m_absorbed_mg) == n
    assert len(r.m_remaining_mg) == n
    assert len(r.c_systemic_mg_L) == n


def test_m_remaining_starts_at_dose():
    dose = 15.0
    r = default_sim(dose_mg=dose)
    assert r.m_remaining_mg[0] == pytest.approx(dose)


def test_m_absorbed_starts_at_zero():
    r = default_sim()
    assert r.m_absorbed_mg[0] == pytest.approx(0.0)


def test_m_remaining_decreases():
    r = default_sim()
    # Should be non-increasing overall
    assert r.m_remaining_mg[-1] < r.m_remaining_mg[0]


def test_m_absorbed_increases():
    r = default_sim()
    assert r.m_absorbed_mg[-1] > r.m_absorbed_mg[0]


def test_cmax_systemic_positive():
    r = default_sim()
    assert r.cmax_systemic_mg_L > 0.0


def test_f_absorbed_in_0_1():
    r = default_sim()
    assert 0.0 <= r.f_absorbed_total <= 1.0


def test_salivary_loss_non_negative():
    r = default_sim()
    assert r.salivary_loss_pct >= 0.0


def test_t_half_absorption_positive():
    r = default_sim()
    assert r.t_half_absorption_min > 0.0


def test_auc_positive():
    r = default_sim()
    assert r.auc_systemic_mg_h_per_L > 0.0


def test_drug_name_preserved():
    r = simulate_sublingual_absorption(
        drug_name="Nitroglycerin",
        dose_mg=5.0,
        route="sublingual",
        drug_pka=9.0,
        logp=1.62,
        ionization_type="neutral",
        cl_sys_L_per_h=10.0,
    )
    assert r.drug_name == "Nitroglycerin"


def test_route_preserved():
    for route in ["sublingual", "buccal"]:
        r = default_sim(route=route)
        assert r.route == route


# ---------------------------------------------------------------------------
# Physical relationships
# ---------------------------------------------------------------------------


def test_higher_logp_higher_f_absorbed():
    """More lipophilic drug → higher mucosal permeability → more absorbed."""
    r_low = default_sim(logp=0.5, ionization_type="neutral")
    r_high = default_sim(logp=3.0, ionization_type="neutral")
    assert r_high.f_absorbed_total > r_low.f_absorbed_total


def test_buccal_higher_f_absorbed_than_sublingual():
    """Buccal has 2.5x larger area → higher absorption than sublingual."""
    r_sub = default_sim(route="sublingual")
    r_buc = default_sim(route="buccal")
    assert r_buc.f_absorbed_total > r_sub.f_absorbed_total


def test_buccal_higher_cmax_than_sublingual():
    r_sub = default_sim(route="sublingual")
    r_buc = default_sim(route="buccal")
    assert r_buc.cmax_systemic_mg_L > r_sub.cmax_systemic_mg_L


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


def test_dose_linearity_cmax():
    r1 = default_sim(dose_mg=5.0)
    r2 = default_sim(dose_mg=10.0)
    ratio = r2.cmax_systemic_mg_L / r1.cmax_systemic_mg_L
    assert ratio == pytest.approx(2.0, rel=0.01)


def test_dose_linearity_auc():
    r1 = default_sim(dose_mg=5.0)
    r2 = default_sim(dose_mg=10.0)
    ratio = r2.auc_systemic_mg_h_per_L / r1.auc_systemic_mg_h_per_L
    assert ratio == pytest.approx(2.0, rel=0.01)


# ---------------------------------------------------------------------------
# compare_routes_sublingual
# ---------------------------------------------------------------------------


def test_compare_routes_returns_two():
    results = compare_routes_sublingual(
        drug_name="Drug",
        dose_mg=10.0,
        drug_pka=7.4,
        logp=2.0,
        ionization_type="neutral",
        cl_sys_L_per_h=5.0,
        vd_sys_L=10.0,
    )
    assert len(results) == 2


def test_compare_routes_sorted_descending():
    results = compare_routes_sublingual(
        drug_name="Drug",
        dose_mg=10.0,
        drug_pka=7.4,
        logp=2.0,
        ionization_type="neutral",
        cl_sys_L_per_h=5.0,
        vd_sys_L=10.0,
    )
    assert results[0].cmax_systemic_mg_L >= results[1].cmax_systemic_mg_L


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        default_sim(dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        default_sim(dose_mg=-1.0)


def test_validation_bad_route():
    with pytest.raises(ValueError, match="route"):
        simulate_sublingual_absorption(
            drug_name="Drug",
            dose_mg=10.0,
            route="intravenous",
            drug_pka=7.0,
            logp=1.0,
            ionization_type="neutral",
            cl_sys_L_per_h=5.0,
        )


def test_validation_bad_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        simulate_sublingual_absorption(
            drug_name="Drug",
            dose_mg=10.0,
            route="sublingual",
            drug_pka=7.0,
            logp=1.0,
            ionization_type="zwitterion",
            cl_sys_L_per_h=5.0,
        )
