"""Tests for liver zonation pharmacokinetics model — Phase 494."""

from __future__ import annotations

import pytest

from omega_pbpk.core.liver_zonation import (
    LiverZonationResult,
    calculate_outlet_concentration,
    periportal_vs_centrilobular,
    simulate_liver_zonation,
    zone_extraction_ratio,
)

# ---------------------------------------------------------------------------
# zone_extraction_ratio
# ---------------------------------------------------------------------------


def test_zone_er_basic():
    # E = fu*CLint / (Q + fu*CLint)
    e = zone_extraction_ratio(zone_cl_int=9.0, flow_per_zone_L_per_h=1.0, fu_plasma=1.0)
    assert e == pytest.approx(9.0 / 10.0)


def test_zone_er_zero_clint():
    e = zone_extraction_ratio(0.0, 90.0, 0.1)
    assert e == pytest.approx(0.0)


def test_zone_er_between_zero_and_one():
    e = zone_extraction_ratio(5.0, 20.0, 0.2)
    assert 0.0 <= e < 1.0


def test_zone_er_negative_clint_raises():
    with pytest.raises(ValueError, match="cl_int"):
        zone_extraction_ratio(-1.0, 90.0, 0.1)


def test_zone_er_zero_flow_raises():
    with pytest.raises(ValueError, match="flow"):
        zone_extraction_ratio(5.0, 0.0, 0.1)


def test_zone_er_fu_out_of_range_raises():
    with pytest.raises(ValueError, match="fu_plasma"):
        zone_extraction_ratio(5.0, 90.0, 0.0)


# ---------------------------------------------------------------------------
# calculate_outlet_concentration
# ---------------------------------------------------------------------------


def test_outlet_lt_inlet():
    c_out = calculate_outlet_concentration(
        inlet_conc=10.0,
        n_zones=3,
        cl_int_per_zone=[5.0, 5.0, 5.0],
        flow_L_per_h=90.0,
        fu_plasma=0.1,
    )
    assert c_out < 10.0


def test_outlet_non_negative():
    c_out = calculate_outlet_concentration(
        inlet_conc=5.0,
        n_zones=5,
        cl_int_per_zone=[20.0] * 5,
        flow_L_per_h=90.0,
        fu_plasma=0.5,
    )
    assert c_out >= 0.0


def test_outlet_monotone_decreasing_with_clint():
    c1 = calculate_outlet_concentration(1.0, 3, [1.0, 1.0, 1.0], 90.0, 0.1)
    c2 = calculate_outlet_concentration(1.0, 3, [5.0, 5.0, 5.0], 90.0, 0.1)
    c3 = calculate_outlet_concentration(1.0, 3, [20.0, 20.0, 20.0], 90.0, 0.1)
    assert c1 > c2 > c3


def test_outlet_zero_clint_equals_inlet():
    c_out = calculate_outlet_concentration(
        inlet_conc=7.0,
        n_zones=4,
        cl_int_per_zone=[0.0] * 4,
        flow_L_per_h=90.0,
        fu_plasma=0.1,
    )
    assert c_out == pytest.approx(7.0)


def test_outlet_n_zones_zero_raises():
    with pytest.raises(ValueError, match="n_zones"):
        calculate_outlet_concentration(5.0, 0, [], 90.0, 0.1)


def test_outlet_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        calculate_outlet_concentration(5.0, 3, [5.0, 5.0], 90.0, 0.1)


# ---------------------------------------------------------------------------
# simulate_liver_zonation — result type and fields
# ---------------------------------------------------------------------------


def _default_sim(**kwargs):
    defaults = dict(
        drug_name="Drug_A",
        inlet_conc_mg_L=10.0,
        flow_L_per_h=90.0,
        n_zones=5,
        cl_int_per_zone=None,
        fu_plasma=0.1,
        t_end_h=2.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_liver_zonation(**defaults)


def test_returns_liver_zonation_result():
    assert isinstance(_default_sim(), LiverZonationResult)


def test_outlet_conc_always_lt_inlet():
    r = _default_sim(cl_int_per_zone=[5.0] * 5)
    for c in r.outlet_conc_mg_L:
        assert c < 10.0


def test_extraction_ratio_in_valid_range():
    r = _default_sim()
    assert 0.0 <= r.overall_extraction_ratio <= 1.0


def test_cl_hepatic_equals_q_times_er():
    r = _default_sim(flow_L_per_h=90.0)
    assert r.cl_hepatic_L_per_h == pytest.approx(90.0 * r.overall_extraction_ratio, rel=1e-4)


def test_zone_concentrations_length():
    r = _default_sim(n_zones=4)
    assert len(r.zone_concentrations) == 4
    n_steps = len(r.times_h)
    for zc in r.zone_concentrations:
        assert len(zc) == n_steps


# ---------------------------------------------------------------------------
# simulate_liver_zonation — model behaviour
# ---------------------------------------------------------------------------


def test_n_zones_1_well_stirred_equivalence():
    """n_zones=1 should match single well-stirred hepatic model."""
    cl_int = 9.0
    flow = 90.0
    fu = 0.1
    r = simulate_liver_zonation(
        "X",
        inlet_conc_mg_L=1.0,
        flow_L_per_h=flow,
        n_zones=1,
        cl_int_per_zone=[cl_int],
        fu_plasma=fu,
        t_end_h=1.0,
        dt_h=0.5,
    )
    e_expected = fu * cl_int / (flow + fu * cl_int)
    assert r.overall_extraction_ratio == pytest.approx(e_expected, rel=1e-6)


def test_more_zones_approaches_plug_flow():
    """Increasing n_zones (parallel tube model) should increase extraction."""
    total_cl = 10.0
    results_er = []
    for n in [1, 3, 10, 50]:
        cl_zone = total_cl / n
        r = simulate_liver_zonation(
            "X",
            1.0,
            flow_L_per_h=90.0,
            n_zones=n,
            cl_int_per_zone=[cl_zone] * n,
            fu_plasma=0.1,
            t_end_h=1.0,
            dt_h=0.5,
        )
        results_er.append(r.overall_extraction_ratio)
    # Parallel tube > well-stirred for same total CLint when extraction > 0
    # More zones → higher (or equal) ER
    for i in range(len(results_er) - 1):
        assert results_er[i + 1] >= results_er[i] - 1e-9


def test_higher_clint_higher_er():
    r_low = simulate_liver_zonation("X", 1.0, cl_int_per_zone=[1.0] * 3, n_zones=3)
    r_high = simulate_liver_zonation("X", 1.0, cl_int_per_zone=[20.0] * 3, n_zones=3)
    assert r_high.overall_extraction_ratio > r_low.overall_extraction_ratio


def test_outlet_conc_decreases_with_clint():
    r_low = simulate_liver_zonation("X", 5.0, cl_int_per_zone=[1.0] * 3, n_zones=3)
    r_high = simulate_liver_zonation("X", 5.0, cl_int_per_zone=[30.0] * 3, n_zones=3)
    assert r_high.outlet_conc_mg_L[-1] < r_low.outlet_conc_mg_L[-1]


def test_mass_balance_flow_times_delta_conc():
    """flow*(inlet - outlet) should ≈ total CL rate = sum of zone eliminations."""
    cl_int = [5.0, 5.0, 5.0]
    flow = 90.0
    fu = 0.1
    r = simulate_liver_zonation(
        "X",
        inlet_conc_mg_L=10.0,
        flow_L_per_h=flow,
        n_zones=3,
        cl_int_per_zone=cl_int,
        fu_plasma=fu,
        t_end_h=1.0,
        dt_h=0.5,
    )
    # Mass balance at steady state (last time point)
    c_in = 10.0
    c_out = r.outlet_conc_mg_L[-1]
    eliminated_by_flow = flow * (c_in - c_out)
    assert eliminated_by_flow == pytest.approx(r.cl_hepatic_L_per_h * c_in, rel=0.05)


def test_uniform_zonation_index_is_one():
    r = simulate_liver_zonation("X", 1.0, n_zones=3, cl_int_per_zone=[4.0, 4.0, 4.0])
    assert r.zonation_index == pytest.approx(1.0, rel=1e-6)


def test_nonuniform_zonation_index():
    r = simulate_liver_zonation("X", 1.0, n_zones=2, cl_int_per_zone=[8.0, 4.0])
    assert r.zonation_index == pytest.approx(2.0, rel=1e-6)


def test_n_zones_one_has_zonation_index_one():
    r = simulate_liver_zonation("X", 1.0, n_zones=1, cl_int_per_zone=[5.0])
    assert r.zonation_index == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# periportal_vs_centrilobular
# ---------------------------------------------------------------------------


def test_periportal_vs_centrilobular_returns_dict():
    result = periportal_vs_centrilobular("TestDrug", cl_int_total=10.0)
    assert isinstance(result, dict)


def test_periportal_cl_gt_centrilobular_when_factor_gt1():
    d = periportal_vs_centrilobular("D", cl_int_total=10.0, zonation_factor=3.0)
    assert d["cl_int_periportal"] > d["cl_int_centrilobular"]


def test_periportal_eq_centrilobular_when_factor_1():
    d = periportal_vs_centrilobular("D", cl_int_total=10.0, zonation_factor=1.0)
    assert d["cl_int_periportal"] == pytest.approx(d["cl_int_centrilobular"])


def test_periportal_vs_centrilobular_total_clint_conserved():
    d = periportal_vs_centrilobular("D", cl_int_total=12.0, zonation_factor=2.0)
    total = d["cl_int_periportal"] + d["cl_int_centrilobular"]
    assert total == pytest.approx(12.0, rel=1e-6)


def test_periportal_first_er_non_negative():
    d = periportal_vs_centrilobular("D", cl_int_total=10.0)
    assert d["e_periportal_first"] >= 0.0
    assert d["e_centrilobular_first"] >= 0.0


def test_periportal_cl_int_total_zero_raises():
    with pytest.raises(ValueError, match="cl_int_total"):
        periportal_vs_centrilobular("D", cl_int_total=0.0)


def test_periportal_zonation_factor_zero_raises():
    with pytest.raises(ValueError, match="zonation_factor"):
        periportal_vs_centrilobular("D", cl_int_total=10.0, zonation_factor=0.0)


# ---------------------------------------------------------------------------
# simulate_liver_zonation — input validation
# ---------------------------------------------------------------------------


def test_inlet_conc_zero_raises():
    with pytest.raises(ValueError, match="inlet_conc"):
        simulate_liver_zonation("X", inlet_conc_mg_L=0.0)


def test_flow_zero_raises():
    with pytest.raises(ValueError, match="flow"):
        simulate_liver_zonation("X", 1.0, flow_L_per_h=0.0)


def test_n_zones_zero_raises():
    with pytest.raises(ValueError, match="n_zones"):
        simulate_liver_zonation("X", 1.0, n_zones=0)


def test_fu_zero_raises():
    with pytest.raises(ValueError, match="fu_plasma"):
        simulate_liver_zonation("X", 1.0, fu_plasma=0.0)


def test_cl_int_length_mismatch_raises():
    with pytest.raises(ValueError):
        simulate_liver_zonation("X", 1.0, n_zones=3, cl_int_per_zone=[5.0, 5.0])
