"""Tests for Phase 406: pk_simulation_report — PK simulation report generator."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.pk_simulation_report import (
    PKReport,
    compare_simulations,
    generate_pk_report,
)

# ---------------------------------------------------------------------------
# Helpers — generate a simple 1-compartment IV profile
# ---------------------------------------------------------------------------

def _iv_profile(
    dose_mg: float = 100.0,
    cl: float = 2.0,
    vd: float = 20.0,
    t_end: float = 24.0,
    n_pts: int = 241,
) -> tuple[list[float], list[float]]:
    """Analytical 1-cpt IV profile: C(t) = (D/Vd)*exp(-ke*t)."""
    ke = cl / vd
    times = [t_end * i / (n_pts - 1) for i in range(n_pts)]
    concs = [dose_mg / vd * math.exp(-ke * t) for t in times]
    return times, concs


BASE_TIMES, BASE_CONCS = _iv_profile()

BASE_KWARGS = dict(
    drug_name="TestDrug",
    times_h=BASE_TIMES,
    c_plasma=BASE_CONCS,
    dose_mg=100.0,
    route="iv",
    cl_L_per_h=2.0,
    vd_L=20.0,
    interval_h=0.0,
)


# ---------------------------------------------------------------------------
# Test 1: Returns PKReport
# ---------------------------------------------------------------------------
def test_returns_pk_report():
    result = generate_pk_report(**BASE_KWARGS)
    assert isinstance(result, PKReport)


# ---------------------------------------------------------------------------
# Test 2: Drug name preserved
# ---------------------------------------------------------------------------
def test_drug_name_preserved():
    result = generate_pk_report(**BASE_KWARGS)
    assert result.drug_name == "TestDrug"


# ---------------------------------------------------------------------------
# Test 3: Route preserved
# ---------------------------------------------------------------------------
def test_route_preserved():
    result = generate_pk_report(**BASE_KWARGS)
    assert result.route == "iv"


# ---------------------------------------------------------------------------
# Test 4: Dose preserved
# ---------------------------------------------------------------------------
def test_dose_preserved():
    result = generate_pk_report(**BASE_KWARGS)
    assert result.dose_mg == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Test 5: Cmax ~ D/Vd for IV 1-cpt
# ---------------------------------------------------------------------------
def test_cmax_approx_d_over_vd():
    result = generate_pk_report(**BASE_KWARGS)
    expected_cmax = 100.0 / 20.0  # 5 mg/L
    assert result.cmax_mg_L == pytest.approx(expected_cmax, rel=0.05)


# ---------------------------------------------------------------------------
# Test 6: Tmax at t=0 for IV bolus
# ---------------------------------------------------------------------------
def test_tmax_zero_for_iv():
    result = generate_pk_report(**BASE_KWARGS)
    assert result.tmax_h == pytest.approx(0.0, abs=0.2)


# ---------------------------------------------------------------------------
# Test 7: ke ~ CL/Vd
# ---------------------------------------------------------------------------
def test_ke_approx_cl_over_vd():
    result = generate_pk_report(**BASE_KWARGS)
    expected_ke = 2.0 / 20.0  # 0.1
    assert result.ke_per_h == pytest.approx(expected_ke, rel=0.15)


# ---------------------------------------------------------------------------
# Test 8: t_half ~ ln(2)/ke
# ---------------------------------------------------------------------------
def test_t_half_consistent_with_ke():
    result = generate_pk_report(**BASE_KWARGS)
    assert result.t_half_h == pytest.approx(math.log(2) / result.ke_per_h, rel=1e-6)


# ---------------------------------------------------------------------------
# Test 9: AUC_inf ~ D/CL for IV
# ---------------------------------------------------------------------------
def test_auc_inf_approx_d_over_cl():
    result = generate_pk_report(**BASE_KWARGS)
    expected_auc = 100.0 / 2.0  # 50 mg*h/L
    assert result.auc_inf_mg_h_per_L == pytest.approx(expected_auc, rel=0.10)


# ---------------------------------------------------------------------------
# Test 10: MRT > 0
# ---------------------------------------------------------------------------
def test_mrt_positive():
    result = generate_pk_report(**BASE_KWARGS)
    assert result.mrt_h > 0.0


# ---------------------------------------------------------------------------
# Test 11: interval_h=0 → accumulation_ratio=0
# ---------------------------------------------------------------------------
def test_no_interval_zero_accumulation():
    result = generate_pk_report(**BASE_KWARGS)
    assert result.accumulation_ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 12: interval_h=0 → ss_trough=0
# ---------------------------------------------------------------------------
def test_no_interval_zero_ss_trough():
    result = generate_pk_report(**BASE_KWARGS)
    assert result.steady_state_trough_mg_L == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 13: Multi-dose — accumulation ratio > 1
# ---------------------------------------------------------------------------
def test_multidose_accumulation_ratio_gt1():
    kw = dict(BASE_KWARGS)
    kw["interval_h"] = 12.0
    result = generate_pk_report(**kw)
    assert result.accumulation_ratio > 1.0


# ---------------------------------------------------------------------------
# Test 14: Multi-dose — ss trough > 0
# ---------------------------------------------------------------------------
def test_multidose_ss_trough_positive():
    kw = dict(BASE_KWARGS)
    kw["interval_h"] = 12.0
    result = generate_pk_report(**kw)
    assert result.steady_state_trough_mg_L > 0.0


# ---------------------------------------------------------------------------
# Test 15: CL stored correctly
# ---------------------------------------------------------------------------
def test_cl_stored():
    result = generate_pk_report(**BASE_KWARGS)
    assert result.cl_L_per_h == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Test 16: Vd stored correctly
# ---------------------------------------------------------------------------
def test_vd_stored():
    result = generate_pk_report(**BASE_KWARGS)
    assert result.vd_L == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Test 17: notes is tuple
# ---------------------------------------------------------------------------
def test_notes_is_tuple():
    result = generate_pk_report(**BASE_KWARGS)
    assert isinstance(result.notes, tuple)


# ---------------------------------------------------------------------------
# Test 18: PKReport is immutable (frozen)
# ---------------------------------------------------------------------------
def test_pk_report_frozen():
    result = generate_pk_report(**BASE_KWARGS)
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test 19: Validation — empty drug_name
# ---------------------------------------------------------------------------
def test_validation_empty_drug_name():
    kw = dict(BASE_KWARGS)
    kw["drug_name"] = ""
    with pytest.raises(ValueError, match="drug_name"):
        generate_pk_report(**kw)


# ---------------------------------------------------------------------------
# Test 20: Validation — times too short
# ---------------------------------------------------------------------------
def test_validation_times_too_short():
    kw = dict(BASE_KWARGS)
    kw["times_h"] = [0.0]
    kw["c_plasma"] = [5.0]
    with pytest.raises(ValueError, match="times_h"):
        generate_pk_report(**kw)


# ---------------------------------------------------------------------------
# Test 21: Validation — mismatched lengths
# ---------------------------------------------------------------------------
def test_validation_mismatched_lengths():
    kw = dict(BASE_KWARGS)
    kw["c_plasma"] = BASE_CONCS[:-1]
    with pytest.raises(ValueError, match="same length"):
        generate_pk_report(**kw)


# ---------------------------------------------------------------------------
# Test 22: Validation — dose_mg <= 0
# ---------------------------------------------------------------------------
def test_validation_dose_nonpositive():
    kw = dict(BASE_KWARGS)
    kw["dose_mg"] = 0.0
    with pytest.raises(ValueError, match="dose_mg"):
        generate_pk_report(**kw)


# ---------------------------------------------------------------------------
# Test 23: Validation — cl_L_per_h <= 0
# ---------------------------------------------------------------------------
def test_validation_cl_nonpositive():
    kw = dict(BASE_KWARGS)
    kw["cl_L_per_h"] = -1.0
    with pytest.raises(ValueError, match="cl_L_per_h"):
        generate_pk_report(**kw)


# ---------------------------------------------------------------------------
# Test 24: Validation — vd_L <= 0
# ---------------------------------------------------------------------------
def test_validation_vd_nonpositive():
    kw = dict(BASE_KWARGS)
    kw["vd_L"] = 0.0
    with pytest.raises(ValueError, match="vd_L"):
        generate_pk_report(**kw)


# ---------------------------------------------------------------------------
# Test 25: Validation — interval_h < 0
# ---------------------------------------------------------------------------
def test_validation_negative_interval():
    kw = dict(BASE_KWARGS)
    kw["interval_h"] = -1.0
    with pytest.raises(ValueError, match="interval_h"):
        generate_pk_report(**kw)


# ---------------------------------------------------------------------------
# Test 26: compare_simulations — returns dict
# ---------------------------------------------------------------------------
def test_compare_simulations_returns_dict():
    r = generate_pk_report(**BASE_KWARGS)
    result = compare_simulations([r])
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Test 27: compare_simulations — required keys
# ---------------------------------------------------------------------------
def test_compare_simulations_keys():
    r = generate_pk_report(**BASE_KWARGS)
    result = compare_simulations([r])
    for key in ("table", "best_auc", "best_cmax", "longest_t_half", "summary"):
        assert key in result


# ---------------------------------------------------------------------------
# Test 28: compare_simulations — table length matches reports
# ---------------------------------------------------------------------------
def test_compare_simulations_table_length():
    r1 = generate_pk_report(**BASE_KWARGS)
    kw2 = dict(BASE_KWARGS)
    kw2["drug_name"] = "DrugB"
    kw2["cl_L_per_h"] = 5.0
    t2, c2 = _iv_profile(cl=5.0)
    kw2["times_h"] = t2
    kw2["c_plasma"] = c2
    r2 = generate_pk_report(**kw2)
    result = compare_simulations([r1, r2])
    assert len(result["table"]) == 2


# ---------------------------------------------------------------------------
# Test 29: compare_simulations — best_auc is a string (drug_name)
# ---------------------------------------------------------------------------
def test_compare_simulations_best_auc_string():
    r = generate_pk_report(**BASE_KWARGS)
    result = compare_simulations([r])
    assert isinstance(result["best_auc"], str)


# ---------------------------------------------------------------------------
# Test 30: compare_simulations — correct best_auc when two drugs
# ---------------------------------------------------------------------------
def test_compare_simulations_best_auc_correct():
    kw_low_cl = dict(BASE_KWARGS)
    kw_low_cl["drug_name"] = "LowCL"
    kw_low_cl["cl_L_per_h"] = 0.5
    t1, c1 = _iv_profile(cl=0.5)
    kw_low_cl["times_h"] = t1
    kw_low_cl["c_plasma"] = c1
    r_low = generate_pk_report(**kw_low_cl)

    kw_high_cl = dict(BASE_KWARGS)
    kw_high_cl["drug_name"] = "HighCL"
    kw_high_cl["cl_L_per_h"] = 10.0
    t2, c2 = _iv_profile(cl=10.0)
    kw_high_cl["times_h"] = t2
    kw_high_cl["c_plasma"] = c2
    r_high = generate_pk_report(**kw_high_cl)

    result = compare_simulations([r_low, r_high])
    assert result["best_auc"] == "LowCL"


# ---------------------------------------------------------------------------
# Test 31: compare_simulations — empty list raises ValueError
# ---------------------------------------------------------------------------
def test_compare_simulations_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        compare_simulations([])


# ---------------------------------------------------------------------------
# Test 32: compare_simulations — summary is a string
# ---------------------------------------------------------------------------
def test_compare_simulations_summary_string():
    r = generate_pk_report(**BASE_KWARGS)
    result = compare_simulations([r])
    assert isinstance(result["summary"], str)


# ---------------------------------------------------------------------------
# Test 33: compare_simulations — table rows contain drug_name
# ---------------------------------------------------------------------------
def test_compare_simulations_table_has_drug_name():
    r = generate_pk_report(**BASE_KWARGS)
    result = compare_simulations([r])
    assert result["table"][0]["drug_name"] == "TestDrug"


# ---------------------------------------------------------------------------
# Test 34: compare_simulations — table rows contain cmax_mg_L
# ---------------------------------------------------------------------------
def test_compare_simulations_table_has_cmax():
    r = generate_pk_report(**BASE_KWARGS)
    result = compare_simulations([r])
    assert "cmax_mg_L" in result["table"][0]


# ---------------------------------------------------------------------------
# Test 35: Higher CL → lower AUC in report
# ---------------------------------------------------------------------------
def test_higher_cl_lower_auc_in_report():
    t1, c1 = _iv_profile(cl=1.0)
    kw_low = dict(BASE_KWARGS, cl_L_per_h=1.0, times_h=t1, c_plasma=c1)
    t2, c2 = _iv_profile(cl=10.0)
    kw_high = dict(BASE_KWARGS, cl_L_per_h=10.0, times_h=t2, c_plasma=c2)
    r_low = generate_pk_report(**kw_low)
    r_high = generate_pk_report(**kw_high)
    assert r_low.auc_inf_mg_h_per_L > r_high.auc_inf_mg_h_per_L


# ---------------------------------------------------------------------------
# Test 36: MRT ~ 1/ke for 1-cpt IV
# ---------------------------------------------------------------------------
def test_mrt_approx_one_over_ke():
    result = generate_pk_report(**BASE_KWARGS)
    expected_mrt = 1.0 / (2.0 / 20.0)  # 1/ke = Vd/CL = 10 h
    assert result.mrt_h == pytest.approx(expected_mrt, rel=0.15)


# ---------------------------------------------------------------------------
# Test 37: interval_h stored correctly
# ---------------------------------------------------------------------------
def test_interval_h_stored():
    kw = dict(BASE_KWARGS)
    kw["interval_h"] = 8.0
    result = generate_pk_report(**kw)
    assert result.interval_h == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Test 38: Oral route profile — Tmax > 0
# ---------------------------------------------------------------------------
def test_oral_tmax_positive():
    """Use a fake oral profile where Cmax is not at t=0."""
    # Ascending then descending
    times = [float(i) for i in range(25)]
    # Simulate: rises to peak at t=2, then decays
    ke = 0.1
    ka = 1.0
    dose = 100.0
    vd = 20.0
    f = 0.8
    coeff = f * dose * ka / (vd * (ka - ke))
    concs = [coeff * (math.exp(-ke * t) - math.exp(-ka * t)) for t in times]
    concs = [max(c, 0.0) for c in concs]
    kw = dict(BASE_KWARGS)
    kw["route"] = "oral"
    kw["times_h"] = times
    kw["c_plasma"] = concs
    result = generate_pk_report(**kw)
    assert result.tmax_h > 0.0


# ---------------------------------------------------------------------------
# Test 39: AUC > 0 for valid profile
# ---------------------------------------------------------------------------
def test_auc_positive():
    result = generate_pk_report(**BASE_KWARGS)
    assert result.auc_inf_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Test 40: t_half > 0
# ---------------------------------------------------------------------------
def test_t_half_positive():
    result = generate_pk_report(**BASE_KWARGS)
    assert result.t_half_h > 0.0
