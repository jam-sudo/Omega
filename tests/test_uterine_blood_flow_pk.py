"""Tests for Phase 461 — Uterine Blood Flow PK."""

import pytest

from omega_pbpk.core.uterine_blood_flow_pk import (
    UterinePKResult,
    compare_gestational_weeks,
    simulate_uterine_pk,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _default_result(**kwargs) -> UterinePKResult:
    params = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=20.0,
        gestational_week=20,
    )
    params.update(kwargs)
    return simulate_uterine_pk(**params)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _default_result()
    assert isinstance(result, UterinePKResult)


# ---------------------------------------------------------------------------
# Basic field checks
# ---------------------------------------------------------------------------


def test_drug_name_preserved():
    result = _default_result(drug_name="Aspirin")
    assert result.drug_name == "Aspirin"


def test_gestational_week_preserved():
    result = _default_result(gestational_week=30)
    assert result.gestational_week == 30


def test_dose_mg_preserved():
    result = _default_result(dose_mg=50.0)
    assert result.dose_mg == 50.0


# ---------------------------------------------------------------------------
# Time series checks
# ---------------------------------------------------------------------------


def test_times_start_at_zero():
    result = _default_result()
    assert result.times_h[0] == pytest.approx(0.0)


def test_c_systemic_starts_at_dose_over_vd():
    result = _default_result(dose_mg=100.0, vd_sys_L=20.0)
    assert result.c_systemic_mg_L[0] == pytest.approx(100.0 / 20.0, rel=1e-6)


def test_c_systemic_decreases_over_time():
    result = _default_result()
    # IV bolus should cause monotonically decreasing systemic concentration
    c = result.c_systemic_mg_L
    for i in range(1, len(c)):
        assert c[i] <= c[i - 1] + 1e-10


def test_c_uterus_starts_at_zero():
    result = _default_result()
    assert result.c_uterus_mg_L[0] == pytest.approx(0.0)


def test_c_uterus_rises_then_falls():
    result = _default_result()
    # cmax should be > 0 and occur after t=0
    assert result.cmax_uterus_mg_L > 0.0
    assert result.tmax_uterus_h > 0.0


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------


def test_cmax_uterus_positive():
    result = _default_result()
    assert result.cmax_uterus_mg_L > 0.0


def test_tmax_uterus_nonnegative():
    result = _default_result()
    assert result.tmax_uterus_h >= 0.0


def test_auc_uterus_positive():
    result = _default_result()
    assert result.auc_uterus_mg_h_per_L > 0.0


def test_uterine_to_systemic_ratio_nonnegative():
    result = _default_result()
    assert result.uterine_to_systemic_ratio >= 0.0


# ---------------------------------------------------------------------------
# UBF formula
# ---------------------------------------------------------------------------


def test_ubf_formula_gestational_week_20():
    # UBF = (50 + 6*20) mL/min * 60 / 1000 = 170 * 0.06 = 10.2 L/h
    result = _default_result(gestational_week=20)
    expected = (50 + 6 * 20) * 60 / 1000
    assert result.uterine_blood_flow_L_per_h == pytest.approx(expected, rel=1e-6)


def test_ubf_formula_gestational_week_0():
    result = _default_result(gestational_week=0)
    expected = 50 * 60 / 1000  # 3.0 L/h
    assert result.uterine_blood_flow_L_per_h == pytest.approx(expected, rel=1e-6)


def test_ubf_formula_gestational_week_40():
    result = _default_result(gestational_week=40)
    expected = (50 + 6 * 40) * 60 / 1000
    assert result.uterine_blood_flow_L_per_h == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Later gestational week → higher UBF → higher uterine exposure
# ---------------------------------------------------------------------------


def test_later_week_higher_ubf():
    r_early = _default_result(gestational_week=10)
    r_late = _default_result(gestational_week=38)
    assert r_late.uterine_blood_flow_L_per_h > r_early.uterine_blood_flow_L_per_h


def test_later_week_higher_auc():
    r_early = _default_result(gestational_week=5)
    r_late = _default_result(gestational_week=40)
    assert r_late.auc_uterus_mg_h_per_L > r_early.auc_uterus_mg_h_per_L


def test_later_week_higher_cmax():
    r_early = _default_result(gestational_week=5)
    r_late = _default_result(gestational_week=40)
    assert r_late.cmax_uterus_mg_L > r_early.cmax_uterus_mg_L


# ---------------------------------------------------------------------------
# compare_gestational_weeks
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    results = compare_gestational_weeks("Drug", 100.0, 5.0, 20.0, [10, 20, 30])
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_sorted_by_ubf_ascending():
    results = compare_gestational_weeks("Drug", 100.0, 5.0, 20.0, [30, 10, 20])
    ubfs = [r.uterine_blood_flow_L_per_h for r in results]
    assert ubfs == sorted(ubfs)


def test_compare_all_are_uterine_results():
    results = compare_gestational_weeks("Drug", 100.0, 5.0, 20.0, [5, 15, 25])
    for r in results:
        assert isinstance(r, UterinePKResult)


def test_compare_single_week():
    results = compare_gestational_weeks("Drug", 100.0, 5.0, 20.0, [20])
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_uterine_pk("D", 0.0, 5.0, 20.0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_uterine_pk("D", -10.0, 5.0, 20.0)


def test_invalid_cl_sys():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_uterine_pk("D", 100.0, 0.0, 20.0)


def test_invalid_vd_sys():
    with pytest.raises(ValueError, match="vd_sys_L"):
        simulate_uterine_pk("D", 100.0, 5.0, -1.0)


def test_invalid_kp_uterus():
    with pytest.raises(ValueError, match="Kp_uterus"):
        simulate_uterine_pk("D", 100.0, 5.0, 20.0, Kp_uterus=-1.0)


def test_invalid_vd_uterus():
    with pytest.raises(ValueError, match="vd_uterus_L"):
        simulate_uterine_pk("D", 100.0, 5.0, 20.0, vd_uterus_L=0.0)


def test_invalid_gestational_week_negative():
    with pytest.raises(ValueError, match="gestational_week"):
        simulate_uterine_pk("D", 100.0, 5.0, 20.0, gestational_week=-1)


def test_invalid_gestational_week_too_large():
    with pytest.raises(ValueError, match="gestational_week"):
        simulate_uterine_pk("D", 100.0, 5.0, 20.0, gestational_week=43)


def test_notes_is_string():
    result = _default_result()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
