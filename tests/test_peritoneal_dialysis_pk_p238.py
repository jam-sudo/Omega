"""Tests for Phase 238: peritoneal_dialysis_pk_p238."""

import pytest

from omega_pbpk.core.peritoneal_dialysis_pk_p238 import (
    PeritonealDialysisPKResult,
    compare_dwell_times,
    simulate_peritoneal_dialysis_pk,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _default_sim(**kwargs):
    params = dict(
        drug_name="TestDrug",
        dose_mg=500.0,
        dwell_time_h=4.0,
        cl_sys_L_per_h=2.0,
        vp_L=3.0,
    )
    params.update(kwargs)
    return simulate_peritoneal_dialysis_pk(**params)


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_return_type():
    result = _default_sim()
    assert isinstance(result, PeritonealDialysisPKResult)


def test_result_fields_present():
    result = _default_sim()
    assert result.drug_name == "TestDrug"
    assert result.dose_mg == 500.0
    assert result.dwell_time_h == 4.0


# ---------------------------------------------------------------------------
# Plasma concentration tests
# ---------------------------------------------------------------------------


def test_c_plasma_starts_high():
    result = _default_sim()
    # Initial plasma concentration = dose / vp = 500/3 ~ 166.7 mg/L
    assert result.c_plasma_mg_L[0] > 100.0


def test_c_plasma_decreases_overall():
    result = _default_sim()
    # Plasma should be lower at end than at start
    assert result.c_plasma_mg_L[-1] < result.c_plasma_mg_L[0]


def test_cmax_plasma_equals_initial():
    result = _default_sim()
    # For IV bolus, Cmax should be at t=0
    assert result.cmax_plasma_mg_L == pytest.approx(result.c_plasma_mg_L[0], rel=1e-6)


def test_trough_plasma_non_negative():
    result = _default_sim()
    assert result.trough_plasma_mg_L >= 0.0


# ---------------------------------------------------------------------------
# Dialysate concentration tests
# ---------------------------------------------------------------------------


def test_c_dialysate_starts_at_zero():
    result = _default_sim()
    assert result.c_dialysate_mg_L[0] == pytest.approx(0.0)


def test_c_dialysate_rises_during_dwell():
    result = _default_sim()
    # Find index near dwell_time (4h) with dt=0.05 -> step 80
    mid_idx = 40  # ~2h in
    assert result.c_dialysate_mg_L[mid_idx] > 0.0


# ---------------------------------------------------------------------------
# Drug removal tests
# ---------------------------------------------------------------------------


def test_drug_removed_positive():
    result = _default_sim()
    assert result.drug_removed_mg > 0.0


def test_drug_removed_less_than_dose():
    result = _default_sim()
    assert result.drug_removed_mg < result.dose_mg


def test_clearance_pd_non_negative():
    result = _default_sim()
    assert result.clearance_pd_L_per_h >= 0.0


def test_auc_plasma_positive():
    result = _default_sim()
    assert result.auc_plasma_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Time series structural tests
# ---------------------------------------------------------------------------


def test_times_start_at_zero():
    result = _default_sim()
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_monotonically_increasing():
    result = _default_sim()
    for i in range(1, len(result.times_h)):
        assert result.times_h[i] > result.times_h[i - 1]


# ---------------------------------------------------------------------------
# Compare dwell times tests
# ---------------------------------------------------------------------------


def test_compare_dwell_returns_four():
    results = compare_dwell_times("TestDrug", 500.0, 2.0)
    assert len(results) == 4


def test_compare_dwell_sorted_descending():
    results = compare_dwell_times("TestDrug", 500.0, 2.0)
    removals = [r.drug_removed_mg for r in results]
    assert removals == sorted(removals, reverse=True)


def test_compare_dwell_custom_times():
    results = compare_dwell_times("TestDrug", 500.0, 2.0, dwell_times_h=[1.0, 3.0, 6.0])
    assert len(results) == 3


def test_longer_dwell_more_removal():
    # The longer the dwell, the more drug diffuses into dialysate during closed phase
    res_short = simulate_peritoneal_dialysis_pk(
        "TestDrug", 500.0, dwell_time_h=2.0, cl_sys_L_per_h=2.0
    )
    res_long = simulate_peritoneal_dialysis_pk(
        "TestDrug", 500.0, dwell_time_h=8.0, cl_sys_L_per_h=2.0
    )
    assert res_long.drug_removed_mg > res_short.drug_removed_mg


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_peritoneal_dialysis_pk("Drug", -10.0, 4.0, 2.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_peritoneal_dialysis_pk("Drug", 100.0, 4.0, cl_sys_L_per_h=0.0)


def test_invalid_vp_raises():
    with pytest.raises(ValueError, match="vp_L"):
        simulate_peritoneal_dialysis_pk("Drug", 100.0, 4.0, 2.0, vp_L=0.0)


def test_invalid_dwell_raises():
    with pytest.raises(ValueError, match="dwell_time_h"):
        simulate_peritoneal_dialysis_pk("Drug", 100.0, dwell_time_h=0.0, cl_sys_L_per_h=2.0)
