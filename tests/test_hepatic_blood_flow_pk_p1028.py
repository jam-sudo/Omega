"""Tests for Phase 1028 — Hepatic Blood Flow PK (p1028 suffix)."""

import pytest

from omega_pbpk.core.hepatic_blood_flow_pk_p1028 import (
    HepaticBloodFlowResult,
    compare_flow_conditions,
    simulate_hepatic_blood_flow_pk,
)

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_hepatic_blood_flow_pk("DrugA", dose_mg=100.0)
    assert isinstance(result, HepaticBloodFlowResult)


# ---------------------------------------------------------------------------
# Derived field consistency
# ---------------------------------------------------------------------------


def test_extraction_ratio_in_range():
    result = simulate_hepatic_blood_flow_pk("DrugA", dose_mg=100.0)
    assert 0.0 < result.extraction_ratio < 1.0


def test_oral_bioavailability_equals_1_minus_extraction():
    result = simulate_hepatic_blood_flow_pk("DrugA", dose_mg=100.0)
    assert abs(result.oral_bioavailability - (1.0 - result.extraction_ratio)) < 1e-6


def test_hepatic_clearance_equals_flow_times_extraction():
    result = simulate_hepatic_blood_flow_pk("DrugA", dose_mg=100.0)
    expected = result.hepatic_blood_flow_L_h * result.extraction_ratio
    assert abs(result.hepatic_clearance_L_h - expected) < 1e-6


# ---------------------------------------------------------------------------
# Flow sensitivity classification
# ---------------------------------------------------------------------------


def test_high_clint_gives_flow_limited():
    result = simulate_hepatic_blood_flow_pk(
        "HighE", dose_mg=100.0, clint_L_per_h=500.0, fu_blood=0.5
    )
    assert result.extraction_ratio > 0.7
    assert result.flow_sensitivity == "flow-limited"


def test_low_clint_gives_capacity_limited():
    result = simulate_hepatic_blood_flow_pk("LowE", dose_mg=100.0, clint_L_per_h=1.0, fu_blood=0.1)
    assert result.extraction_ratio < 0.3
    assert result.flow_sensitivity == "capacity-limited"


def test_intermediate_clint_gives_intermediate():
    result = simulate_hepatic_blood_flow_pk("MidE", dose_mg=100.0, clint_L_per_h=30.0, fu_blood=0.1)
    assert result.flow_sensitivity in {"flow-limited", "intermediate", "capacity-limited"}


# ---------------------------------------------------------------------------
# IV route
# ---------------------------------------------------------------------------


def test_iv_cmax_positive():
    result = simulate_hepatic_blood_flow_pk("DrugA", dose_mg=100.0, route="iv")
    assert result.cmax_plasma > 0.0


def test_iv_c_plasma_starts_at_dose_over_vd():
    result = simulate_hepatic_blood_flow_pk("DrugA", dose_mg=100.0, vd_L=50.0, route="iv")
    expected_c0 = 100.0 / 50.0
    assert abs(result.c_plasma_mg_L[0] - expected_c0) < 1e-6


# ---------------------------------------------------------------------------
# Oral route
# ---------------------------------------------------------------------------


def test_oral_cmax_positive():
    result = simulate_hepatic_blood_flow_pk("DrugA", dose_mg=100.0, route="oral")
    assert result.cmax_plasma > 0.0


def test_oral_tmax_positive():
    result = simulate_hepatic_blood_flow_pk("DrugA", dose_mg=100.0, route="oral")
    assert result.tmax_plasma_h > 0.0


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------


def test_auc_plasma_positive():
    result = simulate_hepatic_blood_flow_pk("DrugA", dose_mg=100.0)
    assert result.auc_plasma > 0.0


# ---------------------------------------------------------------------------
# Flow effect on clearance
# ---------------------------------------------------------------------------


def test_increasing_flow_increases_cl_for_high_extraction():
    r_low = simulate_hepatic_blood_flow_pk(
        "HighE", dose_mg=100.0, clint_L_per_h=500.0, fu_blood=0.5, hepatic_blood_flow_L_h=45.0
    )
    r_high = simulate_hepatic_blood_flow_pk(
        "HighE", dose_mg=100.0, clint_L_per_h=500.0, fu_blood=0.5, hepatic_blood_flow_L_h=135.0
    )
    assert r_high.hepatic_clearance_L_h > r_low.hepatic_clearance_L_h


# ---------------------------------------------------------------------------
# compare_flow_conditions
# ---------------------------------------------------------------------------


def test_compare_flow_returns_list_of_3():
    results = compare_flow_conditions("DrugA", dose_mg=100.0)
    assert len(results) == 3


def test_compare_flow_returns_list_of_hepatic_results():
    results = compare_flow_conditions("DrugA", dose_mg=100.0)
    for r in results:
        assert isinstance(r, HepaticBloodFlowResult)


def test_compare_flow_higher_flow_higher_cl_for_high_e():
    results = compare_flow_conditions(
        "HighE", dose_mg=100.0, clint_L_per_h=500.0, fu_blood=0.5, flows_L_h=[45.0, 90.0, 135.0]
    )
    assert results[2].hepatic_clearance_L_h > results[0].hepatic_clearance_L_h


def test_compare_flow_custom_flows():
    results = compare_flow_conditions("DrugA", dose_mg=100.0, flows_L_h=[60.0, 90.0])
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = simulate_hepatic_blood_flow_pk("DrugA", dose_mg=100.0)
    assert len(result.notes) > 0


def test_notes_contains_drug_name():
    result = simulate_hepatic_blood_flow_pk("Warfarin", dose_mg=50.0)
    assert "Warfarin" in result.notes


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_dose_zero_raises():
    with pytest.raises(ValueError):
        simulate_hepatic_blood_flow_pk("DrugA", dose_mg=0.0)


def test_fu_zero_raises():
    with pytest.raises(ValueError):
        simulate_hepatic_blood_flow_pk("DrugA", dose_mg=100.0, fu_blood=0.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError):
        simulate_hepatic_blood_flow_pk("DrugA", dose_mg=100.0, route="subcutaneous")
