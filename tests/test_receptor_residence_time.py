"""Tests for prediction/receptor_residence_time.py — Phase 458."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.receptor_residence_time import (
    ReceptorResidenceResult,
    compare_residence_times,
    predict_residence_time,
)

# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = predict_residence_time("DrugA", "TargetX", kon_M_per_s=1e5, koff_per_s=1e-3)
    assert isinstance(result, ReceptorResidenceResult)


def test_frozen_dataclass():
    result = predict_residence_time("DrugA", "TargetX", kon_M_per_s=1e5, koff_per_s=1e-3)
    with pytest.raises(Exception):
        result.drug_name = "Y"  # type: ignore[misc]


def test_drug_name_stored():
    result = predict_residence_time("Ibrutinib", "BTK", kon_M_per_s=1e6, koff_per_s=1e-4)
    assert result.drug_name == "Ibrutinib"


def test_target_name_stored():
    result = predict_residence_time("DrugA", "COX2", kon_M_per_s=1e5, koff_per_s=0.01)
    assert result.target_name == "COX2"


# ---------------------------------------------------------------------------
# Exact RT = 1/koff
# ---------------------------------------------------------------------------


def test_rt_exact_seconds():
    koff = 0.001  # s^-1 -> RT = 1000 s
    result = predict_residence_time("D", "T", kon_M_per_s=1e6, koff_per_s=koff)
    assert math.isclose(result.residence_time_s, 1.0 / koff, rel_tol=1e-12)


def test_rt_minutes_conversion():
    koff = 0.01  # s^-1 -> RT = 100 s = 100/60 min
    result = predict_residence_time("D", "T", kon_M_per_s=1e6, koff_per_s=koff)
    assert math.isclose(result.residence_time_min, (1.0 / koff) / 60.0, rel_tol=1e-12)


def test_rt_hours_conversion():
    koff = 1e-4  # s^-1 -> RT = 10000 s
    result = predict_residence_time("D", "T", kon_M_per_s=1e6, koff_per_s=koff)
    assert math.isclose(result.residence_time_h, (1.0 / koff) / 3600.0, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Kd calculation
# ---------------------------------------------------------------------------


def test_kd_nm_formula():
    kon = 1e6  # M^-1 s^-1
    koff = 1e-3  # s^-1
    # Kd = koff/kon = 1e-9 M = 1 nM
    result = predict_residence_time("D", "T", kon_M_per_s=kon, koff_per_s=koff)
    expected_kd_nM = (koff / kon) * 1e9
    assert math.isclose(result.kd_nM, expected_kd_nM, rel_tol=1e-12)


def test_kd_nm_value_1nm():
    result = predict_residence_time("D", "T", kon_M_per_s=1e6, koff_per_s=1e-3)
    assert math.isclose(result.kd_nM, 1.0, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# t50 = ln(2) * RT
# ---------------------------------------------------------------------------


def test_t50_recovery_exact():
    koff = 0.005  # s^-1
    rt_s = 1.0 / koff
    rt_min = rt_s / 60.0
    expected_t50_min = math.log(2.0) * rt_min
    result = predict_residence_time("D", "T", kon_M_per_s=1e5, koff_per_s=koff)
    assert math.isclose(result.t50_recovery_min, expected_t50_min, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Occupancy class
# ---------------------------------------------------------------------------


def test_occupancy_class_transient():
    # RT < 1 min -> koff must be very fast: RT_s < 60 -> koff > 1/60
    koff = 0.1  # s^-1 -> RT = 10 s = 0.167 min  (transient)
    result = predict_residence_time("D", "T", kon_M_per_s=1e7, koff_per_s=koff)
    assert result.occupancy_class == "transient"


def test_occupancy_class_short():
    # 1 min <= RT < 30 min -> 60 s <= RT_s < 1800 s -> koff in (1/1800, 1/60)
    koff = 1.0 / 600.0  # RT = 600 s = 10 min  (short)
    result = predict_residence_time("D", "T", kon_M_per_s=1e6, koff_per_s=koff)
    assert result.occupancy_class == "short"


def test_occupancy_class_moderate():
    # 30 min <= RT <= 240 min -> RT_s in [1800, 14400] -> koff in [1/14400, 1/1800]
    koff = 1.0 / 3600.0  # RT = 3600 s = 60 min  (moderate)
    result = predict_residence_time("D", "T", kon_M_per_s=1e6, koff_per_s=koff)
    assert result.occupancy_class == "moderate"


def test_occupancy_class_long():
    # RT > 240 min -> koff < 1/14400
    koff = 1.0 / 36000.0  # RT = 36000 s = 600 min  (long)
    result = predict_residence_time("D", "T", kon_M_per_s=1e6, koff_per_s=koff)
    assert result.occupancy_class == "long"


# ---------------------------------------------------------------------------
# Kinetic selectivity
# ---------------------------------------------------------------------------


def test_kinetic_selectivity_no_ref_is_1():
    result = predict_residence_time("D", "T", kon_M_per_s=1e6, koff_per_s=1e-3)
    assert result.kinetic_selectivity_vs_ref == 1.0


def test_kinetic_selectivity_longer_rt_gt_1():
    """Drug koff slower than ref -> KS > 1."""
    koff_drug = 1e-4  # slower koff -> longer RT
    koff_ref = 1e-3
    result = predict_residence_time(
        "D", "T", kon_M_per_s=1e6, koff_per_s=koff_drug, ref_koff_per_s=koff_ref
    )
    assert result.kinetic_selectivity_vs_ref > 1.0


def test_kinetic_selectivity_formula():
    koff_drug = 1e-4
    koff_ref = 1e-3
    result = predict_residence_time(
        "D", "T", kon_M_per_s=1e6, koff_per_s=koff_drug, ref_koff_per_s=koff_ref
    )
    expected_ks = koff_ref / koff_drug  # = 10
    assert math.isclose(result.kinetic_selectivity_vs_ref, expected_ks, rel_tol=1e-12)


def test_kinetic_selectivity_shorter_rt_lt_1():
    """Drug koff faster than ref -> KS < 1."""
    koff_drug = 1e-2  # faster koff -> shorter RT
    koff_ref = 1e-3
    result = predict_residence_time(
        "D", "T", kon_M_per_s=1e6, koff_per_s=koff_drug, ref_koff_per_s=koff_ref
    )
    assert result.kinetic_selectivity_vs_ref < 1.0


# ---------------------------------------------------------------------------
# compare_residence_times
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    drugs = [
        {"drug_name": "A", "target_name": "T", "kon_M_per_s": 1e6, "koff_per_s": 1e-3},
        {"drug_name": "B", "target_name": "T", "kon_M_per_s": 1e6, "koff_per_s": 1e-4},
    ]
    results = compare_residence_times(drugs)
    assert isinstance(results, list)
    assert len(results) == 2


def test_compare_sorted_by_rt_descending():
    drugs = [
        {"drug_name": "Fast", "target_name": "T", "kon_M_per_s": 1e6, "koff_per_s": 0.1},
        {"drug_name": "Mid", "target_name": "T", "kon_M_per_s": 1e6, "koff_per_s": 1e-3},
        {"drug_name": "Slow", "target_name": "T", "kon_M_per_s": 1e6, "koff_per_s": 1e-5},
    ]
    results = compare_residence_times(drugs)
    rts = [r.residence_time_h for r in results]
    assert rts == sorted(rts, reverse=True)


def test_compare_longest_rt_first():
    drugs = [
        {"drug_name": "Short", "target_name": "T", "kon_M_per_s": 1e6, "koff_per_s": 1.0},
        {"drug_name": "Long", "target_name": "T", "kon_M_per_s": 1e6, "koff_per_s": 1e-6},
    ]
    results = compare_residence_times(drugs)
    assert results[0].drug_name == "Long"


def test_compare_empty_list():
    assert compare_residence_times([]) == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_kon_zero_raises():
    with pytest.raises(ValueError):
        predict_residence_time("D", "T", kon_M_per_s=0.0, koff_per_s=1e-3)


def test_invalid_kon_negative_raises():
    with pytest.raises(ValueError):
        predict_residence_time("D", "T", kon_M_per_s=-1e6, koff_per_s=1e-3)


def test_invalid_koff_zero_raises():
    with pytest.raises(ValueError):
        predict_residence_time("D", "T", kon_M_per_s=1e6, koff_per_s=0.0)


def test_invalid_koff_negative_raises():
    with pytest.raises(ValueError):
        predict_residence_time("D", "T", kon_M_per_s=1e6, koff_per_s=-1e-3)


def test_invalid_ref_koff_zero_raises():
    with pytest.raises(ValueError):
        predict_residence_time("D", "T", kon_M_per_s=1e6, koff_per_s=1e-3, ref_koff_per_s=0.0)


def test_invalid_ref_koff_negative_raises():
    with pytest.raises(ValueError):
        predict_residence_time("D", "T", kon_M_per_s=1e6, koff_per_s=1e-3, ref_koff_per_s=-1e-4)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_non_empty():
    result = predict_residence_time("D", "T", kon_M_per_s=1e6, koff_per_s=1e-3)
    assert len(result.notes) > 0


def test_notes_contains_rt():
    result = predict_residence_time("D", "T", kon_M_per_s=1e6, koff_per_s=1e-3)
    assert "RT=" in result.notes
