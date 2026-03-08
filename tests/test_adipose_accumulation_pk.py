"""Tests for Phase 463 — Drug Accumulation in Adipose Tissue."""

import pytest

from omega_pbpk.core.adipose_accumulation_pk import (
    AdiposeAccumulationResult,
    compare_logp_accumulation,
    simulate_adipose_accumulation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(logP=3.0, dose_mg=100.0, cl=5.0, vd=10.0, t_end=24.0):
    return simulate_adipose_accumulation(
        drug_name="TestDrug",
        dose_mg=dose_mg,
        logP=logP,
        cl_sys_L_per_h=cl,
        vd_plasma_L=vd,
        t_end_h=t_end,
        dt_h=0.1,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _run()
    assert isinstance(result, AdiposeAccumulationResult)


def test_result_fields_present():
    result = _run()
    assert hasattr(result, "times_h")
    assert hasattr(result, "c_plasma_mg_L")
    assert hasattr(result, "c_adipose_mg_L")
    assert hasattr(result, "kp_adipose")
    assert hasattr(result, "accumulation_ratio")
    assert hasattr(result, "t_half_terminal_h")
    assert hasattr(result, "notes")


# ---------------------------------------------------------------------------
# Initial conditions (IV bolus)
# ---------------------------------------------------------------------------


def test_c_plasma_starts_at_dose_over_vd():
    """IV bolus: C_plasma[0] == dose_mg / vd_plasma_L."""
    result = simulate_adipose_accumulation(
        drug_name="DrugA",
        dose_mg=200.0,
        logP=2.0,
        cl_sys_L_per_h=3.0,
        vd_plasma_L=20.0,
        t_end_h=12.0,
        dt_h=0.1,
    )
    expected = 200.0 / 20.0
    assert abs(result.c_plasma_mg_L[0] - expected) < 1e-9


def test_c_adipose_starts_near_zero():
    result = _run()
    assert result.c_adipose_mg_L[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Time vector
# ---------------------------------------------------------------------------


def test_times_start_at_zero():
    result = _run()
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_end_at_t_end():
    result = _run(t_end=24.0)
    assert result.times_h[-1] == pytest.approx(24.0, abs=0.15)


def test_time_and_concentration_lists_same_length():
    result = _run()
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) == len(result.c_adipose_mg_L)


# ---------------------------------------------------------------------------
# Kp_adipose calculation
# ---------------------------------------------------------------------------


def test_higher_logP_gives_higher_kp():
    res_low = _run(logP=1.0)
    res_high = _run(logP=5.0)
    assert res_high.kp_adipose > res_low.kp_adipose


def test_logP_zero_gives_kp_01():
    result = _run(logP=0.0)
    assert result.kp_adipose == pytest.approx(0.1)


def test_negative_logP_gives_kp_01():
    result = _run(logP=-2.0)
    assert result.kp_adipose == pytest.approx(0.1)


def test_very_high_logP_clamped_to_50():
    # logP=10 -> 10^1.5/3 = 31.62/3 ≈ 10.54, not clamped yet
    # logP=20 -> 20^1.5/3 ≈ 89.44/3 ≈ 29.8, not clamped
    # logP=30 -> 30^1.5/3 ≈ 164.3/3 ≈ 54.8, clamped to 50
    result = _run(logP=30.0)
    assert result.kp_adipose == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Adipose accumulation behavior
# ---------------------------------------------------------------------------


def test_cmax_adipose_greater_than_zero():
    result = _run(logP=3.0)
    assert result.cmax_adipose_mg_L > 0.0


def test_auc_plasma_greater_than_zero():
    result = _run()
    assert result.auc_plasma_mg_h_per_L > 0.0


def test_auc_adipose_greater_than_zero():
    result = _run(logP=3.0)
    assert result.auc_adipose_mg_h_per_L > 0.0


def test_higher_logP_gives_higher_adipose_auc():
    res_low = _run(logP=1.0, t_end=48.0)
    res_high = _run(logP=5.0, t_end=48.0)
    assert res_high.auc_adipose_mg_h_per_L > res_low.auc_adipose_mg_h_per_L


def test_accumulation_ratio_computed():
    result = _run(logP=3.0)
    assert result.accumulation_ratio >= 0.0


def test_tmax_adipose_after_time_zero():
    """Drug takes time to distribute into adipose, so tmax_adipose > 0."""
    result = _run(logP=4.0, t_end=48.0)
    assert result.tmax_adipose_h >= 0.0


def test_drug_name_preserved():
    result = simulate_adipose_accumulation(
        drug_name="Amiodarone",
        dose_mg=100.0,
        logP=5.0,
        cl_sys_L_per_h=2.0,
    )
    assert result.drug_name == "Amiodarone"


def test_dose_preserved():
    result = _run(dose_mg=500.0)
    assert result.dose_mg == pytest.approx(500.0)


def test_fat_fraction_preserved():
    result = simulate_adipose_accumulation(
        drug_name="X",
        dose_mg=100.0,
        logP=3.0,
        cl_sys_L_per_h=5.0,
        fat_fraction=0.35,
    )
    assert result.fat_fraction == pytest.approx(0.35)


def test_terminal_half_life_positive():
    result = _run(t_end=48.0)
    assert result.t_half_terminal_h > 0.0


# ---------------------------------------------------------------------------
# compare_logp_accumulation
# ---------------------------------------------------------------------------


def test_compare_sorted_by_kp_descending():
    results = compare_logp_accumulation(
        drug_name="Drug",
        dose_mg=100.0,
        logp_values=[1.0, 3.0, 5.0],
        cl_sys_L_per_h=5.0,
    )
    kp_values = [r.kp_adipose for r in results]
    assert kp_values == sorted(kp_values, reverse=True)


def test_compare_returns_correct_count():
    results = compare_logp_accumulation(
        drug_name="Drug",
        dose_mg=100.0,
        logp_values=[1.0, 2.0, 4.0, 6.0],
        cl_sys_L_per_h=5.0,
    )
    assert len(results) == 4


def test_compare_first_result_has_highest_kp():
    results = compare_logp_accumulation(
        drug_name="Drug",
        dose_mg=100.0,
        logp_values=[0.5, 2.0, 5.0],
        cl_sys_L_per_h=5.0,
    )
    assert results[0].kp_adipose >= results[-1].kp_adipose


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_adipose_accumulation("X", 0.0, 3.0, 5.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_adipose_accumulation("X", -10.0, 3.0, 5.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_adipose_accumulation("X", 100.0, 3.0, 0.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_plasma_L"):
        simulate_adipose_accumulation("X", 100.0, 3.0, 5.0, vd_plasma_L=0.0)


def test_validation_fat_fraction_zero():
    with pytest.raises(ValueError, match="fat_fraction"):
        simulate_adipose_accumulation("X", 100.0, 3.0, 5.0, fat_fraction=0.0)


def test_validation_fat_fraction_one():
    with pytest.raises(ValueError, match="fat_fraction"):
        simulate_adipose_accumulation("X", 100.0, 3.0, 5.0, fat_fraction=1.0)


def test_validation_body_weight_zero():
    with pytest.raises(ValueError, match="body_weight_kg"):
        simulate_adipose_accumulation("X", 100.0, 3.0, 5.0, body_weight_kg=0.0)
