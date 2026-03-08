"""
Tests for Phase 400 — Drug Accumulation Index Calculator.
~22 tests covering result types, formulas, steady-state PK,
time-to-SS, regimen comparison, and validation.
"""

import math

import pytest

from omega_pbpk.clinical.accumulation_index_calculator import (
    AccumulationIndexResult,
    calculate_accumulation,
    compare_dosing_regimens,
)

# ---------------------------------------------------------------------------
# Return-type / structure tests
# ---------------------------------------------------------------------------


def test_calculate_accumulation_returns_correct_type():
    r = calculate_accumulation("Drug", dose_mg=100, t_half_h=12, dosing_interval_h=24, vd_L=50)
    assert isinstance(r, AccumulationIndexResult)


def test_result_is_frozen_dataclass():
    r = calculate_accumulation("Drug", dose_mg=100, t_half_h=12, dosing_interval_h=24, vd_L=50)
    with pytest.raises((AttributeError, TypeError)):
        r.accumulation_index = 999.0  # type: ignore[misc]


def test_result_has_required_fields():
    r = calculate_accumulation("Drug", dose_mg=100, t_half_h=12, dosing_interval_h=24, vd_L=50)
    for field in [
        "drug_name",
        "dose_mg",
        "dosing_interval_h",
        "t_half_h",
        "ke_per_h",
        "accumulation_index",
        "trough_accumulation_factor",
        "cmax_accumulation_factor",
        "css_avg_mg_L",
        "css_max_mg_L",
        "css_min_mg_L",
        "fluctuation_pct",
        "doses_to_90pct_ss",
        "time_to_90pct_ss_h",
        "time_to_95pct_ss_h",
        "notes",
    ]:
        assert hasattr(r, field), f"Missing field: {field}"


def test_drug_name_preserved():
    r = calculate_accumulation("Warfarin", dose_mg=5, t_half_h=40, dosing_interval_h=24, vd_L=10)
    assert r.drug_name == "Warfarin"
    assert r.dose_mg == 5.0


# ---------------------------------------------------------------------------
# Formula correctness tests
# ---------------------------------------------------------------------------


def test_ke_formula():
    """ke = ln2 / t_half."""
    r = calculate_accumulation("D", dose_mg=100, t_half_h=10, dosing_interval_h=24, vd_L=50)
    expected_ke = math.log(2) / 10.0
    assert abs(r.ke_per_h - expected_ke) < 1e-9


def test_accumulation_index_formula():
    """Rac = 1 / (1 - exp(-ke * tau))."""
    t_half = 12.0
    tau = 24.0
    ke = math.log(2) / t_half
    expected_rac = 1.0 / (1.0 - math.exp(-ke * tau))
    r = calculate_accumulation("D", dose_mg=100, t_half_h=t_half, dosing_interval_h=tau, vd_L=50)
    assert abs(r.accumulation_index - expected_rac) < 1e-6


def test_css_avg_formula():
    """Css_avg = F * dose / (Vd * ke * tau)."""
    dose = 200.0
    t_half = 8.0
    tau = 12.0
    vd = 40.0
    f = 0.8
    ke = math.log(2) / t_half
    expected_css_avg = f * dose / (vd * ke * tau)
    r = calculate_accumulation(
        "D", dose_mg=dose, t_half_h=t_half, dosing_interval_h=tau, vd_L=vd, f_oral=f
    )
    assert abs(r.css_avg_mg_L - expected_css_avg) < 1e-6


def test_trough_accumulation_equals_rac():
    """trough_accumulation_factor == accumulation_index for 1-cpt approximation."""
    r = calculate_accumulation("D", dose_mg=100, t_half_h=12, dosing_interval_h=24, vd_L=50)
    assert r.trough_accumulation_factor == r.accumulation_index


def test_cmax_accumulation_equals_rac():
    r = calculate_accumulation("D", dose_mg=100, t_half_h=12, dosing_interval_h=24, vd_L=50)
    assert r.cmax_accumulation_factor == r.accumulation_index


# ---------------------------------------------------------------------------
# Accumulation trend tests
# ---------------------------------------------------------------------------


def test_longer_half_life_more_accumulation():
    """Longer t_half with same tau → higher Rac."""
    r_short = calculate_accumulation("D", dose_mg=100, t_half_h=4, dosing_interval_h=12, vd_L=50)
    r_long = calculate_accumulation("D", dose_mg=100, t_half_h=24, dosing_interval_h=12, vd_L=50)
    assert r_long.accumulation_index > r_short.accumulation_index


def test_shorter_interval_more_accumulation():
    """Shorter dosing interval with same t_half → higher Rac."""
    r_freq = calculate_accumulation("D", dose_mg=100, t_half_h=12, dosing_interval_h=6, vd_L=50)
    r_rare = calculate_accumulation("D", dose_mg=100, t_half_h=12, dosing_interval_h=48, vd_L=50)
    assert r_freq.accumulation_index > r_rare.accumulation_index


def test_accumulation_index_gte_1():
    """Rac is always >= 1."""
    r = calculate_accumulation("D", dose_mg=100, t_half_h=2, dosing_interval_h=24, vd_L=50)
    assert r.accumulation_index >= 1.0


# ---------------------------------------------------------------------------
# Time-to-steady-state tests
# ---------------------------------------------------------------------------


def test_time_to_90pct_ss():
    """time_to_90pct_ss = 3.32 * t_half."""
    t_half = 10.0
    r = calculate_accumulation("D", dose_mg=100, t_half_h=t_half, dosing_interval_h=24, vd_L=50)
    assert abs(r.time_to_90pct_ss_h - 3.32 * t_half) < 1e-6


def test_time_to_95pct_ss():
    """time_to_95pct_ss = 4.32 * t_half."""
    t_half = 10.0
    r = calculate_accumulation("D", dose_mg=100, t_half_h=t_half, dosing_interval_h=24, vd_L=50)
    assert abs(r.time_to_95pct_ss_h - 4.32 * t_half) < 1e-6


def test_doses_to_90pct_positive_integer():
    r = calculate_accumulation("D", dose_mg=100, t_half_h=12, dosing_interval_h=24, vd_L=50)
    assert isinstance(r.doses_to_90pct_ss, int)
    assert r.doses_to_90pct_ss >= 1


def test_doses_to_90pct_increases_with_half_life():
    """Longer t_half → more doses needed."""
    r_short = calculate_accumulation("D", dose_mg=100, t_half_h=2, dosing_interval_h=8, vd_L=50)
    r_long = calculate_accumulation("D", dose_mg=100, t_half_h=48, dosing_interval_h=8, vd_L=50)
    assert r_long.doses_to_90pct_ss >= r_short.doses_to_90pct_ss


# ---------------------------------------------------------------------------
# Fluctuation and Css tests
# ---------------------------------------------------------------------------


def test_css_min_non_negative():
    r = calculate_accumulation("D", dose_mg=100, t_half_h=1, dosing_interval_h=24, vd_L=50)
    assert r.css_min_mg_L >= 0.0


def test_fluctuation_pct_non_negative():
    r = calculate_accumulation("D", dose_mg=100, t_half_h=12, dosing_interval_h=24, vd_L=50)
    assert r.fluctuation_pct >= 0.0


def test_high_fluctuation_for_short_half_life():
    """Short t_half vs long tau → high fluctuation."""
    r = calculate_accumulation("D", dose_mg=100, t_half_h=1, dosing_interval_h=24, vd_L=50)
    assert r.fluctuation_pct > 50.0


# ---------------------------------------------------------------------------
# Regimen comparison tests
# ---------------------------------------------------------------------------


def test_compare_dosing_regimens_returns_list():
    results = compare_dosing_regimens("Drug", dose_mg=100, t_half_h=12, vd_L=50)
    assert isinstance(results, list)


def test_compare_dosing_regimens_returns_5_by_default():
    results = compare_dosing_regimens("Drug", dose_mg=100, t_half_h=12, vd_L=50)
    assert len(results) == 5


def test_compare_dosing_regimens_sorted_descending():
    """Shorter intervals → higher Rac → first element."""
    results = compare_dosing_regimens("Drug", dose_mg=100, t_half_h=12, vd_L=50)
    racs = [r.accumulation_index for r in results]
    assert racs == sorted(racs, reverse=True), f"Not sorted: {racs}"


def test_compare_dosing_regimens_custom():
    results = compare_dosing_regimens(
        "Drug", dose_mg=100, t_half_h=12, vd_L=50, regimens=[4.0, 12.0, 24.0]
    )
    assert len(results) == 3


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_validation_zero_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        calculate_accumulation("D", dose_mg=0, t_half_h=12, dosing_interval_h=24, vd_L=50)


def test_validation_zero_half_life():
    with pytest.raises(ValueError, match="t_half_h"):
        calculate_accumulation("D", dose_mg=100, t_half_h=0, dosing_interval_h=24, vd_L=50)


def test_validation_zero_interval():
    with pytest.raises(ValueError, match="dosing_interval_h"):
        calculate_accumulation("D", dose_mg=100, t_half_h=12, dosing_interval_h=0, vd_L=50)


def test_validation_zero_vd():
    with pytest.raises(ValueError, match="vd_L"):
        calculate_accumulation("D", dose_mg=100, t_half_h=12, dosing_interval_h=24, vd_L=0)


def test_validation_f_oral_zero():
    with pytest.raises(ValueError, match="f_oral"):
        calculate_accumulation(
            "D", dose_mg=100, t_half_h=12, dosing_interval_h=24, vd_L=50, f_oral=0.0
        )


def test_validation_f_oral_above_1():
    with pytest.raises(ValueError, match="f_oral"):
        calculate_accumulation(
            "D", dose_mg=100, t_half_h=12, dosing_interval_h=24, vd_L=50, f_oral=1.5
        )
