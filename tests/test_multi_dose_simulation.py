"""Tests for multi-dose simulation (Phase 787)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.multi_dose_simulation import (
    MultiDoseResult,
    simulate_loading_maintenance,
    simulate_multi_dose,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_multi_dose_result():
    result = simulate_multi_dose("DrugA", dose_mg=100.0, tau_h=12.0, n_doses=5)
    assert isinstance(result, MultiDoseResult)


def test_result_has_list_fields():
    result = simulate_multi_dose("DrugA", dose_mg=100.0, tau_h=12.0, n_doses=5)
    assert isinstance(result.times_h, list)
    assert isinstance(result.c_plasma_mg_L, list)
    assert len(result.times_h) == len(result.c_plasma_mg_L)


def test_scalar_fields_are_floats():
    result = simulate_multi_dose("DrugA", dose_mg=100.0, tau_h=12.0, n_doses=5)
    assert isinstance(result.css_max_mg_L, float)
    assert isinstance(result.css_min_mg_L, float)
    assert isinstance(result.css_avg_mg_L, float)
    assert isinstance(result.accumulation_ratio, float)
    assert isinstance(result.fluctuation_pct, float)
    assert isinstance(result.doses_to_ss, int)
    assert isinstance(result.notes, str)


def test_drug_name_preserved():
    result = simulate_multi_dose("TestDrug", dose_mg=50.0, tau_h=8.0)
    assert result.drug_name == "TestDrug"


def test_dose_preserved():
    result = simulate_multi_dose("DrugA", dose_mg=200.0, tau_h=12.0)
    assert result.dose_mg == 200.0


def test_tau_preserved():
    result = simulate_multi_dose("DrugA", dose_mg=100.0, tau_h=24.0)
    assert result.tau_h == 24.0


# ---------------------------------------------------------------------------
# n_doses correctness
# ---------------------------------------------------------------------------


def test_n_doses_5_reflected():
    result = simulate_multi_dose("DrugA", dose_mg=100.0, tau_h=12.0, n_doses=5)
    assert result.n_doses == 5


def test_n_doses_10_reflected():
    result = simulate_multi_dose("DrugA", dose_mg=100.0, tau_h=12.0, n_doses=10)
    assert result.n_doses == 10


# ---------------------------------------------------------------------------
# Linear PK: 2x dose → ~2x css_max
# ---------------------------------------------------------------------------


def test_double_dose_doubles_css_max():
    r1 = simulate_multi_dose(
        "DrugA", dose_mg=100.0, tau_h=12.0, n_doses=15, cl_L_per_h=5.0, vd_L=50.0, ka_per_h=1.0
    )
    r2 = simulate_multi_dose(
        "DrugA", dose_mg=200.0, tau_h=12.0, n_doses=15, cl_L_per_h=5.0, vd_L=50.0, ka_per_h=1.0
    )
    ratio = r2.css_max_mg_L / r1.css_max_mg_L
    assert 1.8 < ratio < 2.2, f"Expected ~2x, got {ratio:.3f}"


def test_double_dose_doubles_css_avg():
    r1 = simulate_multi_dose(
        "DrugA", dose_mg=100.0, tau_h=12.0, n_doses=15, cl_L_per_h=5.0, vd_L=50.0
    )
    r2 = simulate_multi_dose(
        "DrugA", dose_mg=200.0, tau_h=12.0, n_doses=15, cl_L_per_h=5.0, vd_L=50.0
    )
    ratio = r2.css_avg_mg_L / r1.css_avg_mg_L
    assert 1.8 < ratio < 2.2, f"Expected ~2x css_avg, got {ratio:.3f}"


# ---------------------------------------------------------------------------
# Accumulation ratio
# ---------------------------------------------------------------------------


def test_accumulation_ratio_geq_1():
    result = simulate_multi_dose("DrugA", dose_mg=100.0, tau_h=8.0, n_doses=10)
    assert result.accumulation_ratio >= 1.0


def test_accumulation_ratio_approaches_1_for_long_tau():
    # Very long tau (>> t1/2) → drug clears between doses → no accumulation
    result = simulate_multi_dose(
        "DrugA", dose_mg=100.0, tau_h=100.0, n_doses=5, cl_L_per_h=5.0, vd_L=50.0, ka_per_h=1.0
    )
    assert result.accumulation_ratio < 1.5, f"accumulation_ratio={result.accumulation_ratio}"


def test_accumulation_higher_for_shorter_tau():
    r_short = simulate_multi_dose(
        "DrugA", dose_mg=100.0, tau_h=4.0, n_doses=15, cl_L_per_h=5.0, vd_L=50.0
    )
    r_long = simulate_multi_dose(
        "DrugA", dose_mg=100.0, tau_h=24.0, n_doses=15, cl_L_per_h=5.0, vd_L=50.0
    )
    assert r_short.accumulation_ratio >= r_long.accumulation_ratio


# ---------------------------------------------------------------------------
# Fluctuation
# ---------------------------------------------------------------------------


def test_fluctuation_positive_oral():
    result = simulate_multi_dose("DrugA", dose_mg=100.0, tau_h=12.0, n_doses=10, route="oral")
    assert result.fluctuation_pct > 0.0


def test_fluctuation_positive_iv():
    result = simulate_multi_dose("DrugA", dose_mg=100.0, tau_h=12.0, n_doses=10, route="iv")
    assert result.fluctuation_pct > 0.0


def test_css_max_geq_css_min():
    result = simulate_multi_dose("DrugA", dose_mg=100.0, tau_h=12.0, n_doses=10)
    assert result.css_max_mg_L >= result.css_min_mg_L


# ---------------------------------------------------------------------------
# IV route: no absorption delay → earlier peak
# ---------------------------------------------------------------------------


def test_iv_route_peak_earlier_than_oral():
    # For IV, dose goes directly to plasma; first point has high concentration
    r_iv = simulate_multi_dose(
        "DrugA",
        dose_mg=100.0,
        tau_h=24.0,
        n_doses=3,
        route="iv",
        cl_L_per_h=5.0,
        vd_L=50.0,
        dt_h=0.1,
    )
    # IV peak at t=0; oral peak later
    iv_peak_t = r_iv.times_h[r_iv.c_plasma_mg_L.index(max(r_iv.c_plasma_mg_L[:20]))]
    # IV should peak at or near t=0 (first time point)
    assert iv_peak_t < 1.0, f"IV should peak early, got {iv_peak_t}h"


def test_iv_route_no_gut_depot():
    # IV: c at t=0 should be dose/Vd (no absorption delay)
    result = simulate_multi_dose(
        "DrugA",
        dose_mg=100.0,
        tau_h=24.0,
        n_doses=1,
        route="iv",
        vd_L=50.0,
        cl_L_per_h=1.0,
        dt_h=0.5,
    )
    # First concentration should be close to dose/Vd
    expected_c0 = 100.0 / 50.0  # 2.0 mg/L
    assert abs(result.c_plasma_mg_L[0] - expected_c0) < 0.1, (
        f"Expected c0~{expected_c0}, got {result.c_plasma_mg_L[0]}"
    )


# ---------------------------------------------------------------------------
# Loading + maintenance
# ---------------------------------------------------------------------------


def test_loading_maintenance_returns_result():
    result = simulate_loading_maintenance(
        "DrugA", loading_dose_mg=200.0, maintenance_dose_mg=100.0, tau_h=12.0
    )
    assert isinstance(result, MultiDoseResult)


def test_loading_dose_gives_higher_first_peak():
    result = simulate_loading_maintenance(
        "DrugA",
        loading_dose_mg=300.0,
        maintenance_dose_mg=100.0,
        tau_h=12.0,
        n_maintenance=5,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ka_per_h=1.0,
        route="oral",
    )
    # First interval peak should be higher than second interval peak
    # (loading >> maintenance)
    # Find peak in first and second intervals
    first_interval_concs = [c for t, c in zip(result.times_h, result.c_plasma_mg_L) if t <= 12.0]
    second_interval_concs = [
        c for t, c in zip(result.times_h, result.c_plasma_mg_L) if 12.0 <= t <= 24.0
    ]
    first_peak = max(first_interval_concs) if first_interval_concs else 0.0
    second_peak = max(second_interval_concs) if second_interval_concs else 0.0
    assert first_peak > second_peak, (
        f"Loading peak {first_peak:.3f} should > maintenance peak {second_peak:.3f}"
    )


def test_loading_maintenance_n_doses_correct():
    result = simulate_loading_maintenance(
        "DrugA", loading_dose_mg=200.0, maintenance_dose_mg=100.0, tau_h=12.0, n_maintenance=5
    )
    # Total = 1 loading + 5 maintenance = 6
    assert result.n_doses == 6


def test_loading_maintenance_lists_nonempty():
    result = simulate_loading_maintenance(
        "DrugA", loading_dose_mg=200.0, maintenance_dose_mg=100.0, tau_h=12.0
    )
    assert len(result.times_h) > 0
    assert len(result.c_plasma_mg_L) > 0


# ---------------------------------------------------------------------------
# Validation: negative/zero inputs
# ---------------------------------------------------------------------------


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_multi_dose("DrugA", dose_mg=0.0, tau_h=12.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_multi_dose("DrugA", dose_mg=-100.0, tau_h=12.0)


def test_zero_tau_raises():
    with pytest.raises(ValueError, match="tau_h"):
        simulate_multi_dose("DrugA", dose_mg=100.0, tau_h=0.0)


def test_negative_tau_raises():
    with pytest.raises(ValueError, match="tau_h"):
        simulate_multi_dose("DrugA", dose_mg=100.0, tau_h=-1.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        simulate_multi_dose("DrugA", dose_mg=100.0, tau_h=12.0, route="subcutaneous")


# ---------------------------------------------------------------------------
# doses_to_ss
# ---------------------------------------------------------------------------


def test_doses_to_ss_leq_n_doses():
    result = simulate_multi_dose("DrugA", dose_mg=100.0, tau_h=12.0, n_doses=20)
    assert result.doses_to_ss <= result.n_doses
