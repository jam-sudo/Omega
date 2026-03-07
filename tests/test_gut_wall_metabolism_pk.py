"""Tests for Phase 869 — Gut Wall Metabolism PBPK."""

import pytest

from omega_pbpk.core.gut_wall_metabolism_pk import (
    GutWallMetabolismResult,
    sensitivity_gut_clint,
    simulate_gut_wall_metabolism,
)

# ---------------------------------------------------------------------------
# Basic return type and field checks
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_gut_wall_metabolism("DrugA", dose_mg=100.0)
    assert isinstance(result, GutWallMetabolismResult)


def test_fields_preserved():
    result = simulate_gut_wall_metabolism(
        "TestDrug", dose_mg=50.0, fa=0.8, clint_gut_mL_per_min=15.0, clint_hep_mL_per_min=80.0
    )
    assert result.drug_name == "TestDrug"
    assert result.dose_mg == 50.0
    assert result.fa == 0.8
    assert result.clint_gut_mL_per_min == 15.0
    assert result.clint_hep_mL_per_min == 80.0


def test_times_and_concs_same_length():
    result = simulate_gut_wall_metabolism("DrugA", dose_mg=100.0, t_end_h=12.0, dt_h=0.5)
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) > 0


def test_times_start_at_zero():
    result = simulate_gut_wall_metabolism("DrugA", dose_mg=100.0)
    assert result.times_h[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Bioavailability computations
# ---------------------------------------------------------------------------


def test_fg_in_zero_one():
    result = simulate_gut_wall_metabolism("DrugA", dose_mg=100.0)
    assert 0.0 <= result.fg_gut_availability <= 1.0


def test_fh_in_zero_one():
    result = simulate_gut_wall_metabolism("DrugA", dose_mg=100.0)
    assert 0.0 <= result.fh_hepatic_availability <= 1.0


def test_f_equals_fa_times_fg_times_fh():
    result = simulate_gut_wall_metabolism(
        "DrugA", dose_mg=100.0, fa=0.9, clint_gut_mL_per_min=20.0, clint_hep_mL_per_min=100.0
    )
    expected_f = result.fa * result.fg_gut_availability * result.fh_hepatic_availability
    assert result.f_bioavailability == pytest.approx(expected_f, rel=1e-6)


def test_effective_dose_equals_dose_times_f():
    result = simulate_gut_wall_metabolism("DrugA", dose_mg=200.0)
    assert result.effective_dose_mg == pytest.approx(
        result.dose_mg * result.f_bioavailability, rel=1e-6
    )


# ---------------------------------------------------------------------------
# Effect of CLint_gut on bioavailability
# ---------------------------------------------------------------------------


def test_higher_clint_gut_lower_fg():
    r_low = simulate_gut_wall_metabolism("DrugA", dose_mg=100.0, clint_gut_mL_per_min=5.0)
    r_high = simulate_gut_wall_metabolism("DrugA", dose_mg=100.0, clint_gut_mL_per_min=100.0)
    assert r_high.fg_gut_availability < r_low.fg_gut_availability


def test_higher_clint_gut_lower_f():
    r_low = simulate_gut_wall_metabolism("DrugA", dose_mg=100.0, clint_gut_mL_per_min=5.0)
    r_high = simulate_gut_wall_metabolism("DrugA", dose_mg=100.0, clint_gut_mL_per_min=200.0)
    assert r_high.f_bioavailability < r_low.f_bioavailability


def test_high_extraction_drug_gut_er_approaches_one():
    """CLint_gut >> Q_villous → gut extraction ratio approaches 1."""
    result = simulate_gut_wall_metabolism(
        "HighER",
        dose_mg=100.0,
        clint_gut_mL_per_min=100000.0,
        q_villous_mL_per_min=18.0,
    )
    assert result.gut_extraction_ratio > 0.99


def test_low_extraction_drug_fg_approaches_one():
    """CLint_gut << Q_villous → Fg approaches 1."""
    result = simulate_gut_wall_metabolism(
        "LowER",
        dose_mg=100.0,
        clint_gut_mL_per_min=0.001,
        q_villous_mL_per_min=18.0,
    )
    assert result.fg_gut_availability > 0.999


def test_zero_clint_gut_fg_equals_one():
    """CLint_gut = 0 → no gut metabolism → Fg = 1."""
    result = simulate_gut_wall_metabolism("NoGutMet", dose_mg=100.0, clint_gut_mL_per_min=0.0)
    assert result.fg_gut_availability == pytest.approx(1.0, abs=1e-9)
    assert result.gut_extraction_ratio == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Extraction ratios and dominance
# ---------------------------------------------------------------------------


def test_extraction_dominant_gut():
    """High gut CLint but low hepatic CLint → gut dominant."""
    result = simulate_gut_wall_metabolism(
        "GutDom",
        dose_mg=100.0,
        clint_gut_mL_per_min=1000.0,
        q_villous_mL_per_min=18.0,
        clint_hep_mL_per_min=1.0,
        q_hepatic_mL_per_min=1500.0,
    )
    assert result.extraction_dominant == "gut"


def test_extraction_dominant_hepatic():
    """Low gut CLint but high hepatic CLint → hepatic dominant."""
    result = simulate_gut_wall_metabolism(
        "HepDom",
        dose_mg=100.0,
        clint_gut_mL_per_min=0.1,
        q_villous_mL_per_min=18.0,
        clint_hep_mL_per_min=5000.0,
        q_hepatic_mL_per_min=1500.0,
    )
    assert result.extraction_dominant == "hepatic"


def test_extraction_dominant_equal():
    """Same ER → equal."""
    # Same ratio: CLint/Q same for both
    result = simulate_gut_wall_metabolism(
        "EqualDom",
        dose_mg=100.0,
        clint_gut_mL_per_min=18.0,
        q_villous_mL_per_min=18.0,
        clint_hep_mL_per_min=1500.0,
        q_hepatic_mL_per_min=1500.0,
        fug=1.0,
        fup=1.0,
    )
    # Both ERs = 0.5
    assert result.extraction_dominant == "equal"


def test_gut_er_plus_fg_equals_one():
    result = simulate_gut_wall_metabolism("DrugA", dose_mg=100.0)
    assert result.gut_extraction_ratio + result.fg_gut_availability == pytest.approx(1.0, abs=1e-9)


def test_hep_er_plus_fh_equals_one():
    result = simulate_gut_wall_metabolism("DrugA", dose_mg=100.0)
    assert result.hepatic_extraction_ratio + result.fh_hepatic_availability == pytest.approx(
        1.0, abs=1e-9
    )


# ---------------------------------------------------------------------------
# PK profile checks
# ---------------------------------------------------------------------------


def test_cmax_positive():
    result = simulate_gut_wall_metabolism("DrugA", dose_mg=100.0)
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = simulate_gut_wall_metabolism("DrugA", dose_mg=100.0)
    assert result.auc_mg_h_per_L > 0.0


def test_tmax_within_simulation_window():
    result = simulate_gut_wall_metabolism("DrugA", dose_mg=100.0, t_end_h=24.0)
    assert 0.0 < result.tmax_h <= 24.0


def test_notes_is_string():
    result = simulate_gut_wall_metabolism("DrugA", dose_mg=100.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


def test_sensitivity_returns_list():
    results = sensitivity_gut_clint("DrugA", 100.0, [5.0, 20.0, 100.0])
    assert isinstance(results, list)
    assert len(results) == 3


def test_sensitivity_sorted_descending_by_f():
    results = sensitivity_gut_clint("DrugA", 100.0, [1.0, 50.0, 500.0])
    f_values = [r.f_bioavailability for r in results]
    assert f_values == sorted(f_values, reverse=True)


def test_sensitivity_single_value():
    results = sensitivity_gut_clint("DrugA", 100.0, [20.0])
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_gut_wall_metabolism("DrugA", dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_gut_wall_metabolism("DrugA", dose_mg=-10.0)


def test_validation_fa_zero():
    with pytest.raises(ValueError, match="fa must be in"):
        simulate_gut_wall_metabolism("DrugA", dose_mg=100.0, fa=0.0)


def test_validation_fa_greater_than_one():
    with pytest.raises(ValueError, match="fa must be in"):
        simulate_gut_wall_metabolism("DrugA", dose_mg=100.0, fa=1.1)


def test_validation_clint_gut_negative():
    with pytest.raises(ValueError, match="clint_gut_mL_per_min must be >= 0"):
        simulate_gut_wall_metabolism("DrugA", dose_mg=100.0, clint_gut_mL_per_min=-1.0)


def test_validation_q_villous_zero():
    with pytest.raises(ValueError, match="q_villous_mL_per_min must be > 0"):
        simulate_gut_wall_metabolism("DrugA", dose_mg=100.0, q_villous_mL_per_min=0.0)
