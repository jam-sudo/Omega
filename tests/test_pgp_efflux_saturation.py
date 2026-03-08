"""Tests for Phase 870 — P-gp Efflux Saturation Model."""

import pytest

from omega_pbpk.prediction.pgp_efflux_saturation import (
    PGPEffluxSaturationResult,
    compare_dose_levels,
    simulate_pgp_efflux_saturation,
)

# ---------------------------------------------------------------------------
# Basic return type and field checks
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_pgp_efflux_saturation("DrugB", dose_mg=10.0)
    assert isinstance(result, PGPEffluxSaturationResult)


def test_fields_preserved():
    result = simulate_pgp_efflux_saturation(
        "TestDrug", dose_mg=20.0, vmax_pgp_mg_per_h=3.0, km_pgp_mg_per_L=5.0
    )
    assert result.drug_name == "TestDrug"
    assert result.dose_mg == 20.0
    assert result.vmax_pgp_mg_per_h == 3.0
    assert result.km_pgp_mg_per_L == 5.0


def test_times_and_c_blood_same_length():
    result = simulate_pgp_efflux_saturation("DrugB", dose_mg=10.0, t_end_h=6.0, dt_h=0.1)
    assert len(result.times_h) == len(result.c_blood_mg_L)
    assert len(result.times_h) > 0


def test_times_and_a_dissolved_same_length():
    result = simulate_pgp_efflux_saturation("DrugB", dose_mg=10.0)
    assert len(result.times_h) == len(result.a_dissolved_mg)


def test_times_start_at_zero():
    result = simulate_pgp_efflux_saturation("DrugB", dose_mg=10.0)
    assert result.times_h[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# PK metrics
# ---------------------------------------------------------------------------


def test_cmax_positive():
    result = simulate_pgp_efflux_saturation("DrugB", dose_mg=10.0)
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = simulate_pgp_efflux_saturation("DrugB", dose_mg=10.0)
    assert result.auc_mg_h_per_L > 0.0


def test_tmax_within_simulation_window():
    result = simulate_pgp_efflux_saturation("DrugB", dose_mg=10.0, t_end_h=12.0)
    assert 0.0 <= result.tmax_h <= 12.0


def test_f_absorbed_in_zero_one():
    result = simulate_pgp_efflux_saturation("DrugB", dose_mg=10.0)
    assert 0.0 <= result.f_absorbed <= 1.0


def test_notes_is_string():
    result = simulate_pgp_efflux_saturation("DrugB", dose_mg=10.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# P-gp effect on absorption
# ---------------------------------------------------------------------------


def test_no_pgp_higher_f_absorbed():
    """vmax=0 (no P-gp) → higher f_absorbed than with active P-gp."""
    r_no_pgp = simulate_pgp_efflux_saturation("DrugB", dose_mg=5.0, vmax_pgp_mg_per_h=0.0)
    r_with_pgp = simulate_pgp_efflux_saturation("DrugB", dose_mg=5.0, vmax_pgp_mg_per_h=5.0)
    assert r_no_pgp.f_absorbed > r_with_pgp.f_absorbed


def test_no_pgp_higher_auc():
    """vmax=0 → higher AUC than with active P-gp efflux."""
    r_no_pgp = simulate_pgp_efflux_saturation("DrugB", dose_mg=5.0, vmax_pgp_mg_per_h=0.0)
    r_with_pgp = simulate_pgp_efflux_saturation("DrugB", dose_mg=5.0, vmax_pgp_mg_per_h=5.0)
    assert r_no_pgp.auc_mg_h_per_L > r_with_pgp.auc_mg_h_per_L


def test_pgp_saturation_at_cmax_in_zero_one():
    result = simulate_pgp_efflux_saturation("DrugB", dose_mg=10.0)
    assert 0.0 <= result.pgp_saturation_at_cmax <= 1.0


def test_pgp_saturation_zero_when_no_pgp():
    """When vmax=0, pgp_saturation should be 0."""
    result = simulate_pgp_efflux_saturation("DrugB", dose_mg=10.0, vmax_pgp_mg_per_h=0.0)
    assert result.pgp_saturation_at_cmax == 0.0


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


def test_high_dose_not_dose_proportional():
    """Dose >> Km*V_gi → dose_proportional=False."""
    # km=10 mg/L, v_gi=0.25 L → Km*V_gi = 2.5 mg → dose >> 2.5
    result = simulate_pgp_efflux_saturation(
        "DrugB", dose_mg=1000.0, km_pgp_mg_per_L=10.0, v_gi_L=0.25
    )
    assert result.dose_proportional is False


def test_low_dose_is_dose_proportional():
    """Dose << Km*V_gi → dose_proportional=True."""
    # km=100 mg/L, v_gi=1.0 L → Km*V_gi = 100 mg → dose=1.0 << 100
    result = simulate_pgp_efflux_saturation("DrugB", dose_mg=1.0, km_pgp_mg_per_L=100.0, v_gi_L=1.0)
    assert result.dose_proportional is True


# ---------------------------------------------------------------------------
# Dose as solution vs. dissolution
# ---------------------------------------------------------------------------


def test_dose_as_solution_initial_dissolved():
    """When dose_as_solution=True, initial a_dissolved = dose_mg."""
    result = simulate_pgp_efflux_saturation("DrugB", dose_mg=10.0, dose_as_solution=True)
    assert result.a_dissolved_mg[0] == pytest.approx(10.0, rel=1e-6)


def test_dose_as_suspension_initial_dissolved_zero():
    """When dose_as_solution=False, initial a_dissolved = 0 (all undissolved)."""
    result = simulate_pgp_efflux_saturation("DrugB", dose_mg=10.0, dose_as_solution=False)
    assert result.a_dissolved_mg[0] == pytest.approx(0.0, abs=1e-9)


def test_solution_higher_cmax_than_suspension():
    """Immediate dissolution (solution) should give higher Cmax than slow dissolution."""
    r_sol = simulate_pgp_efflux_saturation("DrugB", dose_mg=10.0, dose_as_solution=True)
    r_sus = simulate_pgp_efflux_saturation("DrugB", dose_mg=10.0, dose_as_solution=False)
    assert r_sol.cmax_mg_L >= r_sus.cmax_mg_L


# ---------------------------------------------------------------------------
# compare_dose_levels
# ---------------------------------------------------------------------------


def test_compare_dose_levels_returns_list():
    results = compare_dose_levels("DrugB", [5.0, 50.0, 500.0])
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_dose_levels_sorted_ascending():
    results = compare_dose_levels("DrugB", [500.0, 5.0, 50.0])
    doses = [r.dose_mg for r in results]
    assert doses == sorted(doses)


def test_compare_dose_levels_single():
    results = compare_dose_levels("DrugB", [10.0])
    assert len(results) == 1
    assert results[0].dose_mg == 10.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_pgp_efflux_saturation("DrugB", dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_pgp_efflux_saturation("DrugB", dose_mg=-5.0)


def test_validation_ka_passive_zero():
    with pytest.raises(ValueError, match="ka_passive_per_h must be > 0"):
        simulate_pgp_efflux_saturation("DrugB", dose_mg=10.0, ka_passive_per_h=0.0)


def test_validation_ka_passive_negative():
    with pytest.raises(ValueError, match="ka_passive_per_h must be > 0"):
        simulate_pgp_efflux_saturation("DrugB", dose_mg=10.0, ka_passive_per_h=-0.1)


def test_validation_vmax_pgp_negative():
    with pytest.raises(ValueError, match="vmax_pgp_mg_per_h must be >= 0"):
        simulate_pgp_efflux_saturation("DrugB", dose_mg=10.0, vmax_pgp_mg_per_h=-1.0)


def test_validation_km_pgp_zero():
    with pytest.raises(ValueError, match="km_pgp_mg_per_L must be > 0"):
        simulate_pgp_efflux_saturation("DrugB", dose_mg=10.0, km_pgp_mg_per_L=0.0)
