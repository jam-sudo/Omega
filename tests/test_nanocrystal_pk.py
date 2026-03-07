"""Tests for Phase 769 — Nanocrystal Dissolution PK."""

import pytest

from omega_pbpk.core.nanocrystal_pk import (
    NanocrystalPKResult,
    compare_particle_sizes_nano,
    simulate_nanocrystal_pk,
)

# ---------------------------------------------------------------------------
# Basic return type and shape
# ---------------------------------------------------------------------------


def test_returns_nanocrystal_pk_result():
    result = simulate_nanocrystal_pk("DrugX", dose_mg=10.0)
    assert isinstance(result, NanocrystalPKResult)


def test_times_list_non_empty():
    result = simulate_nanocrystal_pk("DrugX", dose_mg=10.0)
    assert len(result.times_h) > 0


def test_c_plasma_list_length_matches_times():
    result = simulate_nanocrystal_pk("DrugX", dose_mg=10.0)
    assert len(result.c_plasma_mg_L) == len(result.times_h)


def test_c_solution_list_length_matches_times():
    result = simulate_nanocrystal_pk("DrugX", dose_mg=10.0)
    assert len(result.c_solution_mg_L) == len(result.times_h)


def test_a_crystal_list_length_matches_times():
    result = simulate_nanocrystal_pk("DrugX", dose_mg=10.0)
    assert len(result.a_crystal_mg) == len(result.times_h)


# ---------------------------------------------------------------------------
# Scalar outputs
# ---------------------------------------------------------------------------


def test_cmax_positive():
    result = simulate_nanocrystal_pk("DrugX", dose_mg=10.0)
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = simulate_nanocrystal_pk("DrugX", dose_mg=10.0)
    assert result.auc_mg_h_per_L > 0.0


def test_k_diss_effective_positive():
    result = simulate_nanocrystal_pk("DrugX", dose_mg=10.0, particle_radius_nm=500.0)
    assert result.k_diss_effective > 0.0


def test_fraction_dissolved_between_0_and_1():
    result = simulate_nanocrystal_pk("DrugX", dose_mg=10.0)
    assert 0.0 <= result.fraction_dissolved <= 1.0


def test_notes_nonempty():
    result = simulate_nanocrystal_pk("DrugX", dose_mg=10.0)
    assert isinstance(result.notes, str) and len(result.notes) > 0


def test_drug_name_stored():
    result = simulate_nanocrystal_pk("Ibuprofen", dose_mg=5.0)
    assert result.drug_name == "Ibuprofen"


def test_dose_stored():
    result = simulate_nanocrystal_pk("DrugX", dose_mg=25.0)
    assert result.dose_mg == 25.0


def test_particle_radius_stored():
    result = simulate_nanocrystal_pk("DrugX", dose_mg=10.0, particle_radius_nm=200.0)
    assert result.particle_radius_nm == 200.0


# ---------------------------------------------------------------------------
# Physical behaviour
# ---------------------------------------------------------------------------


def test_smaller_particle_gives_higher_auc():
    """Nanocrystals (100 nm) should give higher AUC than larger (1000 nm)."""
    res_small = simulate_nanocrystal_pk("DrugX", dose_mg=10.0, particle_radius_nm=100.0)
    res_large = simulate_nanocrystal_pk("DrugX", dose_mg=10.0, particle_radius_nm=1000.0)
    assert res_small.auc_mg_h_per_L > res_large.auc_mg_h_per_L


def test_auc_enhancement_vs_coarse_greater_than_1():
    """Nanocrystals should outperform coarse particles in AUC."""
    result = simulate_nanocrystal_pk("DrugX", dose_mg=10.0, particle_radius_nm=500.0)
    assert result.auc_enhancement_vs_coarse > 1.0


def test_higher_dose_gives_higher_auc():
    res_low = simulate_nanocrystal_pk("DrugX", dose_mg=5.0)
    res_high = simulate_nanocrystal_pk("DrugX", dose_mg=20.0)
    assert res_high.auc_mg_h_per_L > res_low.auc_mg_h_per_L


def test_higher_dose_gives_higher_cmax():
    res_low = simulate_nanocrystal_pk("DrugX", dose_mg=5.0)
    res_high = simulate_nanocrystal_pk("DrugX", dose_mg=20.0)
    assert res_high.cmax_mg_L > res_low.cmax_mg_L


def test_tmax_within_simulation_window():
    result = simulate_nanocrystal_pk("DrugX", dose_mg=10.0, t_end_h=12.0)
    assert 0.0 <= result.tmax_h <= 12.0


# ---------------------------------------------------------------------------
# compare_particle_sizes_nano
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    results = compare_particle_sizes_nano("DrugX", dose_mg=10.0, radii_nm=[100.0, 500.0, 1000.0])
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_sorted_by_auc_descending():
    results = compare_particle_sizes_nano("DrugX", dose_mg=10.0, radii_nm=[100.0, 500.0, 2000.0])
    aucs = [r.auc_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_single_radius():
    results = compare_particle_sizes_nano("DrugX", dose_mg=10.0, radii_nm=[300.0])
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_dose_zero_raises():
    with pytest.raises(ValueError):
        simulate_nanocrystal_pk("DrugX", dose_mg=0.0)


def test_dose_negative_raises():
    with pytest.raises(ValueError):
        simulate_nanocrystal_pk("DrugX", dose_mg=-5.0)


def test_particle_radius_zero_raises():
    with pytest.raises(ValueError):
        simulate_nanocrystal_pk("DrugX", dose_mg=10.0, particle_radius_nm=0.0)


def test_c_sat_zero_raises():
    with pytest.raises(ValueError):
        simulate_nanocrystal_pk("DrugX", dose_mg=10.0, c_sat_mg_L=0.0)


def test_cl_zero_raises():
    with pytest.raises(ValueError):
        simulate_nanocrystal_pk("DrugX", dose_mg=10.0, cl_L_per_h=0.0)
