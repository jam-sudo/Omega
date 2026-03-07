"""Tests for core/inhalation_dosimetry.py — Phase 829."""

import math

import pytest

from omega_pbpk.core.inhalation_dosimetry import (
    InhalationDosimetryResult,
    compare_particle_sizes,
    simulate_inhalation_pk,
)

# ---------------------------------------------------------------------------
# Return type and field structure
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_inhalation_pk("salbutamol", nominal_dose_mg=0.1)
    assert isinstance(result, InhalationDosimetryResult)


def test_field_types():
    result = simulate_inhalation_pk("salbutamol", nominal_dose_mg=0.1)
    assert isinstance(result.drug_name, str)
    assert isinstance(result.nominal_dose_mg, float)
    assert isinstance(result.particle_size_MMAD_um, float)
    assert isinstance(result.emitted_dose_mg, float)
    assert isinstance(result.lung_deposited_fraction, float)
    assert isinstance(result.oropharyngeal_fraction, float)
    assert isinstance(result.alveolar_fraction, float)
    assert isinstance(result.bronchial_fraction, float)
    assert isinstance(result.swallowed_fraction, float)
    assert isinstance(result.times_h, list)
    assert isinstance(result.c_plasma_mg_L, list)
    assert isinstance(result.cmax_mg_L, float)
    assert isinstance(result.tmax_h, float)
    assert isinstance(result.auc_mg_h_per_L, float)
    assert isinstance(result.lung_depot_initial_mg, float)
    assert isinstance(result.lung_t_half_h, float)
    assert isinstance(result.systemic_bioavailability, float)
    assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# Deposition fraction constraints
# ---------------------------------------------------------------------------


def test_lung_and_oropharyngeal_sum_to_one():
    result = simulate_inhalation_pk("drug_a", nominal_dose_mg=1.0, MMAD_um=3.0)
    total = result.lung_deposited_fraction + result.oropharyngeal_fraction
    assert abs(total - 1.0) < 1e-9


def test_lung_and_oropharyngeal_sum_to_one_large_mmad():
    result = simulate_inhalation_pk("drug_b", nominal_dose_mg=1.0, MMAD_um=10.0)
    total = result.lung_deposited_fraction + result.oropharyngeal_fraction
    assert abs(total - 1.0) < 1e-9


def test_lung_and_oropharyngeal_sum_to_one_small_mmad():
    result = simulate_inhalation_pk("drug_c", nominal_dose_mg=1.0, MMAD_um=0.5)
    total = result.lung_deposited_fraction + result.oropharyngeal_fraction
    assert abs(total - 1.0) < 1e-9


def test_alveolar_plus_bronchial_equals_lung():
    result = simulate_inhalation_pk("drug_d", nominal_dose_mg=1.0, MMAD_um=3.0)
    assert (
        abs(result.alveolar_fraction + result.bronchial_fraction - result.lung_deposited_fraction)
        < 1e-9
    )


def test_alveolar_is_60pct_of_lung():
    result = simulate_inhalation_pk("drug_e", nominal_dose_mg=1.0, MMAD_um=3.0)
    assert abs(result.alveolar_fraction - result.lung_deposited_fraction * 0.6) < 1e-9


def test_swallowed_is_80pct_of_oropharyngeal():
    result = simulate_inhalation_pk("drug_f", nominal_dose_mg=1.0, MMAD_um=3.0)
    assert abs(result.swallowed_fraction - result.oropharyngeal_fraction * 0.8) < 1e-9


# ---------------------------------------------------------------------------
# Particle size effect on lung deposition
# ---------------------------------------------------------------------------


def test_smaller_mmad_higher_lung_deposition():
    res_small = simulate_inhalation_pk("drug", nominal_dose_mg=1.0, MMAD_um=1.0)
    res_large = simulate_inhalation_pk("drug", nominal_dose_mg=1.0, MMAD_um=8.0)
    assert res_small.lung_deposited_fraction > res_large.lung_deposited_fraction


def test_mmad_1_lung_deposition_is_0p5():
    result = simulate_inhalation_pk("drug", nominal_dose_mg=1.0, MMAD_um=1.0)
    assert abs(result.lung_deposited_fraction - 0.5) < 1e-9


def test_mmad_clamped_at_minimum_for_large_particles():
    result = simulate_inhalation_pk("drug", nominal_dose_mg=1.0, MMAD_um=20.0)
    assert result.lung_deposited_fraction >= 0.05


# ---------------------------------------------------------------------------
# Plasma PK metrics
# ---------------------------------------------------------------------------


def test_cmax_positive():
    result = simulate_inhalation_pk("salbutamol", nominal_dose_mg=0.1)
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = simulate_inhalation_pk("salbutamol", nominal_dose_mg=0.1)
    assert result.auc_mg_h_per_L > 0.0


def test_tmax_within_simulation_range():
    result = simulate_inhalation_pk("salbutamol", nominal_dose_mg=0.1, t_end_h=12.0)
    assert 0.0 <= result.tmax_h <= 12.0


def test_times_and_concentrations_same_length():
    result = simulate_inhalation_pk("drug", nominal_dose_mg=1.0)
    assert len(result.times_h) == len(result.c_plasma_mg_L)


# ---------------------------------------------------------------------------
# Lung metrics
# ---------------------------------------------------------------------------


def test_lung_t_half_formula():
    ka = 2.0
    result = simulate_inhalation_pk("drug", nominal_dose_mg=1.0, ka_lung_per_h=ka)
    expected = math.log(2.0) / ka
    assert abs(result.lung_t_half_h - expected) < 1e-9


def test_lung_t_half_different_ka():
    ka = 5.0
    result = simulate_inhalation_pk("drug", nominal_dose_mg=1.0, ka_lung_per_h=ka)
    expected = math.log(2.0) / ka
    assert abs(result.lung_t_half_h - expected) < 1e-9


def test_lung_depot_initial_proportional_to_dose():
    res1 = simulate_inhalation_pk("drug", nominal_dose_mg=1.0, MMAD_um=3.0)
    res2 = simulate_inhalation_pk("drug", nominal_dose_mg=2.0, MMAD_um=3.0)
    assert abs(res2.lung_depot_initial_mg / res1.lung_depot_initial_mg - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# Bioavailability
# ---------------------------------------------------------------------------


def test_systemic_bioavailability_in_range():
    result = simulate_inhalation_pk("drug", nominal_dose_mg=1.0)
    assert 0.0 < result.systemic_bioavailability <= 1.0


def test_emitted_dose_is_80pct_of_nominal():
    result = simulate_inhalation_pk("drug", nominal_dose_mg=10.0)
    assert abs(result.emitted_dose_mg - 8.0) < 1e-9


# ---------------------------------------------------------------------------
# compare_particle_sizes
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    results = compare_particle_sizes("drug", nominal_dose_mg=1.0, MMAD_values=[1.0, 3.0, 8.0])
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_sorted_by_lung_deposition_descending():
    results = compare_particle_sizes("drug", nominal_dose_mg=1.0, MMAD_values=[1.0, 3.0, 5.0, 8.0])
    fracs = [r.lung_deposited_fraction for r in results]
    assert fracs == sorted(fracs, reverse=True)


def test_compare_all_results_are_correct_type():
    results = compare_particle_sizes("drug", nominal_dose_mg=1.0, MMAD_values=[2.0, 4.0])
    for r in results:
        assert isinstance(r, InhalationDosimetryResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_negative_dose_raises():
    with pytest.raises(ValueError):
        simulate_inhalation_pk("drug", nominal_dose_mg=-1.0)


def test_zero_dose_raises():
    with pytest.raises(ValueError):
        simulate_inhalation_pk("drug", nominal_dose_mg=0.0)


def test_negative_mmad_raises():
    with pytest.raises(ValueError):
        simulate_inhalation_pk("drug", nominal_dose_mg=1.0, MMAD_um=-2.0)


def test_zero_mmad_raises():
    with pytest.raises(ValueError):
        simulate_inhalation_pk("drug", nominal_dose_mg=1.0, MMAD_um=0.0)


def test_negative_cl_raises():
    with pytest.raises(ValueError):
        simulate_inhalation_pk("drug", nominal_dose_mg=1.0, cl_L_per_h=-1.0)


def test_negative_vd_raises():
    with pytest.raises(ValueError):
        simulate_inhalation_pk("drug", nominal_dose_mg=1.0, vd_L=-10.0)
