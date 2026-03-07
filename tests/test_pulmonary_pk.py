"""Tests for the pulmonary drug delivery PBPK model."""

from __future__ import annotations

import pytest

from omega_pbpk.core.pulmonary_pk import (
    PulmonaryPKResult,
    compare_particle_sizes,
    simulate_pulmonary_pk,
)

# ---- Fixtures ----


@pytest.fixture()
def default_result() -> PulmonaryPKResult:
    return simulate_pulmonary_pk("TestDrug", dose_mg=10.0)


@pytest.fixture()
def double_dose_result() -> PulmonaryPKResult:
    return simulate_pulmonary_pk("TestDrug", dose_mg=20.0)


# ---- 1. Valid result with defaults ----


def test_valid_result_returned(default_result: PulmonaryPKResult) -> None:
    assert isinstance(default_result, PulmonaryPKResult)


# ---- 2-6. Validation errors ----


def test_dose_mg_le_zero_raises() -> None:
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_pulmonary_pk("X", dose_mg=0.0)


def test_dose_mg_negative_raises() -> None:
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_pulmonary_pk("X", dose_mg=-1.0)


def test_particle_size_le_zero_raises() -> None:
    with pytest.raises(ValueError, match="particle_size_um"):
        simulate_pulmonary_pk("X", dose_mg=10.0, particle_size_um=0.0)


def test_k_lung_abs_le_zero_raises() -> None:
    with pytest.raises(ValueError, match="k_lung_abs_per_h"):
        simulate_pulmonary_pk("X", dose_mg=10.0, k_lung_abs_per_h=-0.1)


def test_cl_sys_le_zero_raises() -> None:
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_pulmonary_pk("X", dose_mg=10.0, cl_sys_L_per_h=0.0)


def test_vd_sys_le_zero_raises() -> None:
    with pytest.raises(ValueError, match="vd_sys_L"):
        simulate_pulmonary_pk("X", dose_mg=10.0, vd_sys_L=-5.0)


# ---- 7-8. Stored metadata ----


def test_route_stored_correctly() -> None:
    res = simulate_pulmonary_pk("X", dose_mg=10.0, route="nebulized")
    assert res.route == "nebulized"


def test_drug_name_stored_correctly(default_result: PulmonaryPKResult) -> None:
    assert default_result.drug_name == "TestDrug"


# ---- 9. Array length consistency ----


def test_times_length_matches_concentrations(default_result: PulmonaryPKResult) -> None:
    n = len(default_result.times_h)
    assert len(default_result.c_airway_mg_L) == n
    assert len(default_result.c_lung_tissue_mg_L) == n
    assert len(default_result.c_plasma_mg_L) == n


# ---- 10. Airway concentration starts high and decreases ----


def test_airway_starts_high_and_decreases(default_result: PulmonaryPKResult) -> None:
    c = default_result.c_airway_mg_L
    assert c[0] > c[-1]
    assert c[0] == default_result.cmax_airway


# ---- 11. Lung tissue peaks then declines ----


def test_lung_tissue_peaks_then_declines(default_result: PulmonaryPKResult) -> None:
    c = default_result.c_lung_tissue_mg_L
    peak_idx = c.index(max(c))
    assert peak_idx > 0
    assert c[-1] < max(c)


# ---- 12. Plasma peaks after lung ----


def test_plasma_peaks_after_lung(default_result: PulmonaryPKResult) -> None:
    lung_peak_idx = default_result.c_lung_tissue_mg_L.index(max(default_result.c_lung_tissue_mg_L))
    plasma_peak_idx = default_result.c_plasma_mg_L.index(max(default_result.c_plasma_mg_L))
    assert plasma_peak_idx >= lung_peak_idx


# ---- 13. Larger particles -> lower lung Cmax ----


def test_larger_particles_lower_lung_cmax(default_result: PulmonaryPKResult) -> None:
    large = simulate_pulmonary_pk("X", dose_mg=10.0, particle_size_um=8.0)
    assert large.cmax_lung_tissue < default_result.cmax_lung_tissue


# ---- 14. Small particles -> lower deposition than optimal ----


def test_small_particles_lower_deposition(default_result: PulmonaryPKResult) -> None:
    small = simulate_pulmonary_pk("X", dose_mg=10.0, particle_size_um=0.5)
    assert small.cmax_lung_tissue < default_result.cmax_lung_tissue


# ---- 15-16. Linear dose scaling ----


def test_dose_scaling_cmax_plasma(
    default_result: PulmonaryPKResult,
    double_dose_result: PulmonaryPKResult,
) -> None:
    ratio = double_dose_result.cmax_plasma / default_result.cmax_plasma
    assert abs(ratio - 2.0) < 0.05


def test_dose_scaling_auc_plasma(
    default_result: PulmonaryPKResult,
    double_dose_result: PulmonaryPKResult,
) -> None:
    ratio = double_dose_result.auc_plasma / default_result.auc_plasma
    assert abs(ratio - 2.0) < 0.05


# ---- 17-18. compare_particle_sizes ----


def test_compare_particle_sizes_count() -> None:
    sizes = [1.0, 3.0, 5.0, 8.0]
    results = compare_particle_sizes("X", 10.0, sizes)
    assert len(results) == len(sizes)


def test_compare_particle_sizes_values() -> None:
    sizes = [1.5, 4.0]
    results = compare_particle_sizes("X", 10.0, sizes)
    assert results[0].particle_size_um == 1.5
    assert results[1].particle_size_um == 4.0


# ---- 19. Higher mucociliary clearance -> lower lung AUC ----


def test_higher_mucociliary_clearance_lower_lung_auc() -> None:
    low_clear = simulate_pulmonary_pk("X", dose_mg=10.0, mucociliary_clearance_per_h=0.1)
    high_clear = simulate_pulmonary_pk("X", dose_mg=10.0, mucociliary_clearance_per_h=2.0)
    assert high_clear.auc_lung_tissue < low_clear.auc_lung_tissue


# ---- 20. Higher k_lung_abs -> higher plasma Cmax ----


def test_higher_k_lung_abs_higher_plasma_cmax() -> None:
    slow = simulate_pulmonary_pk("X", dose_mg=10.0, k_lung_abs_per_h=0.5)
    fast = simulate_pulmonary_pk("X", dose_mg=10.0, k_lung_abs_per_h=5.0)
    assert fast.cmax_plasma > slow.cmax_plasma


# ---- 21. lung_to_plasma_ratio > 0 ----


def test_lung_to_plasma_ratio_positive(default_result: PulmonaryPKResult) -> None:
    assert default_result.lung_to_plasma_ratio > 0


# ---- 22. pulmonary_bioavailability between 0 and 1 ----


def test_pulmonary_bioavailability_range(default_result: PulmonaryPKResult) -> None:
    assert 0.0 < default_result.pulmonary_bioavailability < 1.0


# ---- 23. AUC values positive ----


def test_auc_values_positive(default_result: PulmonaryPKResult) -> None:
    assert default_result.auc_lung_tissue > 0
    assert default_result.auc_plasma > 0


# ---- 24. Cmax values positive ----


def test_cmax_values_positive(default_result: PulmonaryPKResult) -> None:
    assert default_result.cmax_airway > 0
    assert default_result.cmax_lung_tissue > 0
    assert default_result.cmax_plasma > 0


# ---- 25. Notes string non-empty ----


def test_notes_non_empty(default_result: PulmonaryPKResult) -> None:
    assert len(default_result.notes) > 0
    assert "TestDrug" in default_result.notes
