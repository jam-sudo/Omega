"""Tests for Phase 1076: Drug Nanocrystal PK (nanocrystal_pk_p1076.py)."""

import pytest

from omega_pbpk.core.nanocrystal_pk_p1076 import (
    NanocrystalPKResult1076,
    compare_particle_sizes,
    simulate_nanocrystal_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result(size_nm: float = 200.0, dose_mg: float = 100.0) -> NanocrystalPKResult1076:
    return simulate_nanocrystal_pk("TestDrug", dose_mg, particle_size_nm=size_nm)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    res = make_result()
    assert isinstance(res, NanocrystalPKResult1076)


def test_fields_present():
    res = make_result()
    for field in [
        "drug_name",
        "dose_mg",
        "particle_size_nm",
        "times_h",
        "c_plasma_mg_L",
        "a_crystal_mg",
        "a_dissolved_mg",
        "cmax",
        "tmax_h",
        "auc",
        "f_dissolved_at_4h",
        "dissolution_t50_h",
        "enhancement_vs_bulk",
        "notes",
    ]:
        assert hasattr(res, field), f"Missing field: {field}"


# ---------------------------------------------------------------------------
# c_plasma starts at 0
# ---------------------------------------------------------------------------


def test_c_plasma_starts_at_zero():
    res = make_result()
    assert res.c_plasma_mg_L[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# cmax > 0
# ---------------------------------------------------------------------------


def test_cmax_positive():
    res = make_result()
    assert res.cmax > 0.0


# ---------------------------------------------------------------------------
# Smaller particles → higher AUC
# ---------------------------------------------------------------------------


def test_smaller_particles_higher_auc():
    small = simulate_nanocrystal_pk("Drug", 100.0, particle_size_nm=200.0)
    large = simulate_nanocrystal_pk("Drug", 100.0, particle_size_nm=5000.0)
    assert small.auc > large.auc


# ---------------------------------------------------------------------------
# Smaller particles → faster dissolution_t50
# ---------------------------------------------------------------------------


def test_smaller_particles_faster_dissolution():
    small = simulate_nanocrystal_pk("Drug", 100.0, particle_size_nm=200.0)
    large = simulate_nanocrystal_pk("Drug", 100.0, particle_size_nm=5000.0)
    assert small.dissolution_t50_h <= large.dissolution_t50_h


# ---------------------------------------------------------------------------
# enhancement_vs_bulk > 1 for nanocrystals (size < 10000 nm)
# ---------------------------------------------------------------------------


def test_enhancement_vs_bulk_above_one():
    res = simulate_nanocrystal_pk("Drug", 100.0, particle_size_nm=200.0)
    assert res.enhancement_vs_bulk > 1.0


def test_bulk_enhancement_is_one():
    """At exactly bulk size (10000 nm), enhancement should be ~1.0."""
    res = simulate_nanocrystal_pk("Drug", 100.0, particle_size_nm=10000.0)
    assert res.enhancement_vs_bulk == pytest.approx(1.0, rel=0.01)


# ---------------------------------------------------------------------------
# f_dissolved_at_4h in [0, 1]
# ---------------------------------------------------------------------------


def test_f_dissolved_at_4h_range():
    res = make_result()
    assert 0.0 <= res.f_dissolved_at_4h <= 1.0


def test_small_particle_more_dissolved_at_4h():
    small = simulate_nanocrystal_pk("Drug", 100.0, particle_size_nm=200.0)
    large = simulate_nanocrystal_pk("Drug", 100.0, particle_size_nm=8000.0)
    assert small.f_dissolved_at_4h >= large.f_dissolved_at_4h


# ---------------------------------------------------------------------------
# compare_particle_sizes: correct count & sorted descending
# ---------------------------------------------------------------------------


def test_compare_particle_sizes_count():
    sizes = [200.0, 500.0, 1000.0, 5000.0]
    results = compare_particle_sizes("Drug", 100.0, sizes)
    assert len(results) == 4


def test_compare_particle_sizes_sorted_auc_descending():
    sizes = [200.0, 1000.0, 5000.0]
    results = compare_particle_sizes("Drug", 100.0, sizes)
    for i in range(len(results) - 1):
        assert results[i].auc >= results[i + 1].auc


def test_compare_particle_sizes_smallest_first():
    """After sorting by AUC desc, smallest particle should be first."""
    sizes = [200.0, 5000.0]
    results = compare_particle_sizes("Drug", 100.0, sizes)
    assert results[0].particle_size_nm < results[1].particle_size_nm


# ---------------------------------------------------------------------------
# a_crystal starts at dose_mg
# ---------------------------------------------------------------------------


def test_a_crystal_starts_at_dose():
    res = make_result(dose_mg=200.0)
    assert res.a_crystal_mg[0] == pytest.approx(200.0, rel=1e-3)


# ---------------------------------------------------------------------------
# a_crystal monotone decreasing (approximately)
# ---------------------------------------------------------------------------


def test_a_crystal_monotone_decreasing():
    res = make_result()
    for i in range(1, len(res.a_crystal_mg)):
        # Allow tiny floating point tolerance
        assert res.a_crystal_mg[i] <= res.a_crystal_mg[i - 1] + 1e-9


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_zero():
    with pytest.raises(ValueError):
        simulate_nanocrystal_pk("Drug", 0.0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError):
        simulate_nanocrystal_pk("Drug", -50.0)


def test_invalid_particle_size_zero():
    with pytest.raises(ValueError):
        simulate_nanocrystal_pk("Drug", 100.0, particle_size_nm=0.0)


def test_invalid_particle_size_too_large():
    with pytest.raises(ValueError):
        simulate_nanocrystal_pk("Drug", 100.0, particle_size_nm=200_000.0)


def test_invalid_cl():
    with pytest.raises(ValueError):
        simulate_nanocrystal_pk("Drug", 100.0, cl_L_per_h=0.0)


def test_invalid_vd():
    with pytest.raises(ValueError):
        simulate_nanocrystal_pk("Drug", 100.0, vd_L=-1.0)


# ---------------------------------------------------------------------------
# Additional correctness
# ---------------------------------------------------------------------------


def test_drug_name_stored():
    res = simulate_nanocrystal_pk("Ibuprofen", 200.0)
    assert res.drug_name == "Ibuprofen"


def test_dose_stored():
    res = simulate_nanocrystal_pk("Drug", 150.0)
    assert res.dose_mg == pytest.approx(150.0)


def test_particle_size_stored():
    res = simulate_nanocrystal_pk("Drug", 100.0, particle_size_nm=400.0)
    assert res.particle_size_nm == pytest.approx(400.0)


def test_auc_positive():
    res = make_result()
    assert res.auc > 0.0


def test_times_start_at_zero():
    res = make_result()
    assert res.times_h[0] == pytest.approx(0.0)


def test_notes_nonempty():
    res = make_result()
    assert len(res.notes) > 0


def test_tmax_within_simulation():
    res = make_result()
    assert 0.0 <= res.tmax_h <= 24.0
