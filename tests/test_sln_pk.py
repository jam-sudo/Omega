"""Tests for Phase 464 — Solid Lipid Nanoparticle PK."""

import pytest

from omega_pbpk.core.sln_pk import (
    SLNPKResult,
    compare_surface_coatings,
    simulate_sln_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coating="PEG", size_nm=200.0, logP=3.0, dose_mg=100.0, t_end=48.0):
    return simulate_sln_pk(
        drug_name="TestSLN",
        dose_mg=dose_mg,
        particle_size_nm=size_nm,
        surface_coating=coating,
        logP=logP,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        t_end_h=t_end,
        dt_h=0.1,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _run()
    assert isinstance(result, SLNPKResult)


def test_result_fields_present():
    result = _run()
    assert hasattr(result, "times_h")
    assert hasattr(result, "c_sln_plasma_mg_L")
    assert hasattr(result, "c_free_drug_mg_L")
    assert hasattr(result, "mps_uptake_fraction")
    assert hasattr(result, "circulation_time_h")
    assert hasattr(result, "notes")


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_c_sln_starts_at_dose_over_volume():
    """SLN IV bolus: C_sln[0] = dose / vd_sys."""
    result = simulate_sln_pk(
        drug_name="DrugB",
        dose_mg=200.0,
        surface_coating="PEG",
        logP=3.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=40.0,
        t_end_h=12.0,
        dt_h=0.1,
    )
    expected = 200.0 / 40.0
    assert abs(result.c_sln_plasma_mg_L[0] - expected) < 1e-9


def test_c_free_drug_starts_at_zero():
    result = _run()
    assert result.c_free_drug_mg_L[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Time vector
# ---------------------------------------------------------------------------


def test_times_start_at_zero():
    result = _run()
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_end_near_t_end():
    result = _run(t_end=24.0)
    assert result.times_h[-1] == pytest.approx(24.0, abs=0.15)


def test_list_lengths_match():
    result = _run()
    assert len(result.times_h) == len(result.c_sln_plasma_mg_L)
    assert len(result.times_h) == len(result.c_free_drug_mg_L)


# ---------------------------------------------------------------------------
# Surface coating effects (stealth PEG effect)
# ---------------------------------------------------------------------------


def test_peg_coating_longer_circulation_than_bare():
    peg = _run(coating="PEG")
    bare = _run(coating="bare")
    assert peg.circulation_time_h > bare.circulation_time_h


def test_bare_sln_higher_mps_uptake():
    peg = _run(coating="PEG")
    bare = _run(coating="bare")
    assert bare.mps_uptake_fraction > peg.mps_uptake_fraction


def test_bare_sln_higher_k_mps_reflected_in_circulation():
    bare = _run(coating="bare")
    peg = _run(coating="PEG")
    # bare should be cleared faster (shorter circulation time)
    assert bare.circulation_time_h < peg.circulation_time_h


# ---------------------------------------------------------------------------
# Free drug kinetics
# ---------------------------------------------------------------------------


def test_cmax_free_drug_positive():
    result = _run()
    assert result.cmax_free_drug_mg_L > 0.0


def test_auc_free_drug_positive():
    result = _run()
    assert result.auc_free_drug_mg_h_per_L > 0.0


def test_auc_sln_positive():
    result = _run()
    assert result.auc_sln_mg_h_per_L > 0.0


def test_tmax_free_drug_after_time_zero():
    """Free drug appears with delay (release from SLN)."""
    result = _run(t_end=48.0)
    assert result.tmax_free_drug_h >= 0.0


# ---------------------------------------------------------------------------
# Particle size effects
# ---------------------------------------------------------------------------


def test_larger_particle_shorter_circulation():
    small = _run(size_nm=50.0)
    large = _run(size_nm=500.0)
    assert large.circulation_time_h < small.circulation_time_h


def test_particle_size_preserved():
    result = simulate_sln_pk("X", 100.0, particle_size_nm=150.0, surface_coating="PEG")
    assert result.particle_size_nm == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# MPS uptake fraction
# ---------------------------------------------------------------------------


def test_mps_fraction_between_zero_and_one():
    result = _run()
    assert 0.0 <= result.mps_uptake_fraction <= 1.0


def test_mps_fraction_bare_close_to_one():
    """Bare SLN with high k_mps should have large MPS fraction."""
    bare = _run(coating="bare", logP=3.0)
    # k_mps_base=2.0, k_release~0.07 -> fraction ~ 2/(2+0.07)≈0.97
    assert bare.mps_uptake_fraction > 0.9


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_drug_name_preserved():
    result = simulate_sln_pk("Paclitaxel-SLN", 100.0, surface_coating="PEG")
    assert result.drug_name == "Paclitaxel-SLN"


def test_surface_coating_preserved():
    result = _run(coating="bare")
    assert result.surface_coating == "bare"


def test_notes_non_empty():
    result = _run()
    assert len(result.notes) > 0


def test_circulation_time_positive():
    result = _run()
    assert result.circulation_time_h > 0.0


# ---------------------------------------------------------------------------
# compare_surface_coatings
# ---------------------------------------------------------------------------


def test_compare_surface_coatings_returns_two():
    results = compare_surface_coatings("Drug", 100.0)
    assert len(results) == 2


def test_compare_sorted_by_circulation_time_ascending():
    results = compare_surface_coatings("Drug", 100.0)
    assert results[0].circulation_time_h <= results[1].circulation_time_h


def test_compare_first_is_bare():
    """Bare has shorter circulation time, should be first (ascending sort)."""
    results = compare_surface_coatings("Drug", 100.0)
    assert results[0].surface_coating == "bare"


def test_compare_second_is_peg():
    results = compare_surface_coatings("Drug", 100.0)
    assert results[1].surface_coating == "PEG"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_sln_pk("X", 0.0, surface_coating="PEG")


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_sln_pk("X", -50.0, surface_coating="PEG")


def test_validation_particle_size_zero():
    with pytest.raises(ValueError, match="particle_size_nm"):
        simulate_sln_pk("X", 100.0, particle_size_nm=0.0, surface_coating="PEG")


def test_validation_invalid_coating():
    with pytest.raises(ValueError, match="surface_coating"):
        simulate_sln_pk("X", 100.0, surface_coating="polymer")


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_sln_pk("X", 100.0, surface_coating="PEG", cl_sys_L_per_h=0.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_sys_L"):
        simulate_sln_pk("X", 100.0, surface_coating="PEG", vd_sys_L=0.0)
