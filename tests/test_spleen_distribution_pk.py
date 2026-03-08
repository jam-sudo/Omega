"""Tests for Phase 1031 — Drug Spleen Distribution PK."""

import pytest

from omega_pbpk.core.spleen_distribution_pk import (
    SpleenDistributionResult,
    simulate_spleen_distribution,
)

# ── Basic return type ───────────────────────────────────────────────────────


def test_returns_correct_type():
    result = simulate_spleen_distribution("TestDrug", dose_mg=10.0, particle_size_nm=0)
    assert isinstance(result, SpleenDistributionResult)


def test_initial_plasma_concentration():
    dose_mg = 100.0
    vd = 50.0
    result = simulate_spleen_distribution("DrugA", dose_mg=dose_mg, particle_size_nm=0, vd_sys_L=vd)
    assert abs(result.c_plasma_mg_L[0] - dose_mg / vd) < 1e-6


def test_initial_spleen_zero():
    result = simulate_spleen_distribution("DrugA", dose_mg=10.0, particle_size_nm=0)
    assert result.c_spleen_mg_L[0] == 0.0


def test_initial_red_pulp_zero():
    result = simulate_spleen_distribution("DrugA", dose_mg=10.0, particle_size_nm=0)
    assert result.c_red_pulp_mg_L[0] == 0.0


# ── Spleen kinetics ────────────────────────────────────────────────────────


def test_spleen_has_peak():
    result = simulate_spleen_distribution("DrugA", dose_mg=10.0, particle_size_nm=0, t_end_h=48.0)
    assert result.cmax_spleen > 0.0


def test_auc_plasma_positive():
    result = simulate_spleen_distribution("DrugA", dose_mg=10.0, particle_size_nm=0)
    assert result.auc_plasma > 0.0


def test_auc_spleen_positive():
    result = simulate_spleen_distribution("DrugA", dose_mg=10.0, particle_size_nm=0)
    assert result.auc_spleen > 0.0


def test_tmax_spleen_is_nonnegative():
    result = simulate_spleen_distribution("DrugA", dose_mg=10.0, particle_size_nm=0)
    assert result.tmax_spleen_h >= 0.0


# ── MPS uptake ─────────────────────────────────────────────────────────────


def test_mps_uptake_fraction_in_range():
    result = simulate_spleen_distribution("DrugA", dose_mg=10.0, particle_size_nm=100)
    assert 0.0 <= result.mps_uptake_fraction <= 1.0


def test_no_mps_small_molecule_red_pulp_near_zero():
    """Without MPS (particle_size_nm=0, mps_rate=0): red pulp stays essentially 0."""
    result = simulate_spleen_distribution(
        "SmallMol",
        dose_mg=10.0,
        particle_size_nm=0,
        mps_rate_per_h=0.0,
    )
    assert max(result.c_red_pulp_mg_L) < 1e-9


def test_nanoparticle_mps_uptake_nonzero():
    """100 nm nanoparticles should have nonzero MPS uptake."""
    result = simulate_spleen_distribution("NanoDrug", dose_mg=10.0, particle_size_nm=100.0)
    assert result.mps_uptake_fraction > 0.0


def test_150nm_higher_mps_than_10nm():
    """150 nm is at Gaussian peak; 10 nm should have lower MPS uptake."""
    r_peak = simulate_spleen_distribution("Nano150", dose_mg=10.0, particle_size_nm=150.0)
    r_small = simulate_spleen_distribution("Nano10", dose_mg=10.0, particle_size_nm=10.0)
    assert r_peak.mps_uptake_fraction > r_small.mps_uptake_fraction


# ── Partition coefficient effect ────────────────────────────────────────────


def test_higher_kp_spleen_increases_spleen_concentration():
    r_low = simulate_spleen_distribution("Drug", dose_mg=10.0, particle_size_nm=0, kp_spleen=1.0)
    r_high = simulate_spleen_distribution("Drug", dose_mg=10.0, particle_size_nm=0, kp_spleen=5.0)
    assert r_high.auc_spleen > r_low.auc_spleen


# ── Ratio ──────────────────────────────────────────────────────────────────


def test_spleen_to_plasma_ratio_positive():
    result = simulate_spleen_distribution("DrugA", dose_mg=10.0, particle_size_nm=0)
    assert result.spleen_to_plasma_ratio > 0.0


# ── Notes ─────────────────────────────────────────────────────────────────


def test_notes_nonempty():
    result = simulate_spleen_distribution("DrugA", dose_mg=10.0, particle_size_nm=0)
    assert len(result.notes) > 0


def test_notes_nanoparticle_mention():
    result = simulate_spleen_distribution("NP", dose_mg=10.0, particle_size_nm=100.0)
    assert "Nanoparticle" in result.notes or "100" in result.notes


# ── Time vector ────────────────────────────────────────────────────────────


def test_time_vector_length_matches_concentrations():
    result = simulate_spleen_distribution("Drug", dose_mg=5.0, particle_size_nm=0, t_end_h=10.0)
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) == len(result.c_spleen_mg_L)
    assert len(result.times_h) == len(result.c_red_pulp_mg_L)


# ── Validation errors ──────────────────────────────────────────────────────


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_spleen_distribution("Drug", dose_mg=0.0)


def test_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_spleen_distribution("Drug", dose_mg=10.0, cl_sys_L_per_h=0.0)


def test_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_sys_L"):
        simulate_spleen_distribution("Drug", dose_mg=10.0, vd_sys_L=0.0)


def test_kp_zero_raises():
    with pytest.raises(ValueError, match="kp_spleen"):
        simulate_spleen_distribution("Drug", dose_mg=10.0, kp_spleen=0.0)


def test_negative_particle_size_raises():
    with pytest.raises(ValueError, match="particle_size_nm"):
        simulate_spleen_distribution("Drug", dose_mg=10.0, particle_size_nm=-5.0)


# ── Edge cases ─────────────────────────────────────────────────────────────


def test_explicit_mps_rate():
    """Explicitly setting mps_rate_per_h increases red pulp accumulation."""
    r_no_mps = simulate_spleen_distribution(
        "Drug", dose_mg=10.0, particle_size_nm=0, mps_rate_per_h=0.0
    )
    r_mps = simulate_spleen_distribution(
        "Drug", dose_mg=10.0, particle_size_nm=0, mps_rate_per_h=0.5
    )
    assert r_mps.mps_uptake_fraction > r_no_mps.mps_uptake_fraction
