"""Tests for SC depot formulation PK model (Phase 977)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.subcutaneous_depot_pk_p977 import (
    SubcutaneousDepotResult,
    simulate_sc_depot,
)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_result_type():
    r = simulate_sc_depot("drug_a", dose_mg=100.0)
    assert isinstance(r, SubcutaneousDepotResult)


def test_drug_name_preserved():
    r = simulate_sc_depot("mylong_acting", dose_mg=50.0)
    assert r.drug_name == "mylong_acting"


def test_dose_preserved():
    r = simulate_sc_depot("drug_a", dose_mg=75.0)
    assert r.dose_mg == pytest.approx(75.0)


def test_particle_size_preserved():
    r = simulate_sc_depot("drug_a", dose_mg=100.0, particle_size_um=200.0)
    assert r.particle_size_um == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_c_plasma_initial_zero():
    r = simulate_sc_depot("drug_a", dose_mg=100.0)
    assert r.c_plasma_mg_L[0] == pytest.approx(0.0)


def test_a_depot_initial_equals_dose():
    r = simulate_sc_depot("drug_a", dose_mg=100.0)
    assert r.a_depot_mg[0] == pytest.approx(100.0)


def test_a_depot_monotonically_decreases():
    r = simulate_sc_depot("drug_a", dose_mg=100.0, particle_size_um=50.0)
    for i in range(1, len(r.a_depot_mg)):
        assert r.a_depot_mg[i] <= r.a_depot_mg[i - 1] + 1e-10


# ---------------------------------------------------------------------------
# PK metric validity
# ---------------------------------------------------------------------------


def test_cmax_positive():
    r = simulate_sc_depot("drug_a", dose_mg=100.0)
    assert r.cmax_mg_L > 0.0


def test_tmax_positive():
    r = simulate_sc_depot("drug_a", dose_mg=100.0)
    assert r.tmax_h > 0.0


def test_auc_positive():
    r = simulate_sc_depot("drug_a", dose_mg=100.0)
    assert r.auc_mg_h_per_L > 0.0


def test_f_absorbed_in_valid_range():
    r = simulate_sc_depot("drug_a", dose_mg=100.0)
    assert 0.0 < r.f_absorbed <= 1.0


def test_t_half_absorption_positive():
    r = simulate_sc_depot("drug_a", dose_mg=100.0)
    assert r.t_half_absorption_h > 0.0


def test_notes_nonempty():
    r = simulate_sc_depot("drug_a", dose_mg=100.0)
    assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Particle size effects
# ---------------------------------------------------------------------------


def test_large_particles_longer_tmax():
    """Larger particles dissolve slower → later Tmax."""
    r_small = simulate_sc_depot("drug_a", dose_mg=100.0, particle_size_um=10.0)
    r_large = simulate_sc_depot("drug_a", dose_mg=100.0, particle_size_um=500.0)
    assert r_large.tmax_h > r_small.tmax_h


# ---------------------------------------------------------------------------
# Solubility effects
# ---------------------------------------------------------------------------


def test_higher_solubility_faster_absorption():
    """Higher solubility → faster dissolution → smaller tmax."""
    r_low = simulate_sc_depot("drug_a", dose_mg=100.0, solubility_mg_mL=0.1)
    r_high = simulate_sc_depot("drug_a", dose_mg=100.0, solubility_mg_mL=10.0)
    assert r_high.tmax_h < r_low.tmax_h


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


def test_double_dose_doubles_cmax():
    r1 = simulate_sc_depot("drug_a", dose_mg=100.0)
    r2 = simulate_sc_depot("drug_a", dose_mg=200.0)
    assert r2.cmax_mg_L == pytest.approx(2.0 * r1.cmax_mg_L, rel=1e-4)


def test_double_dose_doubles_auc():
    r1 = simulate_sc_depot("drug_a", dose_mg=100.0)
    r2 = simulate_sc_depot("drug_a", dose_mg=200.0)
    assert r2.auc_mg_h_per_L == pytest.approx(2.0 * r1.auc_mg_h_per_L, rel=1e-4)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_sc_depot("drug_a", dose_mg=0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_sc_depot("drug_a", dose_mg=-50.0)


def test_invalid_particle_size_raises():
    with pytest.raises(ValueError, match="particle_size_um"):
        simulate_sc_depot("drug_a", dose_mg=100.0, particle_size_um=0.0)


def test_negative_particle_size_raises():
    with pytest.raises(ValueError, match="particle_size_um"):
        simulate_sc_depot("drug_a", dose_mg=100.0, particle_size_um=-10.0)


def test_invalid_solubility_raises():
    with pytest.raises(ValueError, match="solubility_mg_mL"):
        simulate_sc_depot("drug_a", dose_mg=100.0, solubility_mg_mL=0.0)


def test_invalid_vd_raises():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_sc_depot("drug_a", dose_mg=100.0, vd_L=0.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_sc_depot("drug_a", dose_mg=100.0, cl_L_per_h=0.0)
