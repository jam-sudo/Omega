"""Tests for Phase 928 — Nanosuspension PK Model."""

import math
import pytest

from omega_pbpk.core.nanosuspension_pk import (
    NanosuspensionPKResult,
    simulate_nanosuspension,
    compare_particle_sizes,
)


# ── Basic return type and structure ──────────────────────────────────────────

def test_return_type():
    result = simulate_nanosuspension("NanoA", dose_mg=100.0)
    assert isinstance(result, NanosuspensionPKResult)


def test_c_plasma_starts_at_zero():
    result = simulate_nanosuspension("NanoA", dose_mg=100.0)
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-9)


def test_cmax_positive():
    result = simulate_nanosuspension("NanoA", dose_mg=100.0)
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = simulate_nanosuspension("NanoA", dose_mg=100.0)
    assert result.auc_mg_h_per_L > 0.0


def test_tmax_nonnegative():
    result = simulate_nanosuspension("NanoA", dose_mg=100.0)
    assert result.tmax_h >= 0.0


def test_m_undissolved_starts_at_dose():
    result = simulate_nanosuspension("NanoA", dose_mg=200.0)
    assert result.m_undissolved_mg[0] == pytest.approx(200.0, abs=1e-6)


def test_m_undissolved_nonincreasing():
    result = simulate_nanosuspension("NanoA", dose_mg=100.0)
    for i in range(1, len(result.m_undissolved_mg)):
        assert result.m_undissolved_mg[i] <= result.m_undissolved_mg[i - 1] + 1e-9


def test_fraction_dissolved_at_2h_in_range():
    result = simulate_nanosuspension("NanoA", dose_mg=100.0)
    assert 0.0 <= result.fraction_dissolved_at_2h <= 1.0


def test_fraction_absorbed_in_range():
    result = simulate_nanosuspension("NanoA", dose_mg=100.0)
    assert 0.0 <= result.fraction_absorbed <= 1.0


# ── Particle size effect ──────────────────────────────────────────────────────

def test_smaller_particle_faster_dissolution():
    result_nano = simulate_nanosuspension("NanoA", dose_mg=100.0, particle_radius_um=0.2)
    result_conv = simulate_nanosuspension("NanoA", dose_mg=100.0, particle_radius_um=50.0)
    assert result_nano.fraction_dissolved_at_2h > result_conv.fraction_dissolved_at_2h


def test_smaller_particle_higher_auc():
    result_nano = simulate_nanosuspension("NanoA", dose_mg=100.0, particle_radius_um=0.2)
    result_conv = simulate_nanosuspension("NanoA", dose_mg=100.0, particle_radius_um=50.0)
    assert result_nano.auc_mg_h_per_L > result_conv.auc_mg_h_per_L


# ── compare_particle_sizes ────────────────────────────────────────────────────

def test_compare_returns_4_results_default():
    results = compare_particle_sizes("NanoA", dose_mg=100.0)
    assert len(results) == 4


def test_compare_sorted_by_auc_descending():
    results = compare_particle_sizes("NanoA", dose_mg=100.0)
    aucs = [r.auc_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_smallest_radius_highest_auc():
    results = compare_particle_sizes("NanoA", dose_mg=100.0, radii_um=[0.2, 0.5, 5.0, 50.0])
    smallest_radius_result = min(results, key=lambda x: x.particle_radius_um)
    assert smallest_radius_result.auc_mg_h_per_L == results[0].auc_mg_h_per_L


# ── Dose linearity ────────────────────────────────────────────────────────────

def test_double_dose_approx_double_cmax():
    r1 = simulate_nanosuspension("NanoA", dose_mg=100.0, particle_radius_um=0.5)
    r2 = simulate_nanosuspension("NanoA", dose_mg=200.0, particle_radius_um=0.5)
    # Linear PK: 2x dose → ~2x Cmax (within 5% tolerance due to dissolution nonlinearity)
    assert r2.cmax_mg_L == pytest.approx(r1.cmax_mg_L * 2.0, rel=0.1)


# ── Other attributes ──────────────────────────────────────────────────────────

def test_dissolution_rate_constant_positive():
    result = simulate_nanosuspension("NanoA", dose_mg=100.0)
    assert result.dissolution_rate_constant > 0.0


def test_notes_nonempty():
    result = simulate_nanosuspension("NanoA", dose_mg=100.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ── Validation ────────────────────────────────────────────────────────────────

def test_validation_dose_mg_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_nanosuspension("NanoA", dose_mg=0.0)


def test_validation_dose_mg_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_nanosuspension("NanoA", dose_mg=-100.0)


def test_validation_particle_radius_zero():
    with pytest.raises(ValueError, match="particle_radius_um"):
        simulate_nanosuspension("NanoA", dose_mg=100.0, particle_radius_um=0.0)


def test_validation_particle_radius_negative():
    with pytest.raises(ValueError, match="particle_radius_um"):
        simulate_nanosuspension("NanoA", dose_mg=100.0, particle_radius_um=-1.0)


def test_validation_cs_zero():
    with pytest.raises(ValueError, match="cs_mg_mL"):
        simulate_nanosuspension("NanoA", dose_mg=100.0, cs_mg_mL=0.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_nanosuspension("NanoA", dose_mg=100.0, cl_L_per_h=0.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_nanosuspension("NanoA", dose_mg=100.0, vd_L=0.0)


def test_validation_vgut_zero():
    with pytest.raises(ValueError, match="vgut_L"):
        simulate_nanosuspension("NanoA", dose_mg=100.0, vgut_L=0.0)


# ── Custom radii ──────────────────────────────────────────────────────────────

def test_compare_custom_radii_returns_3():
    results = compare_particle_sizes("NanoA", dose_mg=100.0, radii_um=[0.1, 1.0, 10.0])
    assert len(results) == 3


# ── Times length ──────────────────────────────────────────────────────────────

def test_times_h_length_matches_ode_steps():
    t_end_h = 24.0
    dt_h = 0.1
    result = simulate_nanosuspension("NanoA", dose_mg=100.0, t_end_h=t_end_h, dt_h=dt_h)
    expected_steps = int(round(t_end_h / dt_h)) + 1
    assert len(result.times_h) == expected_steps
