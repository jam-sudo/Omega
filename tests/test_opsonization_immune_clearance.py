"""Tests for Phase 1052 — Drug Opsonization and Immune Clearance."""

import pytest

from omega_pbpk.prediction.opsonization_immune_clearance import (
    OpsonizationResult,
    compare_surface_coatings,
    predict_opsonization,
)


def _default(**kwargs) -> OpsonizationResult:
    params = dict(
        particle_name="TestNP",
        particle_size_nm=100.0,
        surface_coating="none",
        zeta_potential_mV=-10.0,
        material="PLGA",
        peg_density_chains_per_nm2=0.0,
    )
    params.update(kwargs)
    return predict_opsonization(**params)


# --- Return type (frozen) ---
def test_returns_opsonization_result():
    result = _default()
    assert isinstance(result, OpsonizationResult)


def test_result_is_frozen():
    result = _default()
    with pytest.raises(Exception):
        result.complement_activation_score = 99.0  # type: ignore[misc]


# --- Complement activation ---
def test_complement_activation_in_range():
    result = _default()
    assert 0 <= result.complement_activation_score <= 100


def test_zwitterionic_lower_complement_than_none():
    zw = _default(surface_coating="zwitterionic")
    none = _default(surface_coating="none")
    assert zw.complement_activation_score < none.complement_activation_score


# --- Phagocytosis rate ---
def test_phagocytosis_rate_positive():
    result = _default()
    assert result.phagocytosis_rate_per_h > 0


# --- MPS clearance ---
def test_mps_clearance_t50_positive():
    result = _default()
    assert result.mps_clearance_t50_h > 0


# --- Circulation half-life ---
def test_circulation_half_life_positive():
    result = _default()
    assert result.circulation_half_life_h > 0


# --- PEG shielding ---
def test_peg_shielding_in_range():
    result = _default()
    assert 0 <= result.peg_shielding_efficiency <= 1


def test_peg_shielding_zero_for_no_coating():
    result = _default(surface_coating="none")
    assert result.peg_shielding_efficiency == 0.0


def test_peg20k_longer_half_life_than_none():
    peg = _default(surface_coating="PEG_20k", peg_density_chains_per_nm2=0.5)
    none = _default(surface_coating="none")
    assert peg.circulation_half_life_h > none.circulation_half_life_h


# --- Renal clearance ---
def test_renal_clearance_in_range():
    result = _default()
    assert 0 <= result.renal_clearance_contribution <= 1


def test_small_particles_high_renal():
    result = _default(particle_size_nm=5.0)
    assert result.renal_clearance_contribution > 0.8


def test_large_particles_zero_renal():
    result = _default(particle_size_nm=200.0)
    assert result.renal_clearance_contribution == 0.0


# --- Overall clearance class ---
def test_clearance_class_valid():
    result = _default()
    assert result.overall_clearance_class in {"slow", "moderate", "fast"}


def test_clearance_class_consistency_with_half_life():
    result = _default()
    t = result.circulation_half_life_h
    cls = result.overall_clearance_class
    if t > 24:
        assert cls == "slow"
    elif t >= 6:
        assert cls == "moderate"
    else:
        assert cls == "fast"


# --- compare_surface_coatings ---
def test_compare_surface_coatings_returns_six():
    results = compare_surface_coatings("NP", 100.0)
    assert len(results) == 6


def test_compare_surface_coatings_sorted_descending():
    results = compare_surface_coatings("NP", 100.0)
    half_lives = [r.circulation_half_life_h for r in results]
    assert half_lives == sorted(half_lives, reverse=True)


# --- Notes ---
def test_notes_nonempty():
    result = _default()
    assert len(result.notes) > 0


# --- Validation errors ---
def test_raises_on_zero_size():
    with pytest.raises(ValueError):
        predict_opsonization("NP", particle_size_nm=0.0)


def test_raises_on_invalid_coating():
    with pytest.raises(ValueError):
        predict_opsonization("NP", particle_size_nm=100.0, surface_coating="PTFE")


def test_raises_on_invalid_material():
    with pytest.raises(ValueError):
        predict_opsonization("NP", particle_size_nm=100.0, material="carbon")


def test_raises_on_negative_peg_density():
    with pytest.raises(ValueError):
        predict_opsonization("NP", particle_size_nm=100.0, peg_density_chains_per_nm2=-0.1)
