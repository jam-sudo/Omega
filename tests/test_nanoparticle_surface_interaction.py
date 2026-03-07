"""Tests for Phase 968 — Nanoparticle surface interaction prediction."""

import pytest

from omega_pbpk.prediction.nanoparticle_surface_interaction import (
    NanoparticleSurfaceResult,
    compare_surface_modifications,
    predict_surface_interaction,
)


def _default_result(**kwargs):
    defaults = dict(
        nanoparticle_name="test_NP",
        diameter_nm=100.0,
        zeta_mV=-10.0,
        contact_angle_deg=70.0,
        is_pegylated=False,
        k_release_per_h=0.1,
        baseline_t_half_h=24.0,
    )
    defaults.update(kwargs)
    return predict_surface_interaction(**defaults)


def test_return_type():
    result = _default_result()
    assert isinstance(result, NanoparticleSurfaceResult)


def test_frozen_dataclass():
    result = _default_result()
    with pytest.raises((AttributeError, TypeError)):
        result.corona_score = 99.0  # type: ignore


def test_corona_score_in_range():
    result = _default_result()
    assert 0 <= result.corona_score <= 100


def test_mps_clearance_factor_gte_1():
    result = _default_result()
    assert result.mps_clearance_factor >= 1.0


def test_effective_release_rate_positive():
    result = _default_result()
    assert result.effective_release_rate_per_h > 0


def test_opsonization_risk_valid():
    result = _default_result()
    assert result.opsonization_risk in {"high", "moderate", "low"}


def test_circulation_half_life_positive():
    result = _default_result()
    assert result.circulation_half_life_h > 0


def test_targeting_efficiency_positive():
    result = _default_result()
    assert result.targeting_efficiency > 0


def test_notes_nonempty():
    result = _default_result()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


def test_pegylated_lower_corona():
    """PEGylated → lower corona_score vs non-PEGylated (same params)."""
    r_normal = _default_result(is_pegylated=False, contact_angle_deg=70, zeta_mV=-10)
    r_peg = _default_result(is_pegylated=True, contact_angle_deg=70, zeta_mV=-10)
    assert r_peg.corona_score < r_normal.corona_score


def test_pegylated_longer_circulation():
    """PEGylated → longer circulation_half_life_h."""
    r_normal = _default_result(is_pegylated=False)
    r_peg = _default_result(is_pegylated=True)
    assert r_peg.circulation_half_life_h > r_normal.circulation_half_life_h


def test_high_contact_angle_higher_corona():
    """High contact_angle (>60) → higher corona_score vs low contact angle."""
    r_hydrophilic = _default_result(contact_angle_deg=20.0)
    r_hydrophobic = _default_result(contact_angle_deg=80.0)
    assert r_hydrophobic.corona_score > r_hydrophilic.corona_score


def test_high_zeta_higher_corona():
    """High |zeta_mV| (30) → higher corona than near-neutral (5)."""
    r_neutral = _default_result(zeta_mV=-5.0)
    r_charged = _default_result(zeta_mV=-30.0)
    assert r_charged.corona_score > r_neutral.corona_score


def test_small_particle_higher_corona():
    """Small particle (20nm) → higher corona than large (300nm)."""
    r_small = _default_result(diameter_nm=20.0)
    r_large = _default_result(diameter_nm=300.0)
    assert r_small.corona_score > r_large.corona_score


def test_high_corona_score_high_risk():
    """corona_score > 60 → opsonization_risk 'high'."""
    # Force high corona: hydrophobic + charged + no PEG + small
    r = predict_surface_interaction(
        nanoparticle_name="high_risk",
        diameter_nm=20.0,
        zeta_mV=-30.0,
        contact_angle_deg=80.0,
        is_pegylated=False,
    )
    assert r.corona_score > 60
    assert r.opsonization_risk == "high"


def test_low_corona_score_low_risk():
    """corona_score <= 30 → opsonization_risk 'low'."""
    # Force low corona: PEGylated + neutral + hydrophilic + large
    r = predict_surface_interaction(
        nanoparticle_name="low_risk",
        diameter_nm=300.0,
        zeta_mV=-5.0,
        contact_angle_deg=20.0,
        is_pegylated=True,
    )
    assert r.corona_score <= 30
    assert r.opsonization_risk == "low"


def test_compare_surface_modifications_returns_3():
    results = compare_surface_modifications("test_NP", 100.0)
    assert len(results) == 3


def test_compare_sorted_descending_half_life():
    results = compare_surface_modifications("test_NP", 100.0)
    half_lives = [r.circulation_half_life_h for r in results]
    assert half_lives == sorted(half_lives, reverse=True)


def test_compare_pegylated_longest_circulation():
    """PEGylated modification has longest circulation_half_life_h."""
    results = compare_surface_modifications("test_NP", 100.0)
    # First result (sorted descending) should be PEGylated
    assert results[0].is_pegylated is True


def test_validation_diameter_zero():
    with pytest.raises(ValueError):
        predict_surface_interaction("NP", 0.0, -10.0)


def test_validation_diameter_negative():
    with pytest.raises(ValueError):
        predict_surface_interaction("NP", -50.0, -10.0)


def test_validation_k_release_zero():
    with pytest.raises(ValueError):
        predict_surface_interaction("NP", 100.0, -10.0, k_release_per_h=0.0)


def test_validation_baseline_t_half_zero():
    with pytest.raises(ValueError):
        predict_surface_interaction("NP", 100.0, -10.0, baseline_t_half_h=0.0)


def test_mps_factor_formula():
    """mps_clearance_factor = 1 + corona_score / 50."""
    result = _default_result()
    expected = 1.0 + result.corona_score / 50.0
    assert abs(result.mps_clearance_factor - expected) < 1e-9


def test_targeting_efficiency_formula():
    """targeting_efficiency = 1 / mps_clearance_factor."""
    result = _default_result()
    assert abs(result.targeting_efficiency - 1.0 / result.mps_clearance_factor) < 1e-9


def test_compare_custom_modifications():
    mods = [
        {"zeta_mV": 0, "contact_angle_deg": 50, "is_pegylated": False},
        {"zeta_mV": -25, "contact_angle_deg": 30, "is_pegylated": True},
    ]
    results = compare_surface_modifications("custom_NP", 80.0, modifications=mods)
    assert len(results) == 2
    half_lives = [r.circulation_half_life_h for r in results]
    assert half_lives == sorted(half_lives, reverse=True)
