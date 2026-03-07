"""Tests for Phase 1093 — Pulmonary Drug Absorption (Inhaled Therapeutics)."""

import pytest

from omega_pbpk.core.pulmonary_absorption_pk import (
    PulmonaryAbsorptionResult,
    optimize_particle_size,
    simulate_pulmonary_absorption,
)


# ---------------------------------------------------------------------------
# Basic return-type and structure tests
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_pulmonary_absorption("salbutamol", dose_mg=2.5)
    assert isinstance(result, PulmonaryAbsorptionResult)


def test_drug_name_preserved():
    result = simulate_pulmonary_absorption("budesonide", dose_mg=1.0)
    assert result.drug_name == "budesonide"


def test_dose_preserved():
    result = simulate_pulmonary_absorption("fluticasone", dose_mg=0.5)
    assert result.dose_mg == pytest.approx(0.5)


def test_particle_size_preserved():
    result = simulate_pulmonary_absorption("salmeterol", dose_mg=0.05, particle_size_um=4.0)
    assert result.particle_size_um == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Initial condition tests
# ---------------------------------------------------------------------------


def test_c_plasma_starts_at_zero():
    result = simulate_pulmonary_absorption("drug_x", dose_mg=5.0)
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0)


def test_a_alveolar_starts_positive_small_particle():
    """Small particles (≤3 µm) have alveolar fraction = 0.6."""
    result = simulate_pulmonary_absorption("drug_x", dose_mg=5.0, particle_size_um=2.0)
    assert result.a_alveolar_mg[0] == pytest.approx(5.0 * 0.6)


def test_a_tb_starts_positive_small_particle():
    result = simulate_pulmonary_absorption("drug_x", dose_mg=5.0, particle_size_um=2.0)
    assert result.a_tb_mg[0] == pytest.approx(5.0 * 0.1)


def test_large_particle_no_lung_deposition():
    """Particles >10 µm go entirely to GI; lung amounts start at 0."""
    result = simulate_pulmonary_absorption("drug_x", dose_mg=5.0, particle_size_um=12.0)
    assert result.a_alveolar_mg[0] == pytest.approx(0.0)
    assert result.a_tb_mg[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Monotonicity tests
# ---------------------------------------------------------------------------


def test_a_alveolar_decreases_monotonically():
    """Alveolar amount should decrease monotonically (only absorption, no input)."""
    result = simulate_pulmonary_absorption("drug_x", dose_mg=5.0, particle_size_um=2.0)
    for i in range(1, len(result.a_alveolar_mg)):
        assert result.a_alveolar_mg[i] <= result.a_alveolar_mg[i - 1] + 1e-9


# ---------------------------------------------------------------------------
# Bioavailability / fraction tests
# ---------------------------------------------------------------------------


def test_f_total_absorbed_in_unit_interval():
    result = simulate_pulmonary_absorption("drug_x", dose_mg=5.0)
    assert 0.0 <= result.f_total_absorbed <= 1.0


def test_lung_selectivity_ratio_in_unit_interval():
    result = simulate_pulmonary_absorption("drug_x", dose_mg=5.0)
    assert 0.0 <= result.lung_selectivity_ratio <= 1.0


def test_small_particle_higher_lung_selectivity_than_large():
    """2 µm particles should have higher lung selectivity than 10 µm (which go to GI)."""
    r_small = simulate_pulmonary_absorption("drug_x", dose_mg=5.0, particle_size_um=2.0)
    r_large = simulate_pulmonary_absorption("drug_x", dose_mg=5.0, particle_size_um=10.0)
    assert r_small.lung_selectivity_ratio > r_large.lung_selectivity_ratio


def test_f_gi_absorbed_large_particle():
    """Particles >10 µm → only GI absorption → lung fraction = 0."""
    result = simulate_pulmonary_absorption("drug_x", dose_mg=5.0, particle_size_um=12.0)
    assert result.f_lung_absorbed == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# PK metrics
# ---------------------------------------------------------------------------


def test_auc_positive():
    result = simulate_pulmonary_absorption("drug_x", dose_mg=5.0)
    assert result.auc > 0.0


def test_cmax_positive():
    result = simulate_pulmonary_absorption("drug_x", dose_mg=5.0)
    assert result.cmax > 0.0


def test_cmax_equals_max_c_plasma():
    result = simulate_pulmonary_absorption("drug_x", dose_mg=5.0)
    assert result.cmax == pytest.approx(max(result.c_plasma_mg_L))


def test_tmax_in_valid_range():
    result = simulate_pulmonary_absorption("drug_x", dose_mg=5.0, t_end_h=12.0)
    assert 0.0 <= result.tmax_h <= 12.0


def test_notes_is_string():
    result = simulate_pulmonary_absorption("drug_x", dose_mg=5.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# optimize_particle_size tests
# ---------------------------------------------------------------------------


def test_optimize_particle_size_returns_result():
    result = optimize_particle_size("salbutamol", dose_mg=2.5)
    assert isinstance(result, PulmonaryAbsorptionResult)


def test_optimize_particle_size_lung_selectivity_is_highest():
    """Optimized result should have lung_selectivity_ratio >= all candidate sizes."""
    best = optimize_particle_size("drug_x", dose_mg=5.0)
    for size in [2.0, 4.0, 7.0, 12.0]:
        r = simulate_pulmonary_absorption("drug_x", dose_mg=5.0, particle_size_um=size)
        assert best.lung_selectivity_ratio >= r.lung_selectivity_ratio - 1e-9


def test_optimize_particle_size_uses_smallest_size():
    """Smallest particles (2 µm) maximise lung selectivity — best should be ≤5 µm."""
    best = optimize_particle_size("drug_x", dose_mg=5.0)
    assert best.particle_size_um <= 5.0


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_pulmonary_absorption("drug_x", dose_mg=0.0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_pulmonary_absorption("drug_x", dose_mg=-1.0)


def test_invalid_particle_size():
    with pytest.raises(ValueError, match="particle_size_um"):
        simulate_pulmonary_absorption("drug_x", dose_mg=5.0, particle_size_um=-1.0)


def test_invalid_cl():
    with pytest.raises(ValueError, match="cl"):
        simulate_pulmonary_absorption("drug_x", dose_mg=5.0, cl_L_per_h=0.0)


def test_invalid_vd():
    with pytest.raises(ValueError, match="vd"):
        simulate_pulmonary_absorption("drug_x", dose_mg=5.0, vd_L=0.0)
