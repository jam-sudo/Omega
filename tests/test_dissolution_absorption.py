"""Tests for Phase 208: dissolution_absorption module."""

import pytest

from omega_pbpk.biopharmaceutics.dissolution_absorption import (
    DissAbsResult,
    compare_formulation_da,
    simulate_dissolution_absorption,
)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


def test_returns_diss_abs_result():
    """Returns correct type with expected fields."""
    result = simulate_dissolution_absorption(
        formulation_name="IR",
        dose_mg=100.0,
        particle_size_um=10.0,
    )
    assert isinstance(result, DissAbsResult)
    assert result.formulation_name == "IR"
    assert result.dose_mg == 100.0
    assert result.particle_size_um == 10.0


def test_output_array_lengths_consistent():
    """All time-series arrays have the same length."""
    result = simulate_dissolution_absorption("IR", 100.0, 10.0, t_end_h=6.0, dt_h=0.1)
    n = len(result.times_h)
    assert len(result.undissolved_mg) == n
    assert len(result.dissolved_mg) == n
    assert len(result.absorbed_mg) == n
    assert len(result.c_plasma_mg_L) == n


def test_mass_conservation_approximately():
    """Undissolved + dissolved + absorbed ≈ dose at all times."""
    result = simulate_dissolution_absorption("IR", 100.0, 10.0, t_end_h=6.0, dt_h=0.05)
    for u, d, a in zip(result.undissolved_mg, result.dissolved_mg, result.absorbed_mg):
        total = u + d + a
        assert abs(total - 100.0) < 1.0, f"Mass not conserved: {total}"


def test_fraction_absorbed_between_0_and_1():
    """fraction_absorbed is in [0, 1]."""
    result = simulate_dissolution_absorption("IR", 100.0, 10.0)
    assert 0.0 <= result.fraction_absorbed <= 1.0


def test_cmax_positive():
    """Cmax is positive for non-zero dose."""
    result = simulate_dissolution_absorption("IR", 100.0, 10.0)
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    """AUC is positive for non-zero dose."""
    result = simulate_dissolution_absorption("IR", 100.0, 10.0)
    assert result.auc_mg_L_h > 0.0


# ---------------------------------------------------------------------------
# Physical behaviour
# ---------------------------------------------------------------------------


def test_small_particle_higher_fraction_absorbed_than_large():
    """Small particle → higher fraction_absorbed than large particle."""
    small = simulate_dissolution_absorption(
        "Small",
        100.0,
        particle_size_um=5.0,
        solubility_mg_mL=1.0,
        t_end_h=12.0,
    )
    large = simulate_dissolution_absorption(
        "Large",
        100.0,
        particle_size_um=500.0,
        solubility_mg_mL=1.0,
        t_end_h=12.0,
    )
    assert small.fraction_absorbed > large.fraction_absorbed


def test_high_solubility_more_absorbed_than_low():
    """High solubility → faster dissolution, more absorbed."""
    high_sol = simulate_dissolution_absorption(
        "HighSol",
        100.0,
        particle_size_um=50.0,
        solubility_mg_mL=10.0,
        t_end_h=12.0,
    )
    low_sol = simulate_dissolution_absorption(
        "LowSol",
        100.0,
        particle_size_um=50.0,
        solubility_mg_mL=0.01,
        t_end_h=12.0,
    )
    assert high_sol.fraction_absorbed > low_sol.fraction_absorbed


def test_large_particle_low_solubility_dissolution_rate_limited():
    """Large particle + low solubility → dissolution_rate_limited=True."""
    result = simulate_dissolution_absorption(
        "Coarse",
        100.0,
        particle_size_um=200.0,
        solubility_mg_mL=0.1,
        t_end_h=12.0,
    )
    assert result.dissolution_rate_limited is True


def test_small_particle_high_solubility_not_dissolution_rate_limited():
    """Small particle + high solubility → dissolution_rate_limited=False."""
    result = simulate_dissolution_absorption(
        "Fine",
        100.0,
        particle_size_um=10.0,
        solubility_mg_mL=5.0,
        t_end_h=12.0,
    )
    assert result.dissolution_rate_limited is False


def test_linear_pk_double_dose_double_cmax():
    """Linear PK: 2x dose → approximately 2x Cmax."""
    r1 = simulate_dissolution_absorption("D100", 100.0, 10.0, t_end_h=12.0)
    r2 = simulate_dissolution_absorption("D200", 200.0, 10.0, t_end_h=12.0)
    ratio = r2.cmax_mg_L / r1.cmax_mg_L
    assert 1.8 <= ratio <= 2.2, f"Cmax ratio {ratio} not near 2x"


# ---------------------------------------------------------------------------
# compare_formulation_da
# ---------------------------------------------------------------------------


def test_compare_formulation_sorted_descending():
    """compare_formulation_da returns results sorted by fraction_absorbed desc."""
    results = compare_formulation_da(
        "Form",
        100.0,
        particle_sizes_um=[500.0, 100.0, 10.0, 1.0],
        solubility_mg_mL=0.5,
        t_end_h=12.0,
    )
    fracs = [r.fraction_absorbed for r in results]
    assert fracs == sorted(fracs, reverse=True)


def test_compare_formulation_count():
    """compare_formulation_da returns one result per particle size."""
    results = compare_formulation_da(
        "Form",
        100.0,
        particle_sizes_um=[10.0, 50.0, 200.0],
    )
    assert len(results) == 3


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    """dose_mg <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_dissolution_absorption("F", dose_mg=0.0, particle_size_um=10.0)


def test_invalid_particle_size_raises():
    """particle_size_um <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="particle_size_um"):
        simulate_dissolution_absorption("F", dose_mg=100.0, particle_size_um=-5.0)


def test_invalid_solubility_raises():
    """solubility_mg_mL <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="solubility"):
        simulate_dissolution_absorption(
            "F", dose_mg=100.0, particle_size_um=10.0, solubility_mg_mL=0.0
        )


def test_invalid_cl_raises():
    """cl_L_per_h <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_dissolution_absorption("F", dose_mg=100.0, particle_size_um=10.0, cl_L_per_h=-1.0)
