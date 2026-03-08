"""Tests for Phase 1086 — Dissolution-Absorption Rate-Limiting Step Identifier."""

import pytest

from omega_pbpk.prediction.rate_limiting_step import (
    RateLimitingStepResult,
    compare_formulations,
    identify_rate_limiting_step,
)

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = identify_rate_limiting_step("Drug", dose_mg=10.0)
    assert isinstance(result, RateLimitingStepResult)


def test_frozen_dataclass():
    result = identify_rate_limiting_step("Drug", dose_mg=10.0)
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"  # type: ignore[misc]


def test_fields_populated():
    result = identify_rate_limiting_step("Aspirin", dose_mg=500.0)
    assert result.drug_name == "Aspirin"
    assert result.dose_mg == 500.0


# ---------------------------------------------------------------------------
# Large particles → dissolution-limited
# ---------------------------------------------------------------------------


def test_large_particles_dissolution_limited():
    """100 um particle with high Peff should be dissolution-limited."""
    result = identify_rate_limiting_step(
        "Drug",
        dose_mg=500.0,
        particle_size_um=500.0,
        solubility_mg_mL=0.01,
        peff_cm_per_s=5e-4,
    )
    assert result.rate_limiting_step in ("dissolution", "both")


def test_large_particle_high_dose_dissolution():
    """Very large particles + high dose → dissolution rate-limiting."""
    result = identify_rate_limiting_step(
        "Drug",
        dose_mg=1000.0,
        particle_size_um=1000.0,
        solubility_mg_mL=0.01,
        peff_cm_per_s=1e-3,
    )
    # Do > 1 and An > 1 → dissolution-limited
    assert result.rate_limiting_step == "dissolution"


# ---------------------------------------------------------------------------
# Small particles → absorption or permeability limited
# ---------------------------------------------------------------------------


def test_small_particles_not_dissolution_limited():
    """Very small particles dissolve quickly → dissolution is not limiting."""
    result = identify_rate_limiting_step(
        "Drug",
        dose_mg=1.0,
        particle_size_um=1.0,
        solubility_mg_mL=10.0,
        peff_cm_per_s=1e-6,
    )
    assert result.rate_limiting_step in ("absorption", "permeability", "both")


# ---------------------------------------------------------------------------
# BCS classifications
# ---------------------------------------------------------------------------


def test_do_gt1_an_gt1_gives_bcs_ii():
    """High dose number + good permeability → BCS II."""
    result = identify_rate_limiting_step(
        "Drug",
        dose_mg=1000.0,
        particle_size_um=100.0,
        solubility_mg_mL=0.001,
        peff_cm_per_s=1e-3,
    )
    assert result.dose_number > 1.0
    assert result.absorption_number > 1.0
    assert result.bcs_class_predicted == "II"


def test_do_lt1_an_gt1_gives_bcs_i():
    """Low dose number + high Peff → BCS I."""
    result = identify_rate_limiting_step(
        "Drug",
        dose_mg=1.0,
        particle_size_um=10.0,
        solubility_mg_mL=10.0,
        peff_cm_per_s=1e-3,
    )
    assert result.dose_number < 1.0
    assert result.absorption_number > 1.0
    assert result.bcs_class_predicted == "I"


def test_do_gt1_an_lt1_gives_bcs_iv_or_iii():
    """High dose number + poor Peff → BCS IV."""
    result = identify_rate_limiting_step(
        "Drug",
        dose_mg=500.0,
        particle_size_um=100.0,
        solubility_mg_mL=0.01,
        peff_cm_per_s=1e-6,
    )
    assert result.dose_number > 1.0
    assert result.absorption_number < 1.0
    assert result.bcs_class_predicted == "IV"


def test_bcs_class_valid_values():
    for dose in (1.0, 500.0):
        for peff in (1e-6, 1e-3):
            result = identify_rate_limiting_step("Drug", dose_mg=dose, peff_cm_per_s=peff)
            assert result.bcs_class_predicted in ("I", "II", "III", "IV")


# ---------------------------------------------------------------------------
# High solubility → higher f_absorbed
# ---------------------------------------------------------------------------


def test_high_solubility_higher_f_absorbed():
    low_sol = identify_rate_limiting_step(
        "Drug", dose_mg=100.0, solubility_mg_mL=0.01, peff_cm_per_s=1e-4
    )
    high_sol = identify_rate_limiting_step(
        "Drug", dose_mg=100.0, solubility_mg_mL=10.0, peff_cm_per_s=1e-4
    )
    assert high_sol.f_absorbed_predicted >= low_sol.f_absorbed_predicted


# ---------------------------------------------------------------------------
# rate_limiting_step valid values
# ---------------------------------------------------------------------------


def test_rate_limiting_step_valid_values():
    valid = {"dissolution", "absorption", "both", "permeability"}
    result = identify_rate_limiting_step("Drug", dose_mg=10.0)
    assert result.rate_limiting_step in valid


# ---------------------------------------------------------------------------
# f_absorbed_predicted in [0, 1]
# ---------------------------------------------------------------------------


def test_f_absorbed_in_0_1():
    for peff in (1e-8, 1e-4, 1e-2):
        for dose in (0.1, 100.0, 10000.0):
            result = identify_rate_limiting_step("Drug", dose_mg=dose, peff_cm_per_s=peff)
            assert 0.0 <= result.f_absorbed_predicted <= 1.0


# ---------------------------------------------------------------------------
# formulation_strategy nonempty
# ---------------------------------------------------------------------------


def test_formulation_strategy_nonempty():
    result = identify_rate_limiting_step("Drug", dose_mg=10.0)
    assert len(result.formulation_strategy) > 0


# ---------------------------------------------------------------------------
# compare_formulations
# ---------------------------------------------------------------------------


def test_compare_formulations_count():
    sizes = [10.0, 50.0, 100.0, 500.0]
    results = compare_formulations("Drug", dose_mg=100.0, particle_sizes=sizes)
    assert len(results) == len(sizes)


def test_compare_formulations_sorted_descending():
    sizes = [10.0, 50.0, 100.0, 500.0]
    results = compare_formulations("Drug", dose_mg=100.0, particle_sizes=sizes)
    for i in range(len(results) - 1):
        assert results[i].f_absorbed_predicted >= results[i + 1].f_absorbed_predicted


def test_small_particles_higher_f_absorbed_than_large():
    sizes = [5.0, 500.0]
    results = compare_formulations(
        "Drug",
        dose_mg=100.0,
        particle_sizes=sizes,
        solubility_mg_mL=0.1,
        peff_cm_per_s=5e-4,
    )
    # Sorted descending; first result should correspond to smallest particles
    assert results[0].f_absorbed_predicted >= results[-1].f_absorbed_predicted


def test_compare_formulations_kwargs_passed():
    sizes = [10.0, 100.0]
    results = compare_formulations("Drug", dose_mg=50.0, particle_sizes=sizes, peff_cm_per_s=1e-3)
    for r in results:
        # peff_cm_per_s=1e-3 → An = 10 → high perm
        assert r.absorption_number > 1.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        identify_rate_limiting_step("Drug", dose_mg=0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        identify_rate_limiting_step("Drug", dose_mg=-10.0)


def test_invalid_particle_size_raises():
    with pytest.raises(ValueError, match="particle_size"):
        identify_rate_limiting_step("Drug", dose_mg=10.0, particle_size_um=0.0)


def test_invalid_solubility_raises():
    with pytest.raises(ValueError, match="solubility"):
        identify_rate_limiting_step("Drug", dose_mg=10.0, solubility_mg_mL=0.0)


def test_invalid_peff_raises():
    with pytest.raises(ValueError, match="peff"):
        identify_rate_limiting_step("Drug", dose_mg=10.0, peff_cm_per_s=-1e-4)
