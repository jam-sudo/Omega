"""Tests for Phase 341 — Drug Stability in Freeze-Thaw Cycles."""

import pytest

from omega_pbpk.prediction.freeze_thaw_stability import (
    FreezeTHawStabilityResult,
    simulate_freeze_thaw_stability,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _default(**kwargs):
    defaults = dict(
        drug_name="TestMab",
        n_cycles=5,
        initial_concentration_mg_mL=10.0,
        k_aggregation_per_cycle=0.02,
        k_denaturation_per_cycle=0.01,
        excipient_protection_factor=0.0,
        protein_type="mAb",
        freeze_rate_C_per_min=1.0,
        thaw_rate_C_per_min=2.0,
    )
    defaults.update(kwargs)
    return simulate_freeze_thaw_stability(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        result = _default()
        assert isinstance(result, FreezeTHawStabilityResult)

    def test_is_plain_dataclass_not_frozen(self):
        result = _default()
        # Plain dataclass (not frozen) — mutation should work
        result.drug_name = "Modified"
        assert result.drug_name == "Modified"


# ---------------------------------------------------------------------------
# Basic structural invariants
# ---------------------------------------------------------------------------


class TestStructure:
    def test_cycle_numbers_length_matches_n_cycles(self):
        result = _default(n_cycles=7)
        assert len(result.cycle_numbers) == 7

    def test_purity_per_cycle_length_matches_n_cycles(self):
        result = _default(n_cycles=7)
        assert len(result.purity_per_cycle) == 7

    def test_aggregation_per_cycle_length_matches_n_cycles(self):
        result = _default(n_cycles=7)
        assert len(result.aggregation_per_cycle) == 7

    def test_cycle_numbers_start_at_1(self):
        result = _default(n_cycles=3)
        assert result.cycle_numbers[0] == 1

    def test_cycle_numbers_sequential(self):
        result = _default(n_cycles=5)
        assert result.cycle_numbers == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Purity invariants
# ---------------------------------------------------------------------------


class TestPurity:
    def test_final_purity_in_valid_range(self):
        result = _default()
        assert 0.0 <= result.final_purity_pct <= 100.0

    def test_purity_decreases_with_cycles(self):
        result = _default(n_cycles=10, k_aggregation_per_cycle=0.05, k_denaturation_per_cycle=0.03)
        # Each subsequent cycle should have lower or equal purity
        for i in range(1, len(result.purity_per_cycle)):
            assert result.purity_per_cycle[i] <= result.purity_per_cycle[i - 1] + 1e-9

    def test_more_cycles_lower_purity(self):
        r5 = _default(n_cycles=5)
        r10 = _default(n_cycles=10)
        assert r10.final_purity_pct <= r5.final_purity_pct

    def test_higher_excipient_protection_higher_purity(self):
        r_low = _default(excipient_protection_factor=0.0)
        r_high = _default(excipient_protection_factor=0.8)
        assert r_high.final_purity_pct >= r_low.final_purity_pct

    def test_full_excipient_protection_minimal_degradation(self):
        # With full protection, only freeze/thaw stress remains
        result = _default(
            excipient_protection_factor=1.0,
            k_aggregation_per_cycle=0.1,
            k_denaturation_per_cycle=0.1,
        )
        # Should still degrade slightly from physical stresses
        assert result.final_purity_pct > 0.0


# ---------------------------------------------------------------------------
# Stability class
# ---------------------------------------------------------------------------


class TestStabilityClass:
    def test_stability_class_valid_values(self):
        for protein_type in ("mAb", "peptide", "small_molecule"):
            result = _default(protein_type=protein_type)
            assert result.stability_class in {"stable", "borderline", "unstable"}

    def test_stable_when_high_purity(self):
        # Very low degradation → stable
        result = _default(
            n_cycles=1,
            k_aggregation_per_cycle=0.001,
            k_denaturation_per_cycle=0.001,
            excipient_protection_factor=0.9,
        )
        assert result.stability_class == "stable"

    def test_unstable_when_low_purity(self):
        # High degradation rates → unstable
        result = _default(
            n_cycles=20,
            k_aggregation_per_cycle=0.3,
            k_denaturation_per_cycle=0.3,
            excipient_protection_factor=0.0,
        )
        assert result.stability_class == "unstable"


# ---------------------------------------------------------------------------
# Stress effects
# ---------------------------------------------------------------------------


class TestStressEffects:
    def test_slower_freeze_more_degradation(self):
        r_fast = _default(freeze_rate_C_per_min=10.0)
        r_slow = _default(freeze_rate_C_per_min=0.5)
        assert r_slow.final_purity_pct <= r_fast.final_purity_pct

    def test_slower_thaw_more_degradation(self):
        r_fast = _default(thaw_rate_C_per_min=10.0)
        r_slow = _default(thaw_rate_C_per_min=0.5)
        assert r_slow.final_purity_pct <= r_fast.final_purity_pct


# ---------------------------------------------------------------------------
# Recommended max cycles
# ---------------------------------------------------------------------------


class TestRecommendedMaxCycles:
    def test_recommended_max_cycles_at_least_1(self):
        result = _default()
        assert result.recommended_max_cycles >= 1

    def test_recommended_max_cycles_at_most_20(self):
        # Even with zero degradation it caps at 20
        result = _default(
            k_aggregation_per_cycle=0.0,
            k_denaturation_per_cycle=0.0,
            excipient_protection_factor=1.0,
        )
        assert result.recommended_max_cycles <= 20


# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------


class TestConcentration:
    def test_final_concentration_leq_initial(self):
        result = _default()
        assert result.final_concentration_mg_mL <= result.initial_concentration_mg_mL

    def test_final_concentration_proportional_to_purity(self):
        result = _default(initial_concentration_mg_mL=20.0)
        expected = result.initial_concentration_mg_mL * result.final_purity_pct / 100.0
        assert abs(result.final_concentration_mg_mL - expected) < 1e-9


# ---------------------------------------------------------------------------
# Excipient effectiveness
# ---------------------------------------------------------------------------


class TestExcipientEffectiveness:
    def test_excipient_effectiveness_pct_zero(self):
        result = _default(excipient_protection_factor=0.0)
        assert result.excipient_effectiveness_pct == 0.0

    def test_excipient_effectiveness_pct_full(self):
        result = _default(excipient_protection_factor=1.0)
        assert result.excipient_effectiveness_pct == 100.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_n_cycles_too_low(self):
        with pytest.raises(ValueError, match="n_cycles"):
            _default(n_cycles=0)

    def test_n_cycles_too_high(self):
        with pytest.raises(ValueError, match="n_cycles"):
            _default(n_cycles=21)

    def test_k_aggregation_negative(self):
        with pytest.raises(ValueError, match="k_aggregation_per_cycle"):
            _default(k_aggregation_per_cycle=-0.01)

    def test_k_aggregation_too_high(self):
        with pytest.raises(ValueError, match="k_aggregation_per_cycle"):
            _default(k_aggregation_per_cycle=1.5)

    def test_excipient_negative(self):
        with pytest.raises(ValueError, match="excipient_protection_factor"):
            _default(excipient_protection_factor=-0.1)

    def test_excipient_too_high(self):
        with pytest.raises(ValueError, match="excipient_protection_factor"):
            _default(excipient_protection_factor=1.1)

    def test_invalid_protein_type(self):
        with pytest.raises(ValueError, match="protein_type"):
            _default(protein_type="unknown_type")
