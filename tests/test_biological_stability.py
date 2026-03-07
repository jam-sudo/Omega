"""Tests for biological stability module (Phase 299)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.biological_stability import (
    BloodStabilityResult,
    GIStabilityResult,
    MicrosomalStabilityResult,
    PlasmaStabilityResult,
    blood_stability,
    gi_fluid_stability,
    microsomal_stability,
    plasma_stability,
)

# ---------------------------------------------------------------------------
# plasma_stability
# ---------------------------------------------------------------------------


class TestPlasmaStability:
    def test_returns_plasma_stability_result(self):
        result = plasma_stability("drug_A", t_half_plasma_min=60.0)
        assert isinstance(result, PlasmaStabilityResult)

    def test_provided_t_half_used_directly(self):
        result = plasma_stability("drug_A", t_half_plasma_min=45.0)
        assert result.t_half_plasma_min == pytest.approx(45.0)

    def test_k_deg_from_t_half(self):
        result = plasma_stability("drug_A", t_half_plasma_min=60.0)
        expected_k = math.log(2) / 60.0
        assert result.k_deg_per_min == pytest.approx(expected_k, rel=1e-6)

    def test_pct_remaining_30min(self):
        result = plasma_stability("drug_A", t_half_plasma_min=60.0)
        k = math.log(2) / 60.0
        expected = 100.0 * math.exp(-k * 30.0)
        assert result.pct_remaining_30min == pytest.approx(expected, rel=1e-5)

    def test_pct_remaining_60min(self):
        result = plasma_stability("drug_A", t_half_plasma_min=60.0)
        # At t_half, exactly 50% should remain
        assert result.pct_remaining_60min == pytest.approx(50.0, rel=1e-5)

    def test_pct_remaining_120min(self):
        result = plasma_stability("drug_A", t_half_plasma_min=60.0)
        # After 2 half-lives, 25% remains
        assert result.pct_remaining_120min == pytest.approx(25.0, rel=1e-5)

    def test_stability_class_very_stable(self):
        result = plasma_stability("drug_A", t_half_plasma_min=200.0)
        assert result.stability_class == "very_stable"

    def test_stability_class_stable(self):
        result = plasma_stability("drug_A", t_half_plasma_min=90.0)
        assert result.stability_class == "stable"

    def test_stability_class_moderate(self):
        result = plasma_stability("drug_A", t_half_plasma_min=45.0)
        assert result.stability_class == "moderate"

    def test_stability_class_unstable(self):
        result = plasma_stability("drug_A", t_half_plasma_min=15.0)
        assert result.stability_class == "unstable"

    def test_estimated_t_half_when_none(self):
        result = plasma_stability("drug_B", logP=2.0, fu_plasma=0.1)
        expected_t = 30.0 * math.exp(0.2 * 2.0) * (0.1**0.3)
        assert result.t_half_plasma_min == pytest.approx(expected_t, rel=1e-6)

    def test_fu_plasma_validation(self):
        with pytest.raises(ValueError):
            plasma_stability("drug_A", fu_plasma=0.0)
        with pytest.raises(ValueError):
            plasma_stability("drug_A", fu_plasma=1.5)

    def test_t_half_zero_validation(self):
        with pytest.raises(ValueError):
            plasma_stability("drug_A", t_half_plasma_min=0.0)

    def test_t_half_negative_validation(self):
        with pytest.raises(ValueError):
            plasma_stability("drug_A", t_half_plasma_min=-10.0)

    def test_compound_name_preserved(self):
        result = plasma_stability("test_compound", t_half_plasma_min=30.0)
        assert result.compound_name == "test_compound"

    def test_frozen_dataclass(self):
        result = plasma_stability("drug_A", t_half_plasma_min=60.0)
        with pytest.raises(Exception):
            result.t_half_plasma_min = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# microsomal_stability
# ---------------------------------------------------------------------------


class TestMicrosomalStability:
    def test_returns_microsomal_result(self):
        result = microsomal_stability("drug_B")
        assert isinstance(result, MicrosomalStabilityResult)

    def test_extraction_ratio_between_0_and_1(self):
        result = microsomal_stability("drug_B", clint_uL_per_min_per_mg=5.0)
        assert 0 < result.extraction_ratio < 1

    def test_cl_hepatic_positive(self):
        result = microsomal_stability("drug_B")
        assert result.cl_hepatic_L_per_h > 0

    def test_t_half_microsomes_positive(self):
        result = microsomal_stability("drug_B")
        assert result.t_half_microsomes_min > 0

    def test_high_clint_high_er(self):
        low_er = microsomal_stability("drug_B", clint_uL_per_min_per_mg=1.0).extraction_ratio
        high_er = microsomal_stability("drug_B", clint_uL_per_min_per_mg=1000.0).extraction_ratio
        assert high_er > low_er

    def test_er_approaches_1_with_very_high_clint(self):
        result = microsomal_stability("drug_B", clint_uL_per_min_per_mg=1e6)
        assert result.extraction_ratio > 0.99

    def test_stability_class_stable_low_clint(self):
        result = microsomal_stability("drug_B", clint_uL_per_min_per_mg=5.0)
        assert result.stability_class == "stable"

    def test_stability_class_unstable_high_clint(self):
        result = microsomal_stability("drug_B", clint_uL_per_min_per_mg=50.0)
        assert result.stability_class == "unstable"

    def test_clint_validation(self):
        with pytest.raises(ValueError):
            microsomal_stability("drug_B", clint_uL_per_min_per_mg=0.0)

    def test_fu_mic_validation(self):
        with pytest.raises(ValueError):
            microsomal_stability("drug_B", fu_mic=0.0)

    def test_compound_name_preserved(self):
        result = microsomal_stability("compound_X")
        assert result.compound_name == "compound_X"

    def test_frozen_dataclass(self):
        result = microsomal_stability("drug_B")
        with pytest.raises(Exception):
            result.cl_hepatic_L_per_h = 99.0  # type: ignore[misc]

    def test_fu_mic_reduces_clhep(self):
        result_full = microsomal_stability("drug_B", fu_mic=1.0)
        result_low = microsomal_stability("drug_B", fu_mic=0.1)
        assert result_low.cl_hepatic_L_per_h < result_full.cl_hepatic_L_per_h


# ---------------------------------------------------------------------------
# gi_fluid_stability
# ---------------------------------------------------------------------------


class TestGIFluidStability:
    def test_returns_gi_result(self):
        result = gi_fluid_stability("drug_C")
        assert isinstance(result, GIStabilityResult)

    def test_default_four_ph_values(self):
        result = gi_fluid_stability("drug_C")
        assert len(result.ph_values) == 4
        assert len(result.k_deg_per_h) == 4
        assert len(result.pct_remaining_2h) == 4

    def test_pct_remaining_between_0_and_100(self):
        result = gi_fluid_stability("drug_C")
        for pct in result.pct_remaining_2h:
            assert 0 <= pct <= 100

    def test_acid_labile_compound_stomach_labile(self):
        # pka_acid < 4 → acid-labile
        result = gi_fluid_stability("acid_drug", pka_acid=2.0)
        # stomach (pH 1.5) should have highest k_deg among defaults
        assert result.most_labile_region == "stomach"

    def test_neutral_compound_stability(self):
        # No acid or base lability, small logP → nearly stable
        result = gi_fluid_stability("neutral_drug", logP=0.5)
        for pct in result.pct_remaining_2h:
            assert pct > 99  # very stable

    def test_custom_ph_values(self):
        result = gi_fluid_stability("drug_C", ph_values=[3.0, 7.4])
        assert result.ph_values == [3.0, 7.4]
        assert len(result.k_deg_per_h) == 2

    def test_empty_ph_raises(self):
        with pytest.raises(ValueError):
            gi_fluid_stability("drug_C", ph_values=[])

    def test_k_deg_non_negative(self):
        result = gi_fluid_stability("drug_C")
        for k in result.k_deg_per_h:
            assert k >= 0

    def test_compound_name_preserved(self):
        result = gi_fluid_stability("compound_Y")
        assert result.compound_name == "compound_Y"

    def test_frozen_dataclass(self):
        result = gi_fluid_stability("drug_C")
        with pytest.raises(Exception):
            result.most_labile_region = "xyz"  # type: ignore[misc]

    def test_base_labile_high_ph_more_labile(self):
        # base-labile compound: higher pH → higher k_base
        result = gi_fluid_stability("base_drug", pka_base=9.0)
        # ileum (pH 7.0) should have highest k_deg
        assert result.most_labile_region == "ileum"


# ---------------------------------------------------------------------------
# blood_stability
# ---------------------------------------------------------------------------


class TestBloodStability:
    def test_returns_blood_result(self):
        result = blood_stability("drug_D")
        assert isinstance(result, BloodStabilityResult)

    def test_provided_t_half(self):
        result = blood_stability("drug_D", t_half_blood_min=30.0)
        assert result.t_half_blood_min == pytest.approx(30.0)

    def test_ester_short_t_half(self):
        result = blood_stability("ester_drug", is_ester=True, logP=2.0)
        assert result.t_half_blood_min <= 60.0
        assert result.t_half_blood_min >= 5.0

    def test_non_ester_longer_t_half(self):
        ester_result = blood_stability("drug", is_ester=True, logP=2.0)
        non_ester_result = blood_stability("drug", is_ester=False, logP=2.0)
        assert non_ester_result.t_half_blood_min > ester_result.t_half_blood_min

    def test_pct_remaining_30min_decreasing(self):
        result = blood_stability("drug_D", t_half_blood_min=10.0)
        expected = 100.0 * math.exp(-math.log(2) / 10.0 * 30.0)
        assert result.pct_remaining_30min == pytest.approx(expected, rel=1e-5)

    def test_is_ester_flag_preserved(self):
        result = blood_stability("drug_D", is_ester=True)
        assert result.is_ester is True

    def test_t_half_zero_raises(self):
        with pytest.raises(ValueError):
            blood_stability("drug_D", t_half_blood_min=0.0)

    def test_t_half_negative_raises(self):
        with pytest.raises(ValueError):
            blood_stability("drug_D", t_half_blood_min=-5.0)

    def test_compound_name_preserved(self):
        result = blood_stability("compound_Z")
        assert result.compound_name == "compound_Z"

    def test_frozen_dataclass(self):
        result = blood_stability("drug_D")
        with pytest.raises(Exception):
            result.t_half_blood_min = 99.0  # type: ignore[misc]

    def test_k_deg_positive(self):
        result = blood_stability("drug_D", t_half_blood_min=30.0)
        assert result.k_deg_per_min > 0
