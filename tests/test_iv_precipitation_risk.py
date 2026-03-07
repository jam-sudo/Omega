"""Tests for Phase 520 — IV Precipitation Risk Assessment."""

import pytest

from omega_pbpk.biopharmaceutics.iv_precipitation_risk import (
    IVCompatibilityResult,
    assess_iv_compatibility,
    calculate_solubility_at_ph,
    ph_stability_window,
)

# ---------------------------------------------------------------------------
# calculate_solubility_at_ph
# ---------------------------------------------------------------------------


class TestCalculateSolubilityAtPh:
    def test_weak_acid_higher_solubility_at_high_ph(self):
        low_ph = calculate_solubility_at_ph(4.0, 2.0, 1.0, "weak_acid", 3.0)
        high_ph = calculate_solubility_at_ph(4.0, 2.0, 1.0, "weak_acid", 7.0)
        assert high_ph > low_ph

    def test_weak_base_higher_solubility_at_low_ph(self):
        low_ph = calculate_solubility_at_ph(8.0, 2.0, 1.0, "weak_base", 4.0)
        high_ph = calculate_solubility_at_ph(8.0, 2.0, 1.0, "weak_base", 10.0)
        assert low_ph > high_ph

    def test_neutral_constant_solubility(self):
        s1 = calculate_solubility_at_ph(7.0, 2.0, 5.0, "neutral", 4.0)
        s2 = calculate_solubility_at_ph(7.0, 2.0, 5.0, "neutral", 9.0)
        assert abs(s1 - s2) < 1e-9

    def test_neutral_equals_intrinsic(self):
        s = calculate_solubility_at_ph(7.0, 2.0, 3.5, "neutral", 7.0)
        assert abs(s - 3.5) < 1e-9

    def test_weak_acid_at_pka_double_solubility(self):
        # At pH == pKa: S = s_intrinsic * (1 + 10^0) = 2 * s_intrinsic
        s = calculate_solubility_at_ph(7.0, 2.0, 1.0, "weak_acid", 7.0)
        assert abs(s - 2.0) < 0.001

    def test_weak_base_at_pka_double_solubility(self):
        s = calculate_solubility_at_ph(7.0, 2.0, 1.0, "weak_base", 7.0)
        assert abs(s - 2.0) < 0.001

    def test_invalid_intrinsic_solubility(self):
        with pytest.raises(ValueError, match="s_intrinsic"):
            calculate_solubility_at_ph(7.0, 2.0, -1.0, "weak_acid", 7.0)

    def test_invalid_drug_type(self):
        with pytest.raises(ValueError, match="drug_type"):
            calculate_solubility_at_ph(7.0, 2.0, 1.0, "amphoteric", 7.0)

    def test_zero_intrinsic_solubility_raises(self):
        with pytest.raises(ValueError):
            calculate_solubility_at_ph(7.0, 2.0, 0.0, "neutral", 7.0)


# ---------------------------------------------------------------------------
# assess_iv_compatibility
# ---------------------------------------------------------------------------


class TestAssessIvCompatibility:
    def _make_high_risk(self):
        """Drug A precipitates: conc (5 mg/mL) > solubility at pH 7 (weak acid, pKa=9, s=0.01)."""
        return assess_iv_compatibility(
            drug_a_name="DrugA",
            drug_b_name="DrugB",
            conc_a_mg_mL=5.0,
            conc_b_mg_mL=0.1,
            ph_mix=7.0,
            pka_a=9.0,
            pka_b=4.0,
            logP_a=2.0,
            logP_b=1.0,
            s_intrinsic_a=0.01,  # very low → precipitates
            s_intrinsic_b=10.0,
            drug_type_a="weak_acid",
            drug_type_b="weak_base",
        )

    def _make_low_risk(self):
        """Both drugs well soluble at mix pH."""
        return assess_iv_compatibility(
            drug_a_name="SafeA",
            drug_b_name="SafeB",
            conc_a_mg_mL=1.0,
            conc_b_mg_mL=1.0,
            ph_mix=7.0,
            pka_a=4.0,
            pka_b=10.0,
            logP_a=1.0,
            logP_b=1.0,
            s_intrinsic_a=100.0,
            s_intrinsic_b=100.0,
            drug_type_a="weak_acid",
            drug_type_b="weak_base",
        )

    def test_high_risk_when_conc_exceeds_solubility(self):
        result = self._make_high_risk()
        assert result.risk == "high"

    def test_precipitates_a_flag(self):
        result = self._make_high_risk()
        assert result.precipitates_a is True

    def test_low_risk_both_soluble(self):
        result = self._make_low_risk()
        assert result.risk == "low"

    def test_neither_precipitates_in_low_risk(self):
        result = self._make_low_risk()
        assert result.precipitates_a is False
        assert result.precipitates_b is False

    def test_result_is_frozen_dataclass(self):
        result = self._make_low_risk()
        assert isinstance(result, IVCompatibilityResult)
        with pytest.raises((AttributeError, TypeError)):
            result.risk = "high"  # type: ignore[misc]

    def test_drug_names_preserved(self):
        result = self._make_low_risk()
        assert result.drug_a_name == "SafeA"
        assert result.drug_b_name == "SafeB"

    def test_solubility_values_positive(self):
        result = self._make_low_risk()
        assert result.sol_a_mg_mL > 0
        assert result.sol_b_mg_mL > 0

    def test_recommendation_not_empty(self):
        result = self._make_high_risk()
        assert len(result.recommendation) > 0

    def test_negative_conc_raises(self):
        with pytest.raises(ValueError, match="conc_a"):
            assess_iv_compatibility("A", "B", -1.0, 1.0, 7.0, 4.0, 8.0, 1.0, 1.0, 1.0, 1.0)

    def test_drug_b_precipitates(self):
        # DrugB: conc=5, intrinsic=0.001 (weak_base at low pH = very soluble, but at high pH...)
        # weak_base at pH 7, pKa=4 → sol = 0.001 * (1 + 10^(4-7)) = 0.001 * 1.001 ≈ 0.001
        result = assess_iv_compatibility(
            drug_a_name="A",
            drug_b_name="B",
            conc_a_mg_mL=0.0,
            conc_b_mg_mL=5.0,
            ph_mix=7.0,
            pka_a=4.0,
            pka_b=4.0,
            logP_a=1.0,
            logP_b=1.0,
            s_intrinsic_a=100.0,
            s_intrinsic_b=0.001,
            drug_type_a="weak_acid",
            drug_type_b="weak_base",
        )
        assert result.precipitates_b is True
        assert result.risk == "high"


# ---------------------------------------------------------------------------
# ph_stability_window
# ---------------------------------------------------------------------------


class TestPhStabilityWindow:
    def test_returns_expected_keys(self):
        result = ph_stability_window(4.0, "weak_acid", 1.0, 2.0)
        assert "ph_values" in result
        assert "solubility_mg_mL" in result
        assert "stable_ph_min" in result
        assert "stable_ph_max" in result

    def test_ph_values_length(self):
        result = ph_stability_window(4.0, "weak_acid", 1.0, 2.0, n_points=50)
        assert len(result["ph_values"]) == 50
        assert len(result["solubility_mg_mL"]) == 50

    def test_weak_acid_stable_at_high_ph(self):
        # weak acid: more soluble at high pH
        result = ph_stability_window(4.0, "weak_acid", 1.0, target_conc_mg_mL=5.0)
        assert result["stable_ph_max"] is not None
        assert result["stable_ph_max"] > 5.0  # stable only at higher pH

    def test_weak_base_stable_at_low_ph(self):
        # weak base pKa=8: more soluble at low pH
        result = ph_stability_window(8.0, "weak_base", 1.0, target_conc_mg_mL=5.0)
        assert result["stable_ph_min"] is not None
        assert result["stable_ph_min"] < 7.0

    def test_neutral_constant_across_range(self):
        # neutral drug with high intrinsic solubility: stable everywhere
        result = ph_stability_window(7.0, "neutral", 100.0, target_conc_mg_mL=50.0)
        assert result["stable_ph_min"] is not None
        assert result["stable_ph_max"] is not None

    def test_impossible_conc_returns_none(self):
        # intrinsic=0.001, target=1000 → never stable
        result = ph_stability_window(7.0, "neutral", 0.001, target_conc_mg_mL=1000.0)
        assert result["stable_ph_min"] is None
        assert result["stable_ph_max"] is None

    def test_ph_range_respected(self):
        result = ph_stability_window(4.0, "weak_acid", 1.0, 2.0, ph_range=(5.0, 8.0))
        assert result["ph_values"][0] == pytest.approx(5.0)
        assert result["ph_values"][-1] == pytest.approx(8.0)

    def test_invalid_intrinsic_solubility(self):
        with pytest.raises(ValueError):
            ph_stability_window(4.0, "weak_acid", 0.0, 1.0)

    def test_invalid_target_conc(self):
        with pytest.raises(ValueError):
            ph_stability_window(4.0, "weak_acid", 1.0, -1.0)
