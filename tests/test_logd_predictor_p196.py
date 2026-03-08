"""Tests for Phase 196 — LogD Predictor (logd_predictor_p196)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.logd_predictor_p196 import (
    LogDResult,
    predict_logd,
    screen_logd_profile,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_acid(**kwargs) -> LogDResult:
    params = dict(
        drug_name="TestAcid",
        logP=2.0,
        pka=7.0,
        ionization_type="acid",
    )
    params.update(kwargs)
    return predict_logd(**params)


def _default_base(**kwargs) -> LogDResult:
    params = dict(
        drug_name="TestBase",
        logP=2.0,
        pka=7.0,
        ionization_type="base",
    )
    params.update(kwargs)
    return predict_logd(**params)


def _default_neutral(**kwargs) -> LogDResult:
    params = dict(
        drug_name="TestNeutral",
        logP=2.0,
        pka=7.0,
        ionization_type="neutral",
    )
    params.update(kwargs)
    return predict_logd(**params)


# ---------------------------------------------------------------------------
# Return type and basic structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_logd_result(self):
        result = _default_acid()
        assert isinstance(result, LogDResult)

    def test_logd_result_is_dataclass(self):
        import dataclasses

        assert dataclasses.is_dataclass(LogDResult)

    def test_drug_name_preserved(self):
        result = predict_logd("Aspirin", 1.19, 3.5, "acid")
        assert result.drug_name == "Aspirin"

    def test_logp_preserved(self):
        result = _default_acid(logP=3.0)
        assert result.logP == pytest.approx(3.0)

    def test_pka_preserved(self):
        result = _default_acid(pka=6.5)
        assert result.pka == pytest.approx(6.5)

    def test_ionization_type_preserved(self):
        result = _default_acid()
        assert result.ionization_type == "acid"

    def test_ph_values_is_list(self):
        result = _default_acid()
        assert isinstance(result.ph_values, list)

    def test_logd_values_is_list(self):
        result = _default_acid()
        assert isinstance(result.logd_values, list)

    def test_ph_and_logd_same_length(self):
        result = _default_acid()
        assert len(result.ph_values) == len(result.logd_values)

    def test_custom_ph_values(self):
        ph_vals = [5.0, 6.5, 7.4, 8.0]
        result = predict_logd("Drug", 2.0, 7.0, "acid", ph_values=ph_vals)
        assert result.ph_values == ph_vals
        assert len(result.logd_values) == 4


# ---------------------------------------------------------------------------
# Acid behavior: logD decreases with increasing pH
# ---------------------------------------------------------------------------


class TestAcidBehavior:
    def test_logd_decreases_with_ph_for_acid(self):
        """Acids are deprotonated (more ionized) at higher pH → lower logD."""
        ph_vals = [4.0, 7.0, 10.0]
        result = predict_logd("Acid", 3.0, 7.0, "acid", ph_values=ph_vals)
        assert result.logd_values[0] > result.logd_values[1] > result.logd_values[2]

    def test_acid_logd_below_logp(self):
        """logD ≤ logP for ionizable acids (ionization reduces effective logP)."""
        result = _default_acid(logP=2.5, pka=5.0)
        assert result.logd_at_ph74 <= result.logP + 1e-9

    def test_acid_logd_at_low_ph_close_to_logp(self):
        """At very low pH (acid fully protonated), logD ≈ logP."""
        # At pH 1.2, pKa=7: logD = logP - log10(1 + 10^(1.2-7)) = logP - ~0
        result = predict_logd("Acid", 2.0, 7.0, "acid")
        # logD(1.2) should be very close to logP
        assert abs(result.logd_at_ph12 - result.logP) < 0.01


# ---------------------------------------------------------------------------
# Base behavior: logD increases with increasing pH
# ---------------------------------------------------------------------------


class TestBaseBehavior:
    def test_logd_increases_with_ph_for_base(self):
        """Bases are protonated (more ionized) at lower pH → lower logD at low pH."""
        ph_vals = [4.0, 7.0, 10.0]
        result = predict_logd("Base", 3.0, 7.0, "base", ph_values=ph_vals)
        assert result.logd_values[0] < result.logd_values[1] < result.logd_values[2]

    def test_base_logd_below_logp(self):
        """logD ≤ logP for ionizable bases."""
        result = _default_base(logP=3.0, pka=8.0)
        assert result.logd_at_ph74 <= result.logP + 1e-9

    def test_base_logd_at_high_ph_close_to_logp(self):
        """At very high pH (base fully deprotonated), logD ≈ logP."""
        result = predict_logd("Base", 2.0, 4.0, "base")
        # logD(7.4) should be close to logP when pKa << pH
        assert result.logd_at_ph74 > result.logP - 0.1


# ---------------------------------------------------------------------------
# Neutral behavior: logD constant
# ---------------------------------------------------------------------------


class TestNeutralBehavior:
    def test_neutral_logd_constant(self):
        """Neutral compounds: logD = logP at all pH values."""
        ph_vals = [1.0, 4.0, 7.4, 10.0, 13.0]
        result = predict_logd("Neutral", 2.5, 7.0, "neutral", ph_values=ph_vals)
        for ld in result.logd_values:
            assert ld == pytest.approx(result.logP)

    def test_neutral_logd_at_74_equals_logp(self):
        result = _default_neutral(logP=1.8)
        assert result.logd_at_ph74 == pytest.approx(1.8)

    def test_neutral_logd_at_68_equals_logp(self):
        result = _default_neutral(logP=2.2)
        assert result.logd_at_ph68 == pytest.approx(2.2)


# ---------------------------------------------------------------------------
# Special pH values
# ---------------------------------------------------------------------------


class TestSpecialPH:
    def test_logd_at_ph74_is_float(self):
        result = _default_acid()
        assert isinstance(result.logd_at_ph74, float)

    def test_logd_at_ph68_is_float(self):
        result = _default_acid()
        assert isinstance(result.logd_at_ph68, float)

    def test_logd_at_ph12_is_float(self):
        result = _default_acid()
        assert isinstance(result.logd_at_ph12, float)

    def test_logd_le_logp(self):
        """logD must be ≤ logP for ionizable compounds (ionization can only reduce)."""
        for ion_type in ("acid", "base"):
            result = predict_logd("Drug", 3.0, 7.0, ion_type)
            assert result.logd_at_ph74 <= result.logP + 1e-9, (
                f"logD({ion_type}) at pH 7.4 should be ≤ logP"
            )


# ---------------------------------------------------------------------------
# Lipophilicity classification
# ---------------------------------------------------------------------------


class TestLipophilicityClass:
    def test_hydrophilic_class(self):
        # logD at 7.4 < 0 → hydrophilic
        # For acid with logP=0.0, pKa=5: logD(7.4) = 0 - log10(1+10^2.4) ≈ -2.4
        result = predict_logd("HydroAcid", 0.0, 5.0, "acid")
        assert result.lipophilicity_class_at_74 == "hydrophilic"

    def test_optimal_class(self):
        # logD at 7.4 in [0, 3] → optimal
        # Neutral with logP=1.5 → logD = 1.5
        result = predict_logd("OptDrug", 1.5, 7.0, "neutral")
        assert result.lipophilicity_class_at_74 == "optimal"

    def test_lipophilic_class(self):
        # logD > 3 → lipophilic
        result = predict_logd("LipoDrug", 4.0, 7.0, "neutral")
        assert result.lipophilicity_class_at_74 == "lipophilic"

    def test_valid_class_values(self):
        valid = {"hydrophilic", "optimal", "lipophilic"}
        for ion in ("acid", "base", "neutral"):
            result = predict_logd("Drug", 2.0, 7.0, ion)
            assert result.lipophilicity_class_at_74 in valid


# ---------------------------------------------------------------------------
# Optimal absorption range
# ---------------------------------------------------------------------------


class TestOptimalAbsorptionRange:
    def test_optimal_range_type_none_or_tuple(self):
        result = _default_acid()
        assert result.optimal_absorption_ph_range is None or isinstance(
            result.optimal_absorption_ph_range, tuple
        )

    def test_optimal_range_present_for_drug_with_good_logd(self):
        # Neutral drug with logP=2.0 → logD=2.0 at all pH → range should exist
        result = predict_logd("NeutralGood", 2.0, 7.0, "neutral")
        assert result.optimal_absorption_ph_range is not None


# ---------------------------------------------------------------------------
# screen_logd_profile
# ---------------------------------------------------------------------------


class TestScreenLogdProfile:
    def _make_drugs(self):
        return [
            {"drug_name": "DrugA", "logP": 2.0, "pka": 7.0, "ionization_type": "neutral"},
            {"drug_name": "DrugB", "logP": 4.5, "pka": 7.0, "ionization_type": "neutral"},
            {"drug_name": "DrugC", "logP": 0.5, "pka": 7.0, "ionization_type": "neutral"},
        ]

    def test_sorted_by_logd_74_descending(self):
        drugs = self._make_drugs()
        results = screen_logd_profile(drugs)
        logd_vals = [r.logd_at_ph74 for r in results]
        assert logd_vals == sorted(logd_vals, reverse=True)

    def test_returns_correct_count(self):
        drugs = self._make_drugs()
        results = screen_logd_profile(drugs)
        assert len(results) == len(drugs)

    def test_all_results_are_logd_result(self):
        drugs = self._make_drugs()
        results = screen_logd_profile(drugs)
        for r in results:
            assert isinstance(r, LogDResult)

    def test_empty_list_returns_empty(self):
        results = screen_logd_profile([])
        assert results == []

    def test_single_drug(self):
        drugs = [{"drug_name": "Only", "logP": 2.0, "pka": 7.0, "ionization_type": "acid"}]
        results = screen_logd_profile(drugs)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_ionization_type_raises(self):
        with pytest.raises(ValueError, match="ionization_type"):
            predict_logd("Drug", 2.0, 7.0, "zwitterion")

    def test_logp_too_low_raises(self):
        with pytest.raises(ValueError, match="logP"):
            predict_logd("Drug", -6.0, 7.0, "acid")

    def test_logp_too_high_raises(self):
        with pytest.raises(ValueError, match="logP"):
            predict_logd("Drug", 11.0, 7.0, "acid")

    def test_pka_negative_raises(self):
        with pytest.raises(ValueError, match="pKa"):
            predict_logd("Drug", 2.0, -1.0, "acid")

    def test_pka_above_14_raises(self):
        with pytest.raises(ValueError, match="pKa"):
            predict_logd("Drug", 2.0, 15.0, "acid")

    def test_valid_boundary_logp(self):
        # Should not raise
        result = predict_logd("Drug", -5.0, 7.0, "neutral")
        assert result is not None
        result2 = predict_logd("Drug", 10.0, 7.0, "neutral")
        assert result2 is not None

    def test_valid_boundary_pka(self):
        # Should not raise
        result = predict_logd("Drug", 2.0, 0.0, "acid")
        assert result is not None
        result2 = predict_logd("Drug", 2.0, 14.0, "base")
        assert result2 is not None
