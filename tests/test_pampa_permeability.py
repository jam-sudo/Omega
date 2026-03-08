"""Tests for Phase 541: PAMPA permeability prediction."""

import pytest

from omega_pbpk.prediction.pampa_permeability import (
    PAMPAPermeabilityResult,
    predict_pampa_permeability,
    screen_pampa,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_neutral(logP=2.0, mw_Da=300.0, pKa=7.0):
    return predict_pampa_permeability("Drug", logP, mw_Da, pKa, "neutral")


# ---------------------------------------------------------------------------
# Return type and field presence
# ---------------------------------------------------------------------------
class TestReturnType:
    def test_returns_pampa_result(self):
        result = _make_neutral()
        assert isinstance(result, PAMPAPermeabilityResult)

    def test_frozen_dataclass(self):
        result = _make_neutral()
        with pytest.raises((AttributeError, TypeError)):
            result.logP = 99.0  # type: ignore[misc]

    def test_all_fields_present(self):
        result = _make_neutral()
        for attr in [
            "drug_name",
            "logP",
            "mw_Da",
            "pKa",
            "drug_type",
            "peff_cm_per_s",
            "peff_apparent_cm_per_s",
            "absorption_number",
            "fa_predicted",
            "bcs_permeability_class",
            "fa_classification",
            "notes",
        ]:
            assert hasattr(result, attr), f"Missing field: {attr}"

    def test_drug_name_stored(self):
        result = predict_pampa_permeability("Aspirin", 1.0, 180.0, 3.5, "acid")
        assert result.drug_name == "Aspirin"

    def test_drug_type_stored(self):
        result = predict_pampa_permeability("TestBase", 2.0, 300.0, 8.0, "base")
        assert result.drug_type == "base"


# ---------------------------------------------------------------------------
# Peff calculation
# ---------------------------------------------------------------------------
class TestPeffCalculation:
    def test_higher_logP_higher_peff(self):
        low = predict_pampa_permeability("A", 0.0, 300.0, 7.0, "neutral")
        high = predict_pampa_permeability("B", 4.0, 300.0, 7.0, "neutral")
        assert high.peff_cm_per_s > low.peff_cm_per_s

    def test_higher_mw_lower_peff(self):
        light = predict_pampa_permeability("A", 2.0, 200.0, 7.0, "neutral")
        heavy = predict_pampa_permeability("B", 2.0, 600.0, 7.0, "neutral")
        assert light.peff_cm_per_s > heavy.peff_cm_per_s

    def test_peff_clamp_lower(self):
        # Very lipophobic, very large MW → should clamp to 1e-8
        result = predict_pampa_permeability("X", -5.0, 1000.0, 7.0, "neutral")
        assert result.peff_cm_per_s >= 1e-8

    def test_peff_clamp_upper(self):
        # Very lipophilic, tiny MW → should clamp to 1e-4
        result = predict_pampa_permeability("X", 10.0, 50.0, 7.0, "neutral")
        assert result.peff_cm_per_s <= 1e-4

    def test_peff_positive(self):
        result = _make_neutral()
        assert result.peff_cm_per_s > 0.0


# ---------------------------------------------------------------------------
# pH effect on apparent Peff
# ---------------------------------------------------------------------------
class TestPHEffect:
    def test_acid_at_low_ph_higher_apparent_peff(self):
        """Acid at low pH: less ionized → higher apparent Peff."""
        acid_low_ph = predict_pampa_permeability("Acid", 2.0, 200.0, 4.0, "acid", ph_intestine=3.0)
        acid_high_ph = predict_pampa_permeability("Acid", 2.0, 200.0, 4.0, "acid", ph_intestine=7.0)
        assert acid_low_ph.peff_apparent_cm_per_s > acid_high_ph.peff_apparent_cm_per_s

    def test_acid_apparent_peff_less_than_or_equal_base_peff(self):
        """Acid at intestinal pH 6.5 with pKa=4 should be largely ionized → lower apparent Peff."""
        acid = predict_pampa_permeability("Acid", 3.0, 250.0, 4.0, "acid")
        base = predict_pampa_permeability("Base", 3.0, 250.0, 4.0, "base")
        assert acid.peff_apparent_cm_per_s < base.peff_apparent_cm_per_s

    def test_neutral_apparent_equals_peff(self):
        result = predict_pampa_permeability("Neutral", 2.0, 300.0, 7.0, "neutral")
        assert result.peff_apparent_cm_per_s == result.peff_cm_per_s

    def test_base_apparent_equals_peff(self):
        result = predict_pampa_permeability("Base", 2.0, 300.0, 8.0, "base")
        assert result.peff_apparent_cm_per_s == result.peff_cm_per_s

    def test_acid_fully_ionized_apparent_peff_near_min(self):
        """Acid with pKa=2 at pH 6.5 → almost fully ionized."""
        result = predict_pampa_permeability("Acid", 2.0, 300.0, 2.0, "acid", ph_intestine=6.5)
        # Nearly all ionized, apparent Peff should be near minimum or << base peff
        base = predict_pampa_permeability("Base", 2.0, 300.0, 2.0, "base", ph_intestine=6.5)
        assert result.peff_apparent_cm_per_s < base.peff_apparent_cm_per_s


# ---------------------------------------------------------------------------
# Absorption number and fa
# ---------------------------------------------------------------------------
class TestAbsorptionMetrics:
    def test_an_positive(self):
        result = _make_neutral(logP=3.0)
        assert result.absorption_number > 0.0

    def test_fa_between_0_and_1(self):
        result = _make_neutral()
        assert 0.0 <= result.fa_predicted <= 1.0

    def test_fa_increases_with_peff(self):
        low_perm = predict_pampa_permeability("A", 0.0, 400.0, 7.0, "neutral")
        high_perm = predict_pampa_permeability("B", 4.0, 200.0, 7.0, "neutral")
        assert high_perm.fa_predicted >= low_perm.fa_predicted

    def test_fa_from_an_monotonic(self):
        """Higher An → higher fa."""
        r1 = predict_pampa_permeability("A", 1.0, 300.0, 7.0, "neutral")
        r2 = predict_pampa_permeability("B", 3.0, 300.0, 7.0, "neutral")
        assert r2.absorption_number >= r1.absorption_number
        assert r2.fa_predicted >= r1.fa_predicted


# ---------------------------------------------------------------------------
# BCS classification
# ---------------------------------------------------------------------------
class TestBCSClassification:
    def test_bcs_high_permeability(self):
        """logP=4, MW=200 → Peff should be well above 1e-6 → high."""
        result = predict_pampa_permeability("HighPerm", 4.0, 200.0, 7.0, "neutral")
        assert result.bcs_permeability_class == "high"

    def test_bcs_low_permeability(self):
        """logP=-2, MW=500 → very low Peff → low."""
        result = predict_pampa_permeability("LowPerm", -2.0, 500.0, 7.0, "neutral")
        assert result.bcs_permeability_class == "low"

    def test_bcs_class_is_string(self):
        result = _make_neutral()
        assert isinstance(result.bcs_permeability_class, str)
        assert result.bcs_permeability_class in {"high", "low"}


# ---------------------------------------------------------------------------
# fa classification
# ---------------------------------------------------------------------------
class TestFaClassification:
    def test_complete_absorption_high_perm(self):
        result = predict_pampa_permeability("HighPerm", 5.0, 150.0, 7.0, "neutral")
        assert result.fa_classification == "complete"

    def test_low_absorption_low_perm(self):
        result = predict_pampa_permeability("LowPerm", -3.0, 600.0, 7.0, "neutral")
        assert result.fa_classification == "low"

    def test_fa_class_is_valid_string(self):
        result = _make_neutral()
        assert result.fa_classification in {"complete", "moderate", "low"}

    def test_fa_class_boundary_complete(self):
        """fa > 0.9 → complete."""
        result = predict_pampa_permeability("X", 5.0, 150.0, 7.0, "neutral")
        if result.fa_predicted > 0.9:
            assert result.fa_classification == "complete"

    def test_fa_class_boundary_low(self):
        """fa < 0.5 → low."""
        result = predict_pampa_permeability("X", -3.0, 700.0, 7.0, "neutral")
        if result.fa_predicted < 0.5:
            assert result.fa_classification == "low"


# ---------------------------------------------------------------------------
# Screen function
# ---------------------------------------------------------------------------
class TestScreenPampa:
    def _compounds(self):
        return [
            {
                "drug_name": "LowPerm",
                "logP": -1.0,
                "mw_Da": 500.0,
                "pKa": 7.0,
                "drug_type": "neutral",
            },
            {
                "drug_name": "HighPerm",
                "logP": 4.0,
                "mw_Da": 200.0,
                "pKa": 7.0,
                "drug_type": "neutral",
            },
            {
                "drug_name": "MidPerm",
                "logP": 1.5,
                "mw_Da": 350.0,
                "pKa": 7.0,
                "drug_type": "neutral",
            },
        ]

    def test_screen_returns_list(self):
        result = screen_pampa(self._compounds())
        assert isinstance(result, list)

    def test_screen_sorted_by_peff_descending(self):
        results = screen_pampa(self._compounds())
        peff_vals = [r.peff_cm_per_s for r in results]
        assert peff_vals == sorted(peff_vals, reverse=True)

    def test_screen_returns_pampa_results(self):
        results = screen_pampa(self._compounds())
        for r in results:
            assert isinstance(r, PAMPAPermeabilityResult)

    def test_screen_first_is_highest_perm(self):
        results = screen_pampa(self._compounds())
        assert results[0].drug_name == "HighPerm"

    def test_screen_empty_list(self):
        assert screen_pampa([]) == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
class TestValidation:
    def test_mw_zero_raises(self):
        with pytest.raises(ValueError):
            predict_pampa_permeability("Drug", 2.0, 0.0, 7.0, "neutral")

    def test_mw_negative_raises(self):
        with pytest.raises(ValueError):
            predict_pampa_permeability("Drug", 2.0, -100.0, 7.0, "neutral")

    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError):
            predict_pampa_permeability("Drug", 2.0, 300.0, 7.0, "zwitterion")

    def test_valid_drug_types_no_error(self):
        for dtype in ("acid", "base", "neutral"):
            predict_pampa_permeability("Drug", 2.0, 300.0, 7.0, dtype)
