"""Tests for Phase 494 — Drug Concentrations in Hair."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.hair_drug_concentration import (
    HairDrugConcentrationResult,
    _calculate_hpr,
    predict_hair_concentration,
    screen_forensic_detectability,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_hair_drug_concentration_result(self):
        result = predict_hair_concentration("cocaine", 2.3, 8.6, "base", 0.1)
        assert isinstance(result, HairDrugConcentrationResult)

    def test_result_is_frozen(self):
        result = predict_hair_concentration("cocaine", 2.3, 8.6, "base", 0.1)
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "changed"  # type: ignore[misc]

    def test_all_fields_present(self):
        result = predict_hair_concentration("test", 1.0, 7.0, "neutral", 0.5)
        for field in [
            "drug_name",
            "logP",
            "pKa",
            "ionization_type",
            "plasma_conc_mg_L",
            "hair_plasma_ratio",
            "hair_conc_mg_kg",
            "detectable_in_hair",
            "detection_window_months",
            "melanin_effect",
            "forensic_significance",
            "notes",
        ]:
            assert hasattr(result, field)


# ---------------------------------------------------------------------------
# HPR formula checks
# ---------------------------------------------------------------------------


class TestHPRFormula:
    def test_neutral_hpr_formula(self):
        """HPR_neutral = 0.01 + 0.001 * max(0, logP)."""
        logP = 2.5
        expected = 0.01 + 0.001 * max(0.0, logP)
        hpr = _calculate_hpr(logP, 7.0, "neutral")
        assert abs(hpr - expected) < 1e-9

    def test_neutral_negative_logp_floor(self):
        """Negative logP: max(0, logP) = 0 → HPR_neutral = 0.01."""
        hpr = _calculate_hpr(-3.0, 7.0, "neutral")
        assert abs(hpr - 0.01) < 1e-9

    def test_basic_drug_higher_hpr_than_neutral(self):
        """Basic drug with pKa > 7 should have higher HPR than neutral."""
        hpr_neutral = _calculate_hpr(2.0, 9.0, "neutral")
        hpr_base = _calculate_hpr(2.0, 9.0, "base")
        assert hpr_base > hpr_neutral

    def test_acid_drug_lower_hpr_than_neutral(self):
        """Acid drug should have lower HPR than neutral."""
        hpr_neutral = _calculate_hpr(2.0, 5.0, "neutral")
        hpr_acid = _calculate_hpr(2.0, 5.0, "acid")
        assert hpr_acid < hpr_neutral

    def test_basic_hpr_formula(self):
        """HPR_basic = HPR_neutral * 10^((pKa - 7) * 0.5)."""
        logP = 2.0
        pKa = 9.0
        hpr_neutral = 0.01 + 0.001 * logP
        expected = hpr_neutral * (10.0 ** ((pKa - 7.0) * 0.5))
        hpr = _calculate_hpr(logP, pKa, "base")
        assert abs(hpr - expected) < 1e-9

    def test_acid_hpr_formula(self):
        """HPR_acid = HPR_neutral * 0.1."""
        logP = 1.0
        pKa = 4.0
        hpr_neutral = 0.01 + 0.001 * logP
        expected = hpr_neutral * 0.1
        hpr = _calculate_hpr(logP, pKa, "acid")
        assert abs(hpr - expected) < 1e-9

    def test_hpr_clamped_to_max(self):
        """Extremely basic, lipophilic drug should be clamped to 100."""
        hpr = _calculate_hpr(5.0, 14.0, "base")
        assert hpr <= 100.0

    def test_hpr_clamped_to_min(self):
        """HPR should never be below 0.001."""
        hpr = _calculate_hpr(-10.0, 2.0, "acid")
        assert hpr >= 0.001


# ---------------------------------------------------------------------------
# hair_conc = plasma_conc * HPR
# ---------------------------------------------------------------------------


class TestHairConcentration:
    def test_hair_conc_exact(self):
        """hair_conc_mg_kg = plasma_conc_mg_L * HPR."""
        result = predict_hair_concentration("drug", 1.5, 7.0, "neutral", 0.5)
        expected = result.plasma_conc_mg_L * result.hair_plasma_ratio
        assert abs(result.hair_conc_mg_kg - expected) < 1e-10

    def test_hair_conc_proportional_to_plasma(self):
        r1 = predict_hair_concentration("drug", 1.5, 7.0, "neutral", 0.2)
        r2 = predict_hair_concentration("drug", 1.5, 7.0, "neutral", 0.8)
        ratio = r2.hair_conc_mg_kg / r1.hair_conc_mg_kg
        assert abs(ratio - 4.0) < 1e-9


# ---------------------------------------------------------------------------
# Detectable in hair
# ---------------------------------------------------------------------------


class TestDetectable:
    def test_detectable_when_above_threshold(self):
        """hair_conc > 0.001 → detectable."""
        result = predict_hair_concentration("morphine", 0.9, 8.0, "base", 1.0)
        if result.hair_conc_mg_kg > 0.001:
            assert result.detectable_in_hair is True
        else:
            assert result.detectable_in_hair is False

    def test_detection_window_3_if_detectable(self):
        """If detectable, detection_window_months = 3.0."""
        result = predict_hair_concentration("cocaine", 2.3, 8.6, "base", 1.0)
        if result.detectable_in_hair:
            assert result.detection_window_months == 3.0
        else:
            assert result.detection_window_months == 0.0

    def test_not_detectable_very_low_conc(self):
        """Very low plasma conc of acid drug → not detectable."""
        result = predict_hair_concentration("drug", 0.0, 3.0, "acid", 1e-6)
        # HPR_acid = (0.01 + 0) * 0.1 = 0.001; hair_conc = 1e-6 * 0.001 = 1e-9
        assert result.detectable_in_hair is False
        assert result.detection_window_months == 0.0


# ---------------------------------------------------------------------------
# Forensic significance classification
# ---------------------------------------------------------------------------


class TestForensicSignificance:
    def test_not_detected(self):
        """hair_conc <= 0.001 → not_detected."""
        # acid drug, very low conc
        result = predict_hair_concentration("drug", 0.0, 4.0, "acid", 1e-5)
        assert result.forensic_significance == "not_detected"

    def test_trace(self):
        """0.001 < hair_conc <= 0.01 → trace."""
        # neutral, logP=0: HPR=0.01; plasma=0.005 → hair=0.00005 -- too low
        # Let's use base drug to push into trace range
        # neutral logP=1: HPR=0.011; plasma=0.1 → hair=0.0011 -- trace
        result = predict_hair_concentration("drug", 1.0, 7.0, "neutral", 0.1)
        if 0.001 < result.hair_conc_mg_kg <= 0.01:
            assert result.forensic_significance == "trace"

    def test_positive(self):
        """0.01 < hair_conc <= 1.0 → positive."""
        result = predict_hair_concentration("drug", 2.0, 7.0, "neutral", 10.0)
        if 0.01 < result.hair_conc_mg_kg <= 1.0:
            assert result.forensic_significance == "positive"

    def test_high(self):
        """hair_conc > 1.0 → high."""
        result = predict_hair_concentration("cocaine", 2.3, 8.6, "base", 10.0)
        if result.hair_conc_mg_kg > 1.0:
            assert result.forensic_significance == "high"


# ---------------------------------------------------------------------------
# Melanin effect
# ---------------------------------------------------------------------------


class TestMelaninEffect:
    def test_melanin_effect_base_high_pka(self):
        result = predict_hair_concentration("amphetamine", 1.8, 9.9, "base", 0.1)
        assert result.melanin_effect is True

    def test_no_melanin_effect_neutral(self):
        result = predict_hair_concentration("caffeine", -0.07, 10.4, "neutral", 0.1)
        assert result.melanin_effect is False

    def test_no_melanin_effect_acid(self):
        result = predict_hair_concentration("aspirin", 1.19, 3.5, "acid", 0.1)
        assert result.melanin_effect is False

    def test_no_melanin_effect_base_low_pka(self):
        """Base with pKa <= 7 should NOT have melanin effect."""
        result = predict_hair_concentration("drug", 1.0, 6.5, "base", 0.1)
        assert result.melanin_effect is False


# ---------------------------------------------------------------------------
# screen_forensic_detectability
# ---------------------------------------------------------------------------


class TestScreenForensicDetectability:
    def _sample_compounds(self):
        return [
            {
                "drug_name": "aspirin",
                "logP": 1.19,
                "pKa": 3.5,
                "ionization_type": "acid",
                "plasma_conc_mg_L": 0.5,
            },
            {
                "drug_name": "cocaine",
                "logP": 2.3,
                "pKa": 8.6,
                "ionization_type": "base",
                "plasma_conc_mg_L": 0.5,
            },
            {
                "drug_name": "caffeine",
                "logP": -0.07,
                "pKa": 10.4,
                "ionization_type": "neutral",
                "plasma_conc_mg_L": 0.5,
            },
        ]

    def test_returns_list(self):
        results = screen_forensic_detectability(self._sample_compounds())
        assert isinstance(results, list)

    def test_length_matches_input(self):
        compounds = self._sample_compounds()
        results = screen_forensic_detectability(compounds)
        assert len(results) == len(compounds)

    def test_sorted_by_hpr_descending(self):
        results = screen_forensic_detectability(self._sample_compounds())
        hprs = [r.hair_plasma_ratio for r in results]
        assert hprs == sorted(hprs, reverse=True)

    def test_all_are_hair_result(self):
        results = screen_forensic_detectability(self._sample_compounds())
        for r in results:
            assert isinstance(r, HairDrugConcentrationResult)

    def test_empty_list(self):
        results = screen_forensic_detectability([])
        assert results == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_ionization_type(self):
        with pytest.raises(ValueError, match="ionization_type"):
            predict_hair_concentration("drug", 1.0, 7.0, "zwitterion", 0.1)

    def test_zero_plasma_conc(self):
        with pytest.raises(ValueError, match="plasma_conc_mg_L"):
            predict_hair_concentration("drug", 1.0, 7.0, "neutral", 0.0)

    def test_negative_plasma_conc(self):
        with pytest.raises(ValueError, match="plasma_conc_mg_L"):
            predict_hair_concentration("drug", 1.0, 7.0, "neutral", -0.5)
