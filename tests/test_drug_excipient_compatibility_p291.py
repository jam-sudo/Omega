"""Tests for Phase 291 — Drug-Excipient Compatibility Predictor."""

import pytest

from omega_pbpk.prediction.drug_excipient_compatibility_p291 import (
    CompatibilityResult,
    predict_compatibility,
    screen_excipient_panel,
)

_VALID_CLASSES = {"excellent", "good", "moderate", "poor"}
_ALL_EXCIPIENTS = [
    "MCC",
    "lactose",
    "starch",
    "povidone",
    "HPMC",
    "citric_acid",
    "Na_bicarbonate",
    "Mg_stearate",
    "talc",
    "SDS",
]


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnTypeAndFields:
    def test_returns_compatibility_result(self):
        result = predict_compatibility("DrugA", 5.0, "acid", 1.5, "MCC")
        assert isinstance(result, CompatibilityResult)

    def test_drug_name_preserved(self):
        result = predict_compatibility("DrugX", 6.0, "neutral", 2.0, "talc")
        assert result.drug_name == "DrugX"

    def test_excipient_name_preserved(self):
        result = predict_compatibility("DrugX", 6.0, "neutral", 2.0, "talc")
        assert result.excipient_name == "talc"

    def test_pka_preserved(self):
        result = predict_compatibility("DrugX", 4.5, "acid", 1.0, "MCC")
        assert result.drug_pka == pytest.approx(4.5)

    def test_logp_preserved(self):
        result = predict_compatibility("DrugX", 4.5, "acid", 2.5, "MCC")
        assert result.drug_logp == pytest.approx(2.5)

    def test_ionization_type_preserved(self):
        result = predict_compatibility("DrugX", 9.0, "base", 1.0, "MCC")
        assert result.drug_ionization_type == "base"

    def test_risk_factors_is_list(self):
        result = predict_compatibility("DrugX", 6.0, "neutral", 2.0, "MCC")
        assert isinstance(result.risk_factors, list)

    def test_notes_is_str(self):
        result = predict_compatibility("DrugX", 6.0, "neutral", 2.0, "MCC")
        assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# Score bounds
# ---------------------------------------------------------------------------


class TestScoreBounds:
    @pytest.mark.parametrize("exc", _ALL_EXCIPIENTS)
    def test_score_in_range(self, exc):
        result = predict_compatibility("DrugA", 5.0, "acid", 1.5, exc)
        assert 0.0 <= result.compatibility_score <= 100.0

    @pytest.mark.parametrize("exc", _ALL_EXCIPIENTS)
    def test_class_valid(self, exc):
        result = predict_compatibility("DrugA", 5.0, "acid", 1.5, exc)
        assert result.compatibility_class in _VALID_CLASSES


# ---------------------------------------------------------------------------
# MCC: high score for neutral drug (minimal risks)
# ---------------------------------------------------------------------------


class TestMCCHighScore:
    def test_mcc_neutral_drug_excellent(self):
        # Neutral drug, logP > 1 (low moisture), pKa <= 8 and logP <= 3 (low redox)
        result = predict_compatibility("NeutralDrug", 7.0, "neutral", 2.0, "MCC")
        assert result.compatibility_score >= 80.0
        assert result.compatibility_class == "excellent"

    def test_mcc_moisture_risk_minimal(self):
        result = predict_compatibility("NeutralDrug", 7.0, "neutral", 2.0, "MCC")
        assert result.moisture_risk_score == pytest.approx(0.1 * 10.0)


# ---------------------------------------------------------------------------
# lactose: lower score for amine/high-pka drug (redox risk)
# ---------------------------------------------------------------------------


class TestLactoseRedoxRisk:
    def test_lactose_high_pka_drug_lower_score(self):
        # pKa > 8 → drug is redox susceptible; lactose redox_risk=0.4 → 25 pts
        result_high = predict_compatibility("AmineDrug", 9.0, "base", 2.0, "lactose")
        assert result_high.redox_risk_score == pytest.approx(0.4 * 25.0)

    def test_lactose_redox_risk_present(self):
        result = predict_compatibility("AmineDrug", 9.5, "base", 4.0, "lactose")
        assert "redox degradation" in result.risk_factors


# ---------------------------------------------------------------------------
# citric_acid: lower score for basic drug (acid-base conflict)
# ---------------------------------------------------------------------------


class TestCitricAcidConflict:
    def test_basic_drug_citric_acid_conflict(self):
        # base with pKa > 7, citric_acid ph_effect=-2.0 < -1.0 → conflict +20
        result = predict_compatibility("BaseDrug", 8.5, "base", 1.5, "citric_acid")
        assert "acid-base conflict" in result.risk_factors
        assert result.compatibility_score < 90.0

    def test_neutral_drug_citric_acid_no_conflict(self):
        result = predict_compatibility("NeutralDrug", 7.0, "neutral", 2.0, "citric_acid")
        assert "acid-base conflict" not in result.risk_factors

    def test_acid_drug_na_bicarbonate_conflict(self):
        # acid with pKa < 7, Na_bicarbonate ph_effect=2.0 > 1.0 → conflict +20
        result = predict_compatibility("AcidDrug", 4.0, "acid", 1.5, "Na_bicarbonate")
        assert "acid-base conflict" in result.risk_factors


# ---------------------------------------------------------------------------
# Moisture sensitivity scoring
# ---------------------------------------------------------------------------


class TestMoistureSensitivity:
    def test_hydrophilic_drug_higher_moisture_score(self):
        # logP < 1.0 → multiplier 30
        result = predict_compatibility("HydrophilicDrug", 5.0, "neutral", 0.5, "povidone")
        assert result.moisture_risk_score == pytest.approx(0.5 * 30.0)

    def test_lipophilic_drug_lower_moisture_score(self):
        # logP >= 1.0 → multiplier 10
        result = predict_compatibility("LipophilicDrug", 5.0, "neutral", 3.0, "povidone")
        assert result.moisture_risk_score == pytest.approx(0.5 * 10.0)


# ---------------------------------------------------------------------------
# screen_excipient_panel
# ---------------------------------------------------------------------------


class TestScreenExcipientPanel:
    def test_returns_all_10_excipients(self):
        results = screen_excipient_panel("DrugA", 5.0, "neutral", 2.0)
        assert len(results) == 10

    def test_sorted_descending(self):
        results = screen_excipient_panel("DrugA", 5.0, "neutral", 2.0)
        scores = [r.compatibility_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_results_are_compatibility_result(self):
        results = screen_excipient_panel("DrugA", 5.0, "neutral", 2.0)
        for r in results:
            assert isinstance(r, CompatibilityResult)

    def test_all_excipients_covered(self):
        results = screen_excipient_panel("DrugA", 5.0, "neutral", 2.0)
        names = {r.excipient_name for r in results}
        assert names == set(_ALL_EXCIPIENTS)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_excipient(self):
        with pytest.raises(ValueError, match="not in database"):
            predict_compatibility("DrugA", 5.0, "acid", 1.5, "unknown_excipient")

    def test_pka_too_low(self):
        with pytest.raises(ValueError, match="drug_pka"):
            predict_compatibility("DrugA", 0.0, "acid", 1.5, "MCC")

    def test_pka_too_high(self):
        with pytest.raises(ValueError, match="drug_pka"):
            predict_compatibility("DrugA", 15.0, "acid", 1.5, "MCC")

    def test_logp_too_low(self):
        with pytest.raises(ValueError, match="drug_logp"):
            predict_compatibility("DrugA", 5.0, "acid", -6.0, "MCC")

    def test_logp_too_high(self):
        with pytest.raises(ValueError, match="drug_logp"):
            predict_compatibility("DrugA", 5.0, "acid", 11.0, "MCC")

    def test_invalid_ionization_type(self):
        with pytest.raises(ValueError, match="drug_ionization_type"):
            predict_compatibility("DrugA", 5.0, "zwitterion", 1.5, "MCC")

    def test_screen_invalid_pka(self):
        with pytest.raises(ValueError):
            screen_excipient_panel("DrugA", -1.0, "neutral", 2.0)
