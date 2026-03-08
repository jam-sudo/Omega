"""
Tests for Phase 211 — excipient_compatibility_matrix.py
~25 tests covering return types, field values, rules, sorting, and validation.
"""

import pytest

from omega_pbpk.prediction.excipient_compatibility_matrix import (
    CompatibilityResult,
    assess_compatibility,
    screen_excipient_set,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AMINE_DRUG = {"has_amine": True, "has_carboxyl": False, "oxidation_sensitive": False}
ACID_DRUG = {"has_amine": False, "has_carboxyl": True, "oxidation_sensitive": False}
OX_DRUG = {"has_amine": False, "has_carboxyl": False, "oxidation_sensitive": True}
INERT_DRUG = {"has_amine": False, "has_carboxyl": False, "oxidation_sensitive": False}


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_compatibility_result(self):
        result = assess_compatibility("DrugA", AMINE_DRUG, ["lactose"])
        assert isinstance(result, CompatibilityResult)

    def test_result_is_frozen(self):
        result = assess_compatibility("DrugA", AMINE_DRUG, ["lactose"])
        with pytest.raises(Exception):
            result.drug_name = "other"  # type: ignore[misc]

    def test_excipients_tested_is_tuple(self):
        result = assess_compatibility("DrugA", AMINE_DRUG, ["lactose", "talc"])
        assert isinstance(result.excipients_tested, tuple)

    def test_compatibility_scores_is_tuple(self):
        result = assess_compatibility("DrugA", AMINE_DRUG, ["lactose"])
        assert isinstance(result.compatibility_scores, tuple)

    def test_flagged_combinations_is_tuple(self):
        result = assess_compatibility("DrugA", AMINE_DRUG, ["lactose"])
        assert isinstance(result.flagged_combinations, tuple)


# ---------------------------------------------------------------------------
# 2. Field preservation
# ---------------------------------------------------------------------------


class TestFieldPreservation:
    def test_drug_name_preserved(self):
        result = assess_compatibility("MyDrug", INERT_DRUG, ["mannitol"])
        assert result.drug_name == "MyDrug"

    def test_excipients_tested_matches_input(self):
        excipients = ["lactose", "povidone"]
        result = assess_compatibility("D", AMINE_DRUG, excipients)
        assert result.excipients_tested == tuple(excipients)

    def test_scores_length_matches_excipients(self):
        excipients = ["lactose", "talc", "mannitol"]
        result = assess_compatibility("D", OX_DRUG, excipients)
        assert len(result.compatibility_scores) == len(excipients)


# ---------------------------------------------------------------------------
# 3. Incompatibility rules
# ---------------------------------------------------------------------------


class TestIncompatibilityRules:
    def test_amine_lactose_high_risk(self):
        result = assess_compatibility("AmineDrug", AMINE_DRUG, ["lactose"])
        assert result.max_score == 3
        assert result.overall_risk == "high"

    def test_amine_starch_high_risk(self):
        """starch is polysaccharide → Maillard → HIGH"""
        result = assess_compatibility("AmineDrug", AMINE_DRUG, ["starch"])
        assert result.max_score == 3
        assert result.overall_risk == "high"

    def test_acid_calcium_phosphate_moderate(self):
        result = assess_compatibility("AcidDrug", ACID_DRUG, ["calcium_phosphate"])
        assert result.max_score == 2
        assert result.overall_risk == "moderate"

    def test_amine_magnesium_stearate_moderate(self):
        result = assess_compatibility("AmineDrug", AMINE_DRUG, ["magnesium_stearate"])
        assert result.max_score == 2
        assert result.overall_risk == "moderate"

    def test_oxidation_sensitive_talc_low(self):
        result = assess_compatibility("OxDrug", OX_DRUG, ["talc"])
        assert result.max_score == 1
        assert result.overall_risk == "low"

    def test_inert_drug_no_risk(self):
        result = assess_compatibility("InertDrug", INERT_DRUG, ["lactose", "talc"])
        assert result.max_score == 0
        assert result.overall_risk == "none"

    def test_cellulose_no_issue_for_amine(self):
        result = assess_compatibility("AmineDrug", AMINE_DRUG, ["microcrystalline_cellulose"])
        assert result.compatibility_scores == (0,)

    def test_polymer_no_issue(self):
        result = assess_compatibility("D", AMINE_DRUG, ["povidone"])
        assert result.compatibility_scores == (0,)


# ---------------------------------------------------------------------------
# 4. overall_risk matches max score
# ---------------------------------------------------------------------------


class TestOverallRiskMatchesMaxScore:
    def test_max_score_3_is_high(self):
        result = assess_compatibility("D", AMINE_DRUG, ["lactose"])
        assert result.overall_risk == "high"
        assert result.max_score == 3

    def test_max_score_2_is_moderate(self):
        result = assess_compatibility("D", ACID_DRUG, ["calcium_phosphate"])
        assert result.overall_risk == "moderate"
        assert result.max_score == 2

    def test_max_score_1_is_low(self):
        result = assess_compatibility("D", OX_DRUG, ["talc"])
        assert result.overall_risk == "low"
        assert result.max_score == 1

    def test_max_score_0_is_none(self):
        result = assess_compatibility("D", INERT_DRUG, ["mannitol"])
        assert result.overall_risk == "none"
        assert result.max_score == 0


# ---------------------------------------------------------------------------
# 5. Flagged combinations
# ---------------------------------------------------------------------------


class TestFlaggedCombinations:
    def test_flagged_nonempty_when_incompatible(self):
        result = assess_compatibility("D", AMINE_DRUG, ["lactose"])
        assert len(result.flagged_combinations) > 0

    def test_flagged_empty_when_compatible(self):
        result = assess_compatibility("D", INERT_DRUG, ["mannitol"])
        assert len(result.flagged_combinations) == 0

    def test_flagged_contains_excipient_name(self):
        result = assess_compatibility("D", AMINE_DRUG, ["lactose"])
        assert any("lactose" in flag for flag in result.flagged_combinations)


# ---------------------------------------------------------------------------
# 6. screen_excipient_set
# ---------------------------------------------------------------------------


class TestScreenExcipientSet:
    def test_sorted_by_risk_ascending(self):
        sets = [
            ["lactose"],  # HIGH (3)
            ["mannitol"],  # NONE (0)
            ["calcium_phosphate"],  # MODERATE (2) — only with acid drug
            ["talc"],  # LOW (1) — only with oxidation-sensitive
        ]
        # Use amine drug: lactose=3, MgSt=2 (not in sets above), talc=0 for amine
        # For amine: lactose=3, mannitol=0, calcium_phosphate=0, talc=0
        results = screen_excipient_set("D", AMINE_DRUG, sets)
        scores = [r.max_score for r in results]
        assert scores == sorted(scores)

    def test_returns_list(self):
        results = screen_excipient_set("D", INERT_DRUG, [["mannitol"], ["talc"]])
        assert isinstance(results, list)

    def test_number_of_results_equals_number_of_sets(self):
        sets = [["lactose"], ["mannitol"], ["povidone"]]
        results = screen_excipient_set("D", AMINE_DRUG, sets)
        assert len(results) == 3

    def test_safest_first(self):
        sets = [["lactose"], ["mannitol"]]  # high vs none
        results = screen_excipient_set("D", AMINE_DRUG, sets)
        # safest (none/mannitol) should come first
        assert results[0].max_score <= results[-1].max_score


# ---------------------------------------------------------------------------
# 7. Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_excipient_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            assess_compatibility("D", AMINE_DRUG, [])

    def test_unknown_excipient_raises(self):
        with pytest.raises(ValueError, match="Unknown excipient"):
            assess_compatibility("D", AMINE_DRUG, ["silica_aerogel"])

    def test_unknown_excipient_in_mixed_list_raises(self):
        with pytest.raises(ValueError):
            assess_compatibility("D", AMINE_DRUG, ["lactose", "unknown_excipient"])
