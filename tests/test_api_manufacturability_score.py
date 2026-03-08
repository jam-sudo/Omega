"""
Tests for Phase 202 — API Manufacturability Score
"""

import pytest

from omega_pbpk.prediction.api_manufacturability_score import (
    ManufacturabilityResult,
    score_manufacturability,
    screen_api_candidates,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs):
    params = dict(
        drug_name="TestAPI",
        mw_Da=350.0,
        mp_C=165.0,
        logP=2.0,
        solubility_mg_mL=1.0,
        hygroscopicity="non",
        crystallinity="crystalline",
    )
    params.update(kwargs)
    return score_manufacturability(**params)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_manufacturability_result(self):
        result = _default()
        assert isinstance(result, ManufacturabilityResult)

    def test_drug_name_preserved(self):
        result = _default(drug_name="Aspirin")
        assert result.drug_name == "Aspirin"

    def test_mw_preserved(self):
        result = _default(mw_Da=400.0)
        assert result.mw_Da == pytest.approx(400.0)

    def test_mp_preserved(self):
        result = _default(mp_C=150.0)
        assert result.mp_C == pytest.approx(150.0)

    def test_logP_preserved(self):
        result = _default(logP=3.5)
        assert result.logP == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Score ranges
# ---------------------------------------------------------------------------


class TestScoreRanges:
    def test_total_score_in_range(self):
        result = _default()
        assert 0.0 <= result.total_score <= 100.0

    def test_mw_score_in_range(self):
        result = _default()
        assert 0.0 <= result.mw_score <= 20.0

    def test_mp_score_in_range(self):
        result = _default()
        assert 0.0 <= result.mp_score <= 20.0

    def test_solubility_score_in_range(self):
        result = _default()
        assert 0.0 <= result.solubility_score <= 20.0

    def test_hygroscopicity_score_in_range(self):
        result = _default()
        assert 0.0 <= result.hygroscopicity_score <= 20.0

    def test_crystallinity_score_in_range(self):
        result = _default()
        assert 0.0 <= result.crystallinity_score <= 20.0


# ---------------------------------------------------------------------------
# Individual component scoring
# ---------------------------------------------------------------------------


class TestComponentScores:
    def test_optimal_mw_score(self):
        result = _default(mw_Da=350.0)  # 200-500 range
        assert result.mw_score == pytest.approx(20.0)

    def test_optimal_mp_score(self):
        result = _default(mp_C=165.0)  # 130-200 range
        assert result.mp_score == pytest.approx(20.0)

    def test_high_solubility_score(self):
        result = _default(solubility_mg_mL=10.0)
        assert result.solubility_score == pytest.approx(20.0)

    def test_solubility_moderate_band(self):
        result = _default(solubility_mg_mL=0.5)
        assert result.solubility_score == pytest.approx(15.0)

    def test_solubility_low_band(self):
        result = _default(solubility_mg_mL=0.05)
        assert result.solubility_score == pytest.approx(8.0)

    def test_solubility_very_low_band(self):
        result = _default(solubility_mg_mL=0.001)
        assert result.solubility_score == pytest.approx(2.0)

    def test_non_hygroscopic_score(self):
        result = _default(hygroscopicity="non")
        assert result.hygroscopicity_score == pytest.approx(20.0)

    def test_slight_hygroscopic_score(self):
        result = _default(hygroscopicity="slight")
        assert result.hygroscopicity_score == pytest.approx(16.0)

    def test_moderate_hygroscopic_score(self):
        result = _default(hygroscopicity="moderate")
        assert result.hygroscopicity_score == pytest.approx(10.0)

    def test_high_hygroscopic_score(self):
        result = _default(hygroscopicity="high")
        assert result.hygroscopicity_score == pytest.approx(4.0)

    def test_crystalline_score(self):
        result = _default(crystallinity="crystalline")
        assert result.crystallinity_score == pytest.approx(20.0)

    def test_semi_crystalline_score(self):
        result = _default(crystallinity="semi_crystalline")
        assert result.crystallinity_score == pytest.approx(14.0)

    def test_amorphous_score(self):
        result = _default(crystallinity="amorphous")
        assert result.crystallinity_score == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Manufacturability classification
# ---------------------------------------------------------------------------


class TestClassification:
    def test_excellent_class_for_near_optimal_drug(self):
        # All optimal → total = 100
        result = _default(
            mw_Da=350.0,
            mp_C=165.0,
            solubility_mg_mL=5.0,
            hygroscopicity="non",
            crystallinity="crystalline",
        )
        assert result.manufacturability_class == "excellent"
        assert result.total_score >= 80.0

    def test_challenging_class_for_poor_drug(self):
        result = _default(
            mw_Da=900.0,  # heavy penalty
            mp_C=300.0,  # too high
            solubility_mg_mL=0.001,
            hygroscopicity="high",
            crystallinity="amorphous",
        )
        assert result.manufacturability_class == "challenging"
        assert result.total_score < 60.0

    def test_good_class_threshold(self):
        # Lower score by using elevated mp, moderate hygro, semi-crystalline
        result = _default(
            mw_Da=350.0,
            mp_C=230.0,
            solubility_mg_mL=0.5,
            hygroscopicity="moderate",
            crystallinity="semi_crystalline",
        )
        assert result.manufacturability_class in ("good", "challenging", "excellent")


# ---------------------------------------------------------------------------
# Top issue identification
# ---------------------------------------------------------------------------


class TestTopIssue:
    def test_top_issue_is_string(self):
        result = _default()
        assert isinstance(result.top_issue, str)

    def test_top_issue_identifies_solubility(self):
        result = _default(solubility_mg_mL=0.001)  # solubility_score=2 — lowest
        assert result.top_issue == "solubility"

    def test_top_issue_identifies_crystallinity(self):
        # amorphous (6) lower than all others at defaults
        result = _default(solubility_mg_mL=5.0, crystallinity="amorphous")
        assert result.top_issue == "crystallinity"


# ---------------------------------------------------------------------------
# Screen candidates
# ---------------------------------------------------------------------------


class TestScreenCandidates:
    def test_returns_list(self):
        candidates = [
            dict(
                drug_name="A",
                mw_Da=350.0,
                mp_C=165.0,
                logP=2.0,
                solubility_mg_mL=1.0,
                hygroscopicity="non",
                crystallinity="crystalline",
            ),
            dict(
                drug_name="B",
                mw_Da=700.0,
                mp_C=280.0,
                logP=5.0,
                solubility_mg_mL=0.001,
                hygroscopicity="high",
                crystallinity="amorphous",
            ),
        ]
        results = screen_api_candidates(candidates)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_sorted_by_total_score_descending(self):
        candidates = [
            dict(
                drug_name="A",
                mw_Da=700.0,
                mp_C=280.0,
                logP=5.0,
                solubility_mg_mL=0.001,
                hygroscopicity="high",
                crystallinity="amorphous",
            ),
            dict(
                drug_name="B",
                mw_Da=350.0,
                mp_C=165.0,
                logP=2.0,
                solubility_mg_mL=1.0,
                hygroscopicity="non",
                crystallinity="crystalline",
            ),
        ]
        results = screen_api_candidates(candidates)
        assert results[0].total_score >= results[1].total_score

    def test_best_candidate_first(self):
        candidates = [
            dict(
                drug_name="Poor",
                mw_Da=900.0,
                mp_C=300.0,
                logP=6.0,
                solubility_mg_mL=0.001,
                hygroscopicity="high",
                crystallinity="amorphous",
            ),
            dict(
                drug_name="Good",
                mw_Da=350.0,
                mp_C=165.0,
                logP=2.0,
                solubility_mg_mL=1.0,
                hygroscopicity="non",
                crystallinity="crystalline",
            ),
        ]
        results = screen_api_candidates(candidates)
        assert results[0].drug_name == "Good"


# ---------------------------------------------------------------------------
# Validation / edge cases
# ---------------------------------------------------------------------------


class TestValidation:
    def test_mw_zero_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            score_manufacturability("Drug", 0.0, 165.0, 2.0, 1.0, "non", "crystalline")

    def test_mw_negative_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            score_manufacturability("Drug", -100.0, 165.0, 2.0, 1.0, "non", "crystalline")

    def test_invalid_hygroscopicity_raises(self):
        with pytest.raises(ValueError, match="hygroscopicity"):
            score_manufacturability("Drug", 350.0, 165.0, 2.0, 1.0, "unknown", "crystalline")

    def test_invalid_crystallinity_raises(self):
        with pytest.raises(ValueError, match="crystallinity"):
            score_manufacturability("Drug", 350.0, 165.0, 2.0, 1.0, "non", "gel")
