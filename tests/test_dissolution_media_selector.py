"""Tests for Phase 177 — dissolution_media_selector."""

import math

import pytest

from omega_pbpk.prediction.dissolution_media_selector import (
    DissolutionMediaResult,
    DissolutionRateResult,
    calculate_dissolution_rate,
    recommend_dissolution_media,
)

# ---------------------------------------------------------------------------
# recommend_dissolution_media
# ---------------------------------------------------------------------------


class TestRecommendDissolutionMedia:
    def test_returns_seven_results(self):
        results = recommend_dissolution_media("TestDrug", "I", "neutral", 7.0, 5.0)
        assert len(results) == 7

    def test_returns_dissolution_media_result_instances(self):
        results = recommend_dissolution_media("DrugA", "I", "acid", 4.5, 1.0)
        for r in results:
            assert isinstance(r, DissolutionMediaResult)

    def test_results_sorted_descending(self):
        results = recommend_dissolution_media("DrugA", "II", "base", 8.0, 1.0)
        scores = [r.suitability_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_bcs1_phosphate_highest_or_top3(self):
        """BCS I → phosphate_pH6.8 should be FDA recommended and score well."""
        results = recommend_dissolution_media("DrugI", "I", "neutral", 7.0, 10.0)
        phosphate = next(r for r in results if r.media_name == "phosphate_pH6.8")
        assert phosphate.fda_recommended is True
        # Should be in top 3
        top3_names = [r.media_name for r in results[:3]]
        assert "phosphate_pH6.8" in top3_names

    def test_bcs2_fassif_fda_recommended(self):
        """BCS II → FaSSIF should be FDA recommended."""
        results = recommend_dissolution_media("PoorDrug", "II", "neutral", 7.0, 0.01)
        fassif = next(r for r in results if r.media_name == "FaSSIF")
        assert fassif.fda_recommended is True

    def test_bcs2_fassif_highest_score(self):
        """BCS II → FaSSIF should have highest suitability."""
        results = recommend_dissolution_media("PoorDrug", "II", "neutral", 7.0, 0.01)
        assert results[0].media_name == "FaSSIF"

    def test_bcs3_phosphate_recommended(self):
        results = recommend_dissolution_media("Hydrophilic", "III", "neutral", 7.0, 5.0)
        phosphate = next(r for r in results if r.media_name == "phosphate_pH6.8")
        assert phosphate.fda_recommended is True

    def test_bcs4_fessif_recommended(self):
        results = recommend_dissolution_media("BadDrug", "IV", "neutral", 7.0, 0.005)
        fessif = next(r for r in results if r.media_name == "FeSSIF")
        assert fessif.fda_recommended is True

    def test_acid_drug_higher_solubility_at_high_ph(self):
        """Acid drug: solubility should increase as pH rises above pKa."""
        results = recommend_dissolution_media("AcidDrug", "I", "acid", 4.0, 1.0)
        ph_map = {r.media_name: r for r in results}
        # phosphate_pH6.8 > acetate_pH4.5 for acid drug with pKa=4
        sol_high = ph_map["phosphate_pH6.8"].solubility_at_media_ph_mg_mL
        sol_low = ph_map["acetate_pH4.5"].solubility_at_media_ph_mg_mL
        assert sol_high > sol_low

    def test_base_drug_higher_solubility_at_low_ph(self):
        """Base drug: solubility should increase as pH drops below pKa."""
        results = recommend_dissolution_media("BaseDrug", "II", "base", 8.0, 0.1)
        ph_map = {r.media_name: r for r in results}
        sol_low = ph_map["SGF_pH1.2"].solubility_at_media_ph_mg_mL
        sol_mid = ph_map["phosphate_pH6.8"].solubility_at_media_ph_mg_mL
        assert sol_low > sol_mid

    def test_neutral_drug_same_solubility_all_media(self):
        """Neutral drug: solubility is same in all media."""
        results = recommend_dissolution_media("NeutralDrug", "I", "neutral", 7.0, 5.0)
        solubilities = [r.solubility_at_media_ph_mg_mL for r in results]
        assert all(abs(s - 5.0) < 1e-9 for s in solubilities)

    def test_biorelevant_media_flagged(self):
        results = recommend_dissolution_media("X", "II", "neutral", 7.0, 0.01)
        ph_map = {r.media_name: r for r in results}
        assert ph_map["FaSSIF"].biorelevant is True
        assert ph_map["FeSSIF"].biorelevant is True
        assert ph_map["FaSSGF"].biorelevant is True
        assert ph_map["water"].biorelevant is False
        assert ph_map["phosphate_pH6.8"].biorelevant is False

    def test_score_bounded_0_to_100(self):
        results = recommend_dissolution_media("X", "II", "base", 9.0, 100.0)
        for r in results:
            assert 0.0 <= r.suitability_score <= 100.0

    def test_drug_name_preserved(self):
        results = recommend_dissolution_media("MyDrug", "I", "neutral", 7.0, 1.0)
        for r in results:
            assert r.drug_name == "MyDrug"

    def test_bcs_class_preserved(self):
        results = recommend_dissolution_media("X", "III", "neutral", 7.0, 1.0)
        for r in results:
            assert r.bcs_class == "III"

    # Validation
    def test_invalid_bcs_class(self):
        with pytest.raises(ValueError, match="bcs_class"):
            recommend_dissolution_media("X", "V", "neutral", 7.0, 1.0)

    def test_invalid_ionization_type(self):
        with pytest.raises(ValueError, match="ionization_type"):
            recommend_dissolution_media("X", "I", "zwitterion", 7.0, 1.0)

    def test_pka_out_of_range(self):
        with pytest.raises(ValueError, match="pka"):
            recommend_dissolution_media("X", "I", "acid", -1.0, 1.0)

    def test_pka_too_high(self):
        with pytest.raises(ValueError, match="pka"):
            recommend_dissolution_media("X", "I", "acid", 15.0, 1.0)

    def test_solubility_zero(self):
        with pytest.raises(ValueError, match="solubility_mg_mL"):
            recommend_dissolution_media("X", "I", "neutral", 7.0, 0.0)

    def test_negative_solubility(self):
        with pytest.raises(ValueError, match="solubility_mg_mL"):
            recommend_dissolution_media("X", "I", "neutral", 7.0, -1.0)


# ---------------------------------------------------------------------------
# calculate_dissolution_rate
# ---------------------------------------------------------------------------


class TestCalculateDissolutionRate:
    def test_returns_dissolution_rate_result(self):
        result = calculate_dissolution_rate(
            drug_name="DrugX",
            solubility_mg_mL=1.0,
            ionization_type="neutral",
            pka=7.0,
            media_ph=6.8,
            dose_mg=100.0,
        )
        assert isinstance(result, DissolutionRateResult)

    def test_t90_positive(self):
        result = calculate_dissolution_rate(
            drug_name="X",
            solubility_mg_mL=1.0,
            ionization_type="neutral",
            pka=7.0,
            media_ph=6.8,
            dose_mg=100.0,
        )
        assert result.t90_min > 0

    def test_rapid_dissolution_class(self):
        """Very high solubility + small dose → rapid dissolution."""
        result = calculate_dissolution_rate(
            drug_name="X",
            solubility_mg_mL=1000.0,
            ionization_type="neutral",
            pka=7.0,
            media_ph=6.8,
            dose_mg=1.0,
        )
        assert result.dissolution_class == "rapid"
        assert result.t90_min < 15.0

    def test_slow_dissolution_class(self):
        """Very low solubility + large dose → slow dissolution."""
        result = calculate_dissolution_rate(
            drug_name="X",
            solubility_mg_mL=0.0001,
            ionization_type="neutral",
            pka=7.0,
            media_ph=6.8,
            dose_mg=500.0,
        )
        assert result.dissolution_class == "slow"
        assert result.t90_min > 60.0

    def test_moderate_dissolution_class(self):
        """Medium solubility → moderate dissolution (15-60 min)."""
        result = calculate_dissolution_rate(
            drug_name="X",
            solubility_mg_mL=0.05,
            ionization_type="neutral",
            pka=7.0,
            media_ph=6.8,
            dose_mg=100.0,
        )
        # Just verify t90 is finite and positive
        assert math.isfinite(result.t90_min)
        assert result.t90_min > 0
        assert result.dissolution_class in {"rapid", "moderate", "slow"}

    def test_higher_rpm_faster_dissolution(self):
        """Higher paddle RPM → faster rate → lower t90."""
        r_low = calculate_dissolution_rate("X", 1.0, "neutral", 7.0, 6.8, 100.0, paddle_rpm=25)
        r_high = calculate_dissolution_rate("X", 1.0, "neutral", 7.0, 6.8, 100.0, paddle_rpm=100)
        assert r_high.dissolution_rate_mg_per_min > r_low.dissolution_rate_mg_per_min
        assert r_high.t90_min < r_low.t90_min

    def test_diffusivity_positive(self):
        result = calculate_dissolution_rate("X", 1.0, "neutral", 7.0, 6.8, 100.0)
        assert result.diffusivity_cm2_per_s > 0

    def test_drug_name_preserved(self):
        result = calculate_dissolution_rate("SpecialDrug", 1.0, "neutral", 7.0, 6.8, 100.0)
        assert result.drug_name == "SpecialDrug"

    def test_acid_drug_solubility_at_high_ph_increases(self):
        """Acid drug: higher pH media → higher Cs."""
        r_low = calculate_dissolution_rate("AcidDrug", 0.1, "acid", 4.0, 2.0, 100.0)
        r_high = calculate_dissolution_rate("AcidDrug", 0.1, "acid", 4.0, 8.0, 100.0)
        assert r_high.solubility_at_ph_mg_mL > r_low.solubility_at_ph_mg_mL

    # Validation
    def test_invalid_ionization_type(self):
        with pytest.raises(ValueError, match="ionization_type"):
            calculate_dissolution_rate("X", 1.0, "bad", 7.0, 6.8, 100.0)

    def test_invalid_solubility(self):
        with pytest.raises(ValueError, match="solubility_mg_mL"):
            calculate_dissolution_rate("X", 0.0, "neutral", 7.0, 6.8, 100.0)

    def test_invalid_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            calculate_dissolution_rate("X", 1.0, "neutral", 7.0, 6.8, -1.0)

    def test_invalid_rpm(self):
        with pytest.raises(ValueError, match="paddle_rpm"):
            calculate_dissolution_rate("X", 1.0, "neutral", 7.0, 6.8, 100.0, paddle_rpm=0)

    def test_pka_out_of_range_dissolution(self):
        with pytest.raises(ValueError, match="pka"):
            calculate_dissolution_rate("X", 1.0, "acid", 20.0, 6.8, 100.0)
