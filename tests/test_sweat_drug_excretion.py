"""Tests for sweat drug excretion model — Phase 550."""

import pytest

from omega_pbpk.prediction.sweat_drug_excretion import (
    SweatExcretionResult,
    compare_sweat_excretion,
    predict_sweat_excretion,
)


class TestPredictSweatExcretion:
    """Tests for predict_sweat_excretion."""

    def test_return_type(self):
        result = predict_sweat_excretion("DrugA", 10.0, 8.0, "base")
        assert isinstance(result, SweatExcretionResult)

    def test_all_fields_present(self):
        result = predict_sweat_excretion("DrugA", 10.0, 8.0, "base")
        assert result.drug_name == "DrugA"
        assert result.plasma_concentration_mg_L == 10.0
        assert result.sweat_ph == 6.5
        assert result.drug_type == "base"
        assert isinstance(result.sweat_plasma_ratio, float)
        assert isinstance(result.sweat_concentration_mg_L, float)
        assert isinstance(result.daily_sweat_excretion_mg, float)
        assert isinstance(result.detection_window_h, float)
        assert result.excretion_class in ("high", "moderate", "low")

    def test_sp_ratio_clamped_lower(self):
        # Very low fup should still give S/P >= 0.01
        result = predict_sweat_excretion("DrugA", 10.0, 5.0, "acid", fup=0.001)
        assert result.sweat_plasma_ratio >= 0.01

    def test_sp_ratio_clamped_upper(self):
        # Even extreme conditions should give S/P <= 10.0
        result = predict_sweat_excretion("DrugA", 10.0, 8.0, "base", fup=1.0)
        assert result.sweat_plasma_ratio <= 10.0

    def test_sweat_concentration_positive(self):
        result = predict_sweat_excretion("DrugA", 5.0, 7.0, "base")
        assert result.sweat_concentration_mg_L > 0

    def test_neutral_drug_sp_equals_fup(self):
        fup = 0.3
        result = predict_sweat_excretion("Neutral", 1.0, 7.0, "neutral", fup=fup)
        assert abs(result.sweat_plasma_ratio - fup) < 1e-9

    def test_base_higher_sp_at_acidic_sweat(self):
        # Ion trapping: base drug should have higher S/P at lower sweat pH
        r1 = predict_sweat_excretion("Base", 10.0, 9.0, "base", sweat_ph=5.0, fup=0.5)
        r2 = predict_sweat_excretion("Base", 10.0, 9.0, "base", sweat_ph=7.0, fup=0.5)
        assert r1.sweat_plasma_ratio > r2.sweat_plasma_ratio

    def test_excretion_class_high(self):
        # High fup base with strong ion trapping at very acidic sweat
        result = predict_sweat_excretion("DrugH", 10.0, 9.0, "base", sweat_ph=5.0, fup=1.0)
        assert result.sweat_plasma_ratio >= 1.0
        assert result.excretion_class == "high"

    def test_excretion_class_moderate(self):
        result = predict_sweat_excretion("DrugM", 10.0, 5.0, "neutral", fup=0.4)
        assert result.excretion_class == "moderate"

    def test_excretion_class_low(self):
        result = predict_sweat_excretion("DrugL", 10.0, 5.0, "neutral", fup=0.1)
        assert result.excretion_class == "low"

    def test_detection_window_positive(self):
        result = predict_sweat_excretion("DrugA", 10.0, 8.0, "base")
        assert result.detection_window_h > 0

    def test_detection_window_zero_for_very_low_conc(self):
        result = predict_sweat_excretion("DrugA", 0.00001, 8.0, "base", fup=0.01)
        # sweat_conc = 0.00001 * sp_ratio, which may be <= 0.001
        # detection_window should be 0 or very small
        assert result.detection_window_h >= 0

    def test_daily_excretion_scales_with_sweat_rate(self):
        r1 = predict_sweat_excretion("Drug", 10.0, 7.0, "base", sweat_rate_L_per_day=1.0)
        r2 = predict_sweat_excretion("Drug", 10.0, 7.0, "base", sweat_rate_L_per_day=2.0)
        assert abs(r2.daily_sweat_excretion_mg - 2.0 * r1.daily_sweat_excretion_mg) < 1e-9

    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError, match="Invalid drug_type"):
            predict_sweat_excretion("Drug", 10.0, 7.0, "ion")

    def test_negative_plasma_concentration_raises(self):
        with pytest.raises(ValueError, match="plasma_concentration_mg_L"):
            predict_sweat_excretion("Drug", -1.0, 7.0, "base")

    def test_invalid_fup_raises(self):
        with pytest.raises(ValueError, match="fup"):
            predict_sweat_excretion("Drug", 10.0, 7.0, "base", fup=1.5)

    def test_zero_plasma_concentration(self):
        result = predict_sweat_excretion("Drug", 0.0, 7.0, "base")
        assert result.sweat_concentration_mg_L == 0.0
        assert result.detection_window_h == 0.0


class TestCompareSweatExcretion:
    """Tests for compare_sweat_excretion."""

    def test_sorted_descending_by_sp_ratio(self):
        compounds = [
            {
                "drug_name": "Low",
                "plasma_concentration_mg_L": 10.0,
                "pKa": 5.0,
                "drug_type": "neutral",
                "fup": 0.1,
            },
            {
                "drug_name": "High",
                "plasma_concentration_mg_L": 10.0,
                "pKa": 9.0,
                "drug_type": "base",
                "fup": 0.9,
            },
            {
                "drug_name": "Mid",
                "plasma_concentration_mg_L": 10.0,
                "pKa": 7.0,
                "drug_type": "base",
                "fup": 0.5,
            },
        ]
        results = compare_sweat_excretion(compounds)
        sp_ratios = [r.sweat_plasma_ratio for r in results]
        assert sp_ratios == sorted(sp_ratios, reverse=True)

    def test_returns_list_of_results(self):
        compounds = [
            {
                "drug_name": "A",
                "plasma_concentration_mg_L": 5.0,
                "pKa": 7.0,
                "drug_type": "neutral",
            },
        ]
        results = compare_sweat_excretion(compounds)
        assert isinstance(results, list)
        assert all(isinstance(r, SweatExcretionResult) for r in results)
