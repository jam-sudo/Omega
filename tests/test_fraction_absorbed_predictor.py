"""Tests for Phase 706 — Fraction absorbed predictor."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.fraction_absorbed_predictor import (
    FaResult,
    predict_fa_bcs,
    predict_fa_from_papp,
    predict_fa_from_peff,
    screen_fa,
)


class TestPredictFaFromPeff:
    """Tests for predict_fa_from_peff function."""

    def test_returns_float(self):
        result = predict_fa_from_peff(1e-5)
        assert isinstance(result, float)

    def test_result_in_range_zero_one(self):
        """Result should be in [0, 1]."""
        result = predict_fa_from_peff(1e-5)
        assert 0.0 <= result <= 1.0

    def test_high_peff_near_one(self):
        """Very high Peff should give fa near 1.0."""
        result = predict_fa_from_peff(1e-4)
        assert result >= 0.9

    def test_very_low_peff_near_zero(self):
        """Very low Peff should give fa near 0."""
        result = predict_fa_from_peff(1e-7)
        assert result < 0.1

    def test_higher_peff_higher_fa(self):
        """Higher Peff should always give higher or equal fa."""
        fa_low = predict_fa_from_peff(1e-6)
        fa_high = predict_fa_from_peff(1e-4)
        assert fa_high >= fa_low

    def test_longer_transit_higher_fa(self):
        """Longer transit time should increase fa for same Peff."""
        fa_short = predict_fa_from_peff(1e-5, transit_time_h=1.0)
        fa_long = predict_fa_from_peff(1e-5, transit_time_h=5.0)
        assert fa_long >= fa_short

    def test_clamp_at_one(self):
        """fa should not exceed 1.0 even for very high Peff."""
        result = predict_fa_from_peff(1.0)  # Extremely high
        assert result <= 1.0

    def test_clamp_at_zero(self):
        """fa should not go below 0 even for zero Peff."""
        result = predict_fa_from_peff(0.0)
        assert result >= 0.0


class TestPredictFaFromPapp:
    """Tests for predict_fa_from_papp function."""

    def test_returns_float(self):
        result = predict_fa_from_papp(10.0)
        assert isinstance(result, float)

    def test_result_in_range_zero_one(self):
        """Result should be in [0, 1]."""
        result = predict_fa_from_papp(10.0)
        assert 0.0 <= result <= 1.0

    def test_high_papp_high_fa(self):
        """High Papp (30 x10^-6 cm/s) should give high fa."""
        result = predict_fa_from_papp(30.0)
        assert result > 0.7

    def test_low_papp_low_fa(self):
        """Very low Papp (0.01 x10^-6 cm/s) should give low fa."""
        result = predict_fa_from_papp(0.01)
        assert result < 0.5

    def test_higher_papp_higher_fa(self):
        """Higher Papp should give higher or equal fa."""
        fa_low = predict_fa_from_papp(0.5)
        fa_high = predict_fa_from_papp(50.0)
        assert fa_high >= fa_low


class TestPredictFaBcs:
    """Tests for predict_fa_bcs function."""

    def test_returns_fa_result(self):
        result = predict_fa_bcs(logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=1.0)
        assert isinstance(result, FaResult)

    def test_fa_predicted_in_range(self):
        """fa_predicted should be in [0, 1]."""
        result = predict_fa_bcs(logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=1.0)
        assert 0.0 <= result.fa_predicted <= 1.0

    def test_bcs_class_i_high_fa(self):
        """BCS Class I (high perm, high sol) should give high fa > 0.9."""
        # High logP → high perm; high solubility → Do << 1
        result = predict_fa_bcs(
            logP=3.0,
            mw=200.0,
            psa=30.0,
            solubility_mg_per_mL=10.0,
            dose_mg=50.0,
        )
        assert result.bcs_class == "I"
        assert result.fa_predicted > 0.9

    def test_bcs_class_iv_low_fa(self):
        """BCS Class IV (low perm, low sol) should give low fa < 0.3."""
        # Low logP → low perm; low solubility → Do >> 1
        result = predict_fa_bcs(
            logP=-2.0,
            mw=500.0,
            psa=150.0,
            solubility_mg_per_mL=0.001,
            dose_mg=500.0,
        )
        assert result.bcs_class == "IV"
        assert result.fa_predicted < 0.3

    def test_bcs_class_valid_values(self):
        """BCS class should be one of I, II, III, IV."""
        result = predict_fa_bcs(logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=1.0)
        assert result.bcs_class in {"I", "II", "III", "IV"}

    def test_absorption_class_nonempty(self):
        """Absorption class should not be empty."""
        result = predict_fa_bcs(logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=1.0)
        assert len(result.absorption_class) > 0

    def test_absorption_class_valid_values(self):
        """Absorption class should be one of the expected strings."""
        result = predict_fa_bcs(logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=1.0)
        assert result.absorption_class in {"complete", "high", "moderate", "low", "poor"}

    def test_high_do_low_fa_dissolution(self):
        """Do > 10 should give fa_dissolution < 0.2."""
        # Do = dose / (sol * vol) = 1000 / (0.1 * 250) = 40
        result = predict_fa_bcs(
            logP=2.0,
            mw=300.0,
            psa=60.0,
            solubility_mg_per_mL=0.1,
            dose_mg=1000.0,
            dose_volume_mL=250.0,
        )
        assert result.do_number > 10
        assert result.fa_dissolution < 0.2

    def test_do_number_formula(self):
        """Do number should equal dose / (sol * vol)."""
        result = predict_fa_bcs(
            logP=2.0,
            mw=300.0,
            psa=60.0,
            solubility_mg_per_mL=2.0,
            dose_mg=100.0,
            dose_volume_mL=250.0,
        )
        expected_do = 100.0 / (2.0 * 250.0)
        assert abs(result.do_number - expected_do) < 1e-9

    def test_compound_name_stored(self):
        """Compound name should be stored in result."""
        result = predict_fa_bcs(
            logP=2.0,
            mw=300.0,
            psa=60.0,
            solubility_mg_per_mL=1.0,
            compound_name="TestCompound",
        )
        assert result.compound_name == "TestCompound"

    def test_notes_nonempty(self):
        """Notes field should not be empty."""
        result = predict_fa_bcs(logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=1.0)
        assert len(result.notes) > 0

    def test_validation_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            predict_fa_bcs(logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=1.0, dose_mg=0.0)

    def test_validation_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            predict_fa_bcs(logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=1.0, dose_mg=-10.0)

    def test_validation_solubility_zero_raises(self):
        with pytest.raises(ValueError, match="solubility"):
            predict_fa_bcs(logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=0.0)

    def test_validation_solubility_negative_raises(self):
        with pytest.raises(ValueError, match="solubility"):
            predict_fa_bcs(logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=-1.0)


class TestScreenFa:
    """Tests for screen_fa function."""

    def test_returns_list_of_fa_result(self):
        compounds = [
            {"logP": 2.0, "mw": 300.0, "psa": 60.0, "solubility_mg_per_mL": 1.0},
            {"logP": 3.0, "mw": 200.0, "psa": 40.0, "solubility_mg_per_mL": 5.0},
        ]
        results = screen_fa(compounds)
        assert all(isinstance(r, FaResult) for r in results)

    def test_sorted_descending_by_fa(self):
        """Results should be sorted by fa_predicted descending."""
        compounds = [
            {"logP": -2.0, "mw": 600.0, "psa": 160.0, "solubility_mg_per_mL": 0.01},
            {"logP": 3.0, "mw": 200.0, "psa": 30.0, "solubility_mg_per_mL": 10.0},
            {"logP": 1.0, "mw": 350.0, "psa": 90.0, "solubility_mg_per_mL": 0.5},
        ]
        results = screen_fa(compounds)
        fa_values = [r.fa_predicted for r in results]
        assert fa_values == sorted(fa_values, reverse=True)

    def test_screen_fa_length_matches_input(self):
        """Output length should match input length."""
        compounds = [
            {"logP": 2.0, "mw": 300.0, "psa": 60.0, "solubility_mg_per_mL": 1.0},
            {"logP": 1.5, "mw": 250.0, "psa": 80.0, "solubility_mg_per_mL": 2.0},
            {"logP": 0.5, "mw": 400.0, "psa": 120.0, "solubility_mg_per_mL": 0.5},
        ]
        results = screen_fa(compounds)
        assert len(results) == 3

    def test_screen_fa_uses_compound_name(self):
        """Compound names should be preserved in results."""
        compounds = [
            {
                "logP": 2.0,
                "mw": 300.0,
                "psa": 60.0,
                "solubility_mg_per_mL": 1.0,
                "compound_name": "MyDrug",
            },
        ]
        results = screen_fa(compounds)
        assert results[0].compound_name == "MyDrug"
