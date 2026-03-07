"""Tests for Phase 792: Solubility Predictor p792 variant."""

import pytest

from omega_pbpk.prediction.solubility_predictor_p792 import (
    SolubilityPredictionResult,
    predict_solubility,
    screen_solubility,
)

# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_solubility_prediction_result(self):
        result = predict_solubility("CompA", logp=1.0, mw_da=300.0)
        assert isinstance(result, SolubilityPredictionResult)

    def test_result_is_frozen(self):
        result = predict_solubility("CompA", logp=1.0, mw_da=300.0)
        with pytest.raises((AttributeError, TypeError)):
            result.logp = 99.0  # type: ignore[misc]

    def test_all_fields_present(self):
        result = predict_solubility("CompA", logp=1.0, mw_da=300.0)
        assert result.compound_name == "CompA"
        assert result.logp == 1.0
        assert result.mw_da == 300.0
        assert isinstance(result.solubility_class, str)
        assert isinstance(result.bcs_solubility, str)
        assert isinstance(result.formulation_recommendation, str)
        assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# Core formula tests
# ---------------------------------------------------------------------------


class TestCoreFormulas:
    def test_s_intrinsic_positive(self):
        result = predict_solubility("CompA", logp=1.0, mw_da=300.0)
        assert result.s_intrinsic_mg_L > 0.0

    def test_s_ph7_positive(self):
        result = predict_solubility("CompA", logp=1.0, mw_da=300.0)
        assert result.s_ph7_mg_L > 0.0

    def test_low_logp_higher_solubility_than_high_logp(self):
        """Low logP compound should have higher intrinsic solubility."""
        result_low = predict_solubility("LowLogP", logp=0.5, mw_da=300.0, mp_celsius=150.0)
        result_high = predict_solubility("HighLogP", logp=5.0, mw_da=300.0, mp_celsius=150.0)
        assert result_low.s_intrinsic_mg_L > result_high.s_intrinsic_mg_L

    def test_high_mp_lower_solubility_than_low_mp(self):
        """Higher melting point → lower solubility (same logP)."""
        result_low_mp = predict_solubility("A", logp=2.0, mw_da=300.0, mp_celsius=50.0)
        result_high_mp = predict_solubility("B", logp=2.0, mw_da=300.0, mp_celsius=250.0)
        assert result_low_mp.s_intrinsic_mg_L > result_high_mp.s_intrinsic_mg_L

    def test_gse_formula_at_reference_point(self):
        """At logP=0.5, MP=25 → logS = 0.0 mol/L → S = 1 mol/L."""
        mw = 200.0
        result = predict_solubility("Test", logp=0.5, mw_da=mw, mp_celsius=25.0)
        expected_mg_L = 1.0 * mw * 1000.0  # 200000 mg/L
        assert abs(result.s_intrinsic_mg_L - expected_mg_L) < 1.0

    def test_logS_intrinsic_consistent_with_s(self):
        result = predict_solubility("CompA", logp=2.0, mw_da=300.0)
        expected = 10.0**result.logS_intrinsic
        assert abs(result.s_intrinsic_mg_L - expected) < 0.01


# ---------------------------------------------------------------------------
# pH-dependent solubility tests
# ---------------------------------------------------------------------------


class TestIonization:
    def test_acid_s_ph7_greater_than_intrinsic_when_ph_gt_pka(self):
        """Acid ionizes at pH > pKa → more soluble at pH 7.4 than intrinsic."""
        result = predict_solubility("Acid", logp=2.0, mw_da=300.0, pka=5.0, ionization_type="acid")
        assert result.s_ph7_mg_L > result.s_intrinsic_mg_L

    def test_base_s_ph7_greater_than_intrinsic_when_pka_gt_ph(self):
        """Base pKa=9 > pH=7.4 → ionized → s_ph7 > s_intrinsic."""
        result = predict_solubility("Base", logp=2.0, mw_da=300.0, pka=9.0, ionization_type="base")
        assert result.s_ph7_mg_L > result.s_intrinsic_mg_L

    def test_neutral_s_ph7_equals_intrinsic(self):
        """Neutral compound: pH correction factor = 1."""
        result = predict_solubility("Neutral", logp=2.0, mw_da=300.0, ionization_type="neutral")
        assert abs(result.s_ph7_mg_L - result.s_intrinsic_mg_L) < 1e-9

    def test_acid_at_exact_pka_factor_two(self):
        """At pH == pKa, factor = 1 + 10^0 = 2 → s_ph7 = 2 * s_intrinsic."""
        result = predict_solubility(
            "AcidExact", logp=1.0, mw_da=200.0, mp_celsius=100.0, pka=7.4, ionization_type="acid"
        )
        assert abs(result.s_ph7_mg_L - 2.0 * result.s_intrinsic_mg_L) < 0.01


# ---------------------------------------------------------------------------
# Solubility class tests
# ---------------------------------------------------------------------------


class TestSolubilityClass:
    def test_high_class_above_100_mg_L(self):
        """logP=-2, MP=25: very high solubility."""
        result = predict_solubility("HighSol", logp=-2.0, mw_da=100.0, mp_celsius=25.0)
        assert result.solubility_class == "high"

    def test_very_low_class_high_logp(self):
        """Very high logP → very low solubility."""
        result = predict_solubility("VeryLow", logp=8.0, mw_da=500.0, mp_celsius=200.0)
        assert result.solubility_class == "very_low"

    def test_solubility_class_consistent_with_value(self):
        result = predict_solubility("CompA", logp=2.0, mw_da=300.0)
        s = result.s_ph7_mg_L
        if s >= 100.0:
            assert result.solubility_class == "high"
        elif s >= 10.0:
            assert result.solubility_class == "moderate"
        elif s >= 1.0:
            assert result.solubility_class == "low"
        else:
            assert result.solubility_class == "very_low"


# ---------------------------------------------------------------------------
# BCS and dose number tests
# ---------------------------------------------------------------------------


class TestBCSAndDoseNumber:
    def test_bcs_high_when_dose_number_lt_1(self):
        """Very soluble compound: Do < 1 → BCS_high."""
        result = predict_solubility("HighSol", logp=-3.0, mw_da=100.0, mp_celsius=25.0, dose_mg=1.0)
        assert result.bcs_solubility == "BCS_high"
        assert result.dose_number < 1.0

    def test_bcs_low_when_dose_number_ge_1(self):
        """Poorly soluble compound: Do >= 1 → BCS_low."""
        result = predict_solubility(
            "PoorSol", logp=6.0, mw_da=400.0, mp_celsius=200.0, dose_mg=100.0
        )
        assert result.bcs_solubility == "BCS_low"
        assert result.dose_number >= 1.0

    def test_dose_number_formula(self):
        """Do = dose_mg / (s_ph7_mg_L * 0.250 L)."""
        result = predict_solubility("CompA", logp=2.0, mw_da=300.0, dose_mg=50.0)
        expected_do = 50.0 / (result.s_ph7_mg_L * 0.250)
        assert abs(result.dose_number - expected_do) < 1e-9


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_mw_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="mw_da"):
            predict_solubility("CompA", logp=1.0, mw_da=0.0)

    def test_mw_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="mw_da"):
            predict_solubility("CompA", logp=1.0, mw_da=-100.0)


# ---------------------------------------------------------------------------
# Screen function tests
# ---------------------------------------------------------------------------


class TestScreenSolubility:
    def test_screen_returns_list(self):
        compounds = [
            {"compound_name": "A", "logp": 1.0, "mw_da": 200.0},
            {"compound_name": "B", "logp": 3.0, "mw_da": 300.0},
        ]
        results = screen_solubility(compounds)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_screen_sorted_descending(self):
        compounds = [
            {"compound_name": "LowSol", "logp": 5.0, "mw_da": 400.0, "mp_celsius": 200.0},
            {"compound_name": "HighSol", "logp": 0.0, "mw_da": 150.0, "mp_celsius": 50.0},
        ]
        results = screen_solubility(compounds)
        assert results[0].s_intrinsic_mg_L >= results[1].s_intrinsic_mg_L

    def test_screen_empty_list(self):
        results = screen_solubility([])
        assert results == []

    def test_screen_returns_correct_type(self):
        compounds = [{"compound_name": "X", "logp": 2.0, "mw_da": 300.0}]
        results = screen_solubility(compounds)
        assert all(isinstance(r, SolubilityPredictionResult) for r in results)

    def test_screen_three_compounds_all_returned(self):
        compounds = [
            {"compound_name": "A", "logp": 1.0, "mw_da": 200.0},
            {"compound_name": "B", "logp": 2.0, "mw_da": 300.0},
            {"compound_name": "C", "logp": 4.0, "mw_da": 400.0},
        ]
        results = screen_solubility(compounds)
        assert len(results) == 3

    def test_screen_optional_fields(self):
        compounds = [
            {
                "compound_name": "WithOpts",
                "logp": 2.0,
                "mw_da": 250.0,
                "mp_celsius": 100.0,
                "pka": 5.0,
                "ionization_type": "acid",
                "dose_mg": 50.0,
            }
        ]
        results = screen_solubility(compounds)
        assert len(results) == 1
        assert results[0].compound_name == "WithOpts"
