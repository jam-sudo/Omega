"""Tests for Phase 488 — Formulation Stability Predictor."""

import pytest

from omega_pbpk.prediction.formulation_stability_predictor import (
    FormulationStabilityResult,
    assess_formulation_stability,
    predict_hydrolysis_rate,
    predict_oxidation_risk,
    predict_photodegradation,
)

# ---------------------------------------------------------------------------
# predict_hydrolysis_rate
# ---------------------------------------------------------------------------


class TestPredictHydrolysisRate:
    def test_basic_returns_float(self):
        rate = predict_hydrolysis_rate(7.0, 7.0)
        assert isinstance(rate, float)
        assert rate > 0

    def test_extreme_low_ph_higher_rate(self):
        rate_neutral = predict_hydrolysis_rate(7.0, 7.0)
        rate_acid = predict_hydrolysis_rate(7.0, 1.0)
        assert rate_acid > rate_neutral

    def test_extreme_high_ph_higher_rate(self):
        rate_neutral = predict_hydrolysis_rate(7.0, 7.0)
        rate_base = predict_hydrolysis_rate(7.0, 11.0)
        assert rate_base > rate_neutral

    def test_ph_2_greater_than_ph_4(self):
        rate_low = predict_hydrolysis_rate(7.0, 2.0)
        rate_high = predict_hydrolysis_rate(7.0, 4.0)
        assert rate_low > rate_high

    def test_higher_temperature_faster_degradation(self):
        rate_25 = predict_hydrolysis_rate(7.0, 7.0, temperature_C=25.0)
        rate_40 = predict_hydrolysis_rate(7.0, 7.0, temperature_C=40.0)
        assert rate_40 > rate_25

    def test_arrhenius_increases_with_temperature(self):
        rate_25 = predict_hydrolysis_rate(7.0, 5.0, temperature_C=25.0)
        rate_60 = predict_hydrolysis_rate(7.0, 5.0, temperature_C=60.0)
        assert rate_60 > rate_25 * 5  # significantly faster

    def test_invalid_ph_zero(self):
        with pytest.raises(ValueError):
            predict_hydrolysis_rate(7.0, 0.0)

    def test_invalid_ph_negative(self):
        with pytest.raises(ValueError):
            predict_hydrolysis_rate(7.0, -1.0)

    def test_invalid_temperature_zero(self):
        with pytest.raises(ValueError):
            predict_hydrolysis_rate(7.0, 7.0, temperature_C=0.0)

    def test_neutral_ph_minimum_rate(self):
        rate_neutral = predict_hydrolysis_rate(7.0, 7.0)
        rate_acid = predict_hydrolysis_rate(7.0, 2.0)
        rate_base = predict_hydrolysis_rate(7.0, 12.0)
        assert rate_neutral < rate_acid
        assert rate_neutral < rate_base


# ---------------------------------------------------------------------------
# predict_oxidation_risk
# ---------------------------------------------------------------------------


class TestPredictOxidationRisk:
    def test_low_risk_profile(self):
        risk = predict_oxidation_risk(logP=0.5, n_double_bonds=0)
        assert risk == "low"

    def test_high_logP_and_sulfur_high_risk(self):
        risk = predict_oxidation_risk(logP=4.0, n_double_bonds=0, has_sulfur=True)
        assert risk == "high"

    def test_many_double_bonds_high_risk(self):
        risk = predict_oxidation_risk(logP=0.0, n_double_bonds=5)
        assert risk == "high"

    def test_sulfur_increases_risk(self):
        risk_no_s = predict_oxidation_risk(logP=2.0, n_double_bonds=1, has_sulfur=False)
        risk_s = predict_oxidation_risk(logP=2.0, n_double_bonds=1, has_sulfur=True)
        risk_levels = {"low": 0, "moderate": 1, "high": 2}
        assert risk_levels[risk_s] >= risk_levels[risk_no_s]

    def test_phenol_increases_risk(self):
        risk_no_ph = predict_oxidation_risk(logP=1.0, n_double_bonds=0, n_phenol=0)
        risk_ph = predict_oxidation_risk(logP=1.0, n_double_bonds=0, n_phenol=2)
        risk_levels = {"low": 0, "moderate": 1, "high": 2}
        assert risk_levels[risk_ph] >= risk_levels[risk_no_ph]

    def test_returns_valid_category(self):
        risk = predict_oxidation_risk(3.0, 2, True, 1)
        assert risk in ("low", "moderate", "high")

    def test_moderate_logP_moderate_risk_possible(self):
        risk = predict_oxidation_risk(logP=2.0, n_double_bonds=1)
        assert risk in ("moderate", "high")


# ---------------------------------------------------------------------------
# predict_photodegradation
# ---------------------------------------------------------------------------


class TestPredictPhotodegradation:
    def test_zero_chromophores_zero_degradation(self):
        frac = predict_photodegradation(0, uv_exposure_h=100.0)
        assert frac == 0.0

    def test_more_chromophores_more_degradation(self):
        frac1 = predict_photodegradation(1, uv_exposure_h=100.0)
        frac5 = predict_photodegradation(5, uv_exposure_h=100.0)
        assert frac5 > frac1

    def test_longer_exposure_more_degradation(self):
        frac_short = predict_photodegradation(3, uv_exposure_h=10.0)
        frac_long = predict_photodegradation(3, uv_exposure_h=500.0)
        assert frac_long > frac_short

    def test_fraction_bounded_0_to_1(self):
        frac = predict_photodegradation(100, uv_exposure_h=10000.0)
        assert 0.0 <= frac <= 1.0

    def test_zero_exposure_zero_degradation(self):
        frac = predict_photodegradation(5, uv_exposure_h=0.0)
        assert frac == pytest.approx(0.0, abs=1e-9)

    def test_invalid_negative_chromophores(self):
        with pytest.raises(ValueError):
            predict_photodegradation(-1)

    def test_invalid_negative_exposure(self):
        with pytest.raises(ValueError):
            predict_photodegradation(3, uv_exposure_h=-10.0)


# ---------------------------------------------------------------------------
# assess_formulation_stability
# ---------------------------------------------------------------------------


class TestAssessFormulationStability:
    def _stable_result(self):
        return assess_formulation_stability(
            drug_name="StableDrug",
            ph=7.0,
            temperature_C=25.0,
            humidity_pct=40.0,
        )

    def test_returns_correct_type(self):
        result = self._stable_result()
        assert isinstance(result, FormulationStabilityResult)

    def test_frozen_dataclass(self):
        result = self._stable_result()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "other"  # type: ignore[misc]

    def test_stable_conditions_good_or_excellent_stability(self):
        result = self._stable_result()
        assert result.overall_stability in ("excellent", "good")

    def test_high_humidity_high_humidity_risk(self):
        result = assess_formulation_stability("Drug", ph=7.0, temperature_C=25.0, humidity_pct=80.0)
        assert result.humidity_risk == "high"

    def test_low_humidity_low_humidity_risk(self):
        result = assess_formulation_stability("Drug", ph=7.0, temperature_C=25.0, humidity_pct=20.0)
        assert result.humidity_risk == "low"

    def test_extreme_ph_high_hydrolysis_risk(self):
        result = assess_formulation_stability("Drug", ph=1.0, temperature_C=25.0, humidity_pct=40.0)
        assert result.hydrolysis_risk == "high"

    def test_high_logP_sulfur_high_oxidation(self):
        result = assess_formulation_stability(
            "Drug", ph=7.0, temperature_C=25.0, humidity_pct=40.0, logP=4.0, has_sulfur=True
        )
        assert result.oxidation_risk == "high"

    def test_poor_stability_multiple_high_risks(self):
        result = assess_formulation_stability(
            "Drug",
            ph=1.0,
            temperature_C=40.0,
            humidity_pct=85.0,
            logP=4.0,
            n_double_bonds=4,
            has_sulfur=True,
        )
        assert result.overall_stability == "poor"

    def test_shelf_life_positive(self):
        result = self._stable_result()
        assert result.predicted_shelf_life_months > 0

    def test_high_temperature_reduces_shelf_life(self):
        result_25 = assess_formulation_stability(
            "Drug", ph=7.0, temperature_C=25.0, humidity_pct=40.0
        )
        result_40 = assess_formulation_stability(
            "Drug", ph=7.0, temperature_C=40.0, humidity_pct=40.0
        )
        assert result_40.predicted_shelf_life_months < result_25.predicted_shelf_life_months

    def test_recommended_storage_not_empty(self):
        result = self._stable_result()
        assert isinstance(result.recommended_storage, str)
        assert len(result.recommended_storage) > 0

    def test_notes_contains_ph(self):
        result = self._stable_result()
        assert "pH" in result.notes or "ph" in result.notes.lower()

    def test_drug_name_preserved(self):
        result = assess_formulation_stability(
            "Ibuprofen", ph=6.5, temperature_C=25.0, humidity_pct=50.0
        )
        assert result.drug_name == "Ibuprofen"

    def test_invalid_ph_zero(self):
        with pytest.raises(ValueError):
            assess_formulation_stability("Drug", ph=0.0, temperature_C=25.0, humidity_pct=40.0)

    def test_invalid_temperature_zero(self):
        with pytest.raises(ValueError):
            assess_formulation_stability("Drug", ph=7.0, temperature_C=0.0, humidity_pct=40.0)

    def test_invalid_humidity_gt_100(self):
        with pytest.raises(ValueError):
            assess_formulation_stability("Drug", ph=7.0, temperature_C=25.0, humidity_pct=110.0)

    def test_invalid_humidity_negative(self):
        with pytest.raises(ValueError):
            assess_formulation_stability("Drug", ph=7.0, temperature_C=25.0, humidity_pct=-5.0)

    def test_overall_stability_valid_category(self):
        result = self._stable_result()
        assert result.overall_stability in ("excellent", "good", "fair", "poor")

    def test_hydrolysis_risk_valid_category(self):
        result = self._stable_result()
        assert result.hydrolysis_risk in ("low", "moderate", "high")

    def test_oxidation_risk_valid_category(self):
        result = self._stable_result()
        assert result.oxidation_risk in ("low", "moderate", "high")
