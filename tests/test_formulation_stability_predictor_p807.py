"""Tests for Phase 808 — Formulation Stability Predictor."""

import pytest

from omega_pbpk.prediction.formulation_stability_predictor_p807 import (
    FormulationStabilityResult,
    predict_formulation_stability,
    screen_formulation_stability,
)


class TestReturnType:
    def test_returns_correct_type(self):
        result = predict_formulation_stability("TestDrug")
        assert isinstance(result, FormulationStabilityResult)

    def test_frozen_dataclass(self):
        result = predict_formulation_stability("TestDrug")
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "other"  # type: ignore[misc]

    def test_drug_name_preserved(self):
        result = predict_formulation_stability("Aspirin")
        assert result.drug_name == "Aspirin"

    def test_formulation_type_preserved(self):
        result = predict_formulation_stability("Drug", formulation_type="solution")
        assert result.formulation_type == "solution"

    def test_storage_condition_preserved(self):
        result = predict_formulation_stability("Drug", storage_condition="2-8C")
        assert result.storage_condition == "2-8C"


class TestPositiveValues:
    def test_t90_positive(self):
        result = predict_formulation_stability("Drug")
        assert result.t90_months > 0.0

    def test_degradation_rate_positive(self):
        result = predict_formulation_stability("Drug")
        assert result.degradation_rate_per_month > 0.0

    def test_2yr_potency_in_range(self):
        result = predict_formulation_stability("Drug")
        assert 0.0 <= result.predicted_2yr_potency_pct <= 100.0

    def test_2yr_potency_in_range_unstable(self):
        result = predict_formulation_stability(
            "Drug",
            has_ester=True,
            has_double_bond=True,
            has_phenol=True,
            formulation_type="solution",
            storage_condition="40C_75RH",
        )
        assert 0.0 <= result.predicted_2yr_potency_pct <= 100.0


class TestHydrolysisRisk:
    def test_ester_solution_high_hydrolysis(self):
        result = predict_formulation_stability(
            "Drug",
            has_ester=True,
            formulation_type="solution",
        )
        assert result.hydrolysis_risk == "high"

    def test_ester_high_water_activity_high_hydrolysis(self):
        result = predict_formulation_stability(
            "Drug",
            has_ester=True,
            water_activity=0.8,
        )
        assert result.hydrolysis_risk == "high"

    def test_amide_moderate_hydrolysis(self):
        result = predict_formulation_stability(
            "Drug",
            has_amide=True,
            formulation_type="tablet",
            water_activity=0.3,
        )
        assert result.hydrolysis_risk == "moderate"

    def test_no_alerts_low_hydrolysis(self):
        result = predict_formulation_stability("Drug", formulation_type="tablet")
        assert result.hydrolysis_risk == "low"


class TestOxidationRisk:
    def test_double_bond_tablet_high_oxidation(self):
        result = predict_formulation_stability(
            "Drug",
            has_double_bond=True,
            formulation_type="tablet",
        )
        assert result.oxidation_risk == "high"

    def test_phenol_solution_high_oxidation(self):
        result = predict_formulation_stability(
            "Drug",
            has_phenol=True,
            formulation_type="solution",
        )
        assert result.oxidation_risk == "high"

    def test_no_alerts_low_oxidation(self):
        result = predict_formulation_stability("Drug", formulation_type="lyophilized")
        assert result.oxidation_risk == "low"


class TestStorageConditionEffect:
    def test_frozen_longer_t90_than_40C(self):
        r_frozen = predict_formulation_stability("Drug", storage_condition="frozen")
        r_hot = predict_formulation_stability("Drug", storage_condition="40C_75RH")
        assert r_frozen.t90_months > r_hot.t90_months

    def test_cold_longer_t90_than_room_temp(self):
        r_cold = predict_formulation_stability("Drug", storage_condition="2-8C")
        r_room = predict_formulation_stability("Drug", storage_condition="25C_60RH")
        assert r_cold.t90_months > r_room.t90_months

    def test_frozen_longest_t90(self):
        r_frozen = predict_formulation_stability("Drug", storage_condition="frozen")
        r_25 = predict_formulation_stability("Drug", storage_condition="25C_60RH")
        r_40 = predict_formulation_stability("Drug", storage_condition="40C_75RH")
        assert r_frozen.t90_months > r_25.t90_months
        assert r_25.t90_months > r_40.t90_months


class TestOverallStabilityClass:
    def test_excellent_for_stable_frozen(self):
        result = predict_formulation_stability(
            "Drug",
            storage_condition="frozen",
            formulation_type="lyophilized",
        )
        # t90 = 60 * 3.0 = 180 months → excellent
        assert result.overall_stability_class == "excellent"

    def test_poor_for_very_unstable(self):
        # All 4 risks high: ester+solution=hyd high, double_bond+solution=ox high,
        # double_bond=photo high, aldehyde forces ox high, physical=low for solution
        # Need all 4 high: use emulsion+logp<0 for phys high, aldehyde for ox high
        # hyd: ester+water_activity=0.8 -> high (-12)
        # ox: aldehyde forces high (-12)
        # photo: double_bond -> high (-12)
        # phys: emulsion+logp<0 -> high (-12)
        # t90_base = 60 - 48 = 12; 40C_75RH: 12 * 0.4 = 4.8 < 6 -> poor
        result = predict_formulation_stability(
            "Drug",
            has_ester=True,
            has_aldehyde=True,
            has_double_bond=True,
            formulation_type="emulsion",
            storage_condition="40C_75RH",
            logp=-1.0,
            water_activity=0.8,
        )
        assert result.overall_stability_class == "poor"

    def test_stability_class_valid_values(self):
        result = predict_formulation_stability("Drug")
        assert result.overall_stability_class in ("excellent", "good", "marginal", "poor")


class TestScreenFormulation:
    def test_returns_list(self):
        conditions = [
            {"storage_condition": "25C_60RH"},
            {"storage_condition": "40C_75RH"},
            {"storage_condition": "frozen"},
        ]
        results = screen_formulation_stability("Drug", conditions)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_sorted_by_t90_descending(self):
        conditions = [
            {"storage_condition": "40C_75RH"},
            {"storage_condition": "25C_60RH"},
            {"storage_condition": "frozen"},
        ]
        results = screen_formulation_stability("Drug", conditions)
        t90s = [r.t90_months for r in results]
        assert t90s == sorted(t90s, reverse=True)

    def test_all_results_same_drug(self):
        conditions = [
            {"storage_condition": "25C_60RH"},
            {"storage_condition": "2-8C"},
        ]
        results = screen_formulation_stability("Ibuprofen", conditions)
        assert all(r.drug_name == "Ibuprofen" for r in results)


class TestValidation:
    def test_water_activity_negative_raises_error(self):
        with pytest.raises(ValueError):
            predict_formulation_stability("Drug", water_activity=-0.1)

    def test_water_activity_above_1_raises_error(self):
        with pytest.raises(ValueError):
            predict_formulation_stability("Drug", water_activity=1.1)

    def test_mw_da_zero_raises_error(self):
        with pytest.raises(ValueError):
            predict_formulation_stability("Drug", mw_da=0.0)

    def test_mw_da_negative_raises_error(self):
        with pytest.raises(ValueError):
            predict_formulation_stability("Drug", mw_da=-100.0)

    def test_invalid_formulation_type_raises_error(self):
        with pytest.raises(ValueError):
            predict_formulation_stability("Drug", formulation_type="patch")

    def test_invalid_storage_condition_raises_error(self):
        with pytest.raises(ValueError):
            predict_formulation_stability("Drug", storage_condition="room_temp")
