"""Tests for Phase 642 — Drug Dermal Absorption from Topical Formulation."""

import pytest

from omega_pbpk.prediction.dermal_absorption_p642 import (
    _KP_MAX_CM_H,
    _KP_MIN_CM_H,
    _VALID_FORMULATIONS,
    _VEHICLE_EFFECT_FACTOR,
    DermalAbsorptionResult,
    _potts_guy_kp,
    compare_formulations,
    predict_dermal_absorption,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cream_result():
    return predict_dermal_absorption(
        drug_name="TestDrug",
        dose_applied_mg=500.0,
        application_area_cm2=100.0,
        logP=2.0,
        mw_Da=250.0,
        formulation_type="cream",
        vehicle_conc_mg_mL=10.0,
        t_exposure_h=8.0,
    )


@pytest.fixture
def ointment_result():
    return predict_dermal_absorption(
        drug_name="TestDrug",
        dose_applied_mg=500.0,
        application_area_cm2=100.0,
        logP=2.0,
        mw_Da=250.0,
        formulation_type="ointment",
        vehicle_conc_mg_mL=10.0,
        t_exposure_h=8.0,
    )


@pytest.fixture
def lotion_result():
    return predict_dermal_absorption(
        drug_name="TestDrug",
        dose_applied_mg=500.0,
        application_area_cm2=100.0,
        logP=2.0,
        mw_Da=250.0,
        formulation_type="lotion",
        vehicle_conc_mg_mL=10.0,
        t_exposure_h=8.0,
    )


# ---------------------------------------------------------------------------
# Return type and frozen dataclass
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dermal_absorption_result(self, cream_result):
        assert isinstance(cream_result, DermalAbsorptionResult)

    def test_frozen_dataclass_immutable(self, cream_result):
        with pytest.raises((AttributeError, TypeError)):
            cream_result.drug_name = "Other"  # type: ignore[misc]

    def test_drug_name_stored(self, cream_result):
        assert cream_result.drug_name == "TestDrug"

    def test_formulation_type_stored(self, cream_result):
        assert cream_result.formulation_type == "cream"

    def test_logP_stored(self, cream_result):
        assert cream_result.logP == pytest.approx(2.0)

    def test_mw_Da_stored(self, cream_result):
        assert cream_result.mw_Da == pytest.approx(250.0)

    def test_notes_is_string(self, cream_result):
        assert isinstance(cream_result.notes, str)


# ---------------------------------------------------------------------------
# Potts-Guy model
# ---------------------------------------------------------------------------


class TestPottsGuy:
    def test_kp_positive(self):
        kp = _potts_guy_kp(2.0, 250.0)
        assert kp > 0.0

    def test_kp_clamped_min(self):
        # Very large MW and negative logP → tiny kp → clamped to min
        kp = _potts_guy_kp(-5.0, 2000.0)
        assert kp >= _KP_MIN_CM_H

    def test_kp_clamped_max(self):
        # Very high logP → large kp → clamped to max
        kp = _potts_guy_kp(10.0, 0.0)
        assert kp <= _KP_MAX_CM_H

    def test_kp_higher_logP_higher_kp(self):
        kp_low = _potts_guy_kp(0.0, 300.0)
        kp_high = _potts_guy_kp(3.0, 300.0)
        assert kp_high > kp_low

    def test_kp_higher_mw_lower_kp(self):
        kp_light = _potts_guy_kp(2.0, 200.0)
        kp_heavy = _potts_guy_kp(2.0, 500.0)
        assert kp_light > kp_heavy

    def test_kp_formula_example(self):
        # log(kp) = -2.72 + 0.71*2 - 0.0061*250 = -2.72 + 1.42 - 1.525 = -2.825
        expected = 10 ** (-2.72 + 0.71 * 2.0 - 0.0061 * 250.0)
        expected = max(_KP_MIN_CM_H, min(_KP_MAX_CM_H, expected))
        assert _potts_guy_kp(2.0, 250.0) == pytest.approx(expected, rel=1e-9)

    def test_kp_skin_stored_in_result(self, cream_result):
        expected = _potts_guy_kp(2.0, 250.0)
        assert cream_result.kp_skin_cm_h == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Vehicle enhancement factors
# ---------------------------------------------------------------------------


class TestVehicleFactors:
    def test_ointment_highest_vehicle_factor(self):
        # ointment has factor 3.0
        assert _VEHICLE_EFFECT_FACTOR["ointment"] == 3.0

    def test_lotion_factor_unity(self):
        assert _VEHICLE_EFFECT_FACTOR["lotion"] == 1.0

    def test_ointment_absorbs_more_than_lotion(self, ointment_result, lotion_result):
        assert ointment_result.daily_absorbed_mg > lotion_result.daily_absorbed_mg

    def test_vehicle_factor_stored_in_result(self, cream_result):
        assert cream_result.vehicle_effect_factor == pytest.approx(1.5)

    def test_all_formulations_have_factor(self):
        for ft in _VALID_FORMULATIONS:
            assert ft in _VEHICLE_EFFECT_FACTOR


# ---------------------------------------------------------------------------
# Occlusion effect
# ---------------------------------------------------------------------------


class TestOcclusionEffect:
    def test_occlusion_triples_absorption(self):
        r_open = predict_dermal_absorption("D", 500.0, 100.0, 2.0, 250.0, "lotion", occluded=False)
        r_occluded = predict_dermal_absorption(
            "D", 500.0, 100.0, 2.0, 250.0, "lotion", occluded=True
        )
        assert r_occluded.daily_absorbed_mg == pytest.approx(
            3.0 * r_open.daily_absorbed_mg, rel=1e-9
        )

    def test_occlusion_factor_open(self, lotion_result):
        assert lotion_result.occlusion_factor == pytest.approx(1.0)

    def test_occlusion_factor_occluded(self):
        r = predict_dermal_absorption("D", 500.0, 100.0, 2.0, 250.0, "gel", occluded=True)
        assert r.occlusion_factor == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Daily absorption and exposure
# ---------------------------------------------------------------------------


class TestAbsorptionAndExposure:
    def test_daily_absorbed_positive(self, cream_result):
        assert cream_result.daily_absorbed_mg > 0.0

    def test_systemic_exposure_pct_in_0_100(self, cream_result):
        assert 0.0 <= cream_result.systemic_exposure_pct <= 100.0

    def test_systemic_exposure_pct_calculation(self, cream_result):
        # daily_absorbed / dose * 100
        expected = min(100.0, cream_result.daily_absorbed_mg / 500.0 * 100.0)
        assert cream_result.systemic_exposure_pct == pytest.approx(expected, rel=1e-9)

    def test_larger_area_more_absorbed(self):
        r_small = predict_dermal_absorption("D", 500.0, 50.0, 2.0, 250.0, "cream")
        r_large = predict_dermal_absorption("D", 500.0, 200.0, 2.0, 250.0, "cream")
        assert r_large.daily_absorbed_mg > r_small.daily_absorbed_mg

    def test_t_exposure_below_24h(self):
        # absorption scales with t_exposure when t < 24
        r_4h = predict_dermal_absorption("D", 500.0, 100.0, 2.0, 250.0, "cream", t_exposure_h=4.0)
        r_8h = predict_dermal_absorption("D", 500.0, 100.0, 2.0, 250.0, "cream", t_exposure_h=8.0)
        assert r_8h.daily_absorbed_mg == pytest.approx(2.0 * r_4h.daily_absorbed_mg, rel=1e-9)

    def test_t_exposure_above_24h_capped(self):
        r_24h = predict_dermal_absorption("D", 500.0, 100.0, 2.0, 250.0, "cream", t_exposure_h=24.0)
        r_48h = predict_dermal_absorption("D", 500.0, 100.0, 2.0, 250.0, "cream", t_exposure_h=48.0)
        assert r_24h.daily_absorbed_mg == pytest.approx(r_48h.daily_absorbed_mg, rel=1e-9)


# ---------------------------------------------------------------------------
# Lag time and time to steady state
# ---------------------------------------------------------------------------


class TestLagTime:
    def test_time_to_ss_positive(self, cream_result):
        assert cream_result.time_to_steady_state_h > 0.0

    def test_time_to_ss_three_lag_times(self):
        # time_to_ss = t_lag * 3
        logP, mw_Da = 2.0, 250.0
        kp_skin = _potts_guy_kp(logP, mw_Da)
        t_lag_raw = mw_Da * 0.001 + 1.0 / (kp_skin + 0.01)
        t_lag = max(0.1, min(24.0, t_lag_raw))
        expected_tss = t_lag * 3.0
        r = predict_dermal_absorption("D", 500.0, 100.0, logP, mw_Da, "cream")
        assert r.time_to_steady_state_h == pytest.approx(expected_tss, rel=1e-9)


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------


class TestRiskClassification:
    def test_negligible_risk_small_exposure(self):
        # Very low dose/area → negligible
        r = predict_dermal_absorption(
            "D", 10000.0, 1.0, -2.0, 500.0, "lotion", vehicle_conc_mg_mL=0.01
        )
        assert r.risk_of_systemic_effect == "negligible"

    def test_risk_is_string(self, cream_result):
        assert isinstance(cream_result.risk_of_systemic_effect, str)

    def test_risk_values_valid(self):
        valid_risks = {"negligible", "low", "moderate", "high"}
        for ft in _VALID_FORMULATIONS:
            r = predict_dermal_absorption("D", 500.0, 100.0, 2.0, 250.0, ft)
            assert r.risk_of_systemic_effect in valid_risks


# ---------------------------------------------------------------------------
# Compare formulations
# ---------------------------------------------------------------------------


class TestCompareFormulations:
    @pytest.fixture
    def comparison(self):
        return compare_formulations("D", 500.0, 100.0, 2.0, 250.0)

    def test_compare_returns_list(self, comparison):
        assert isinstance(comparison, list)

    def test_compare_returns_five_results(self, comparison):
        assert len(comparison) == 5

    def test_compare_all_dermal_results(self, comparison):
        for r in comparison:
            assert isinstance(r, DermalAbsorptionResult)

    def test_compare_sorted_descending(self, comparison):
        absorbed = [r.daily_absorbed_mg for r in comparison]
        assert absorbed == sorted(absorbed, reverse=True)

    def test_compare_covers_all_formulations(self, comparison):
        types = {r.formulation_type for r in comparison}
        assert types == _VALID_FORMULATIONS


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_applied_mg"):
            predict_dermal_absorption("D", 0.0, 100.0, 2.0, 250.0, "cream")

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_applied_mg"):
            predict_dermal_absorption("D", -1.0, 100.0, 2.0, 250.0, "cream")

    def test_invalid_area_raises(self):
        with pytest.raises(ValueError, match="application_area_cm2"):
            predict_dermal_absorption("D", 500.0, 0.0, 2.0, 250.0, "cream")

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_dermal_absorption("D", 500.0, 100.0, 2.0, -1.0, "cream")

    def test_invalid_formulation_raises(self):
        with pytest.raises(ValueError, match="formulation_type"):
            predict_dermal_absorption("D", 500.0, 100.0, 2.0, 250.0, "spray")

    def test_invalid_vehicle_conc_raises(self):
        with pytest.raises(ValueError, match="vehicle_conc_mg_mL"):
            predict_dermal_absorption("D", 500.0, 100.0, 2.0, 250.0, "gel", vehicle_conc_mg_mL=0.0)

    def test_all_valid_formulations_accepted(self):
        for ft in _VALID_FORMULATIONS:
            r = predict_dermal_absorption("D", 500.0, 100.0, 2.0, 250.0, ft)
            assert isinstance(r, DermalAbsorptionResult)
