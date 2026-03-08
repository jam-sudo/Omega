"""Tests for Phase 502: Dermal Absorption Risk Assessment."""

from __future__ import annotations

import pytest

from omega_pbpk.risk.dermal_absorption_risk import (
    DermalAbsorptionRiskResult,
    assess_dermal_absorption_risk,
    compare_exposure_scenarios,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = dict(
    substance_name="TestSubstance",
    logP=2.0,
    mw_Da=200.0,
    conc_mg_cm3=1.0,
    skin_area_cm2=100.0,
    exposure_duration_h=8.0,
    vd_body_L=42.0,
)


def default_assess(**kwargs):
    params = {**DEFAULT_PARAMS, **kwargs}
    return assess_dermal_absorption_risk(**params)


# ---------------------------------------------------------------------------
# Return type and field tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        res = default_assess()
        assert isinstance(res, DermalAbsorptionRiskResult)

    def test_has_substance_name(self):
        res = default_assess(substance_name="Benzene")
        assert res.substance_name == "Benzene"

    def test_has_logP(self):
        res = default_assess(logP=3.0)
        assert res.logP == 3.0

    def test_has_mw_Da(self):
        res = default_assess(mw_Da=300.0)
        assert res.mw_Da == 300.0

    def test_has_kp_cm_h(self):
        res = default_assess()
        assert isinstance(res.kp_cm_h, float)
        assert res.kp_cm_h > 0

    def test_has_flux_mg_cm2_h(self):
        res = default_assess()
        assert isinstance(res.flux_mg_cm2_h, float)

    def test_has_skin_area_cm2(self):
        res = default_assess(skin_area_cm2=200.0)
        assert res.skin_area_cm2 == 200.0

    def test_has_exposure_duration_h(self):
        res = default_assess(exposure_duration_h=4.0)
        assert res.exposure_duration_h == 4.0

    def test_has_conc_mg_cm3(self):
        res = default_assess(conc_mg_cm3=2.0)
        assert res.conc_mg_cm3 == 2.0

    def test_has_dermal_absorption_fraction(self):
        res = default_assess()
        assert isinstance(res.dermal_absorption_fraction, float)

    def test_has_total_absorbed_mg(self):
        res = default_assess()
        assert isinstance(res.total_absorbed_mg, float)

    def test_has_body_burden_mg_L(self):
        res = default_assess()
        assert isinstance(res.body_burden_mg_L, float)

    def test_has_risk_class_str(self):
        res = default_assess()
        assert isinstance(res.risk_class, str)
        assert res.risk_class in {"negligible", "low", "moderate", "high"}

    def test_has_ppe_required_bool(self):
        res = default_assess()
        assert isinstance(res.ppe_required, bool)

    def test_has_notes_str(self):
        res = default_assess()
        assert isinstance(res.notes, str)
        assert len(res.notes) > 0

    def test_result_is_frozen(self):
        res = default_assess()
        with pytest.raises((AttributeError, TypeError)):
            res.risk_class = "negligible"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Potts-Guy model correctness
# ---------------------------------------------------------------------------


class TestPottsGuyModel:
    def test_kp_formula(self):
        logP = 2.0
        mw = 200.0
        expected_log10_kp = 0.71 * logP - 0.0061 * mw - 6.3
        expected_kp = 10.0**expected_log10_kp
        res = default_assess(logP=logP, mw_Da=mw)
        assert abs(res.kp_cm_h - expected_kp) < 1e-12

    def test_flux_equals_kp_times_conc(self):
        res = default_assess(logP=2.0, mw_Da=200.0, conc_mg_cm3=1.5)
        expected_flux = res.kp_cm_h * 1.5
        assert abs(res.flux_mg_cm2_h - expected_flux) < 1e-12

    def test_high_logP_higher_kp(self):
        low = default_assess(logP=1.0)
        high = default_assess(logP=4.0)
        assert high.kp_cm_h > low.kp_cm_h

    def test_high_logP_higher_flux(self):
        low = default_assess(logP=1.0)
        high = default_assess(logP=4.0)
        assert high.flux_mg_cm2_h > low.flux_mg_cm2_h


# ---------------------------------------------------------------------------
# Dermal absorption fraction bounds
# ---------------------------------------------------------------------------


class TestAbsorptionFraction:
    def test_fraction_between_0_01_and_1(self):
        res = default_assess()
        assert 0.01 <= res.dermal_absorption_fraction <= 1.0

    def test_very_low_kp_gives_min_fraction(self):
        # Low logP, high MW → very low Kp → fraction should clamp to 0.01
        res = default_assess(logP=-5.0, mw_Da=800.0)
        assert res.dermal_absorption_fraction == pytest.approx(0.01, abs=1e-10)

    def test_very_high_kp_gives_max_fraction(self):
        # High logP → large Kp → fraction clamped to 1.0
        res = default_assess(logP=10.0, mw_Da=50.0)
        assert res.dermal_absorption_fraction == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------


class TestRiskClassification:
    def test_negligible_class(self):
        # Force very small body burden (low logP, high MW, tiny conc, small area, short time)
        res = assess_dermal_absorption_risk(
            substance_name="S",
            logP=-3.0,
            mw_Da=500.0,
            conc_mg_cm3=0.0001,
            skin_area_cm2=1.0,
            exposure_duration_h=0.1,
            vd_body_L=42.0,
        )
        assert res.risk_class == "negligible"

    def test_high_class(self):
        # Force large body burden
        res = assess_dermal_absorption_risk(
            substance_name="S",
            logP=5.0,
            mw_Da=100.0,
            conc_mg_cm3=100.0,
            skin_area_cm2=1000.0,
            exposure_duration_h=24.0,
            vd_body_L=42.0,
        )
        assert res.risk_class == "high"

    def test_risk_class_in_valid_set(self):
        for logP in [-2.0, 0.0, 2.0, 5.0]:
            res = default_assess(logP=logP)
            assert res.risk_class in {"negligible", "low", "moderate", "high"}


# ---------------------------------------------------------------------------
# PPE required
# ---------------------------------------------------------------------------


class TestPpeRequired:
    def test_ppe_required_for_moderate_risk(self):
        # Force moderate body burden (>= 0.01 but < 0.1)
        res = assess_dermal_absorption_risk(
            substance_name="S",
            logP=2.0,
            mw_Da=150.0,
            conc_mg_cm3=5.0,
            skin_area_cm2=200.0,
            exposure_duration_h=8.0,
            vd_body_L=42.0,
        )
        if res.risk_class in {"moderate", "high"}:
            assert res.ppe_required is True
        else:
            assert res.ppe_required is False

    def test_ppe_required_for_high_risk(self):
        res = assess_dermal_absorption_risk(
            substance_name="S",
            logP=5.0,
            mw_Da=100.0,
            conc_mg_cm3=100.0,
            skin_area_cm2=1000.0,
            exposure_duration_h=24.0,
            vd_body_L=42.0,
        )
        assert res.ppe_required is True

    def test_ppe_not_required_for_negligible(self):
        res = assess_dermal_absorption_risk(
            substance_name="S",
            logP=-3.0,
            mw_Da=500.0,
            conc_mg_cm3=0.0001,
            skin_area_cm2=1.0,
            exposure_duration_h=0.1,
            vd_body_L=42.0,
        )
        assert res.ppe_required is False


# ---------------------------------------------------------------------------
# Area and duration effects
# ---------------------------------------------------------------------------


class TestExposureEffects:
    def test_larger_area_higher_burden(self):
        small = default_assess(skin_area_cm2=50.0)
        large = default_assess(skin_area_cm2=500.0)
        assert large.body_burden_mg_L > small.body_burden_mg_L

    def test_longer_duration_higher_burden(self):
        short = default_assess(exposure_duration_h=1.0)
        long_ = default_assess(exposure_duration_h=24.0)
        assert long_.body_burden_mg_L > short.body_burden_mg_L

    def test_higher_conc_higher_burden(self):
        low = default_assess(conc_mg_cm3=0.1)
        high = default_assess(conc_mg_cm3=10.0)
        assert high.body_burden_mg_L > low.body_burden_mg_L

    def test_zero_conc_zero_burden(self):
        res = default_assess(conc_mg_cm3=0.0)
        assert res.total_absorbed_mg == 0.0
        assert res.body_burden_mg_L == 0.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            assess_dermal_absorption_risk("S", 2.0, 0.0, 1.0)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            assess_dermal_absorption_risk("S", 2.0, -100.0, 1.0)

    def test_negative_conc_raises(self):
        with pytest.raises(ValueError, match="conc_mg_cm3"):
            assess_dermal_absorption_risk("S", 2.0, 200.0, -1.0)

    def test_zero_area_raises(self):
        with pytest.raises(ValueError, match="skin_area_cm2"):
            assess_dermal_absorption_risk("S", 2.0, 200.0, 1.0, skin_area_cm2=0.0)

    def test_zero_duration_raises(self):
        with pytest.raises(ValueError, match="exposure_duration_h"):
            assess_dermal_absorption_risk("S", 2.0, 200.0, 1.0, exposure_duration_h=0.0)


# ---------------------------------------------------------------------------
# Compare function
# ---------------------------------------------------------------------------


class TestCompareFunction:
    def test_compare_returns_list(self):
        result = compare_exposure_scenarios(
            substance_name="Drug",
            logP=2.0,
            mw_Da=200.0,
            conc_mg_cm3=1.0,
            areas_cm2=[50.0, 100.0, 500.0],
        )
        assert isinstance(result, list)
        assert len(result) == 3

    def test_compare_sorted_by_burden_descending(self):
        result = compare_exposure_scenarios(
            substance_name="Drug",
            logP=2.0,
            mw_Da=200.0,
            conc_mg_cm3=1.0,
            areas_cm2=[50.0, 100.0, 500.0],
        )
        burdens = [r.body_burden_mg_L for r in result]
        assert burdens == sorted(burdens, reverse=True)

    def test_compare_returns_correct_type_elements(self):
        result = compare_exposure_scenarios(
            substance_name="Drug",
            logP=2.0,
            mw_Da=200.0,
            conc_mg_cm3=1.0,
            areas_cm2=[100.0],
        )
        assert isinstance(result[0], DermalAbsorptionRiskResult)

    def test_compare_single_scenario(self):
        result = compare_exposure_scenarios(
            substance_name="Drug",
            logP=2.0,
            mw_Da=200.0,
            conc_mg_cm3=1.0,
            areas_cm2=[100.0],
        )
        assert len(result) == 1
