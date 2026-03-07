"""Tests for Phase 289 — Topical Formulation Flux Calculator."""

import pytest

from omega_pbpk.prediction.topical_flux_calculator import (
    ENHANCEMENT_FACTORS,
    VALID_FORMULATIONS,
    TopicalFluxResult,
    calculate_topical_flux,
    compare_formulations,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_result(**kwargs) -> TopicalFluxResult:
    defaults = dict(
        drug_name="TestDrug",
        logp=2.0,
        mw=300.0,
        concentration_pct=1.0,
        formulation_type="cream",
        application_area_cm2=100.0,
    )
    defaults.update(kwargs)
    return calculate_topical_flux(**defaults)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_topical_flux_result(self):
        result = make_result()
        assert isinstance(result, TopicalFluxResult)

    def test_frozen(self):
        result = make_result()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "other"  # type: ignore[misc]

    def test_drug_name_preserved(self):
        result = make_result(drug_name="Ibuprofen")
        assert result.drug_name == "Ibuprofen"

    def test_logp_preserved(self):
        result = make_result(logp=3.5)
        assert result.logp == pytest.approx(3.5)

    def test_mw_preserved(self):
        result = make_result(mw=250.0)
        assert result.mw == pytest.approx(250.0)

    def test_concentration_pct_preserved(self):
        result = make_result(concentration_pct=5.0)
        assert result.concentration_pct == pytest.approx(5.0)

    def test_formulation_type_preserved(self):
        result = make_result(formulation_type="gel")
        assert result.formulation_type == "gel"


# ---------------------------------------------------------------------------
# Physics checks
# ---------------------------------------------------------------------------


class TestPhysics:
    def test_flux_positive(self):
        result = make_result()
        assert result.steady_state_flux_ug_per_cm2_per_h > 0

    def test_cumulative_24h_positive(self):
        result = make_result()
        assert result.cumulative_24h_ug > 0

    def test_kp_positive(self):
        result = make_result()
        assert result.kp_cm_per_h > 0
        assert result.kp_eff_cm_per_h > 0

    def test_lag_time_positive(self):
        result = make_result()
        assert result.lag_time_h > 0

    def test_f_sys_in_range(self):
        result = make_result()
        assert 0.0 <= result.f_sys_estimate <= 1.0

    def test_higher_logp_increases_kp(self):
        """logp=2 should give higher Kp than logp=0."""
        r0 = make_result(logp=0.0)
        r2 = make_result(logp=2.0)
        assert r2.kp_cm_per_h > r0.kp_cm_per_h

    def test_higher_mw_decreases_kp(self):
        r_light = make_result(mw=200.0)
        r_heavy = make_result(mw=500.0)
        assert r_light.kp_cm_per_h > r_heavy.kp_cm_per_h

    def test_higher_concentration_increases_flux(self):
        r_low = make_result(concentration_pct=0.5)
        r_high = make_result(concentration_pct=5.0)
        assert r_high.steady_state_flux_ug_per_cm2_per_h > r_low.steady_state_flux_ug_per_cm2_per_h

    def test_higher_area_increases_cumulative(self):
        r_small = make_result(application_area_cm2=10.0)
        r_large = make_result(application_area_cm2=500.0)
        assert r_large.cumulative_24h_ug > r_small.cumulative_24h_ug

    def test_cumulative_proportional_to_area(self):
        r1 = make_result(application_area_cm2=100.0)
        r2 = make_result(application_area_cm2=200.0)
        ratio = r2.cumulative_24h_ug / r1.cumulative_24h_ug
        assert ratio == pytest.approx(2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Formulation comparison
# ---------------------------------------------------------------------------


class TestFormulations:
    def test_ointment_flux_greater_than_cream(self):
        r_cream = make_result(formulation_type="cream")
        r_ointment = make_result(formulation_type="ointment")
        assert (
            r_ointment.steady_state_flux_ug_per_cm2_per_h
            > r_cream.steady_state_flux_ug_per_cm2_per_h
        )

    def test_gel_flux_less_than_cream(self):
        r_cream = make_result(formulation_type="cream")
        r_gel = make_result(formulation_type="gel")
        assert r_gel.steady_state_flux_ug_per_cm2_per_h < r_cream.steady_state_flux_ug_per_cm2_per_h

    def test_lotion_flux_less_than_gel(self):
        r_lotion = make_result(formulation_type="lotion")
        r_gel = make_result(formulation_type="gel")
        assert (
            r_lotion.steady_state_flux_ug_per_cm2_per_h < r_gel.steady_state_flux_ug_per_cm2_per_h
        )

    def test_enhancement_factor_values(self):
        assert ENHANCEMENT_FACTORS["cream"] == pytest.approx(1.0)
        assert ENHANCEMENT_FACTORS["ointment"] == pytest.approx(1.3)
        assert ENHANCEMENT_FACTORS["gel"] == pytest.approx(0.8)
        assert ENHANCEMENT_FACTORS["lotion"] == pytest.approx(0.7)

    def test_enhancement_factor_stored_in_result(self):
        r = make_result(formulation_type="ointment")
        assert r.enhancement_factor == pytest.approx(1.3)

    def test_vehicle_partition_scales_flux(self):
        r1 = make_result(vehicle_skin_partition=1.0)
        r2 = make_result(vehicle_skin_partition=2.0)
        ratio = r2.steady_state_flux_ug_per_cm2_per_h / r1.steady_state_flux_ug_per_cm2_per_h
        assert ratio == pytest.approx(2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Permeability classification
# ---------------------------------------------------------------------------


class TestPermeabilityClass:
    def test_permeability_class_valid_strings(self):
        for ftype in VALID_FORMULATIONS:
            r = make_result(formulation_type=ftype)
            assert r.permeability_class in {"high", "moderate", "low"}

    def test_high_logp_tends_to_higher_class(self):
        r = make_result(logp=4.0, mw=200.0)
        # High logp, low MW → should be high or moderate permeability
        assert r.permeability_class in {"high", "moderate"}

    def test_low_logp_high_mw_tends_low_class(self):
        r = make_result(logp=-3.0, mw=800.0)
        # Very low logp, very high MW → low permeability expected
        assert r.permeability_class == "low"


# ---------------------------------------------------------------------------
# compare_formulations
# ---------------------------------------------------------------------------


class TestCompareFormulations:
    def test_returns_list_of_four(self):
        results = compare_formulations("TestDrug", 2.0, 300.0, 1.0, 100.0)
        assert len(results) == 4

    def test_all_formulation_types_present(self):
        results = compare_formulations("TestDrug", 2.0, 300.0, 1.0, 100.0)
        types = {r.formulation_type for r in results}
        assert types == VALID_FORMULATIONS

    def test_sorted_descending_by_flux(self):
        results = compare_formulations("TestDrug", 2.0, 300.0, 1.0, 100.0)
        fluxes = [r.steady_state_flux_ug_per_cm2_per_h for r in results]
        assert fluxes == sorted(fluxes, reverse=True)

    def test_first_is_highest_flux(self):
        results = compare_formulations("TestDrug", 2.0, 300.0, 1.0, 100.0)
        max_flux = max(r.steady_state_flux_ug_per_cm2_per_h for r in results)
        assert results[0].steady_state_flux_ug_per_cm2_per_h == pytest.approx(max_flux)

    def test_ointment_first(self):
        results = compare_formulations("TestDrug", 2.0, 300.0, 1.0, 100.0)
        # Ointment has highest enhancement factor (1.3) and all else equal
        assert results[0].formulation_type == "ointment"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_logp_too_low(self):
        with pytest.raises(ValueError, match="logp"):
            make_result(logp=-6.0)

    def test_logp_too_high(self):
        with pytest.raises(ValueError, match="logp"):
            make_result(logp=11.0)

    def test_mw_zero(self):
        with pytest.raises(ValueError, match="mw"):
            make_result(mw=0.0)

    def test_mw_negative(self):
        with pytest.raises(ValueError, match="mw"):
            make_result(mw=-50.0)

    def test_concentration_zero(self):
        with pytest.raises(ValueError, match="concentration_pct"):
            make_result(concentration_pct=0.0)

    def test_concentration_over_100(self):
        with pytest.raises(ValueError, match="concentration_pct"):
            make_result(concentration_pct=101.0)

    def test_area_zero(self):
        with pytest.raises(ValueError, match="application_area_cm2"):
            make_result(application_area_cm2=0.0)

    def test_area_negative(self):
        with pytest.raises(ValueError, match="application_area_cm2"):
            make_result(application_area_cm2=-10.0)

    def test_vehicle_partition_zero(self):
        with pytest.raises(ValueError, match="vehicle_skin_partition"):
            make_result(vehicle_skin_partition=0.0)

    def test_invalid_formulation_type(self):
        with pytest.raises(ValueError, match="formulation_type"):
            make_result(formulation_type="patch")


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


class TestNotes:
    def test_notes_is_string(self):
        result = make_result()
        assert isinstance(result.notes, str)

    def test_notes_nonempty(self):
        result = make_result()
        assert len(result.notes) > 0
