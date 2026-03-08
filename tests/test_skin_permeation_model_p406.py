"""
Tests for Phase 406 — Drug Permeation Across Skin Layers
"""

import pytest

from omega_pbpk.core.skin_permeation_model_p406 import (
    _FORMULATION_MODS,
    SkinPermeationResult,
    _potts_guy_kp,
    compare_formulations_skin,
    simulate_skin_permeation,
)

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_skin_permeation_result(self):
        result = simulate_skin_permeation(
            "TestDrug", applied_dose_ug_per_cm2=100.0, logp=2.5, mw_da=300.0
        )
        assert isinstance(result, SkinPermeationResult)

    def test_drug_name_preserved(self):
        result = simulate_skin_permeation(
            "Nicotine", applied_dose_ug_per_cm2=50.0, logp=1.17, mw_da=162.2
        )
        assert result.drug_name == "Nicotine"

    def test_formulation_type_preserved(self):
        result = simulate_skin_permeation(
            "Drug", applied_dose_ug_per_cm2=100.0, logp=2.0, mw_da=250.0, formulation_type="patch"
        )
        assert result.formulation_type == "patch"

    def test_lists_have_same_length(self):
        result = simulate_skin_permeation(
            "Drug", applied_dose_ug_per_cm2=100.0, logp=2.0, mw_da=300.0, t_end_h=12.0, dt_h=0.5
        )
        n = len(result.times_h)
        assert len(result.flux_ug_per_cm2_per_h) == n
        assert len(result.cumulative_permeated_ug_per_cm2) == n
        assert len(result.c_sc_mg_g) == n
        assert len(result.c_dermis_mg_L) == n


# ---------------------------------------------------------------------------
# Physicochemical dependence
# ---------------------------------------------------------------------------


class TestPhysicochemicalDependence:
    def test_high_logp_low_mw_higher_flux(self):
        """High logP / low MW → higher steady-state flux."""
        result_high = simulate_skin_permeation(
            "HighLogP", applied_dose_ug_per_cm2=100.0, logp=4.0, mw_da=200.0
        )
        result_low = simulate_skin_permeation(
            "LowLogP", applied_dose_ug_per_cm2=100.0, logp=0.5, mw_da=500.0
        )
        assert result_high.steady_state_flux > result_low.steady_state_flux

    def test_higher_logp_higher_kp(self):
        """Higher logP should give higher effective permeability coefficient."""
        result_hi = simulate_skin_permeation(
            "Hi", applied_dose_ug_per_cm2=100.0, logp=3.5, mw_da=250.0
        )
        result_lo = simulate_skin_permeation(
            "Lo", applied_dose_ug_per_cm2=100.0, logp=0.0, mw_da=250.0
        )
        assert result_hi.permeability_coefficient_cm_h > result_lo.permeability_coefficient_cm_h

    def test_higher_mw_lower_flux(self):
        """Higher MW should reduce permeability."""
        result_lo = simulate_skin_permeation(
            "SmallMW", applied_dose_ug_per_cm2=100.0, logp=2.0, mw_da=150.0
        )
        result_hi = simulate_skin_permeation(
            "LargeMW", applied_dose_ug_per_cm2=100.0, logp=2.0, mw_da=700.0
        )
        assert result_lo.steady_state_flux > result_hi.steady_state_flux


# ---------------------------------------------------------------------------
# Formulation comparisons
# ---------------------------------------------------------------------------


class TestFormulationEffects:
    def test_solution_gt_ointment_flux(self):
        """Solution should have higher SS flux than ointment (modifier 1.0 > 0.4)."""
        result_sol = simulate_skin_permeation(
            "Drug",
            applied_dose_ug_per_cm2=100.0,
            logp=2.5,
            mw_da=300.0,
            formulation_type="solution",
        )
        result_oint = simulate_skin_permeation(
            "Drug",
            applied_dose_ug_per_cm2=100.0,
            logp=2.5,
            mw_da=300.0,
            formulation_type="ointment",
        )
        assert result_sol.steady_state_flux > result_oint.steady_state_flux

    def test_patch_gt_cream_flux(self):
        """Patch (modifier 1.2) should have higher SS flux than cream (0.6)."""
        result_patch = simulate_skin_permeation(
            "Drug", applied_dose_ug_per_cm2=100.0, logp=2.5, mw_da=300.0, formulation_type="patch"
        )
        result_cream = simulate_skin_permeation(
            "Drug", applied_dose_ug_per_cm2=100.0, logp=2.5, mw_da=300.0, formulation_type="cream"
        )
        assert result_patch.steady_state_flux > result_cream.steady_state_flux

    def test_formulation_modifier_applied_to_kp(self):
        """Permeability coefficient should reflect formulation modifier."""
        base_kp = _potts_guy_kp(logp=2.5, mw_da=300.0)
        result = simulate_skin_permeation(
            "Drug", applied_dose_ug_per_cm2=100.0, logp=2.5, mw_da=300.0, formulation_type="gel"
        )
        expected_kp = base_kp * _FORMULATION_MODS["gel"]
        assert result.permeability_coefficient_cm_h == pytest.approx(expected_kp, rel=1e-6)


# ---------------------------------------------------------------------------
# Lag time
# ---------------------------------------------------------------------------


class TestLagTime:
    def test_lag_time_positive(self):
        result = simulate_skin_permeation(
            "Drug", applied_dose_ug_per_cm2=100.0, logp=2.0, mw_da=300.0
        )
        assert result.lag_time_h > 0

    def test_lag_time_lower_for_higher_kp(self):
        """Higher permeability → shorter lag time (t_lag = 0.1 / effective_kp)."""
        result_hi = simulate_skin_permeation(
            "Hi", applied_dose_ug_per_cm2=100.0, logp=4.0, mw_da=200.0
        )
        result_lo = simulate_skin_permeation(
            "Lo", applied_dose_ug_per_cm2=100.0, logp=0.5, mw_da=500.0
        )
        assert result_hi.lag_time_h <= result_lo.lag_time_h


# ---------------------------------------------------------------------------
# Cumulative permeation
# ---------------------------------------------------------------------------


class TestCumulativePermeation:
    def test_cumulative_monotonically_increasing(self):
        """Cumulative permeated should be non-decreasing over time."""
        result = simulate_skin_permeation(
            "Drug", applied_dose_ug_per_cm2=100.0, logp=2.5, mw_da=300.0, t_end_h=24.0, dt_h=0.5
        )
        cum = result.cumulative_permeated_ug_per_cm2
        for i in range(1, len(cum)):
            assert cum[i] >= cum[i - 1] - 1e-9

    def test_cumulative_starts_at_zero(self):
        result = simulate_skin_permeation(
            "Drug", applied_dose_ug_per_cm2=100.0, logp=2.0, mw_da=300.0
        )
        assert result.cumulative_permeated_ug_per_cm2[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Steady-state flux
# ---------------------------------------------------------------------------


class TestSteadyStateFlux:
    def test_ss_flux_positive(self):
        result = simulate_skin_permeation(
            "Drug", applied_dose_ug_per_cm2=100.0, logp=2.0, mw_da=300.0
        )
        assert result.steady_state_flux > 0

    def test_ss_flux_proportional_to_dose(self):
        """SS flux should scale with applied dose."""
        result1 = simulate_skin_permeation(
            "Drug", applied_dose_ug_per_cm2=100.0, logp=2.0, mw_da=300.0
        )
        result2 = simulate_skin_permeation(
            "Drug", applied_dose_ug_per_cm2=200.0, logp=2.0, mw_da=300.0
        )
        ratio = result2.steady_state_flux / result1.steady_state_flux
        assert ratio == pytest.approx(2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Bioavailability
# ---------------------------------------------------------------------------


class TestBioavailability:
    def test_bioavailability_in_range(self):
        """Bioavailability should be in (0, 100] for a permeable drug."""
        # Use high logP / low MW to ensure lag time is within 24 h simulation window
        result = simulate_skin_permeation(
            "PermeableDrug", applied_dose_ug_per_cm2=100.0, logp=4.0, mw_da=200.0, t_end_h=24.0
        )
        assert 0 < result.bioavailability_transdermal_pct <= 100.0

    def test_higher_dose_same_bioavailability_pct(self):
        """Bioavailability % should be dose-independent for same formulation/drug."""
        result1 = simulate_skin_permeation(
            "Drug", applied_dose_ug_per_cm2=50.0, logp=2.0, mw_da=300.0, t_end_h=12.0
        )
        result2 = simulate_skin_permeation(
            "Drug", applied_dose_ug_per_cm2=100.0, logp=2.0, mw_da=300.0, t_end_h=12.0
        )
        assert result1.bioavailability_transdermal_pct == pytest.approx(
            result2.bioavailability_transdermal_pct, rel=1e-4
        )


# ---------------------------------------------------------------------------
# compare_formulations_skin
# ---------------------------------------------------------------------------


class TestCompareFormulations:
    def test_compare_returns_5_results(self):
        results = compare_formulations_skin(
            "Drug", applied_dose_ug_per_cm2=100.0, logp=2.5, mw_da=300.0
        )
        assert len(results) == 5

    def test_compare_sorted_by_ss_flux_descending(self):
        results = compare_formulations_skin(
            "Drug", applied_dose_ug_per_cm2=100.0, logp=2.5, mw_da=300.0
        )
        ss_fluxes = [r.steady_state_flux for r in results]
        assert ss_fluxes == sorted(ss_fluxes, reverse=True)

    def test_compare_all_formulation_types_present(self):
        results = compare_formulations_skin(
            "Drug", applied_dose_ug_per_cm2=100.0, logp=2.5, mw_da=300.0
        )
        formulations = {r.formulation_type for r in results}
        assert formulations == set(_FORMULATION_MODS.keys())

    def test_compare_all_are_result_type(self):
        results = compare_formulations_skin(
            "Drug", applied_dose_ug_per_cm2=100.0, logp=2.5, mw_da=300.0
        )
        assert all(isinstance(r, SkinPermeationResult) for r in results)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_invalid_dose_zero(self):
        with pytest.raises(ValueError, match="applied_dose"):
            simulate_skin_permeation("Drug", applied_dose_ug_per_cm2=0.0, logp=2.0, mw_da=300.0)

    def test_invalid_dose_negative(self):
        with pytest.raises(ValueError, match="applied_dose"):
            simulate_skin_permeation("Drug", applied_dose_ug_per_cm2=-10.0, logp=2.0, mw_da=300.0)

    def test_invalid_formulation_type(self):
        with pytest.raises(ValueError, match="formulation_type"):
            simulate_skin_permeation(
                "Drug",
                applied_dose_ug_per_cm2=100.0,
                logp=2.0,
                mw_da=300.0,
                formulation_type="spray",
            )

    def test_invalid_mw_zero(self):
        with pytest.raises(ValueError, match="mw_da"):
            simulate_skin_permeation("Drug", applied_dose_ug_per_cm2=100.0, logp=2.0, mw_da=0.0)

    def test_invalid_mw_negative(self):
        with pytest.raises(ValueError, match="mw_da"):
            simulate_skin_permeation("Drug", applied_dose_ug_per_cm2=100.0, logp=2.0, mw_da=-50.0)

    def test_invalid_application_area_zero(self):
        with pytest.raises(ValueError, match="application_area"):
            simulate_skin_permeation(
                "Drug",
                applied_dose_ug_per_cm2=100.0,
                logp=2.0,
                mw_da=300.0,
                application_area_cm2=0.0,
            )
