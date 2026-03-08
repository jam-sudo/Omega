"""Tests for Phase 272: Skin Permeation Model (Fick's diffusion, 3-compartment ODE)."""

import pytest

from omega_pbpk.core.skin_permeation_model_p272 import (
    VEHICLE_PARAMS,
    SkinPermeationResult,
    compare_vehicles,
    simulate_skin_permeation,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _default_sim(**kwargs):
    base = dict(drug_name="test_drug", dose_applied_mg=10.0, logp=2.0, mw=300.0)
    base.update(kwargs)
    return simulate_skin_permeation(**base)


# ---------------------------------------------------------------------------
# Return type and structure tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        result = _default_sim()
        assert isinstance(result, SkinPermeationResult)

    def test_drug_name_preserved(self):
        result = _default_sim(drug_name="nicotine")
        assert result.drug_name == "nicotine"

    def test_vehicle_type_preserved(self):
        result = _default_sim(vehicle_type="cream")
        assert result.vehicle_type == "cream"

    def test_time_arrays_non_empty(self):
        result = _default_sim()
        assert len(result.times_h) > 0
        assert len(result.c_vehicle_mg_mL) == len(result.times_h)
        assert len(result.c_sc_mg_g) == len(result.times_h)
        assert len(result.c_dermis_mg_mL) == len(result.times_h)
        assert len(result.c_plasma_mg_L) == len(result.times_h)


# ---------------------------------------------------------------------------
# Initial condition tests
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_vehicle_starts_positive(self):
        result = _default_sim()
        assert result.c_vehicle_mg_mL[0] > 0.0

    def test_sc_starts_at_zero(self):
        result = _default_sim()
        assert result.c_sc_mg_g[0] == pytest.approx(0.0)

    def test_dermis_starts_at_zero(self):
        result = _default_sim()
        assert result.c_dermis_mg_mL[0] == pytest.approx(0.0)

    def test_plasma_starts_at_zero(self):
        result = _default_sim()
        assert result.c_plasma_mg_L[0] == pytest.approx(0.0)

    def test_vehicle_initial_concentration(self):
        """C_vehicle(0) = dose / V_vehicle = 10mg / 10mL = 1 mg/mL."""
        result = _default_sim(dose_applied_mg=10.0)
        assert result.c_vehicle_mg_mL[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AUC and absorption tests
# ---------------------------------------------------------------------------


class TestAUCAndAbsorption:
    def test_plasma_auc_positive(self):
        result = _default_sim()
        assert result.auc_plasma_mg_h_per_L > 0.0

    def test_dermis_auc_positive(self):
        result = _default_sim()
        assert result.auc_dermis_mg_h_per_mL > 0.0

    def test_cumulative_absorbed_positive(self):
        result = _default_sim()
        assert result.cumulative_absorbed_mg > 0.0

    def test_cumulative_absorbed_less_than_dose(self):
        result = _default_sim(dose_applied_mg=10.0)
        assert result.cumulative_absorbed_mg <= 10.0 * 1.01  # allow tiny float error

    def test_skin_retention_in_range(self):
        result = _default_sim()
        assert 0.0 <= result.skin_retention_pct <= 100.0

    def test_flux_peak_positive(self):
        result = _default_sim()
        assert result.flux_peak_ug_cm2_h > 0.0


# ---------------------------------------------------------------------------
# Vehicle comparison tests
# ---------------------------------------------------------------------------


class TestVehicleComparison:
    def test_solution_higher_auc_than_ointment(self):
        """Solution (enhance=1.0) should deliver more drug than ointment (enhance=0.7)."""
        sol = _default_sim(vehicle_type="solution")
        oint = _default_sim(vehicle_type="ointment")
        assert sol.auc_plasma_mg_h_per_L > oint.auc_plasma_mg_h_per_L

    def test_compare_vehicles_returns_list(self):
        results = compare_vehicles("test_drug", dose_applied_mg=10.0, logp=2.0, mw=300.0)
        assert isinstance(results, list)

    def test_compare_vehicles_returns_four(self):
        results = compare_vehicles("test_drug", dose_applied_mg=10.0, logp=2.0, mw=300.0)
        assert len(results) == 4

    def test_compare_vehicles_sorted_descending(self):
        results = compare_vehicles("test_drug", dose_applied_mg=10.0, logp=2.0, mw=300.0)
        aucs = [r.auc_plasma_mg_h_per_L for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_compare_vehicles_all_types_present(self):
        results = compare_vehicles("test_drug", dose_applied_mg=10.0, logp=2.0, mw=300.0)
        vehicle_types = {r.vehicle_type for r in results}
        assert vehicle_types == set(VEHICLE_PARAMS.keys())

    def test_compare_vehicles_best_is_solution(self):
        """Solution should be top performer (highest enhancement)."""
        results = compare_vehicles("test_drug", dose_applied_mg=10.0, logp=2.0, mw=300.0)
        assert results[0].vehicle_type == "solution"


# ---------------------------------------------------------------------------
# Dynamics tests
# ---------------------------------------------------------------------------


class TestDynamics:
    def test_vehicle_depletes_over_time(self):
        """Vehicle concentration should decrease over time."""
        result = _default_sim()
        assert result.c_vehicle_mg_mL[-1] < result.c_vehicle_mg_mL[0]

    def test_tlag_non_negative(self):
        result = _default_sim()
        assert result.tlag_h >= 0.0

    def test_tlag_less_than_t_end(self):
        result = _default_sim(t_end_h=48.0)
        assert result.tlag_h <= 48.0

    def test_notes_is_string(self):
        result = _default_sim()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_dose_zero(self):
        with pytest.raises(ValueError, match="dose_applied_mg must be > 0"):
            simulate_skin_permeation("x", dose_applied_mg=0.0, logp=2.0, mw=300.0)

    def test_invalid_dose_negative(self):
        with pytest.raises(ValueError):
            simulate_skin_permeation("x", dose_applied_mg=-5.0, logp=2.0, mw=300.0)

    def test_invalid_vehicle_type(self):
        with pytest.raises(ValueError, match="vehicle_type must be one of"):
            simulate_skin_permeation(
                "x", dose_applied_mg=10.0, logp=2.0, mw=300.0, vehicle_type="lotion"
            )

    def test_invalid_t_end(self):
        with pytest.raises(ValueError):
            simulate_skin_permeation("x", dose_applied_mg=10.0, logp=2.0, mw=300.0, t_end_h=0.0)
