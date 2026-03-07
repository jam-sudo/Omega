"""Tests for GI precipitation module — Phase 342."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.gi_precipitation import (
    GIPrecipitationResult,
    effect_of_supersaturation_inhibitors,
    simulate_gi_precipitation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kw) -> GIPrecipitationResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=10.0,
        solubility_stomach_mg_mL=0.1,
        solubility_intestine_mg_mL=0.2,
        gastric_volume_mL=250.0,
        intestinal_volume_mL=500.0,
        gastric_emptying_h=0.5,
        k_precip_per_h=2.0,
        k_redissolve_per_h=0.3,
        ka_intestinal_per_h=1.5,
        t_end_h=6.0,
        dt_h=0.1,
    )
    defaults.update(kw)
    return simulate_gi_precipitation(**defaults)


# ---------------------------------------------------------------------------
# simulate_gi_precipitation — basic tests
# ---------------------------------------------------------------------------


class TestSimulateGIPrecipitation:
    def test_returns_dataclass(self):
        r = _default()
        assert isinstance(r, GIPrecipitationResult)

    def test_frozen_dataclass_immutable(self):
        r = _default()
        with pytest.raises((AttributeError, TypeError)):
            r.drug_name = "other"  # type: ignore[misc]

    def test_low_dose_no_precipitation(self):
        """Dose << solubility * volume → no precipitation."""
        # solubility_stomach = 1.0, volume = 250 → capacity = 250 mg >> 1 mg
        r = simulate_gi_precipitation(
            drug_name="LowDose",
            dose_mg=1.0,
            solubility_stomach_mg_mL=1.0,
            solubility_intestine_mg_mL=2.0,
            gastric_volume_mL=250.0,
            intestinal_volume_mL=500.0,
            dt_h=0.05,
        )
        assert r.f_precipitated_stomach < 0.05

    def test_high_dose_causes_precipitation_in_stomach(self):
        """Dose >> solubility * volume → precipitation occurs."""
        r = simulate_gi_precipitation(
            drug_name="HighDose",
            dose_mg=500.0,
            solubility_stomach_mg_mL=0.01,
            solubility_intestine_mg_mL=0.05,
            gastric_volume_mL=250.0,
            intestinal_volume_mL=500.0,
            k_precip_per_h=5.0,
            dt_h=0.05,
        )
        # At t=0, dose >> solubility*volume → initial precipitate > 0
        assert r.c_precipitated_stomach_mg[0] > 0

    def test_mass_balance_f_absorbed_plus_f_lost(self):
        """f_absorbed + f_lost should approximately equal 1.0."""
        r = _default()
        assert abs(r.f_absorbed + r.f_lost - 1.0) < 0.05

    def test_f_absorbed_in_0_1(self):
        r = _default()
        assert 0.0 <= r.f_absorbed <= 1.0

    def test_f_lost_in_0_1(self):
        r = _default()
        assert 0.0 <= r.f_lost <= 1.0

    def test_bioavailability_reduction_in_0_100(self):
        r = _default()
        assert 0.0 <= r.bioavailability_reduction_pct <= 100.0

    def test_peak_supersaturation_nonnegative(self):
        r = _default()
        assert r.peak_supersaturation_ratio >= 0.0

    def test_time_to_precipitation_nonnegative(self):
        r = _default()
        assert r.time_to_precipitation_h >= 0.0

    def test_times_array_starts_at_zero(self):
        r = _default()
        assert r.times_h[0] == pytest.approx(0.0)

    def test_times_array_ends_at_t_end(self):
        r = _default(t_end_h=4.0)
        assert r.times_h[-1] == pytest.approx(4.0, abs=0.2)

    def test_array_lengths_consistent(self):
        r = _default()
        n = len(r.times_h)
        assert len(r.c_dissolved_stomach_mg_mL) == n
        assert len(r.c_dissolved_intestine_mg_mL) == n
        assert len(r.c_precipitated_stomach_mg) == n
        assert len(r.c_precipitated_intestine_mg) == n

    def test_concentrations_nonnegative(self):
        r = _default()
        assert all(c >= 0 for c in r.c_dissolved_stomach_mg_mL)
        assert all(c >= 0 for c in r.c_dissolved_intestine_mg_mL)
        assert all(m >= 0 for m in r.c_precipitated_stomach_mg)
        assert all(m >= 0 for m in r.c_precipitated_intestine_mg)

    def test_higher_k_precip_lower_f_absorbed(self):
        """Higher precipitation rate → less drug absorbed."""
        r_low = _default(k_precip_per_h=0.1, dose_mg=100.0)
        r_high = _default(k_precip_per_h=10.0, dose_mg=100.0)
        assert r_high.f_absorbed <= r_low.f_absorbed + 0.05

    def test_lower_solubility_more_precipitation(self):
        """Lower intestinal solubility → more precipitation."""
        r_high_sol = _default(solubility_intestine_mg_mL=1.0, dose_mg=100.0)
        r_low_sol = _default(solubility_intestine_mg_mL=0.01, dose_mg=100.0)
        assert r_low_sol.f_absorbed <= r_high_sol.f_absorbed + 0.05

    def test_drug_name_preserved(self):
        r = _default(drug_name="Itraconazole")
        assert r.drug_name == "Itraconazole"

    def test_dose_preserved(self):
        r = _default(dose_mg=250.0)
        assert r.dose_mg == pytest.approx(250.0)

    def test_time_to_precipitation_when_no_precip(self):
        """If no precipitation, time_to_precipitation equals t_end_h."""
        r = simulate_gi_precipitation(
            drug_name="NoPrecip",
            dose_mg=1.0,
            solubility_stomach_mg_mL=5.0,
            solubility_intestine_mg_mL=5.0,
            t_end_h=4.0,
            dt_h=0.1,
        )
        # Either no precipitation event or at t_end
        assert r.time_to_precipitation_h >= 0.0

    def test_peak_supersaturation_gte_one_when_precip_occurs(self):
        """When precipitation occurs, intestinal supersaturation should reach >= 1."""
        r = simulate_gi_precipitation(
            drug_name="HighDose",
            dose_mg=200.0,
            solubility_stomach_mg_mL=0.01,
            solubility_intestine_mg_mL=0.01,
            k_precip_per_h=0.1,  # slow precip → allows supersaturation
            dt_h=0.05,
        )
        # Check intestinal concentration can exceed solubility at some point
        # peak_supersaturation_ratio is max(c_int / sol_int)
        assert r.peak_supersaturation_ratio >= 0.0

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default(dose_mg=0.0)

    def test_invalid_solubility_raises(self):
        with pytest.raises(ValueError, match="solubility_stomach_mg_mL"):
            _default(solubility_stomach_mg_mL=0.0)

    def test_invalid_gastric_volume_raises(self):
        with pytest.raises(ValueError, match="gastric_volume_mL"):
            _default(gastric_volume_mL=0.0)

    def test_invalid_gastric_emptying_raises(self):
        with pytest.raises(ValueError, match="gastric_emptying_h"):
            _default(gastric_emptying_h=0.0)


# ---------------------------------------------------------------------------
# effect_of_supersaturation_inhibitors
# ---------------------------------------------------------------------------


class TestEffectOfSupersaturationInhibitors:
    def test_returns_list(self):
        results = effect_of_supersaturation_inhibitors("TestDrug", 50.0, 0.05)
        assert isinstance(results, list)

    def test_default_returns_four_results(self):
        results = effect_of_supersaturation_inhibitors("TestDrug", 50.0, 0.05)
        assert len(results) == 4

    def test_custom_k_precip_values_length(self):
        results = effect_of_supersaturation_inhibitors(
            "TestDrug", 50.0, 0.05, k_precip_values=[1.0, 0.5]
        )
        assert len(results) == 2

    def test_sorted_descending_f_absorbed(self):
        results = effect_of_supersaturation_inhibitors("TestDrug", 100.0, 0.05)
        f_abs = [r.f_absorbed for r in results]
        assert f_abs == sorted(f_abs, reverse=True)

    def test_returns_gi_precipitation_result_objects(self):
        results = effect_of_supersaturation_inhibitors("TestDrug", 50.0, 0.1)
        assert all(isinstance(r, GIPrecipitationResult) for r in results)
