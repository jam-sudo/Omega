"""Tests for intestinal lymphatic absorption (Phase 59)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.lymphatic import (
    LymphaticAbsorptionResult,
    _lymphatic_fraction,
    compare_food_effect,
    simulate_lymphatic_absorption,
)


class TestLymphaticFraction:
    def test_low_logP_minimal_lymph(self):
        f = _lymphatic_fraction(0.0)
        assert f < 0.02

    def test_high_logP_significant_lymph(self):
        f = _lymphatic_fraction(7.0, solubility_mg_mL=0.01)
        assert f > 0.1

    def test_fraction_clamped_0_to_05(self):
        f_low = _lymphatic_fraction(-5.0)
        f_high = _lymphatic_fraction(10.0, solubility_mg_mL=0.001)
        assert 0.0 <= f_low <= 0.5
        assert 0.0 <= f_high <= 0.5

    def test_high_solubility_reduces_lymph(self):
        f_low_sol = _lymphatic_fraction(5.0, solubility_mg_mL=0.01)
        f_high_sol = _lymphatic_fraction(5.0, solubility_mg_mL=10.0)
        assert f_low_sol > f_high_sol

    def test_midpoint_logP5(self):
        f = _lymphatic_fraction(5.0, solubility_mg_mL=0.001)
        # sol_factor ≈ 0.99, f_base ≈ 0.25 → f ≈ 0.25
        assert abs(f - 0.25) < 0.05


class TestSimulateLymphaticAbsorption:
    def _default(self, logP=2.0, dose_mg=100.0):
        return simulate_lymphatic_absorption(
            drug_name="TestDrug",
            logP=logP,
            dose_mg=dose_mg,
            cl_L_per_h=5.0,
            vd_L=50.0,
            ka_per_h=1.0,
            fup=1.0,
            solubility_mg_mL=1.0,
            t_end_h=24.0,
            dt_h=0.05,
        )

    def test_returns_result_type(self):
        assert isinstance(self._default(), LymphaticAbsorptionResult)

    def test_f_lymph_plus_f_portal_equals_one(self):
        result = self._default()
        assert abs(result.f_lymph + result.f_portal - 1.0) < 1e-9

    def test_auc_portal_plus_lymph_equals_total(self):
        result = self._default()
        assert (
            abs(result.auc_portal_mg_h_L + result.auc_lymph_mg_h_L - result.auc_total_mg_h_L) < 1e-6
        )

    def test_bioavailability_bounds(self):
        result = self._default()
        assert 0.0 < result.bioavailability_total <= 1.0
        assert 0.0 <= result.bioavailability_portal <= 1.0
        assert 0.0 <= result.bioavailability_lymph <= 1.0

    def test_lymph_bypasses_first_pass(self):
        result = self._default()
        assert result.fh_lymph == 1.0

    def test_cmax_positive(self):
        result = self._default()
        assert result.cmax_mg_L > 0.0

    def test_tmax_within_simulation_time(self):
        result = self._default()
        assert 0.0 <= result.tmax_h <= 24.0

    def test_plasma_conc_length(self):
        result = self._default()
        assert len(result.plasma_conc) > 1
        assert len(result.times_h) == len(result.plasma_conc)

    def test_high_logP_increases_lymph_fraction(self):
        low = self._default(logP=0.0)
        high = self._default(logP=6.0)
        assert high.f_lymph > low.f_lymph

    def test_dose_scales_auc(self):
        r100 = self._default(dose_mg=100.0)
        r200 = self._default(dose_mg=200.0)
        ratio = r200.auc_total_mg_h_L / r100.auc_total_mg_h_L
        assert abs(ratio - 2.0) < 0.3

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_lymphatic_absorption("X", 2.0, 0.0, 5.0, 50.0)

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_lymphatic_absorption("X", 2.0, 100.0, 0.0, 50.0)

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_lymphatic_absorption("X", 2.0, 100.0, 5.0, 0.0)

    def test_drug_name_preserved(self):
        result = self._default()
        assert result.drug_name == "TestDrug"
        assert result.logP == 2.0
        assert result.dose_mg == 100.0


class TestFoodEffect:
    def test_compare_food_effect_returns_dict(self):
        result = compare_food_effect(
            drug_name="Lipophilic",
            logP=5.0,
            dose_mg=100.0,
            cl_L_per_h=10.0,
            vd_L=100.0,
        )
        assert "fasted" in result
        assert "fed" in result

    def test_fed_has_higher_lymph_fraction_for_lipophilic(self):
        result = compare_food_effect(
            drug_name="LipoDrug",
            logP=6.0,
            dose_mg=100.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
        )
        assert result["fed"].f_lymph >= result["fasted"].f_lymph

    def test_both_states_have_positive_auc(self):
        result = compare_food_effect(
            drug_name="Drug",
            logP=3.0,
            dose_mg=100.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
        )
        assert result["fasted"].auc_total_mg_h_L > 0
        assert result["fed"].auc_total_mg_h_L > 0
