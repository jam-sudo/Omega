"""Tests for Phase 487 — Drug Excretion Kinetics."""

import math

import pytest

from omega_pbpk.core.excretion_kinetics import (
    ExcretionResult,
    biliary_excretion_rate,
    calculate_fe,
    clearance_from_excretion,
    renal_excretion_rate,
    simulate_excretion,
)

# ---------------------------------------------------------------------------
# renal_excretion_rate
# ---------------------------------------------------------------------------


class TestRenalExcretionRate:
    def test_basic_returns_float(self):
        rate = renal_excretion_rate(10.0, 0.5)
        assert isinstance(rate, float)
        assert rate > 0

    def test_zero_concentration(self):
        assert renal_excretion_rate(0.0, 0.5) == 0.0

    def test_higher_gfr_higher_rate(self):
        rate_low = renal_excretion_rate(10.0, 0.5, gfr_mL_per_min=60.0)
        rate_high = renal_excretion_rate(10.0, 0.5, gfr_mL_per_min=120.0)
        assert rate_high > rate_low

    def test_higher_fu_higher_rate(self):
        rate_low = renal_excretion_rate(10.0, 0.1)
        rate_high = renal_excretion_rate(10.0, 0.9)
        assert rate_high > rate_low

    def test_secretion_increases_rate(self):
        rate_no_sec = renal_excretion_rate(10.0, 0.5, cl_sec_mL_per_min=0.0)
        rate_sec = renal_excretion_rate(10.0, 0.5, cl_sec_mL_per_min=50.0)
        assert rate_sec > rate_no_sec

    def test_full_reabsorption_reduces_rate_to_zero(self):
        rate = renal_excretion_rate(10.0, 0.5, f_reab=1.0)
        assert rate == pytest.approx(0.0, abs=1e-9)

    def test_partial_reabsorption_reduces_rate(self):
        rate_no_reab = renal_excretion_rate(10.0, 0.5, f_reab=0.0)
        rate_reab = renal_excretion_rate(10.0, 0.5, f_reab=0.5)
        assert rate_reab < rate_no_reab

    def test_units_conversion_correct(self):
        # GFR=120 mL/min * fu=1.0 → CL=120 mL/min = 7.2 L/h
        # Rate at C=1 mg/L → 7.2 mg/h
        rate = renal_excretion_rate(1.0, 1.0, gfr_mL_per_min=120.0)
        assert rate == pytest.approx(7.2, rel=1e-6)

    def test_invalid_negative_concentration(self):
        with pytest.raises(ValueError):
            renal_excretion_rate(-1.0, 0.5)

    def test_invalid_fu_gt_1(self):
        with pytest.raises(ValueError):
            renal_excretion_rate(10.0, 1.5)

    def test_invalid_fu_negative(self):
        with pytest.raises(ValueError):
            renal_excretion_rate(10.0, -0.1)


# ---------------------------------------------------------------------------
# biliary_excretion_rate
# ---------------------------------------------------------------------------


class TestBiliaryExcretionRate:
    def test_basic_returns_float(self):
        rate = biliary_excretion_rate(5.0, 0.1, mw=300.0, logP=2.0)
        assert isinstance(rate, float)
        assert rate > 0

    def test_zero_concentration(self):
        assert biliary_excretion_rate(0.0) == 0.0

    def test_mw_above_500_increases_rate(self):
        rate_low = biliary_excretion_rate(5.0, 0.1, mw=400.0, logP=2.0)
        rate_high = biliary_excretion_rate(5.0, 0.1, mw=600.0, logP=2.0)
        assert rate_high > rate_low

    def test_mw_boundary_at_500(self):
        rate_499 = biliary_excretion_rate(1.0, 0.1, mw=499.9, logP=0.0)
        rate_500 = biliary_excretion_rate(1.0, 0.1, mw=500.0, logP=0.0)
        # 500 Da threshold: >= 500 gets 1.5x factor
        assert rate_500 > rate_499

    def test_higher_logP_higher_rate(self):
        rate_low = biliary_excretion_rate(5.0, 0.1, mw=300.0, logP=1.0)
        rate_high = biliary_excretion_rate(5.0, 0.1, mw=300.0, logP=3.0)
        assert rate_high > rate_low

    def test_invalid_negative_concentration(self):
        with pytest.raises(ValueError):
            biliary_excretion_rate(-1.0)

    def test_invalid_negative_cl(self):
        with pytest.raises(ValueError):
            biliary_excretion_rate(5.0, cl_bil_L_per_h=-0.1)

    def test_invalid_zero_mw(self):
        with pytest.raises(ValueError):
            biliary_excretion_rate(5.0, mw=0.0)


# ---------------------------------------------------------------------------
# calculate_fe
# ---------------------------------------------------------------------------


class TestCalculateFe:
    def test_fe_full_renal(self):
        fe = calculate_fe(10.0, 10.0)
        assert fe == pytest.approx(1.0)

    def test_fe_half_renal(self):
        fe = calculate_fe(5.0, 10.0)
        assert fe == pytest.approx(0.5)

    def test_fe_zero_renal(self):
        fe = calculate_fe(0.0, 10.0)
        assert fe == pytest.approx(0.0)

    def test_invalid_negative_cl_renal(self):
        with pytest.raises(ValueError):
            calculate_fe(-1.0, 10.0)

    def test_invalid_zero_cl_total(self):
        with pytest.raises(ValueError):
            calculate_fe(0.0, 0.0)

    def test_invalid_cl_renal_exceeds_total(self):
        with pytest.raises(ValueError):
            calculate_fe(15.0, 10.0)


# ---------------------------------------------------------------------------
# clearance_from_excretion
# ---------------------------------------------------------------------------


class TestClearanceFromExcretion:
    def test_basic_returns_positive(self):
        cl = clearance_from_excretion(120.0, 0.5)
        assert cl > 0

    def test_full_reabsorption_zero_cl(self):
        cl = clearance_from_excretion(120.0, 0.5, f_reabsorption=1.0)
        assert cl == pytest.approx(0.0, abs=1e-9)

    def test_higher_gfr_higher_cl(self):
        cl_low = clearance_from_excretion(60.0, 0.5)
        cl_high = clearance_from_excretion(120.0, 0.5)
        assert cl_high > cl_low

    def test_secretion_increases_cl(self):
        cl_no_sec = clearance_from_excretion(120.0, 0.5, f_secretion=0.0)
        cl_sec = clearance_from_excretion(120.0, 0.5, f_secretion=0.5)
        assert cl_sec > cl_no_sec

    def test_invalid_negative_gfr(self):
        with pytest.raises(ValueError):
            clearance_from_excretion(-10.0, 0.5)

    def test_invalid_fu_gt_1(self):
        with pytest.raises(ValueError):
            clearance_from_excretion(120.0, 1.5)


# ---------------------------------------------------------------------------
# simulate_excretion
# ---------------------------------------------------------------------------


class TestSimulateExcretion:
    def _default_result(self, dose_mg=100.0):
        return simulate_excretion(
            drug_name="TestDrug",
            dose_mg=dose_mg,
            cl_renal_L_per_h=2.0,
            cl_biliary_L_per_h=0.5,
            cl_metabolic_L_per_h=1.5,
            vd_L=70.0,
            fu_plasma=0.1,
            t_end_h=24.0,
            dt_h=0.5,
        )

    def test_returns_excretion_result(self):
        result = self._default_result()
        assert isinstance(result, ExcretionResult)

    def test_mass_balance_approximately_unity(self):
        result = self._default_result()
        total_fe = result.fe_renal + result.fe_biliary + result.fe_metabolic
        # At 24h, most dose should be excreted; fe sums to ~1 eventually
        assert total_fe <= 1.0 + 1e-6

    def test_cumulative_renal_monotonically_increasing(self):
        result = self._default_result()
        for i in range(1, len(result.cumulative_renal_mg)):
            assert result.cumulative_renal_mg[i] >= result.cumulative_renal_mg[i - 1]

    def test_cumulative_biliary_monotonically_increasing(self):
        result = self._default_result()
        for i in range(1, len(result.cumulative_biliary_mg)):
            assert result.cumulative_biliary_mg[i] >= result.cumulative_biliary_mg[i - 1]

    def test_cumulative_metabolized_monotonically_increasing(self):
        result = self._default_result()
        for i in range(1, len(result.cumulative_metabolized_mg)):
            assert result.cumulative_metabolized_mg[i] >= result.cumulative_metabolized_mg[i - 1]

    def test_double_dose_double_cumulative(self):
        res1 = self._default_result(dose_mg=100.0)
        res2 = self._default_result(dose_mg=200.0)
        # Final cumulative amounts should scale linearly
        assert res2.cumulative_renal_mg[-1] == pytest.approx(
            2.0 * res1.cumulative_renal_mg[-1], rel=0.01
        )
        assert res2.cumulative_biliary_mg[-1] == pytest.approx(
            2.0 * res1.cumulative_biliary_mg[-1], rel=0.01
        )

    def test_c_plasma_decreasing(self):
        result = self._default_result()
        for i in range(1, len(result.c_plasma_mg_L)):
            assert result.c_plasma_mg_L[i] <= result.c_plasma_mg_L[i - 1] + 1e-9

    def test_initial_concentration_equals_dose_over_vd(self):
        result = self._default_result(dose_mg=100.0)
        assert result.c_plasma_mg_L[0] == pytest.approx(100.0 / 70.0, rel=1e-6)

    def test_t_half_positive(self):
        result = self._default_result()
        assert result.t_half_h > 0

    def test_t_half_formula(self):
        # CL_total = 2+0.5+1.5 = 4 L/h, Vd=70 → t½ = ln2*70/4
        result = self._default_result()
        expected_t_half = math.log(2) * 70.0 / 4.0
        assert result.t_half_h == pytest.approx(expected_t_half, rel=1e-6)

    def test_higher_renal_cl_higher_fe_renal(self):
        res_low = simulate_excretion("Drug", 100.0, 0.5, 0.5, 1.5, t_end_h=48.0)
        res_high = simulate_excretion("Drug", 100.0, 3.0, 0.5, 1.5, t_end_h=48.0)
        assert res_high.fe_renal > res_low.fe_renal

    def test_drug_name_preserved(self):
        result = simulate_excretion("Aspirin", 500.0, 1.0, 0.2, 0.5)
        assert result.drug_name == "Aspirin"

    def test_invalid_negative_dose(self):
        with pytest.raises(ValueError):
            simulate_excretion("Drug", -100.0, 1.0, 0.5, 0.5)

    def test_invalid_negative_cl_renal(self):
        with pytest.raises(ValueError):
            simulate_excretion("Drug", 100.0, -1.0, 0.5, 0.5)

    def test_invalid_route(self):
        with pytest.raises(ValueError):
            simulate_excretion("Drug", 100.0, 1.0, 0.5, 0.5, route="sublingual")

    def test_oral_route_accepted(self):
        result = simulate_excretion("Drug", 100.0, 1.0, 0.5, 0.5, route="oral")
        assert result is not None

    def test_notes_not_empty(self):
        result = self._default_result()
        assert len(result.notes) > 0

    def test_times_start_at_zero(self):
        result = self._default_result()
        assert result.times_h[0] == pytest.approx(0.0)

    def test_times_end_at_t_end(self):
        result = simulate_excretion("Drug", 100.0, 1.0, 0.5, 0.5, t_end_h=10.0, dt_h=1.0)
        assert result.times_h[-1] == pytest.approx(10.0, abs=0.01)
