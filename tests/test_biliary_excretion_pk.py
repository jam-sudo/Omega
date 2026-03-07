"""Tests for biliary excretion PK model (Phase 729)."""

import math

import pytest

from omega_pbpk.core.biliary_excretion_pk import (
    BiliaryExcretionResult,
    estimate_biliary_cl,
    simulate_biliary_excretion_pk,
)


def make_default() -> BiliaryExcretionResult:
    return simulate_biliary_excretion_pk(
        drug_name="TestDrug",
        dose_mg=100.0,
        mw_da=400.0,
        cl_metabolic_L_per_h=3.0,
        cl_renal_L_per_h=1.0,
        cl_biliary_base_L_per_h=1.0,
        vd_L=50.0,
        t_end_h=24.0,
        dt_h=0.1,
    )


class TestBiliaryExcretionResultType:
    def test_returns_correct_type(self):
        result = make_default()
        assert isinstance(result, BiliaryExcretionResult)

    def test_drug_name_preserved(self):
        result = make_default()
        assert result.drug_name == "TestDrug"

    def test_dose_mg_preserved(self):
        result = make_default()
        assert result.dose_mg == 100.0

    def test_mw_da_preserved(self):
        result = make_default()
        assert result.mw_da == 400.0


class TestIVBolusCmax:
    def test_cmax_equals_dose_over_vd(self):
        """IV bolus: Cmax at t=0 should be dose/Vd."""
        result = simulate_biliary_excretion_pk(dose_mg=200.0, vd_L=50.0, mw_da=400.0)
        expected_cmax = 200.0 / 50.0  # 4.0 mg/L
        assert abs(result.cmax_mg_L - expected_cmax) < 0.01

    def test_cmax_is_first_concentration(self):
        result = make_default()
        assert result.cmax_mg_L == result.c_plasma_mg_L[0]


class TestHalfLife:
    def test_t_half_formula(self):
        """t½ = Vd * ln(2) / CL_total."""
        result = simulate_biliary_excretion_pk(
            mw_da=400.0,  # low MW → CL_bil_actual = 1.0 * 0.1 = 0.1
            cl_metabolic_L_per_h=3.0,
            cl_renal_L_per_h=1.0,
            cl_biliary_base_L_per_h=1.0,
            vd_L=50.0,
        )
        # CL_bil = 0.1, CL_total = 3.0 + 1.0 + 0.1 = 4.1
        expected_t_half = math.log(2) * 50.0 / 4.1
        assert abs(result.t_half_h - expected_t_half) < 0.01


class TestBiliaryCLEstimation:
    def test_high_mw_higher_cl_than_low_mw(self):
        """High MW (600) → higher CL_biliary than low MW (300)."""
        cl_high = estimate_biliary_cl(600.0, 1.0)
        cl_low = estimate_biliary_cl(300.0, 1.0)
        assert cl_high > cl_low

    def test_low_mw_uses_small_factor(self):
        """MW < 500 → CL_bil = base * 0.1, so < 0.5 * base."""
        cl_low = estimate_biliary_cl(400.0, 1.0)
        assert cl_low < 0.5

    def test_estimate_biliary_cl_high_mw_above_base(self):
        """For MW=600 > 500: CL_bil = base * sqrt(600/500) > base."""
        cl = estimate_biliary_cl(600.0, 1.0)
        expected = 1.0 * math.sqrt(600.0 / 500.0)
        assert abs(cl - expected) < 1e-9

    def test_estimate_biliary_cl_low_mw_is_tenth(self):
        """For MW=300 < 500: CL_bil = base * 0.1."""
        cl = estimate_biliary_cl(300.0, 2.0)
        assert abs(cl - 0.2) < 1e-9


class TestFractionEliminated:
    def test_fe_sum_to_one(self):
        """fe_biliary + fe_renal + fe_metabolic should sum to 1.0."""
        result = make_default()
        total = result.fe_biliary + result.fe_renal + result.fe_metabolic
        assert abs(total - 1.0) < 1e-9

    def test_fe_biliary_positive_when_cl_bil_positive(self):
        result = make_default()
        assert result.fe_biliary > 0.0

    def test_fe_renal_correct_ratio(self):
        result = simulate_biliary_excretion_pk(
            mw_da=400.0,
            cl_metabolic_L_per_h=4.0,
            cl_renal_L_per_h=2.0,
            cl_biliary_base_L_per_h=0.0,
        )
        # CL_bil = 0, CL_total = 6.0; fe_renal = 2/6
        assert abs(result.fe_renal - 2.0 / 6.0) < 1e-9

    def test_fe_metabolic_correct_ratio(self):
        result = simulate_biliary_excretion_pk(
            mw_da=400.0,
            cl_metabolic_L_per_h=4.0,
            cl_renal_L_per_h=2.0,
            cl_biliary_base_L_per_h=0.0,
        )
        assert abs(result.fe_metabolic - 4.0 / 6.0) < 1e-9


class TestAUCAndBile:
    def test_auc_positive(self):
        result = make_default()
        assert result.auc_mg_h_per_L > 0.0

    def test_cumulative_bile_positive(self):
        result = make_default()
        assert result.cumulative_bile_mg > 0.0

    def test_times_and_concentrations_same_length(self):
        result = make_default()
        assert len(result.times_h) == len(result.c_plasma_mg_L)
        assert len(result.times_h) == len(result.a_bile_mg)


class TestHighVsLowMWSimulation:
    def test_high_mw_higher_fe_biliary(self):
        """High MW (600 Da) should have higher biliary fraction than low MW (300 Da)."""
        result_high = simulate_biliary_excretion_pk(
            mw_da=600.0,
            cl_biliary_base_L_per_h=1.0,
            cl_metabolic_L_per_h=3.0,
            cl_renal_L_per_h=1.0,
        )
        result_low = simulate_biliary_excretion_pk(
            mw_da=300.0,
            cl_biliary_base_L_per_h=1.0,
            cl_metabolic_L_per_h=3.0,
            cl_renal_L_per_h=1.0,
        )
        assert result_high.fe_biliary > result_low.fe_biliary


class TestLinearDoseScaling:
    def test_double_dose_doubles_cmax(self):
        result_1x = simulate_biliary_excretion_pk(dose_mg=100.0, mw_da=400.0)
        result_2x = simulate_biliary_excretion_pk(dose_mg=200.0, mw_da=400.0)
        assert abs(result_2x.cmax_mg_L / result_1x.cmax_mg_L - 2.0) < 1e-6

    def test_double_dose_doubles_auc(self):
        result_1x = simulate_biliary_excretion_pk(dose_mg=100.0, mw_da=400.0)
        result_2x = simulate_biliary_excretion_pk(dose_mg=200.0, mw_da=400.0)
        assert abs(result_2x.auc_mg_h_per_L / result_1x.auc_mg_h_per_L - 2.0) < 1e-3


class TestNotes:
    def test_notes_nonempty(self):
        result = make_default()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_notes_contains_drug_name(self):
        result = simulate_biliary_excretion_pk(drug_name="Irinotecan", mw_da=600.0)
        assert "Irinotecan" in result.notes


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_biliary_excretion_pk(dose_mg=-10.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_biliary_excretion_pk(dose_mg=0.0)

    def test_all_cl_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_biliary_excretion_pk(
                dose_mg=100.0,
                cl_metabolic_L_per_h=0.0,
                cl_renal_L_per_h=0.0,
                cl_biliary_base_L_per_h=0.0,
            )

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError):
            simulate_biliary_excretion_pk(dose_mg=100.0, vd_L=-5.0)
