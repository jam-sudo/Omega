"""Tests for Phase 228: Biliary Excretion PK Model (p228 suffix)."""

import pytest

from omega_pbpk.core.biliary_excretion_pk_p228 import (
    BiliaryExcretionResult,
    compare_mw_effect,
    simulate_biliary_excretion,
)


class TestReturnType:
    def test_returns_biliary_excretion_result(self):
        result = simulate_biliary_excretion(
            drug_name="TestDrug",
            dose_mg=100.0,
            mw_Da=400.0,
            cl_L_per_h=10.0,
            vd_L=50.0,
            f_reabs=0.0,
        )
        assert isinstance(result, BiliaryExcretionResult)

    def test_result_is_dataclass(self):
        from dataclasses import fields

        result = simulate_biliary_excretion("Drug", 100.0, 400.0, 10.0, 50.0, 0.0)
        assert len(fields(result)) > 0


class TestFieldPreservation:
    def test_drug_name_preserved(self):
        result = simulate_biliary_excretion("MyDrug", 50.0, 350.0, 5.0, 30.0, 0.0)
        assert result.drug_name == "MyDrug"

    def test_dose_mg_preserved(self):
        result = simulate_biliary_excretion("Drug", 200.0, 350.0, 5.0, 30.0, 0.0)
        assert result.dose_mg == pytest.approx(200.0)

    def test_mw_Da_preserved(self):
        result = simulate_biliary_excretion("Drug", 100.0, 450.0, 10.0, 50.0, 0.0)
        assert result.mw_Da == pytest.approx(450.0)

    def test_f_reabs_preserved(self):
        result = simulate_biliary_excretion("Drug", 100.0, 400.0, 10.0, 50.0, 0.3)
        assert result.f_reabs == pytest.approx(0.3)


class TestFBiliaryRange:
    def test_f_biliary_in_0_1(self):
        for mw in [100.0, 300.0, 400.0, 600.0, 900.0]:
            result = simulate_biliary_excretion("Drug", 100.0, mw, 10.0, 50.0, 0.0)
            assert 0.0 <= result.f_biliary <= 1.0

    def test_f_biliary_low_mw(self):
        result = simulate_biliary_excretion("Small", 100.0, 200.0, 10.0, 50.0, 0.0)
        assert result.f_biliary < 0.5  # logistic(0.01*(200-400)) = logistic(-2) < 0.5

    def test_f_biliary_high_mw(self):
        result = simulate_biliary_excretion("Large", 100.0, 600.0, 10.0, 50.0, 0.0)
        assert result.f_biliary > 0.5  # logistic(0.01*(600-400)) = logistic(2) > 0.5


class TestMWEffect:
    def test_high_mw_higher_f_biliary_than_low_mw(self):
        low = simulate_biliary_excretion("Low", 100.0, 200.0, 10.0, 50.0, 0.0)
        high = simulate_biliary_excretion("High", 100.0, 600.0, 10.0, 50.0, 0.0)
        assert high.f_biliary > low.f_biliary

    def test_400_da_midpoint(self):
        result = simulate_biliary_excretion("Mid", 100.0, 400.0, 10.0, 50.0, 0.0)
        assert result.f_biliary == pytest.approx(0.5, abs=1e-6)


class TestPKMetrics:
    def test_cmax_positive(self):
        result = simulate_biliary_excretion("Drug", 100.0, 400.0, 10.0, 50.0, 0.0)
        assert result.cmax_plasma_mg_L > 0.0

    def test_auc_positive(self):
        result = simulate_biliary_excretion("Drug", 100.0, 400.0, 10.0, 50.0, 0.0)
        assert result.auc_plasma_mg_h_per_L > 0.0

    def test_cmax_equals_initial_concentration(self):
        # For IV bolus with no recirculation, Cmax = dose/Vd
        result = simulate_biliary_excretion("Drug", 100.0, 400.0, 10.0, 50.0, 0.0)
        expected_cmax = 100.0 / 50.0  # dose/Vd = 2.0 mg/L
        assert result.cmax_plasma_mg_L == pytest.approx(expected_cmax, rel=0.01)

    def test_times_list_has_entries(self):
        result = simulate_biliary_excretion("Drug", 100.0, 400.0, 10.0, 50.0, 0.0, t_end_h=12.0)
        assert len(result.times_h) > 0

    def test_c_plasma_list_matches_times(self):
        result = simulate_biliary_excretion("Drug", 100.0, 400.0, 10.0, 50.0, 0.0, t_end_h=12.0)
        assert len(result.c_plasma_mg_L) == len(result.times_h)

    def test_a_bile_list_matches_times(self):
        result = simulate_biliary_excretion("Drug", 100.0, 400.0, 10.0, 50.0, 0.0, t_end_h=12.0)
        assert len(result.a_bile_mg) == len(result.times_h)


class TestRecirculation:
    def test_f_reabs_zero_vs_nonzero_auc(self):
        no_reabs = simulate_biliary_excretion("Drug", 100.0, 500.0, 10.0, 50.0, 0.0, t_end_h=72.0)
        with_reabs = simulate_biliary_excretion("Drug", 100.0, 500.0, 10.0, 50.0, 0.8, t_end_h=72.0)
        # With reabsorption, drug returns to plasma → higher AUC
        assert with_reabs.auc_plasma_mg_h_per_L > no_reabs.auc_plasma_mg_h_per_L

    def test_fe_biliary_pct_in_bounds(self):
        result = simulate_biliary_excretion("Drug", 100.0, 600.0, 10.0, 50.0, 0.0)
        assert 0.0 <= result.fe_biliary_pct <= 100.0

    def test_fe_biliary_pct_zero_for_small_mw(self):
        # Very low MW → f_biliary ≈ 0 → fe_biliary_pct ≈ 0
        result = simulate_biliary_excretion("Small", 100.0, 100.0, 10.0, 50.0, 0.0)
        assert result.fe_biliary_pct < 5.0


class TestCompareMWEffect:
    def test_returns_list(self):
        results = compare_mw_effect("Drug", 100.0, 10.0, 50.0, [200.0, 400.0, 600.0])
        assert isinstance(results, list)

    def test_sorted_by_f_biliary_descending(self):
        results = compare_mw_effect("Drug", 100.0, 10.0, 50.0, [200.0, 400.0, 600.0])
        biliary_vals = [r.f_biliary for r in results]
        assert biliary_vals == sorted(biliary_vals, reverse=True)

    def test_high_mw_first_in_sorted(self):
        results = compare_mw_effect("Drug", 100.0, 10.0, 50.0, [200.0, 600.0])
        assert results[0].mw_Da == pytest.approx(600.0)

    def test_single_mw(self):
        results = compare_mw_effect("Drug", 100.0, 10.0, 50.0, [400.0])
        assert len(results) == 1

    def test_all_results_are_biliary_excretion_result(self):
        results = compare_mw_effect("Drug", 100.0, 10.0, 50.0, [200.0, 400.0, 600.0])
        assert all(isinstance(r, BiliaryExcretionResult) for r in results)


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_biliary_excretion("Drug", 0.0, 400.0, 10.0, 50.0, 0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError):
            simulate_biliary_excretion("Drug", -50.0, 400.0, 10.0, 50.0, 0.0)

    def test_mw_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_biliary_excretion("Drug", 100.0, 0.0, 10.0, 50.0, 0.0)

    def test_mw_negative_raises(self):
        with pytest.raises(ValueError):
            simulate_biliary_excretion("Drug", 100.0, -100.0, 10.0, 50.0, 0.0)

    def test_f_reabs_negative_raises(self):
        with pytest.raises(ValueError):
            simulate_biliary_excretion("Drug", 100.0, 400.0, 10.0, 50.0, -0.1)

    def test_f_reabs_greater_than_1_raises(self):
        with pytest.raises(ValueError):
            simulate_biliary_excretion("Drug", 100.0, 400.0, 10.0, 50.0, 1.1)
