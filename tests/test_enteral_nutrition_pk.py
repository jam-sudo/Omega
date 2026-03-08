"""Tests for enteral nutrition PK model (Phase 955)."""

import pytest

from omega_pbpk.core.enteral_nutrition_pk import (
    EnteralNutritionPKResult,
    compare_feeding_modes,
    simulate_en_pk,
)


def _base_result(**kwargs):
    defaults = dict(
        drug_name="phenytoin",
        dose_mg=300.0,
        interaction_category="moderate_interaction",
        feeding_mode="continuous",
        n_doses=2,
        t_end_h=24.0,
        dt_h=0.25,
    )
    defaults.update(kwargs)
    return simulate_en_pk(**defaults)


class TestReturnType:
    def test_return_type(self):
        result = _base_result()
        assert isinstance(result, EnteralNutritionPKResult)


class TestBasicPKValues:
    def test_cmax_positive(self):
        result = _base_result()
        assert result.cmax_mg_L > 0

    def test_auc_positive(self):
        result = _base_result()
        assert result.auc_mg_h_per_L > 0

    def test_fa_effective_in_range(self):
        result = _base_result()
        assert 0.0 <= result.fa_effective <= 1.0

    def test_interaction_factor_in_range(self):
        result = _base_result()
        assert 0.0 <= result.interaction_factor <= 1.0

    def test_auc_ratio_in_range(self):
        result = _base_result()
        assert 0.0 <= result.auc_ratio_vs_fasted <= 2.0

    def test_times_and_concentrations_same_length(self):
        result = _base_result()
        assert len(result.times_h) == len(result.c_plasma_mg_L)


class TestFastedMode:
    def test_fasted_auc_ratio_approx_one(self):
        result = simulate_en_pk(
            drug_name="test",
            dose_mg=100.0,
            interaction_category="moderate_interaction",
            feeding_mode="fasted",
            n_doses=1,
            t_end_h=24.0,
            dt_h=0.25,
        )
        assert abs(result.auc_ratio_vs_fasted - 1.0) < 0.01

    def test_fasted_fa_effective_approx_one(self):
        result = simulate_en_pk(
            drug_name="test",
            dose_mg=100.0,
            interaction_category="no_interaction",
            feeding_mode="fasted",
            n_doses=1,
            t_end_h=24.0,
            dt_h=0.25,
        )
        assert abs(result.fa_effective - 1.0) < 0.01


class TestInteractionCategories:
    def test_high_vs_no_interaction_auc(self):
        high = simulate_en_pk(
            drug_name="drug",
            dose_mg=100.0,
            interaction_category="high_interaction",
            feeding_mode="continuous",
            n_doses=1,
            t_end_h=24.0,
            dt_h=0.25,
        )
        no = simulate_en_pk(
            drug_name="drug",
            dose_mg=100.0,
            interaction_category="no_interaction",
            feeding_mode="continuous",
            n_doses=1,
            t_end_h=24.0,
            dt_h=0.25,
        )
        assert high.auc_mg_h_per_L < no.auc_mg_h_per_L

    def test_no_interaction_fa_effective_approx_one(self):
        result = simulate_en_pk(
            drug_name="drug",
            dose_mg=100.0,
            interaction_category="no_interaction",
            feeding_mode="continuous",
            n_doses=1,
            t_end_h=24.0,
            dt_h=0.25,
        )
        assert abs(result.fa_effective - 1.0) < 0.01

    def test_continuous_high_interaction_auc_ratio_below_half(self):
        result = simulate_en_pk(
            drug_name="phenytoin",
            dose_mg=300.0,
            interaction_category="high_interaction",
            feeding_mode="continuous",
            n_doses=1,
            t_end_h=24.0,
            dt_h=0.25,
        )
        assert result.auc_ratio_vs_fasted < 0.5


class TestNotes:
    def test_notes_non_empty(self):
        result = _base_result()
        assert len(result.notes) > 0


class TestPreservedFields:
    def test_interaction_category_preserved(self):
        result = _base_result(interaction_category="low_interaction")
        assert result.interaction_category == "low_interaction"

    def test_feeding_mode_preserved(self):
        result = _base_result(feeding_mode="cyclic")
        assert result.feeding_mode == "cyclic"

    def test_drug_name_preserved(self):
        result = _base_result(drug_name="ciprofloxacin")
        assert result.drug_name == "ciprofloxacin"


class TestCompareFeedingModes:
    def test_compare_returns_4_results(self):
        results = compare_feeding_modes(
            drug_name="drug",
            dose_mg=100.0,
            interaction_category="moderate_interaction",
            n_doses=1,
            t_end_h=24.0,
            dt_h=0.5,
        )
        assert len(results) == 4

    def test_compare_sorted_by_auc_descending(self):
        results = compare_feeding_modes(
            drug_name="drug",
            dose_mg=100.0,
            interaction_category="moderate_interaction",
            n_doses=1,
            t_end_h=24.0,
            dt_h=0.5,
        )
        aucs = [r.auc_mg_h_per_L for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_fasted_usually_highest_auc_for_interacting_drug(self):
        results = compare_feeding_modes(
            drug_name="phenytoin",
            dose_mg=300.0,
            interaction_category="high_interaction",
            n_doses=1,
            t_end_h=24.0,
            dt_h=0.5,
        )
        # Fasted should have highest AUC
        fasted = next(r for r in results if r.feeding_mode == "fasted")
        assert fasted.auc_mg_h_per_L == results[0].auc_mg_h_per_L


class TestValidation:
    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_en_pk("drug", dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_en_pk("drug", dose_mg=-10.0)

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_en_pk("drug", dose_mg=100.0, cl_L_per_h=0.0)

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_en_pk("drug", dose_mg=100.0, vd_L=-1.0)

    def test_invalid_interaction_category_raises(self):
        with pytest.raises(ValueError, match="interaction_category"):
            simulate_en_pk("drug", dose_mg=100.0, interaction_category="extreme_interaction")

    def test_invalid_feeding_mode_raises(self):
        with pytest.raises(ValueError, match="feeding_mode"):
            simulate_en_pk("drug", dose_mg=100.0, feeding_mode="drip")

    def test_n_doses_zero_raises(self):
        with pytest.raises(ValueError, match="n_doses"):
            simulate_en_pk("drug", dose_mg=100.0, n_doses=0)
