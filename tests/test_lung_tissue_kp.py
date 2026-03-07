"""Tests for prediction/lung_tissue_kp.py (Phase 554)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.lung_tissue_kp import (
    InhaledLungPKResult,
    inhaled_lung_deposit_pk,
    lung_concentration_after_iv,
    predict_kp_lung,
    predict_kp_lung_multiple,
)


class TestPredictKpLung:
    def test_returns_dict_with_required_keys(self):
        result = predict_kp_lung(logP=2.0, pka=7.0, drug_type="neutral", fu_plasma=0.5)
        assert "kp_lung" in result
        assert "kp_lung_to_plasma" in result
        assert "distribution_category" in result

    def test_kp_positive(self):
        result = predict_kp_lung(logP=2.0, pka=7.0, drug_type="neutral", fu_plasma=0.5)
        assert result["kp_lung"] > 0.0

    def test_lipophilic_drug_higher_kp(self):
        r_low = predict_kp_lung(logP=0.0, pka=7.0, drug_type="neutral", fu_plasma=0.5)
        r_high = predict_kp_lung(logP=4.0, pka=7.0, drug_type="neutral", fu_plasma=0.5)
        assert r_high["kp_lung"] > r_low["kp_lung"]

    def test_basic_drug_higher_kp_than_neutral(self):
        r_neutral = predict_kp_lung(logP=2.0, pka=7.0, drug_type="neutral", fu_plasma=0.5)
        r_base = predict_kp_lung(logP=2.0, pka=9.0, drug_type="strong_base", fu_plasma=0.5)
        assert r_base["kp_lung"] > r_neutral["kp_lung"]

    def test_distribution_category_low(self):
        result = predict_kp_lung(logP=-2.0, pka=7.0, drug_type="neutral", fu_plasma=0.9)
        assert result["distribution_category"] == "low"

    def test_distribution_category_high(self):
        result = predict_kp_lung(logP=5.0, pka=9.0, drug_type="strong_base", fu_plasma=0.01)
        assert result["distribution_category"] == "high"

    def test_low_fu_plasma_increases_kp(self):
        r_high_fu = predict_kp_lung(logP=1.0, pka=7.0, drug_type="neutral", fu_plasma=0.9)
        r_low_fu = predict_kp_lung(logP=1.0, pka=7.0, drug_type="neutral", fu_plasma=0.05)
        assert r_low_fu["kp_lung"] > r_high_fu["kp_lung"]

    def test_fu_plasma_zero_raises(self):
        with pytest.raises(ValueError):
            predict_kp_lung(logP=2.0, pka=7.0, drug_type="neutral", fu_plasma=0.0)

    def test_fu_plasma_above_one_raises(self):
        with pytest.raises(ValueError):
            predict_kp_lung(logP=2.0, pka=7.0, drug_type="neutral", fu_plasma=1.5)


class TestPredictKpLungMultiple:
    def _drugs(self):
        return [
            {"name": "DrugA", "logP": 0.5, "pka": 7.0, "drug_type": "neutral", "fu_plasma": 0.8},
            {"name": "DrugB", "logP": 3.0, "pka": 9.0, "drug_type": "weak_base", "fu_plasma": 0.3},
            {
                "name": "DrugC",
                "logP": 5.0,
                "pka": 9.0,
                "drug_type": "strong_base",
                "fu_plasma": 0.1,
            },
        ]

    def test_returns_list(self):
        result = predict_kp_lung_multiple(self._drugs())
        assert isinstance(result, list)

    def test_length_matches_input(self):
        result = predict_kp_lung_multiple(self._drugs())
        assert len(result) == 3

    def test_sorted_descending_kp(self):
        result = predict_kp_lung_multiple(self._drugs())
        kps = [r["kp_lung"] for r in result]
        assert kps == sorted(kps, reverse=True)

    def test_each_result_has_kp_key(self):
        result = predict_kp_lung_multiple(self._drugs())
        for entry in result:
            assert "kp_lung" in entry
            assert "distribution_category" in entry

    def test_empty_list_returns_empty(self):
        result = predict_kp_lung_multiple([])
        assert result == []


class TestLungConcentrationAfterIV:
    def test_returns_dict_with_required_keys(self):
        result = lung_concentration_after_iv(dose_mg=100.0, kp_lung=5.0)
        assert "c_plasma_mg_L" in result
        assert "c_lung_mg_L" in result
        assert "lung_dose_fraction" in result

    def test_c_lung_equals_kp_times_plasma(self):
        result = lung_concentration_after_iv(dose_mg=100.0, kp_lung=3.0, vd_plasma_L=50.0)
        assert result["c_lung_mg_L"] == pytest.approx(3.0 * result["c_plasma_mg_L"])

    def test_c_plasma_equals_dose_over_vd(self):
        result = lung_concentration_after_iv(dose_mg=100.0, kp_lung=2.0, vd_plasma_L=50.0)
        assert result["c_plasma_mg_L"] == pytest.approx(100.0 / 50.0)

    def test_lung_dose_fraction_in_0_1_range(self):
        result = lung_concentration_after_iv(dose_mg=100.0, kp_lung=5.0, vd_plasma_L=50.0)
        assert 0.0 <= result["lung_dose_fraction"] <= 1.0

    def test_higher_kp_higher_lung_fraction(self):
        r_low = lung_concentration_after_iv(dose_mg=100.0, kp_lung=1.0, vd_plasma_L=50.0)
        r_high = lung_concentration_after_iv(dose_mg=100.0, kp_lung=10.0, vd_plasma_L=50.0)
        assert r_high["lung_dose_fraction"] > r_low["lung_dose_fraction"]

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            lung_concentration_after_iv(dose_mg=0.0, kp_lung=5.0)

    def test_zero_kp_raises(self):
        with pytest.raises(ValueError):
            lung_concentration_after_iv(dose_mg=100.0, kp_lung=0.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError):
            lung_concentration_after_iv(dose_mg=100.0, kp_lung=5.0, vd_plasma_L=0.0)


class TestInhaledLungDepositPK:
    def test_returns_result_type(self):
        result = inhaled_lung_deposit_pk("Drug", deposited_dose_mg=1.0, kp_lung=3.0)
        assert isinstance(result, InhaledLungPKResult)

    def test_cmax_plasma_positive(self):
        result = inhaled_lung_deposit_pk("Drug", deposited_dose_mg=1.0, kp_lung=3.0)
        assert result.cmax_plasma > 0.0

    def test_auc_plasma_positive(self):
        result = inhaled_lung_deposit_pk("Drug", deposited_dose_mg=1.0, kp_lung=3.0)
        assert result.auc_plasma > 0.0

    def test_f_absorbed_in_0_1(self):
        result = inhaled_lung_deposit_pk("Drug", deposited_dose_mg=1.0, kp_lung=3.0)
        assert 0.0 <= result.f_absorbed <= 1.0

    def test_times_start_at_zero(self):
        result = inhaled_lung_deposit_pk("Drug", deposited_dose_mg=1.0, kp_lung=3.0)
        assert result.times_h[0] == pytest.approx(0.0)

    def test_lung_amount_decreases_over_time(self):
        result = inhaled_lung_deposit_pk("Drug", deposited_dose_mg=1.0, kp_lung=3.0, t_end_h=12.0)
        assert result.a_lung_mg[-1] < result.a_lung_mg[0]

    def test_double_dose_doubles_cmax(self):
        r1 = inhaled_lung_deposit_pk("Drug", deposited_dose_mg=1.0, kp_lung=3.0)
        r2 = inhaled_lung_deposit_pk("Drug", deposited_dose_mg=2.0, kp_lung=3.0)
        assert r2.cmax_plasma == pytest.approx(2.0 * r1.cmax_plasma, rel=0.05)

    def test_higher_absorption_cl_higher_cmax(self):
        r_slow = inhaled_lung_deposit_pk(
            "Drug", deposited_dose_mg=1.0, kp_lung=3.0, cl_absorption_L_per_h=0.5
        )
        r_fast = inhaled_lung_deposit_pk(
            "Drug", deposited_dose_mg=1.0, kp_lung=3.0, cl_absorption_L_per_h=5.0
        )
        assert r_fast.cmax_plasma > r_slow.cmax_plasma

    def test_high_mucus_clearance_lower_f_absorbed(self):
        r_low_mucus = inhaled_lung_deposit_pk(
            "Drug", deposited_dose_mg=1.0, kp_lung=3.0, cl_mucus_L_per_h=0.1
        )
        r_high_mucus = inhaled_lung_deposit_pk(
            "Drug", deposited_dose_mg=1.0, kp_lung=3.0, cl_mucus_L_per_h=5.0
        )
        assert r_high_mucus.f_absorbed < r_low_mucus.f_absorbed

    def test_notes_nonempty(self):
        result = inhaled_lung_deposit_pk("Drug", deposited_dose_mg=1.0, kp_lung=3.0)
        assert len(result.notes) > 0

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            inhaled_lung_deposit_pk("Drug", deposited_dose_mg=0.0, kp_lung=3.0)

    def test_negative_t_end_raises(self):
        with pytest.raises(ValueError):
            inhaled_lung_deposit_pk("Drug", deposited_dose_mg=1.0, kp_lung=3.0, t_end_h=-1.0)
