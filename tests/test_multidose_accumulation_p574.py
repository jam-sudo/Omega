"""Tests for Phase 574 - Multi-Dose Steady State Accumulation."""

import pytest

from omega_pbpk.clinical.multidose_accumulation_p574 import (
    MultidoseAccumulationResult,
    simulate_multidose_accumulation,
)


def _default_result(**kwargs):
    params = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        ka_per_h=1.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        dosing_interval_h=24.0,
        n_doses=10,
        f_oral=0.8,
    )
    params.update(kwargs)
    return simulate_multidose_accumulation(**params)


class TestReturnTypeAndStructure:
    def test_return_type(self):
        r = _default_result()
        assert isinstance(r, MultidoseAccumulationResult)

    def test_list_lengths_match(self):
        r = _default_result()
        assert len(r.c_systemic_mg_L) == len(r.times_h)

    def test_times_start_at_zero(self):
        r = _default_result()
        assert r.times_h[0] == 0.0

    def test_c_starts_at_zero(self):
        r = _default_result()
        assert r.c_systemic_mg_L[0] == 0.0


class TestAccumulation:
    def test_accumulation_ratio_ge_1(self):
        r = _default_result()
        assert r.accumulation_ratio >= 1.0

    def test_cmax_ss_gt_cmin_ss(self):
        r = _default_result()
        assert r.cmax_ss_mg_L > r.cmin_ss_mg_L

    def test_longer_thalf_higher_accumulation(self):
        """Lower CL → longer t_half → more accumulation."""
        r_short = _default_result(cl_L_per_h=10.0)  # shorter t_half
        r_long = _default_result(cl_L_per_h=1.0)  # longer t_half
        assert r_long.accumulation_ratio > r_short.accumulation_ratio

    def test_fluctuation_pct_positive(self):
        r = _default_result()
        assert r.fluctuation_pct > 0

    def test_doses_to_90pct_ss_ge_1(self):
        r = _default_result()
        assert r.doses_to_90pct_ss >= 1

    def test_doses_to_90pct_ss_le_n_doses(self):
        r = _default_result()
        assert r.doses_to_90pct_ss <= r.n_doses

    def test_t_half_positive(self):
        r = _default_result()
        assert r.t_half_h > 0


class TestDoseLinearity:
    def test_double_dose_doubles_cmax_ss(self):
        r1 = _default_result(dose_mg=100.0)
        r2 = _default_result(dose_mg=200.0)
        ratio = r2.cmax_ss_mg_L / r1.cmax_ss_mg_L
        assert abs(ratio - 2.0) < 0.15  # within 15% tolerance


class TestSteadyStateMetrics:
    def test_cavg_ss_between_cmin_and_cmax(self):
        r = _default_result()
        assert r.cmin_ss_mg_L <= r.cavg_ss_mg_L <= r.cmax_ss_mg_L

    def test_cmax_ss_positive(self):
        r = _default_result()
        assert r.cmax_ss_mg_L > 0

    def test_cmin_ss_non_negative(self):
        r = _default_result()
        assert r.cmin_ss_mg_L >= 0

    def test_drug_name_stored(self):
        r = _default_result(drug_name="Aspirin")
        assert r.drug_name == "Aspirin"

    def test_dose_stored(self):
        r = _default_result(dose_mg=250.0)
        assert r.dose_mg == 250.0

    def test_n_doses_stored(self):
        r = _default_result(n_doses=15)
        assert r.n_doses == 15


class TestMoreDosesMoreAccumulation:
    def test_more_doses_higher_cmax_ss(self):
        r_few = _default_result(n_doses=3, cl_L_per_h=1.0)
        r_many = _default_result(n_doses=20, cl_L_per_h=1.0)
        assert r_many.cmax_ss_mg_L >= r_few.cmax_ss_mg_L


class TestValidation:
    def test_dose_must_be_positive(self):
        with pytest.raises(ValueError):
            _default_result(dose_mg=0)

    def test_ka_must_be_positive(self):
        with pytest.raises(ValueError):
            _default_result(ka_per_h=-1)

    def test_cl_must_be_positive(self):
        with pytest.raises(ValueError):
            _default_result(cl_L_per_h=0)

    def test_vd_must_be_positive(self):
        with pytest.raises(ValueError):
            _default_result(vd_L=-10)

    def test_n_doses_must_be_positive(self):
        with pytest.raises(ValueError):
            _default_result(n_doses=0)
