"""Tests for Phase 807 — Pancreatic PK Model."""

import pytest

from omega_pbpk.core.pancreatic_pk import (
    PancreaticPKResult,
    simulate_pancreatic_pk,
)


class TestReturnType:
    def test_returns_pancreatic_pk_result(self):
        result = simulate_pancreatic_pk("TestDrug", dose_mg=100.0)
        assert isinstance(result, PancreaticPKResult)

    def test_times_h_is_list(self):
        result = simulate_pancreatic_pk("TestDrug", dose_mg=100.0)
        assert isinstance(result.times_h, list)

    def test_c_plasma_is_list(self):
        result = simulate_pancreatic_pk("TestDrug", dose_mg=100.0)
        assert isinstance(result.c_plasma_mg_L, list)

    def test_c_pancreas_is_list(self):
        result = simulate_pancreatic_pk("TestDrug", dose_mg=100.0)
        assert isinstance(result.c_pancreas_mg_L, list)

    def test_drug_name_preserved(self):
        result = simulate_pancreatic_pk("Gemcitabine", dose_mg=500.0)
        assert result.drug_name == "Gemcitabine"

    def test_dose_preserved(self):
        result = simulate_pancreatic_pk("Insulin", dose_mg=50.0)
        assert result.dose_mg == 50.0


class TestPositiveValues:
    def test_cmax_plasma_positive(self):
        result = simulate_pancreatic_pk("Drug", dose_mg=100.0)
        assert result.cmax_plasma > 0.0

    def test_cmax_pancreas_positive(self):
        result = simulate_pancreatic_pk("Drug", dose_mg=100.0)
        assert result.cmax_pancreas > 0.0

    def test_auc_plasma_positive(self):
        result = simulate_pancreatic_pk("Drug", dose_mg=100.0)
        assert result.auc_plasma > 0.0

    def test_auc_pancreas_positive(self):
        result = simulate_pancreatic_pk("Drug", dose_mg=100.0)
        assert result.auc_pancreas > 0.0

    def test_t_half_pancreas_positive(self):
        result = simulate_pancreatic_pk("Drug", dose_mg=100.0)
        assert result.t_half_pancreas_h > 0.0

    def test_portal_delivered_pct_valid_range(self):
        result = simulate_pancreatic_pk("Drug", dose_mg=100.0, f_portal=0.7)
        assert 0.0 <= result.portal_delivered_pct <= 100.0

    def test_arterial_delivered_pct_valid_range(self):
        result = simulate_pancreatic_pk("Drug", dose_mg=100.0, f_portal=0.7)
        assert 0.0 <= result.arterial_delivered_pct <= 100.0

    def test_portal_plus_arterial_equals_100(self):
        result = simulate_pancreatic_pk("Drug", dose_mg=100.0, f_portal=0.6)
        assert abs(result.portal_delivered_pct + result.arterial_delivered_pct - 100.0) < 1e-9


class TestLinearDoseScaling:
    def test_cmax_plasma_scales_with_dose(self):
        r1 = simulate_pancreatic_pk("Drug", dose_mg=100.0)
        r2 = simulate_pancreatic_pk("Drug", dose_mg=200.0)
        ratio = r2.cmax_plasma / r1.cmax_plasma
        assert abs(ratio - 2.0) < 0.01

    def test_auc_plasma_scales_with_dose(self):
        r1 = simulate_pancreatic_pk("Drug", dose_mg=100.0)
        r2 = simulate_pancreatic_pk("Drug", dose_mg=300.0)
        ratio = r2.auc_plasma / r1.auc_plasma
        assert abs(ratio - 3.0) < 0.01

    def test_auc_pancreas_scales_with_dose(self):
        r1 = simulate_pancreatic_pk("Drug", dose_mg=100.0)
        r2 = simulate_pancreatic_pk("Drug", dose_mg=200.0)
        ratio = r2.auc_pancreas / r1.auc_pancreas
        assert abs(ratio - 2.0) < 0.01


class TestRoutes:
    def test_iv_route_works(self):
        result = simulate_pancreatic_pk("Drug", dose_mg=100.0, route="iv")
        assert result.cmax_plasma > 0.0

    def test_oral_route_works(self):
        result = simulate_pancreatic_pk("Drug", dose_mg=100.0, route="oral")
        assert result.cmax_plasma > 0.0

    def test_oral_cmax_less_than_iv_for_same_dose(self):
        r_iv = simulate_pancreatic_pk("Drug", dose_mg=100.0, route="iv", t_end_h=12.0)
        r_oral = simulate_pancreatic_pk(
            "Drug", dose_mg=100.0, route="oral", f_oral=0.5, t_end_h=12.0
        )
        # IV gives immediate high plasma concentration; oral typically lower Cmax
        assert r_iv.cmax_plasma > r_oral.cmax_plasma


class TestKpEffect:
    def test_higher_kp_higher_pancreas_concentration(self):
        r_low = simulate_pancreatic_pk("Drug", dose_mg=100.0, kp_pancreas=1.0)
        r_high = simulate_pancreatic_pk("Drug", dose_mg=100.0, kp_pancreas=5.0)
        assert r_high.cmax_pancreas > r_low.cmax_pancreas


class TestValidation:
    def test_dose_zero_raises_error(self):
        with pytest.raises(ValueError):
            simulate_pancreatic_pk("Drug", dose_mg=0.0)

    def test_dose_negative_raises_error(self):
        with pytest.raises(ValueError):
            simulate_pancreatic_pk("Drug", dose_mg=-50.0)

    def test_f_portal_negative_raises_error(self):
        with pytest.raises(ValueError):
            simulate_pancreatic_pk("Drug", dose_mg=100.0, f_portal=-0.1)

    def test_f_portal_greater_than_1_raises_error(self):
        with pytest.raises(ValueError):
            simulate_pancreatic_pk("Drug", dose_mg=100.0, f_portal=1.5)

    def test_invalid_route_raises_error(self):
        with pytest.raises(ValueError):
            simulate_pancreatic_pk("Drug", dose_mg=100.0, route="subcutaneous")
