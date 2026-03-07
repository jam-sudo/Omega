"""Tests for Phase 242: spinal_cord_pk_p242."""

import pytest

from omega_pbpk.core.spinal_cord_pk_p242 import (
    SpinalCordPKResult,
    compare_spinal_routes,
    simulate_spinal_cord_pk,
)


class TestReturnType:
    def test_returns_dataclass_instance(self):
        result = simulate_spinal_cord_pk("DrugX", dose_mg=10.0, route="iv_bolus", cl_L_per_h=1.0)
        assert isinstance(result, SpinalCordPKResult)

    def test_fields_present(self):
        result = simulate_spinal_cord_pk("DrugX", dose_mg=10.0, route="iv_bolus", cl_L_per_h=1.0)
        for attr in [
            "times_h",
            "c_plasma_mg_L",
            "c_csf_mg_L",
            "c_spinal_mg_L",
            "cmax_plasma_mg_L",
            "cmax_csf_mg_L",
            "cmax_spinal_mg_L",
            "auc_plasma_mg_h_per_L",
            "auc_csf_mg_h_per_L",
            "auc_spinal_mg_h_per_L",
            "penetration_ratio_csf",
            "penetration_ratio_spinal",
        ]:
            assert hasattr(result, attr)


class TestIVBolus:
    def test_iv_plasma_starts_high(self):
        result = simulate_spinal_cord_pk("DrugX", dose_mg=10.0, route="iv_bolus", cl_L_per_h=1.0)
        assert result.c_plasma_mg_L[0] > 0

    def test_iv_csf_starts_at_zero(self):
        result = simulate_spinal_cord_pk("DrugX", dose_mg=10.0, route="iv_bolus", cl_L_per_h=1.0)
        assert result.c_csf_mg_L[0] == pytest.approx(0.0)

    def test_iv_spinal_starts_at_zero(self):
        result = simulate_spinal_cord_pk("DrugX", dose_mg=10.0, route="iv_bolus", cl_L_per_h=1.0)
        assert result.c_spinal_mg_L[0] == pytest.approx(0.0)

    def test_iv_plasma_initial_concentration(self):
        # C_plasma(0) = dose / Vp = 10 / 3.0
        result = simulate_spinal_cord_pk("DrugX", dose_mg=10.0, route="iv_bolus", cl_L_per_h=1.0)
        assert result.c_plasma_mg_L[0] == pytest.approx(10.0 / 3.0, rel=1e-6)


class TestIntrathecal:
    def test_it_csf_starts_high(self):
        result = simulate_spinal_cord_pk("DrugX", dose_mg=1.0, route="intrathecal", cl_L_per_h=1.0)
        assert result.c_csf_mg_L[0] > 0

    def test_it_plasma_starts_at_zero(self):
        result = simulate_spinal_cord_pk("DrugX", dose_mg=1.0, route="intrathecal", cl_L_per_h=1.0)
        assert result.c_plasma_mg_L[0] == pytest.approx(0.0)

    def test_it_spinal_starts_at_zero(self):
        result = simulate_spinal_cord_pk("DrugX", dose_mg=1.0, route="intrathecal", cl_L_per_h=1.0)
        assert result.c_spinal_mg_L[0] == pytest.approx(0.0)

    def test_it_csf_initial_concentration(self):
        # C_csf(0) = dose / V_csf = 1.0 / 0.15
        result = simulate_spinal_cord_pk("DrugX", dose_mg=1.0, route="intrathecal", cl_L_per_h=1.0)
        assert result.c_csf_mg_L[0] == pytest.approx(1.0 / 0.15, rel=1e-6)


class TestCmaxAUC:
    def test_cmax_plasma_positive_iv(self):
        result = simulate_spinal_cord_pk("DrugX", dose_mg=10.0, route="iv_bolus", cl_L_per_h=1.0)
        assert result.cmax_plasma_mg_L > 0

    def test_cmax_csf_positive(self):
        result = simulate_spinal_cord_pk("DrugX", dose_mg=10.0, route="iv_bolus", cl_L_per_h=1.0)
        assert result.cmax_csf_mg_L > 0

    def test_cmax_spinal_positive(self):
        result = simulate_spinal_cord_pk("DrugX", dose_mg=10.0, route="iv_bolus", cl_L_per_h=1.0)
        assert result.cmax_spinal_mg_L > 0

    def test_auc_plasma_positive_iv(self):
        result = simulate_spinal_cord_pk("DrugX", dose_mg=10.0, route="iv_bolus", cl_L_per_h=1.0)
        assert result.auc_plasma_mg_h_per_L > 0

    def test_auc_csf_positive_iv(self):
        result = simulate_spinal_cord_pk("DrugX", dose_mg=10.0, route="iv_bolus", cl_L_per_h=1.0)
        assert result.auc_csf_mg_h_per_L > 0

    def test_auc_spinal_positive_iv(self):
        result = simulate_spinal_cord_pk("DrugX", dose_mg=10.0, route="iv_bolus", cl_L_per_h=1.0)
        assert result.auc_spinal_mg_h_per_L > 0


class TestPenetrationRatios:
    def test_penetration_ratio_csf_nonneg_iv(self):
        # CSF ratio can exceed 1.0 due to volume-scaled transfer (Vp/V_csf = 20x)
        result = simulate_spinal_cord_pk("DrugX", dose_mg=10.0, route="iv_bolus", cl_L_per_h=1.0)
        assert result.penetration_ratio_csf >= 0

    def test_penetration_ratio_spinal_nonneg_iv(self):
        result = simulate_spinal_cord_pk("DrugX", dose_mg=10.0, route="iv_bolus", cl_L_per_h=1.0)
        assert result.penetration_ratio_spinal >= 0


class TestITvsIV:
    def test_it_higher_spinal_auc_than_iv(self):
        iv = simulate_spinal_cord_pk("DrugX", dose_mg=10.0, route="iv_bolus", cl_L_per_h=1.0)
        it = simulate_spinal_cord_pk("DrugX", dose_mg=10.0, route="intrathecal", cl_L_per_h=1.0)
        assert it.auc_spinal_mg_h_per_L > iv.auc_spinal_mg_h_per_L


class TestCompareRoutes:
    def test_returns_2_results(self):
        results = compare_spinal_routes("DrugX", dose_mg=10.0, cl_L_per_h=1.0)
        assert len(results) == 2

    def test_all_are_dataclass_instances(self):
        results = compare_spinal_routes("DrugX", dose_mg=10.0, cl_L_per_h=1.0)
        for r in results:
            assert isinstance(r, SpinalCordPKResult)

    def test_sorted_by_auc_spinal_descending(self):
        results = compare_spinal_routes("DrugX", dose_mg=10.0, cl_L_per_h=1.0)
        aucs = [r.auc_spinal_mg_h_per_L for r in results]
        assert aucs[0] >= aucs[1]

    def test_intrathecal_first_in_comparison(self):
        results = compare_spinal_routes("DrugX", dose_mg=10.0, cl_L_per_h=1.0)
        assert results[0].route == "intrathecal"


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_spinal_cord_pk("X", dose_mg=-1.0, route="iv_bolus", cl_L_per_h=1.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_spinal_cord_pk("X", dose_mg=0.0, route="iv_bolus", cl_L_per_h=1.0)

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_spinal_cord_pk("X", dose_mg=10.0, route="iv_bolus", cl_L_per_h=-1.0)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            simulate_spinal_cord_pk("X", dose_mg=10.0, route="oral", cl_L_per_h=1.0)
