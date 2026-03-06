"""Tests for SC/IM depot absorption PK model."""

import pytest

from omega_pbpk.core.depot_absorption import (
    DepotPKResult,
    DepotRoute,
    compare_routes,
    simulate_depot_pk,
)


class TestDepotRoute:
    def test_sc_value(self):
        assert DepotRoute.SUBCUTANEOUS.value == "sc"

    def test_im_value(self):
        assert DepotRoute.INTRAMUSCULAR.value == "im"


class TestInputValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_depot_pk("X", dose_mg=0, cl_L_per_h=5, vd_L=50, ka_per_h=0.5)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_depot_pk("X", dose_mg=100, cl_L_per_h=0, vd_L=50, ka_per_h=0.5)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_depot_pk("X", dose_mg=100, cl_L_per_h=5, vd_L=0, ka_per_h=0.5)

    def test_ka_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_depot_pk("X", dose_mg=100, cl_L_per_h=5, vd_L=50, ka_per_h=0)

    def test_bioavailability_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_depot_pk(
                "X",
                dose_mg=100,
                cl_L_per_h=5,
                vd_L=50,
                ka_per_h=0.5,
                bioavailability=0,
            )


class TestSCSimulation:
    @pytest.fixture()
    def result(self):
        return simulate_depot_pk("DrugA", dose_mg=100, cl_L_per_h=5, vd_L=50, ka_per_h=0.5)

    def test_cmax_positive(self, result):
        assert result.cmax > 0

    def test_tmax_positive(self, result):
        assert result.tmax_h > 0

    def test_auc_0_t_positive(self, result):
        assert result.auc_0_t > 0

    def test_auc_0_inf_gte_auc_0_t(self, result):
        assert result.auc_0_inf >= result.auc_0_t

    def test_all_cp_nonneg(self, result):
        assert all(c >= 0 for c in result.cp_central)

    def test_all_a_depot_nonneg(self, result):
        assert all(a >= 0 for a in result.a_depot)

    def test_a_depot_starts_at_dose(self, result):
        assert result.a_depot[0] == 100.0

    def test_a_depot_decreases(self, result):
        assert result.a_depot[-1] < result.a_depot[0]

    def test_thalf_abs_positive(self, result):
        assert result.thalf_abs_h > 0

    def test_thalf_elim_positive(self, result):
        assert result.thalf_elim_h > 0

    def test_f_abs_range(self, result):
        assert 0 <= result.f_abs <= 1

    def test_flip_flop_false(self, result):
        # ka=0.5, ke=5/50=0.1, ka > ke so flip_flop=False
        assert result.flip_flop is False


class TestFlipFlop:
    def test_flip_flop_true(self):
        # ka=0.05, ke=5/50=0.1, ka < ke so flip_flop=True
        result = simulate_depot_pk("DrugB", dose_mg=100, cl_L_per_h=5, vd_L=50, ka_per_h=0.05)
        assert result.flip_flop is True


class TestLagTime:
    def test_lag_delays_tmax(self):
        no_lag = simulate_depot_pk(
            "DrugC",
            dose_mg=100,
            cl_L_per_h=5,
            vd_L=50,
            ka_per_h=0.5,
            tlag_h=0,
        )
        with_lag = simulate_depot_pk(
            "DrugC",
            dose_mg=100,
            cl_L_per_h=5,
            vd_L=50,
            ka_per_h=0.5,
            tlag_h=2.0,
        )
        assert with_lag.tmax_h > no_lag.tmax_h


class TestIMFasterThanSC:
    def test_im_tmax_less_than_sc(self):
        sc = simulate_depot_pk(
            "DrugD",
            dose_mg=100,
            cl_L_per_h=5,
            vd_L=50,
            ka_per_h=0.5,
            route=DepotRoute.SUBCUTANEOUS,
        )
        im = simulate_depot_pk(
            "DrugD",
            dose_mg=100,
            cl_L_per_h=5,
            vd_L=50,
            ka_per_h=0.5,
            route=DepotRoute.INTRAMUSCULAR,
        )
        assert im.tmax_h < sc.tmax_h


class TestBioavailability:
    def test_auc_scales_with_f(self):
        result = simulate_depot_pk(
            "DrugE",
            dose_mg=100,
            cl_L_per_h=5,
            vd_L=50,
            ka_per_h=0.5,
            bioavailability=0.5,
        )
        expected_auc = 0.5 * (100 / 5)  # F * dose / CL
        assert abs(result.auc_0_inf - expected_auc) / expected_auc < 0.30


class TestCompareRoutes:
    @pytest.fixture()
    def comparison(self):
        return compare_routes("DrugF", dose_mg=100, cl_L_per_h=5, vd_L=50, ka_per_h=0.5)

    def test_has_sc_key(self, comparison):
        assert "sc" in comparison

    def test_has_im_key(self, comparison):
        assert "im" in comparison

    def test_sc_is_result(self, comparison):
        assert isinstance(comparison["sc"], DepotPKResult)

    def test_im_is_result(self, comparison):
        assert isinstance(comparison["im"], DepotPKResult)

    def test_im_tmax_less_than_sc(self, comparison):
        assert comparison["im"].tmax_h < comparison["sc"].tmax_h

    def test_im_cmax_greater_than_sc(self, comparison):
        assert comparison["im"].cmax > comparison["sc"].cmax


class TestLaggedAbsorptionDelay:
    def test_cp_near_zero_before_lag(self):
        result = simulate_depot_pk(
            "DrugG",
            dose_mg=100,
            cl_L_per_h=5,
            vd_L=50,
            ka_per_h=0.5,
            tlag_h=5.0,
            t_end_h=48.0,
        )
        # cp at t < 5h should be ~0
        for t, cp in zip(result.times_h, result.cp_central, strict=False):
            if t < 5.0:
                assert cp < 1e-6, f"cp={cp} at t={t}h should be ~0 before lag"


class TestResultFields:
    @pytest.fixture()
    def result(self):
        return simulate_depot_pk(
            "DrugH",
            dose_mg=100,
            cl_L_per_h=5,
            vd_L=50,
            ka_per_h=0.5,
            route=DepotRoute.SUBCUTANEOUS,
        )

    def test_route_stored(self, result):
        assert result.route == "sc"

    def test_dose_stored(self, result):
        assert result.dose_mg == 100

    def test_bioavailability_stored(self, result):
        assert result.bioavailability == 1.0

    def test_ka_eff_positive(self, result):
        assert result.ka_eff_per_h > 0
