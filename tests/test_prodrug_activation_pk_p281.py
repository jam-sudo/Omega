"""Tests for Phase 281 — Drug Prodrug Activation Kinetics."""

import pytest

from omega_pbpk.core.prodrug_activation_pk_p281 import (
    ProdrugPKResult,
    compare_conversion_rates,
    simulate_prodrug_pk,
)

# Default parameters for convenience
DEFAULT_PARAMS = dict(
    drug_name="TestProdrug",
    dose_mg=100.0,
    route="oral",
    f_oral_prodrug=0.9,
    cl_prodrug_L_per_h=5.0,
    vd_prodrug_L=20.0,
    k_conv_per_h=0.5,
    cl_active_L_per_h=10.0,
    vd_active_L=30.0,
    t_end_h=24.0,
    dt_h=0.05,
)


def make_result(**overrides) -> ProdrugPKResult:
    params = {**DEFAULT_PARAMS, **overrides}
    return simulate_prodrug_pk(**params)


class TestReturnType:
    def test_returns_prodrug_pk_result(self):
        result = make_result()
        assert isinstance(result, ProdrugPKResult)

    def test_drug_name_preserved(self):
        result = make_result(drug_name="Capecitabine")
        assert result.drug_name == "Capecitabine"

    def test_dose_mg_preserved(self):
        result = make_result(dose_mg=250.0)
        assert result.dose_mg == 250.0

    def test_route_preserved(self):
        result = make_result(route="oral")
        assert result.route == "oral"

    def test_k_conv_preserved(self):
        result = make_result(k_conv_per_h=1.2)
        assert result.k_conv_per_h == 1.2


class TestTimeArray:
    def test_time_array_length(self):
        result = make_result(t_end_h=10.0, dt_h=0.1)
        # n_steps = round(10.0 / 0.1) = 100 steps, so 101 points
        assert len(result.times_h) == 101

    def test_c_prodrug_length_matches_times(self):
        result = make_result()
        assert len(result.c_prodrug_mg_L) == len(result.times_h)

    def test_c_active_length_matches_times(self):
        result = make_result()
        assert len(result.c_active_mg_L) == len(result.times_h)

    def test_time_starts_at_zero(self):
        result = make_result()
        assert result.times_h[0] == pytest.approx(0.0)


class TestProdrugConcentrations:
    def test_cmax_prodrug_positive(self):
        result = make_result()
        assert result.cmax_prodrug_mg_L > 0.0

    def test_auc_prodrug_positive(self):
        result = make_result()
        assert result.auc_prodrug_mg_h_per_L > 0.0

    def test_initial_prodrug_zero_for_oral(self):
        result = make_result(route="oral")
        assert result.c_prodrug_mg_L[0] == pytest.approx(0.0)

    def test_initial_prodrug_positive_for_iv(self):
        result = make_result(route="iv_bolus")
        assert result.c_prodrug_mg_L[0] > 0.0

    def test_tmax_prodrug_nonnegative(self):
        result = make_result()
        assert result.tmax_prodrug_h >= 0.0


class TestActiveConcentrations:
    def test_cmax_active_positive(self):
        result = make_result()
        assert result.cmax_active_mg_L > 0.0

    def test_auc_active_positive(self):
        result = make_result()
        assert result.auc_active_mg_h_per_L > 0.0

    def test_tmax_active_nonnegative(self):
        result = make_result()
        assert result.tmax_active_h >= 0.0

    def test_tmax_active_ge_tmax_prodrug_oral(self):
        """Active drug peaks after or at same time as prodrug for oral dosing."""
        result = make_result(route="oral", k_conv_per_h=0.3)
        assert result.tmax_active_h >= result.tmax_prodrug_h

    def test_initial_active_zero(self):
        result = make_result()
        assert result.c_active_mg_L[0] == pytest.approx(0.0)


class TestActivationEfficiency:
    def test_activation_efficiency_in_0_1(self):
        result = make_result()
        assert 0.0 <= result.activation_efficiency <= 1.0

    def test_higher_k_conv_higher_efficiency(self):
        low = make_result(k_conv_per_h=0.1)
        high = make_result(k_conv_per_h=2.0)
        assert high.activation_efficiency > low.activation_efficiency

    def test_activation_efficiency_formula(self):
        result = make_result()
        expected = result.auc_active_mg_h_per_L / (
            result.auc_active_mg_h_per_L + result.auc_prodrug_mg_h_per_L
        )
        assert result.activation_efficiency == pytest.approx(expected, rel=1e-4)


class TestConversionRateEffect:
    def test_higher_k_conv_higher_cmax_active(self):
        low = make_result(k_conv_per_h=0.1)
        high = make_result(k_conv_per_h=3.0)
        assert high.cmax_active_mg_L > low.cmax_active_mg_L

    def test_higher_k_conv_lower_cmax_prodrug(self):
        low = make_result(k_conv_per_h=0.05)
        high = make_result(k_conv_per_h=5.0)
        assert high.cmax_prodrug_mg_L < low.cmax_prodrug_mg_L


class TestLagTime:
    def test_lag_time_nonnegative(self):
        result = make_result()
        assert result.lag_time_h >= 0.0

    def test_lag_time_less_than_or_equal_tmax_active(self):
        result = make_result()
        assert result.lag_time_h <= result.tmax_active_h


class TestIVRoute:
    def test_iv_route_returns_result(self):
        result = make_result(route="iv_bolus")
        assert isinstance(result, ProdrugPKResult)

    def test_iv_cmax_prodrug_at_t0(self):
        result = make_result(route="iv_bolus", dose_mg=50.0, vd_prodrug_L=20.0)
        expected_c0 = 50.0 / 20.0
        assert result.c_prodrug_mg_L[0] == pytest.approx(expected_c0, rel=1e-3)


class TestCompareConversionRates:
    def test_returns_list(self):
        result = compare_conversion_rates("TestDrug", 100.0, [0.1, 0.5, 1.0], 5.0, 10.0, 20.0, 30.0)
        assert isinstance(result, list)

    def test_sorted_descending_by_cmax_active(self):
        result = compare_conversion_rates(
            "TestDrug", 100.0, [0.1, 0.5, 1.0, 2.0], 5.0, 10.0, 20.0, 30.0
        )
        for i in range(len(result) - 1):
            assert result[i].cmax_active_mg_L >= result[i + 1].cmax_active_mg_L

    def test_length_matches_k_conv_list(self):
        k_list = [0.1, 0.5, 1.0]
        result = compare_conversion_rates("TestDrug", 100.0, k_list, 5.0, 10.0, 20.0, 30.0)
        assert len(result) == len(k_list)


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            make_result(dose_mg=-10.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            make_result(dose_mg=0.0)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            make_result(route="subcutaneous")

    def test_zero_cl_prodrug_raises(self):
        with pytest.raises(ValueError, match="cl_prodrug"):
            make_result(cl_prodrug_L_per_h=0.0)

    def test_zero_k_conv_raises(self):
        with pytest.raises(ValueError, match="k_conv"):
            make_result(k_conv_per_h=0.0)

    def test_zero_vd_prodrug_raises(self):
        with pytest.raises(ValueError, match="vd_prodrug"):
            make_result(vd_prodrug_L=0.0)

    def test_f_oral_zero_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            make_result(f_oral_prodrug=0.0)

    def test_f_oral_above_1_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            make_result(f_oral_prodrug=1.1)

    def test_zero_t_end_raises(self):
        with pytest.raises(ValueError, match="t_end"):
            make_result(t_end_h=0.0)
