"""Tests for omega_pbpk.core.nca — NCA analysis module."""

import math
import pytest
import numpy as np

from omega_pbpk.core.nca import NCAResult, calculate_nca, compare_nca


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mono_profile(c0: float, ke: float, times: list[float]) -> list[float]:
    """Ideal monoexponential IV profile: C(t) = C0 * exp(-ke * t)."""
    return [c0 * math.exp(-ke * t) for t in times]


TIMES_IV = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0]
KE_TRUE = 0.1386  # => t_half ≈ 5 h
C0 = 10.0
CONC_IV = _mono_profile(C0, KE_TRUE, TIMES_IV)
DOSE = 100.0

TIMES_ORAL = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0]
# Rough oral Bateman profile for a quick check (absorption then elimination)
CONC_ORAL = [0.0, 1.2, 2.5, 4.0, 3.5, 2.1, 1.2, 0.4, 0.05]


# ---------------------------------------------------------------------------
# 1. Result type
# ---------------------------------------------------------------------------

class TestResultType:
    def test_returns_nca_result_instance(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert isinstance(res, NCAResult)

    def test_nca_result_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(NCAResult)

    def test_drug_name_stored(self):
        res = calculate_nca("TestDrug", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert res.drug_name == "TestDrug"

    def test_route_stored(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert res.route == "iv"

    def test_dose_stored(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert res.dose_mg == DOSE

    def test_n_points_equals_len_times(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert res.n_points == len(TIMES_IV)

    def test_times_stored(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert list(res.times_h) == TIMES_IV

    def test_conc_stored(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert list(res.conc_mg_L) == pytest.approx(CONC_IV, rel=1e-6)


# ---------------------------------------------------------------------------
# 2. Cmax / Tmax
# ---------------------------------------------------------------------------

class TestCmaxTmax:
    def test_cmax_positive(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert res.cmax_mg_L > 0

    def test_cmax_equals_max_conc(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert res.cmax_mg_L == pytest.approx(max(CONC_IV), rel=1e-6)

    def test_tmax_nonnegative(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert res.tmax_h >= 0

    def test_tmax_at_time_of_max_conc(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        expected_tmax = TIMES_IV[CONC_IV.index(max(CONC_IV))]
        assert res.tmax_h == pytest.approx(expected_tmax, abs=1e-9)

    def test_oral_cmax_positive(self):
        res = calculate_nca("DrugB", TIMES_ORAL, CONC_ORAL, DOSE, route="oral")
        assert res.cmax_mg_L > 0

    def test_oral_tmax_at_peak(self):
        res = calculate_nca("DrugB", TIMES_ORAL, CONC_ORAL, DOSE, route="oral")
        idx = CONC_ORAL.index(max(CONC_ORAL))
        assert res.tmax_h == pytest.approx(TIMES_ORAL[idx], abs=1e-9)


# ---------------------------------------------------------------------------
# 3. AUC
# ---------------------------------------------------------------------------

class TestAUC:
    def test_auc_0_t_positive(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert res.auc_0_t > 0

    def test_auc_0_inf_ge_auc_0_t(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert res.auc_0_inf >= res.auc_0_t

    def test_auc_extrap_pct_in_range(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert 0.0 <= res.auc_extrap_pct <= 100.0

    def test_auc_extrap_small_for_dense_late_samples(self):
        # Long sampling with many late points → small extrapolation
        times = list(np.linspace(0, 48, 25))
        conc = _mono_profile(C0, KE_TRUE, times)
        res = calculate_nca("DrugA", times, conc, DOSE, route="iv")
        assert res.auc_extrap_pct < 10.0

    def test_auc_0_t_trapz_accuracy(self):
        # For monoexponential, analytical AUC_0_t known
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        # trapz estimate should be positive and finite
        assert math.isfinite(res.auc_0_t)
        assert res.auc_0_t > 0


# ---------------------------------------------------------------------------
# 4. PK parameters
# ---------------------------------------------------------------------------

class TestPKParameters:
    def test_mrt_positive(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert res.mrt_h > 0

    def test_t_half_positive(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert res.t_half_h > 0

    def test_ke_positive(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert res.ke_per_h > 0

    def test_lambda_z_r2_in_unit_interval(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert 0.0 <= res.lambda_z_r2 <= 1.0

    def test_cl_positive(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert res.cl_L_per_h > 0

    def test_vd_ss_positive_for_iv(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert res.vd_ss_L > 0

    def test_t_half_relationship_to_ke(self):
        # t_half = 0.693 / ke must hold internally
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert res.t_half_h == pytest.approx(0.693147 / res.ke_per_h, rel=1e-3)


# ---------------------------------------------------------------------------
# 5. Monoexponential recovery accuracy
# ---------------------------------------------------------------------------

class TestMonoexponentialAccuracy:
    """For ideal monoexponential data the estimated t_half should be close
    to the true value (within 10 %)."""

    def test_t_half_within_10pct_of_true(self):
        true_t_half = 0.693147 / KE_TRUE
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert abs(res.t_half_h - true_t_half) / true_t_half < 0.10

    def test_ke_within_10pct_of_true(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert abs(res.ke_per_h - KE_TRUE) / KE_TRUE < 0.10

    def test_lambda_z_r2_near_1_for_monoexp(self):
        # Perfect monoexponential → R² of log-linear regression ≈ 1
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        assert res.lambda_z_r2 > 0.99

    def test_different_ke_changes_t_half(self):
        ke_fast = 0.5
        conc_fast = _mono_profile(C0, ke_fast, TIMES_IV)
        res = calculate_nca("DrugFast", TIMES_IV, conc_fast, DOSE, route="iv")
        expected = 0.693147 / ke_fast
        assert abs(res.t_half_h - expected) / expected < 0.10


# ---------------------------------------------------------------------------
# 6. Validation / error handling
# ---------------------------------------------------------------------------

class TestValidation:
    def test_raises_value_error_if_less_than_3_points(self):
        with pytest.raises(ValueError):
            calculate_nca("DrugA", [0.0, 1.0], [1.0, 0.5], DOSE, route="iv")

    def test_raises_value_error_for_exactly_2_points(self):
        with pytest.raises(ValueError):
            calculate_nca("DrugA", [0.0, 2.0], [5.0, 2.5], DOSE)

    def test_raises_value_error_for_1_point(self):
        with pytest.raises(ValueError):
            calculate_nca("DrugA", [0.0], [5.0], DOSE)

    def test_raises_value_error_for_zero_dose(self):
        with pytest.raises(ValueError):
            calculate_nca("DrugA", TIMES_IV, CONC_IV, 0.0, route="iv")

    def test_raises_value_error_for_negative_dose(self):
        with pytest.raises(ValueError):
            calculate_nca("DrugA", TIMES_IV, CONC_IV, -50.0, route="iv")

    def test_three_points_does_not_raise(self):
        conc3 = _mono_profile(C0, KE_TRUE, [0.0, 2.0, 6.0])
        res = calculate_nca("DrugA", [0.0, 2.0, 6.0], conc3, DOSE, route="iv")
        assert isinstance(res, NCAResult)


# ---------------------------------------------------------------------------
# 7. Default route
# ---------------------------------------------------------------------------

class TestDefaultRoute:
    def test_default_route_is_oral(self):
        res = calculate_nca("DrugB", TIMES_ORAL, CONC_ORAL, DOSE)
        assert res.route == "oral"


# ---------------------------------------------------------------------------
# 8. compare_nca
# ---------------------------------------------------------------------------

class TestCompareNCA:
    def _two_results(self):
        res1 = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        conc2 = _mono_profile(5.0, 0.2, TIMES_IV)
        res2 = calculate_nca("DrugB", TIMES_IV, conc2, 50.0, route="iv")
        return [res1, res2]

    def test_compare_returns_dict(self):
        results = self._two_results()
        out = compare_nca(results)
        assert isinstance(out, dict)

    def test_compare_dict_contains_drug_names(self):
        results = self._two_results()
        out = compare_nca(results)
        assert "DrugA" in out
        assert "DrugB" in out

    def test_compare_single_result(self):
        res = calculate_nca("DrugA", TIMES_IV, CONC_IV, DOSE, route="iv")
        out = compare_nca([res])
        assert "DrugA" in out

    def test_compare_values_are_dicts_or_nca_results(self):
        results = self._two_results()
        out = compare_nca(results)
        for val in out.values():
            assert isinstance(val, (dict, NCAResult))
