"""Tests for Phase 374 — Pharmacokinetic Parameter Calculator (NCA)."""
import math
import pytest

from omega_pbpk.clinical.pk_parameter_calc import (
    NCAResult,
    calculate_nca,
    compare_nca_parameters,
)


# ---------------------------------------------------------------------------
# Helpers to generate monoexponential data
# ---------------------------------------------------------------------------

def _mono_iv_data(C0: float, ke: float, times: list[float]) -> list[float]:
    """Generate monoexponential IV decay concentrations."""
    return [C0 * math.exp(-ke * t) for t in times]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TIMES = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0]
C0 = 10.0    # mg/L
KE = 0.1     # per h
CONCS = _mono_iv_data(C0, KE, TIMES)
DOSE = 100.0  # mg

# Analytical values for monoexponential IV
EXPECTED_AUC_INF = C0 / KE       # = 100 mg*h/L
EXPECTED_T_HALF = math.log(2) / KE  # ≈ 6.93 h
EXPECTED_CL = DOSE / EXPECTED_AUC_INF  # = 1.0 L/h


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

class TestNCAResultStructure:
    def test_returns_nca_result(self):
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv")
        assert isinstance(result, NCAResult)

    def test_frozen_dataclass(self):
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv")
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_drug_name_preserved(self):
        result = calculate_nca("TestDrug", DOSE, TIMES, CONCS, route="iv")
        assert result.drug_name == "TestDrug"

    def test_dose_preserved(self):
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv")
        assert result.dose_mg == DOSE

    def test_route_preserved(self):
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv")
        assert result.route == "iv"


# ---------------------------------------------------------------------------
# Cmax / Tmax tests
# ---------------------------------------------------------------------------

class TestCmaxTmax:
    def test_cmax_equals_max_concentration(self):
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv")
        assert abs(result.cmax_mg_L - max(CONCS)) < 1e-9

    def test_tmax_correct_for_iv(self):
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv")
        # For IV monoexponential, Cmax at t=0
        assert result.tmax_h == TIMES[0]

    def test_tmax_oral(self):
        # Oral: build a profile peaking at t=2
        times_oral = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
        concs_oral = [0.0, 1.5, 3.0, 5.0, 3.5, 1.5, 0.5]
        result = calculate_nca("Drug", 100.0, times_oral, concs_oral, route="oral")
        assert result.tmax_h == 2.0
        assert result.cmax_mg_L == 5.0


# ---------------------------------------------------------------------------
# Lambda_z and t_half tests
# ---------------------------------------------------------------------------

class TestLambdaZ:
    def test_lambda_z_positive(self):
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv")
        assert result.lambda_z_per_h > 0

    def test_lambda_z_close_to_ke(self):
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv", n_terminal_points=4)
        assert abs(result.lambda_z_per_h - KE) < 0.005

    def test_t_half_close_to_expected(self):
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv", n_terminal_points=4)
        assert abs(result.t_half_h - EXPECTED_T_HALF) < 0.1

    def test_t_half_equals_ln2_over_lambda_z(self):
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv")
        expected = math.log(2) / result.lambda_z_per_h
        assert abs(result.t_half_h - expected) < 1e-9


# ---------------------------------------------------------------------------
# AUC tests
# ---------------------------------------------------------------------------

class TestAUC:
    def test_auc_inf_ge_auc_last(self):
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv")
        assert result.auc_inf_mg_h_per_L >= result.auc_last_mg_h_per_L

    def test_auc_inf_close_to_c0_over_ke(self):
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv", n_terminal_points=4)
        # Some deviation expected from trapezoid approximation
        assert abs(result.auc_inf_mg_h_per_L - EXPECTED_AUC_INF) / EXPECTED_AUC_INF < 0.05

    def test_pct_extrapolated_in_range(self):
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv")
        assert 0.0 <= result.pct_auc_extrapolated <= 100.0


# ---------------------------------------------------------------------------
# MRT tests
# ---------------------------------------------------------------------------

class TestMRT:
    def test_mrt_positive(self):
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv")
        assert result.mrt_h > 0

    def test_mrt_greater_than_thalf(self):
        # For 1-cpt IV: MRT = 1/ke = t_half / ln(2), which is always > t_half
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv", n_terminal_points=4)
        assert result.mrt_h > result.t_half_h


# ---------------------------------------------------------------------------
# CL and Vd tests
# ---------------------------------------------------------------------------

class TestCLVd:
    def test_cl_close_to_expected_iv(self):
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv", n_terminal_points=4)
        assert abs(result.cl_obs_L_per_h - EXPECTED_CL) / EXPECTED_CL < 0.05

    def test_vd_ss_positive(self):
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv")
        assert result.vd_ss_L > 0

    def test_vd_ss_equals_cl_times_mrt(self):
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv")
        expected = result.cl_obs_L_per_h * result.mrt_h
        assert abs(result.vd_ss_L - expected) < 1e-9

    def test_cl_oral_with_bioavailability(self):
        times_oral = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
        concs_oral = [2.0, 4.0, 5.0, 3.0, 1.5, 0.5, 0.1]
        result_no_f = calculate_nca("Drug", 100.0, times_oral, concs_oral, route="oral", f_for_oral=None)
        result_with_f = calculate_nca("Drug", 100.0, times_oral, concs_oral, route="oral", f_for_oral=0.5)
        # CL with F=0.5 should be half of apparent CL
        assert abs(result_with_f.cl_obs_L_per_h - 0.5 * result_no_f.cl_obs_L_per_h) < 1e-9


# ---------------------------------------------------------------------------
# Bioavailability tests
# ---------------------------------------------------------------------------

class TestFOral:
    def test_f_oral_none_for_iv(self):
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv")
        assert result.f_oral is None

    def test_f_oral_stored_when_provided(self):
        times_oral = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
        concs_oral = [2.0, 4.0, 5.0, 3.0, 1.5, 0.5]
        result = calculate_nca("Drug", 100.0, times_oral, concs_oral, route="oral", f_for_oral=0.8)
        assert result.f_oral == 0.8

    def test_f_oral_none_when_not_provided_oral(self):
        times_oral = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
        concs_oral = [2.0, 4.0, 5.0, 3.0, 1.5, 0.5]
        result = calculate_nca("Drug", 100.0, times_oral, concs_oral, route="oral")
        assert result.f_oral is None


# ---------------------------------------------------------------------------
# Adequate sampling flag
# ---------------------------------------------------------------------------

class TestAdequateSampling:
    def test_adequate_sampling_true_for_good_data(self):
        # Use more terminal points on perfect monoexponential
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv", n_terminal_points=4)
        assert result.adequate_sampling is True

    def test_n_points_for_lambda_correct(self):
        result = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv", n_terminal_points=3)
        assert result.n_points_for_lambda == 3


# ---------------------------------------------------------------------------
# compare_nca_parameters tests
# ---------------------------------------------------------------------------

class TestCompareNCA:
    def test_returns_dict(self):
        r1 = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv", n_terminal_points=4)
        r2 = calculate_nca("Drug", DOSE * 2, TIMES, _mono_iv_data(C0 * 2, KE, TIMES), route="iv", n_terminal_points=4)
        summary = compare_nca_parameters([r1, r2])
        assert isinstance(summary, dict)

    def test_summary_has_expected_keys(self):
        r1 = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv")
        summary = compare_nca_parameters([r1])
        assert "cl_obs_L_per_h" in summary
        assert "t_half_h" in summary
        assert "auc_inf_mg_h_per_L" in summary
        assert "vd_ss_L" in summary

    def test_summary_stat_keys(self):
        r1 = calculate_nca("Drug", DOSE, TIMES, CONCS, route="iv")
        summary = compare_nca_parameters([r1])
        for param, stats in summary.items():
            assert "mean" in stats
            assert "cv_pct" in stats
            assert "range" in stats

    def test_empty_list_returns_empty_dict(self):
        assert compare_nca_parameters([]) == {}


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            calculate_nca("Drug", DOSE, TIMES, CONCS[:5], route="iv")

    def test_raises_on_fewer_than_3_points(self):
        with pytest.raises(ValueError, match="3"):
            calculate_nca("Drug", DOSE, [0.0, 1.0], [5.0, 3.0], route="iv")

    def test_raises_on_negative_dose(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_nca("Drug", -10.0, TIMES, CONCS, route="iv")

    def test_raises_on_invalid_route(self):
        with pytest.raises(ValueError, match="route"):
            calculate_nca("Drug", DOSE, TIMES, CONCS, route="im")

    def test_raises_on_unsorted_times(self):
        times_bad = TIMES[::-1]
        with pytest.raises(ValueError, match="ascending"):
            calculate_nca("Drug", DOSE, times_bad, CONCS, route="iv")

    def test_raises_on_negative_concentration(self):
        concs_bad = list(CONCS)
        concs_bad[3] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            calculate_nca("Drug", DOSE, TIMES, concs_bad, route="iv")
