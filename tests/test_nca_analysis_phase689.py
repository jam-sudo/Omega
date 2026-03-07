"""Tests for NCA analysis module (Phase 689) — analysis/nca_analysis.py."""

import math

import pytest

from omega_pbpk.analysis.nca_analysis import (
    NCAResult,
    fit_terminal_slope,
    nca_iv_bolus,
    nca_oral,
)

# Simple 1-cpt IV decay: C(t) = C0 * exp(-ke * t)
C0 = 10.0
KE = 0.3
TIMES_IV = [float(i) for i in range(11)]  # 0..10 h
CONCS_IV = [C0 * math.exp(-KE * t) for t in TIMES_IV]
DOSE_MG = 100.0

# Oral profile: rises then falls
TIMES_ORAL = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
CONCS_ORAL = [0.0, 2.0, 5.0, 8.0, 7.5, 6.0, 4.0, 2.5, 1.5, 0.8]


class TestNCAIVBolus:
    def test_returns_nca_result(self):
        result = nca_iv_bolus(TIMES_IV, CONCS_IV, DOSE_MG)
        assert isinstance(result, NCAResult)

    def test_route_is_iv(self):
        result = nca_iv_bolus(TIMES_IV, CONCS_IV, DOSE_MG)
        assert result.route == "iv"

    def test_cmax_equals_first_concentration(self):
        result = nca_iv_bolus(TIMES_IV, CONCS_IV, DOSE_MG)
        assert math.isclose(result.cmax_mg_L, CONCS_IV[0], rel_tol=1e-9)

    def test_tmax_equals_first_time(self):
        result = nca_iv_bolus(TIMES_IV, CONCS_IV, DOSE_MG)
        assert math.isclose(result.tmax_h, TIMES_IV[0], rel_tol=1e-9)

    def test_auc0_t_positive(self):
        result = nca_iv_bolus(TIMES_IV, CONCS_IV, DOSE_MG)
        assert result.auc0_t > 0

    def test_auc0_inf_greater_than_auc0_t(self):
        result = nca_iv_bolus(TIMES_IV, CONCS_IV, DOSE_MG)
        assert result.auc0_inf > result.auc0_t

    def test_lambda_z_positive(self):
        result = nca_iv_bolus(TIMES_IV, CONCS_IV, DOSE_MG)
        assert result.lambda_z > 0

    def test_t_half_positive(self):
        result = nca_iv_bolus(TIMES_IV, CONCS_IV, DOSE_MG)
        assert result.t_half_h > 0

    def test_cl_positive(self):
        result = nca_iv_bolus(TIMES_IV, CONCS_IV, DOSE_MG)
        assert result.cl_L_per_h > 0

    def test_vd_positive(self):
        result = nca_iv_bolus(TIMES_IV, CONCS_IV, DOSE_MG)
        assert result.vd_L > 0

    def test_mrt_positive(self):
        result = nca_iv_bolus(TIMES_IV, CONCS_IV, DOSE_MG)
        assert result.mrt_h > 0

    def test_r_squared_in_range(self):
        result = nca_iv_bolus(TIMES_IV, CONCS_IV, DOSE_MG)
        assert 0.0 <= result.r_squared_terminal <= 1.0

    def test_lambda_z_close_to_ke(self):
        """For perfect 1-cpt data, lambda_z should ≈ ke within 5%."""
        result = nca_iv_bolus(TIMES_IV, CONCS_IV, DOSE_MG)
        assert math.isclose(result.lambda_z, KE, rel_tol=0.05)

    def test_t_half_close_to_expected(self):
        """t_half = ln(2)/ke ≈ 2.31 h."""
        result = nca_iv_bolus(TIMES_IV, CONCS_IV, DOSE_MG)
        expected_thalf = math.log(2) / KE
        assert math.isclose(result.t_half_h, expected_thalf, rel_tol=0.05)

    def test_auc_close_to_c0_over_ke(self):
        """For 1-cpt IV: AUC0_inf ≈ C0/ke = 33.33."""
        result = nca_iv_bolus(TIMES_IV, CONCS_IV, DOSE_MG)
        expected_auc = C0 / KE
        assert math.isclose(result.auc0_inf, expected_auc, rel_tol=0.05)

    def test_mrt_greater_than_t_half(self):
        """For IV bolus: MRT = 1/ke = t_half/ln(2) > t_half."""
        result = nca_iv_bolus(TIMES_IV, CONCS_IV, DOSE_MG)
        assert result.mrt_h > result.t_half_h

    def test_aumc_positive(self):
        result = nca_iv_bolus(TIMES_IV, CONCS_IV, DOSE_MG)
        assert result.aumc0_inf > 0

    def test_notes_nonempty(self):
        result = nca_iv_bolus(TIMES_IV, CONCS_IV, DOSE_MG)
        assert len(result.notes) > 0

    def test_cl_equals_dose_over_auc(self):
        """CL = dose / AUC0_inf."""
        result = nca_iv_bolus(TIMES_IV, CONCS_IV, DOSE_MG)
        expected_cl = DOSE_MG / result.auc0_inf
        assert math.isclose(result.cl_L_per_h, expected_cl, rel_tol=1e-9)

    def test_validation_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            nca_iv_bolus([0.0, 1.0, 2.0], [1.0, 0.5], DOSE_MG)

    def test_validation_too_few_points(self):
        with pytest.raises(ValueError):
            nca_iv_bolus([0.0, 1.0], [10.0, 7.0], DOSE_MG)


class TestNCAOral:
    def test_returns_nca_result(self):
        result = nca_oral(TIMES_ORAL, CONCS_ORAL, DOSE_MG)
        assert isinstance(result, NCAResult)

    def test_route_is_oral(self):
        result = nca_oral(TIMES_ORAL, CONCS_ORAL, DOSE_MG)
        assert result.route == "oral"

    def test_cmax_is_maximum(self):
        result = nca_oral(TIMES_ORAL, CONCS_ORAL, DOSE_MG)
        assert math.isclose(result.cmax_mg_L, max(CONCS_ORAL), rel_tol=1e-9)

    def test_cmax_not_first_point(self):
        """For oral absorption, Cmax should not be at t=0."""
        result = nca_oral(TIMES_ORAL, CONCS_ORAL, DOSE_MG)
        assert result.cmax_mg_L != CONCS_ORAL[0]

    def test_tmax_at_cmax_time(self):
        result = nca_oral(TIMES_ORAL, CONCS_ORAL, DOSE_MG)
        cmax_idx = CONCS_ORAL.index(max(CONCS_ORAL))
        assert math.isclose(result.tmax_h, TIMES_ORAL[cmax_idx], rel_tol=1e-9)

    def test_auc0_t_positive(self):
        result = nca_oral(TIMES_ORAL, CONCS_ORAL, DOSE_MG)
        assert result.auc0_t > 0

    def test_auc0_inf_greater_than_auc0_t(self):
        result = nca_oral(TIMES_ORAL, CONCS_ORAL, DOSE_MG)
        assert result.auc0_inf > result.auc0_t

    def test_lambda_z_positive(self):
        result = nca_oral(TIMES_ORAL, CONCS_ORAL, DOSE_MG)
        assert result.lambda_z > 0

    def test_cl_positive(self):
        result = nca_oral(TIMES_ORAL, CONCS_ORAL, DOSE_MG)
        assert result.cl_L_per_h > 0

    def test_r_squared_in_range(self):
        result = nca_oral(TIMES_ORAL, CONCS_ORAL, DOSE_MG)
        assert 0.0 <= result.r_squared_terminal <= 1.0


class TestFitTerminalSlope:
    def test_returns_tuple(self):
        lz, ic, r2 = fit_terminal_slope(TIMES_IV, CONCS_IV)
        assert isinstance(lz, float)
        assert isinstance(ic, float)
        assert isinstance(r2, float)

    def test_r_squared_in_range(self):
        _, _, r2 = fit_terminal_slope(TIMES_IV, CONCS_IV)
        assert 0.0 <= r2 <= 1.0

    def test_exponential_decay_lambda_z(self):
        """Perfect exponential decay: lambda_z should ≈ ke within 5%."""
        lz, _, _ = fit_terminal_slope(TIMES_IV, CONCS_IV)
        assert math.isclose(lz, KE, rel_tol=0.05)

    def test_high_r_squared_for_perfect_decay(self):
        """Perfect exponential should give R² close to 1.0."""
        _, _, r2 = fit_terminal_slope(TIMES_IV, CONCS_IV)
        assert r2 > 0.99

    def test_too_few_positive_concs_raises(self):
        with pytest.raises(ValueError):
            fit_terminal_slope([0.0, 1.0], [10.0, 5.0])


class TestTrapezoidalAUC:
    def test_simple_flat_profile_auc(self):
        """Flat profile: C=1 for t in [0,1,2] → AUC = 2."""
        times = [0.0, 1.0, 2.0]
        concs = [1.0, 1.0, 1.0]
        result = nca_iv_bolus(times, concs, dose_mg=10.0)
        assert math.isclose(result.auc0_t, 2.0, rel_tol=1e-9)

    def test_linear_decline_auc(self):
        """Linear decline: C=[4,2,0.001] at t=[0,2,4]: area ≈ 6 + 2.001 = 8.001."""
        times = [0.0, 2.0, 4.0]
        concs = [4.0, 2.0, 0.001]
        result = nca_iv_bolus(times, concs, dose_mg=100.0)
        assert math.isclose(result.auc0_t, 8.001, rel_tol=0.01)
