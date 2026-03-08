"""Tests for Phase 295 — Absorption Rate Optimizer."""

import pytest

from omega_pbpk.clinical.absorption_rate_optimizer import (
    AbsorptionOptimizationResult,
    compare_ka_values,
    optimize_ka_for_target_auc,
    optimize_ka_for_target_cmax,
)

# ---------------------------------------------------------------------------
# Shared fixture parameters
# ---------------------------------------------------------------------------

DRUG = "TestDrug"
DOSE = 100.0  # mg
CL = 5.0  # L/h
VD = 50.0  # L
F = 0.8  # dimensionless
KE = CL / VD  # 0.1 h^-1


# ---------------------------------------------------------------------------
# optimize_ka_for_target_cmax
# ---------------------------------------------------------------------------


class TestOptimizeKaForTargetCmax:
    def test_returns_correct_type(self):
        result = optimize_ka_for_target_cmax(DRUG, DOSE, CL, VD, F, 1.0)
        assert isinstance(result, AbsorptionOptimizationResult)

    def test_drug_name_preserved(self):
        result = optimize_ka_for_target_cmax(DRUG, DOSE, CL, VD, F, 1.0)
        assert result.drug_name == DRUG

    def test_dose_preserved(self):
        result = optimize_ka_for_target_cmax(DRUG, DOSE, CL, VD, F, 1.0)
        assert result.dose_mg == DOSE

    def test_optimization_type_is_cmax(self):
        result = optimize_ka_for_target_cmax(DRUG, DOSE, CL, VD, F, 1.0)
        assert result.optimization_type == "cmax"

    def test_target_cmax_stored(self):
        result = optimize_ka_for_target_cmax(DRUG, DOSE, CL, VD, F, 1.5)
        assert result.target_cmax_mg_L == 1.5

    def test_target_auc_is_none(self):
        result = optimize_ka_for_target_cmax(DRUG, DOSE, CL, VD, F, 1.0)
        assert result.target_auc_mg_h_per_L is None

    def test_optimal_ka_in_range(self):
        ka_min, ka_max = 0.1, 5.0
        result = optimize_ka_for_target_cmax(
            DRUG, DOSE, CL, VD, F, 1.0, ka_min_per_h=ka_min, ka_max_per_h=ka_max
        )
        assert ka_min <= result.optimal_ka_per_h <= ka_max

    def test_achieved_cmax_positive(self):
        result = optimize_ka_for_target_cmax(DRUG, DOSE, CL, VD, F, 1.0)
        assert result.achieved_cmax_mg_L > 0.0

    def test_achieved_auc_positive(self):
        result = optimize_ka_for_target_cmax(DRUG, DOSE, CL, VD, F, 1.0)
        assert result.achieved_auc_mg_h_per_L > 0.0

    def test_achieved_tmax_positive(self):
        result = optimize_ka_for_target_cmax(DRUG, DOSE, CL, VD, F, 1.0)
        assert result.achieved_tmax_h > 0.0

    def test_error_pct_nonnegative(self):
        result = optimize_ka_for_target_cmax(DRUG, DOSE, CL, VD, F, 1.0)
        assert result.error_pct >= 0.0

    def test_higher_target_cmax_gives_higher_optimal_ka(self):
        """Higher target Cmax requires faster absorption."""
        r_low = optimize_ka_for_target_cmax(
            DRUG, DOSE, CL, VD, F, 0.5, ka_min_per_h=0.2, ka_max_per_h=8.0, n_steps=200
        )
        r_high = optimize_ka_for_target_cmax(
            DRUG, DOSE, CL, VD, F, 1.2, ka_min_per_h=0.2, ka_max_per_h=8.0, n_steps=200
        )
        assert r_high.optimal_ka_per_h >= r_low.optimal_ka_per_h

    def test_auc_approx_analytical(self):
        """Numerical AUC should be close to F*D/CL (within 5%)."""
        analytical_auc = F * DOSE / CL  # = 16.0
        result = optimize_ka_for_target_cmax(DRUG, DOSE, CL, VD, F, 1.0)
        assert abs(result.achieved_auc_mg_h_per_L - analytical_auc) / analytical_auc < 0.05


# ---------------------------------------------------------------------------
# optimize_ka_for_target_auc
# ---------------------------------------------------------------------------


class TestOptimizeKaForTargetAuc:
    def test_returns_correct_type(self):
        auc_target = F * DOSE / CL
        result = optimize_ka_for_target_auc(DRUG, DOSE, CL, VD, F, auc_target)
        assert isinstance(result, AbsorptionOptimizationResult)

    def test_optimization_type_is_auc(self):
        result = optimize_ka_for_target_auc(DRUG, DOSE, CL, VD, F, 10.0)
        assert result.optimization_type == "auc"

    def test_target_auc_stored(self):
        result = optimize_ka_for_target_auc(DRUG, DOSE, CL, VD, F, 14.0)
        assert result.target_auc_mg_h_per_L == 14.0

    def test_target_cmax_is_none(self):
        result = optimize_ka_for_target_auc(DRUG, DOSE, CL, VD, F, 10.0)
        assert result.target_cmax_mg_L is None

    def test_achieved_auc_equals_analytical(self):
        """AUC = F*D/CL regardless of ka."""
        analytical_auc = F * DOSE / CL
        result = optimize_ka_for_target_auc(DRUG, DOSE, CL, VD, F, 10.0)
        assert abs(result.achieved_auc_mg_h_per_L - analytical_auc) < 1e-6

    def test_optimal_ka_in_range(self):
        ka_min, ka_max = 0.2, 6.0
        result = optimize_ka_for_target_auc(
            DRUG, DOSE, CL, VD, F, 10.0, ka_min_per_h=ka_min, ka_max_per_h=ka_max
        )
        assert ka_min <= result.optimal_ka_per_h <= ka_max

    def test_error_pct_nonnegative(self):
        result = optimize_ka_for_target_auc(DRUG, DOSE, CL, VD, F, 10.0)
        assert result.error_pct >= 0.0

    def test_achieved_cmax_positive(self):
        result = optimize_ka_for_target_auc(DRUG, DOSE, CL, VD, F, 10.0)
        assert result.achieved_cmax_mg_L > 0.0


# ---------------------------------------------------------------------------
# compare_ka_values
# ---------------------------------------------------------------------------


class TestCompareKaValues:
    def test_returns_list(self):
        result = compare_ka_values(DRUG, DOSE, CL, VD, F, [0.5, 1.0, 2.0])
        assert isinstance(result, list)

    def test_correct_length(self):
        ka_list = [0.5, 1.0, 2.0, 5.0]
        result = compare_ka_values(DRUG, DOSE, CL, VD, F, ka_list)
        assert len(result) == len(ka_list)

    def test_sorted_by_cmax_descending(self):
        result = compare_ka_values(DRUG, DOSE, CL, VD, F, [0.2, 0.5, 1.0, 3.0])
        cmaxes = [r.achieved_cmax_mg_L for r in result]
        assert cmaxes == sorted(cmaxes, reverse=True)

    def test_optimization_type_is_sweep(self):
        result = compare_ka_values(DRUG, DOSE, CL, VD, F, [1.0])
        assert result[0].optimization_type == "sweep"

    def test_all_cmax_positive(self):
        result = compare_ka_values(DRUG, DOSE, CL, VD, F, [0.3, 1.0, 5.0])
        assert all(r.achieved_cmax_mg_L > 0 for r in result)

    def test_error_pct_zero(self):
        result = compare_ka_values(DRUG, DOSE, CL, VD, F, [1.0])
        assert result[0].error_pct == 0.0

    def test_drug_name_preserved_in_all(self):
        results = compare_ka_values(DRUG, DOSE, CL, VD, F, [0.5, 1.0])
        assert all(r.drug_name == DRUG for r in results)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            optimize_ka_for_target_cmax(DRUG, 0, CL, VD, F, 1.0)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            optimize_ka_for_target_cmax(DRUG, DOSE, 0, VD, F, 1.0)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            optimize_ka_for_target_cmax(DRUG, DOSE, CL, 0, F, 1.0)

    def test_f_oral_zero_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            optimize_ka_for_target_cmax(DRUG, DOSE, CL, VD, 0.0, 1.0)

    def test_f_oral_above_one_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            optimize_ka_for_target_cmax(DRUG, DOSE, CL, VD, 1.1, 1.0)

    def test_ka_min_zero_raises(self):
        with pytest.raises(ValueError, match="ka_min"):
            optimize_ka_for_target_cmax(DRUG, DOSE, CL, VD, F, 1.0, ka_min_per_h=0)

    def test_ka_max_less_than_min_raises(self):
        with pytest.raises(ValueError, match="ka_max"):
            optimize_ka_for_target_cmax(
                DRUG, DOSE, CL, VD, F, 1.0, ka_min_per_h=2.0, ka_max_per_h=1.0
            )

    def test_n_steps_one_raises(self):
        with pytest.raises(ValueError, match="n_steps"):
            optimize_ka_for_target_cmax(DRUG, DOSE, CL, VD, F, 1.0, n_steps=1)

    def test_compare_empty_list_raises(self):
        with pytest.raises(ValueError):
            compare_ka_values(DRUG, DOSE, CL, VD, F, [])

    def test_compare_negative_ka_raises(self):
        with pytest.raises(ValueError):
            compare_ka_values(DRUG, DOSE, CL, VD, F, [-1.0, 0.5])
