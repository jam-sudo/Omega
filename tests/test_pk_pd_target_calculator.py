"""Tests for Phase 791: PK/PD Target Calculator."""

import math

import pytest

from omega_pbpk.clinical.pk_pd_target_calculator import (
    PKPDTargetResult,
    calculate_pk_pd_target,
    scan_pd_targets,
)

# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_pkpd_target_result(self):
        result = calculate_pk_pd_target("DrugA", 50.0, ec50=1.0)
        assert isinstance(result, PKPDTargetResult)

    def test_result_is_frozen(self):
        result = calculate_pk_pd_target("DrugA", 50.0, ec50=1.0)
        with pytest.raises((AttributeError, TypeError)):
            result.ec50 = 99.0  # type: ignore[misc]

    def test_all_fields_present(self):
        result = calculate_pk_pd_target("DrugA", 50.0, ec50=1.0)
        assert result.drug_name == "DrugA"
        assert result.pd_model == "sigmoid_emax"
        assert result.target_effect_pct == 50.0
        assert result.ec50 == 1.0
        assert result.hill_coef == 1.0
        assert result.emax_pct == 100.0
        assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# Core formula tests
# ---------------------------------------------------------------------------


class TestCoreFormulas:
    def test_required_exposure_positive(self):
        result = calculate_pk_pd_target("DrugA", 50.0, ec50=2.0)
        assert result.required_exposure > 0.0

    def test_at_50pct_hill1_required_equals_ec50(self):
        """At 50% target with hill=1, required exposure = EC50 (Emax=100)."""
        ec50 = 3.5
        result = calculate_pk_pd_target("DrugA", 50.0, ec50=ec50, hill_coef=1.0, emax_pct=100.0)
        assert abs(result.required_exposure - ec50) < 1e-10

    def test_at_50pct_hill2_required_equals_ec50(self):
        """At 50% target with hill=2 and Emax=100, ratio = 50/50 = 1 → C = EC50^(2/2) = EC50."""
        ec50 = 2.0
        result = calculate_pk_pd_target("DrugA", 50.0, ec50=ec50, hill_coef=2.0, emax_pct=100.0)
        assert abs(result.required_exposure - ec50) < 1e-10

    def test_higher_target_requires_higher_exposure(self):
        result_60 = calculate_pk_pd_target("DrugA", 60.0, ec50=1.0)
        result_80 = calculate_pk_pd_target("DrugA", 80.0, ec50=1.0)
        assert result_80.required_exposure > result_60.required_exposure

    def test_higher_ec50_requires_higher_exposure(self):
        result_low = calculate_pk_pd_target("DrugA", 50.0, ec50=1.0)
        result_high = calculate_pk_pd_target("DrugA", 50.0, ec50=5.0)
        assert result_high.required_exposure > result_low.required_exposure

    def test_required_dose_positive(self):
        result = calculate_pk_pd_target("DrugA", 50.0, ec50=1.0)
        assert result.required_dose_mg > 0.0

    def test_required_dose_formula(self):
        """Dose = required_exposure * CL * tau / F."""
        ec50 = 2.0
        cl = 4.0
        tau = 12.0
        f = 0.6
        result = calculate_pk_pd_target(
            "DrugA", 50.0, ec50=ec50, cl_L_per_h=cl, tau_h=tau, f_oral=f
        )
        expected_dose = result.required_exposure * cl * tau / f
        assert abs(result.required_dose_mg - expected_dose) < 1e-6


# ---------------------------------------------------------------------------
# Therapeutic index tests
# ---------------------------------------------------------------------------


class TestTherapeuticIndex:
    def test_therapeutic_index_greater_than_one(self):
        result = calculate_pk_pd_target("DrugA", 50.0, ec50=1.0)
        assert result.therapeutic_index > 1.0

    def test_therapeutic_index_ec90_over_ec10(self):
        """TI = EC90/EC10. With Hill=1, EC90/EC10 = (90/10) / (10/90) = 81."""
        result = calculate_pk_pd_target("DrugA", 50.0, ec50=1.0, hill_coef=1.0, emax_pct=100.0)
        # EC10 = 1 * (10/90) = 0.1111, EC90 = 1 * (90/10) = 9.0
        # TI = 9.0 / 0.1111 = 81
        assert abs(result.therapeutic_index - 81.0) < 0.01

    def test_steeper_hill_narrows_ti(self):
        """Steeper Hill coefficient narrows EC10-EC90 window."""
        result_hill1 = calculate_pk_pd_target("DrugA", 50.0, ec50=1.0, hill_coef=1.0)
        result_hill3 = calculate_pk_pd_target("DrugA", 50.0, ec50=1.0, hill_coef=3.0)
        assert result_hill3.therapeutic_index < result_hill1.therapeutic_index


# ---------------------------------------------------------------------------
# PD margin tests
# ---------------------------------------------------------------------------


class TestPDMargin:
    def test_pd_margin_positive_when_target_below_mtc(self):
        result = calculate_pk_pd_target("DrugA", 50.0, ec50=1.0, mtc_effect_pct=80.0)
        assert result.pd_margin > 0.0

    def test_pd_margin_negative_when_required_exceeds_toxic(self):
        """If required ~EC95 and MTC=80, margin could be negative."""
        # Target=95, MTC=80 → EC_target > EC_toxic
        result = calculate_pk_pd_target(
            "DrugA", 95.0, ec50=1.0, emax_pct=100.0, mtc_effect_pct=80.0
        )
        # EC_target(95) >> EC_toxic(80), so pd_margin < 0
        assert result.pd_margin < 0.0

    def test_pd_margin_inf_when_mtc_equals_emax(self):
        result = calculate_pk_pd_target(
            "DrugA", 50.0, ec50=1.0, emax_pct=100.0, mtc_effect_pct=100.0
        )
        assert math.isinf(result.pd_margin)


# ---------------------------------------------------------------------------
# Trough tests
# ---------------------------------------------------------------------------


class TestTroughCheck:
    def test_trough_met_when_above_required(self):
        result = calculate_pk_pd_target("DrugA", 50.0, ec50=1.0, trough_exposure=10.0)
        assert result.target_met_at_trough is True

    def test_trough_not_met_when_below_required(self):
        result = calculate_pk_pd_target("DrugA", 50.0, ec50=5.0, trough_exposure=0.001)
        assert result.target_met_at_trough is False

    def test_trough_exposure_stored(self):
        result = calculate_pk_pd_target("DrugA", 50.0, ec50=1.0, trough_exposure=3.7)
        assert result.trough_exposure == 3.7

    def test_trough_default_zero(self):
        result = calculate_pk_pd_target("DrugA", 50.0, ec50=1.0)
        assert result.trough_exposure == 0.0


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_target_gte_emax_raises_value_error(self):
        with pytest.raises(ValueError, match="emax"):
            calculate_pk_pd_target("DrugA", 100.0, ec50=1.0, emax_pct=100.0)

    def test_target_equals_emax_raises_value_error(self):
        with pytest.raises(ValueError):
            calculate_pk_pd_target("DrugA", 80.0, ec50=1.0, emax_pct=80.0)

    def test_ec50_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="ec50"):
            calculate_pk_pd_target("DrugA", 50.0, ec50=0.0)

    def test_ec50_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="ec50"):
            calculate_pk_pd_target("DrugA", 50.0, ec50=-1.0)

    def test_hill_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="hill_coef"):
            calculate_pk_pd_target("DrugA", 50.0, ec50=1.0, hill_coef=0.0)

    def test_hill_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="hill_coef"):
            calculate_pk_pd_target("DrugA", 50.0, ec50=1.0, hill_coef=-2.0)

    def test_negative_target_raises_value_error(self):
        with pytest.raises(ValueError):
            calculate_pk_pd_target("DrugA", -10.0, ec50=1.0)


# ---------------------------------------------------------------------------
# Model type tests
# ---------------------------------------------------------------------------


class TestModelType:
    def test_emax_model_uses_hill_one(self):
        """emax model forces hill=1 regardless of hill_coef argument."""
        result_emax = calculate_pk_pd_target(
            "DrugA", 50.0, ec50=1.0, pd_model="emax", hill_coef=3.0
        )
        result_sigmoid = calculate_pk_pd_target(
            "DrugA", 50.0, ec50=1.0, pd_model="sigmoid_emax", hill_coef=1.0
        )
        assert abs(result_emax.required_exposure - result_sigmoid.required_exposure) < 1e-10

    def test_sigmoid_emax_model_stored(self):
        result = calculate_pk_pd_target("DrugA", 50.0, ec50=1.0, pd_model="sigmoid_emax")
        assert result.pd_model == "sigmoid_emax"

    def test_emax_model_stored(self):
        result = calculate_pk_pd_target("DrugA", 50.0, ec50=1.0, pd_model="emax")
        assert result.pd_model == "emax"


# ---------------------------------------------------------------------------
# Scan function tests
# ---------------------------------------------------------------------------


class TestScanPDTargets:
    def test_scan_returns_list(self):
        targets = [20.0, 50.0, 80.0]
        results = scan_pd_targets("DrugA", targets, ec50=1.0)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_scan_returns_pkpd_target_results(self):
        results = scan_pd_targets("DrugA", [50.0], ec50=1.0)
        assert all(isinstance(r, PKPDTargetResult) for r in results)

    def test_scan_exposure_monotonically_increasing(self):
        targets = [20.0, 40.0, 60.0, 80.0]
        results = scan_pd_targets("DrugA", targets, ec50=2.0)
        exposures = [r.required_exposure for r in results]
        assert all(exposures[i] < exposures[i + 1] for i in range(len(exposures) - 1))

    def test_scan_empty_list(self):
        results = scan_pd_targets("DrugA", [], ec50=1.0)
        assert results == []
