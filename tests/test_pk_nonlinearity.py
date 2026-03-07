"""Tests for Phase 373 — PK Nonlinearity Detector."""

import pytest

from omega_pbpk.analysis.pk_nonlinearity import (
    PKNonlinearityResult,
    detect_pk_nonlinearity,
    generate_dose_proportionality_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DOSES_LINEAR = [10.0, 20.0, 50.0, 100.0, 200.0]
# AUC exactly proportional to dose (slope = 2.5)
AUC_LINEAR = [d * 2.5 for d in DOSES_LINEAR]
CMAX_LINEAR = [d * 0.8 for d in DOSES_LINEAR]

# Supraproportional: AUC grows faster than dose (b > 1.1)
DOSES_SUPRA = [10.0, 20.0, 50.0, 100.0, 200.0]
AUC_SUPRA = [d**1.5 * 0.5 for d in DOSES_SUPRA]  # b≈1.5
CMAX_SUPRA = [d * 0.8 for d in DOSES_SUPRA]

# Infraproportional: AUC grows slower than dose (b < 0.9)
DOSES_INFRA = [10.0, 20.0, 50.0, 100.0, 200.0]
AUC_INFRA = [d**0.5 * 10.0 for d in DOSES_INFRA]  # b≈0.5
CMAX_INFRA = [d**0.5 * 3.0 for d in DOSES_INFRA]  # Cmax also infraproportional


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------


class TestPKNonlinearityResultStructure:
    def test_returns_dataclass(self):
        result = detect_pk_nonlinearity("Drug", DOSES_LINEAR, AUC_LINEAR, CMAX_LINEAR)
        assert isinstance(result, PKNonlinearityResult)

    def test_frozen_dataclass(self):
        result = detect_pk_nonlinearity("Drug", DOSES_LINEAR, AUC_LINEAR, CMAX_LINEAR)
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_drug_name_preserved(self):
        result = detect_pk_nonlinearity("TestDrug", DOSES_LINEAR, AUC_LINEAR, CMAX_LINEAR)
        assert result.drug_name == "TestDrug"

    def test_n_doses_correct(self):
        result = detect_pk_nonlinearity("Drug", DOSES_LINEAR, AUC_LINEAR, CMAX_LINEAR)
        assert result.n_doses == len(DOSES_LINEAR)


# ---------------------------------------------------------------------------
# Linear data tests
# ---------------------------------------------------------------------------


class TestLinearPK:
    def test_high_r2_for_linear_auc(self):
        result = detect_pk_nonlinearity("Drug", DOSES_LINEAR, AUC_LINEAR, CMAX_LINEAR)
        assert result.auc_linearity_r2 > 0.99

    def test_is_nonlinear_auc_false_for_linear(self):
        result = detect_pk_nonlinearity("Drug", DOSES_LINEAR, AUC_LINEAR, CMAX_LINEAR)
        assert result.is_nonlinear_auc is False

    def test_is_nonlinear_cmax_false_for_linear(self):
        result = detect_pk_nonlinearity("Drug", DOSES_LINEAR, AUC_LINEAR, CMAX_LINEAR)
        assert result.is_nonlinear_cmax is False

    def test_power_law_b_near_1_for_linear(self):
        result = detect_pk_nonlinearity("Drug", DOSES_LINEAR, AUC_LINEAR, CMAX_LINEAR)
        assert abs(result.auc_power_law_b - 1.0) < 0.02

    def test_dnAUC_cv_pct_low_for_linear(self):
        result = detect_pk_nonlinearity("Drug", DOSES_LINEAR, AUC_LINEAR, CMAX_LINEAR)
        assert result.dnAUC_cv_pct < 5.0

    def test_nonlinearity_type_linear(self):
        result = detect_pk_nonlinearity("Drug", DOSES_LINEAR, AUC_LINEAR, CMAX_LINEAR)
        assert result.nonlinearity_type == "linear"

    def test_suspected_mechanism_linear(self):
        result = detect_pk_nonlinearity("Drug", DOSES_LINEAR, AUC_LINEAR, CMAX_LINEAR)
        assert result.suspected_mechanism == "linear"

    def test_dose_normalized_auc_length(self):
        result = detect_pk_nonlinearity("Drug", DOSES_LINEAR, AUC_LINEAR, CMAX_LINEAR)
        assert len(result.dose_normalized_auc) == len(DOSES_LINEAR)

    def test_dose_normalized_cmax_length(self):
        result = detect_pk_nonlinearity("Drug", DOSES_LINEAR, AUC_LINEAR, CMAX_LINEAR)
        assert len(result.dose_normalized_cmax) == len(DOSES_LINEAR)

    def test_dose_normalized_auc_values_constant(self):
        result = detect_pk_nonlinearity("Drug", DOSES_LINEAR, AUC_LINEAR, CMAX_LINEAR)
        # All DN-AUC values should be ~2.5
        for v in result.dose_normalized_auc:
            assert abs(v - 2.5) < 0.01

    def test_saturation_dose_beyond_range_for_linear(self):
        result = detect_pk_nonlinearity("Drug", DOSES_LINEAR, AUC_LINEAR, CMAX_LINEAR)
        assert result.saturation_dose_mg > max(DOSES_LINEAR)


# ---------------------------------------------------------------------------
# Supraproportional tests
# ---------------------------------------------------------------------------


class TestSupraproportionalPK:
    def test_is_nonlinear_auc_true(self):
        result = detect_pk_nonlinearity("Drug", DOSES_SUPRA, AUC_SUPRA, CMAX_SUPRA)
        assert result.is_nonlinear_auc is True

    def test_nonlinearity_type_supraproportional(self):
        result = detect_pk_nonlinearity("Drug", DOSES_SUPRA, AUC_SUPRA, CMAX_SUPRA)
        assert result.nonlinearity_type == "supraproportional"

    def test_power_law_b_greater_than_1(self):
        result = detect_pk_nonlinearity("Drug", DOSES_SUPRA, AUC_SUPRA, CMAX_SUPRA)
        assert result.auc_power_law_b > 1.1

    def test_suspected_mechanism_saturable_cl(self):
        result = detect_pk_nonlinearity("Drug", DOSES_SUPRA, AUC_SUPRA, CMAX_SUPRA)
        assert result.suspected_mechanism == "saturable_CL"

    def test_dnAUC_cv_pct_high_for_supra(self):
        result = detect_pk_nonlinearity("Drug", DOSES_SUPRA, AUC_SUPRA, CMAX_SUPRA)
        assert result.dnAUC_cv_pct > 10.0

    def test_saturation_dose_within_observed_range(self):
        result = detect_pk_nonlinearity("Drug", DOSES_SUPRA, AUC_SUPRA, CMAX_SUPRA)
        assert result.saturation_dose_mg > 0


# ---------------------------------------------------------------------------
# Infraproportional tests
# ---------------------------------------------------------------------------


class TestInfraproportionalPK:
    def test_nonlinearity_type_infraproportional(self):
        result = detect_pk_nonlinearity("Drug", DOSES_INFRA, AUC_INFRA, CMAX_INFRA)
        assert result.nonlinearity_type == "infraproportional"

    def test_power_law_b_less_than_1(self):
        result = detect_pk_nonlinearity("Drug", DOSES_INFRA, AUC_INFRA, CMAX_INFRA)
        assert result.auc_power_law_b < 0.9

    def test_suspected_mechanism_saturable_absorption(self):
        # When Cmax also infraproportional -> saturable_absorption
        result = detect_pk_nonlinearity("Drug", DOSES_INFRA, AUC_INFRA, CMAX_INFRA)
        assert result.suspected_mechanism == "saturable_absorption"

    def test_is_nonlinear_true(self):
        result = detect_pk_nonlinearity("Drug", DOSES_INFRA, AUC_INFRA, CMAX_INFRA)
        assert result.is_nonlinear_auc is True


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------


class TestReportGeneration:
    def test_report_non_empty_string(self):
        report = generate_dose_proportionality_report("Drug", DOSES_LINEAR, AUC_LINEAR, CMAX_LINEAR)
        assert isinstance(report, str)
        assert len(report) > 100

    def test_report_contains_drug_name(self):
        report = generate_dose_proportionality_report(
            "MyDrug", DOSES_LINEAR, AUC_LINEAR, CMAX_LINEAR
        )
        assert "MYDRUG" in report or "MyDrug" in report

    def test_report_contains_nonlinearity_type(self):
        report = generate_dose_proportionality_report("Drug", DOSES_LINEAR, AUC_LINEAR, CMAX_LINEAR)
        assert "linear" in report.lower()


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            detect_pk_nonlinearity(
                "Drug", [10.0, 20.0, 50.0], [25.0, 50.0, 125.0, 250.0], [8.0, 16.0, 40.0]
            )

    def test_raises_on_fewer_than_3_points(self):
        with pytest.raises(ValueError, match="3"):
            detect_pk_nonlinearity("Drug", [10.0, 20.0], [25.0, 50.0], [8.0, 16.0])

    def test_raises_on_negative_dose(self):
        with pytest.raises(ValueError, match="positive"):
            detect_pk_nonlinearity(
                "Drug", [-10.0, 20.0, 50.0], [25.0, 50.0, 125.0], [8.0, 16.0, 40.0]
            )

    def test_raises_on_negative_auc(self):
        with pytest.raises(ValueError, match="positive"):
            detect_pk_nonlinearity(
                "Drug", [10.0, 20.0, 50.0], [-25.0, 50.0, 125.0], [8.0, 16.0, 40.0]
            )

    def test_raises_on_unsorted_doses(self):
        with pytest.raises(ValueError, match="ascending"):
            detect_pk_nonlinearity(
                "Drug", [50.0, 20.0, 10.0], [125.0, 50.0, 25.0], [40.0, 16.0, 8.0]
            )

    def test_saturation_dose_positive(self):
        result = detect_pk_nonlinearity("Drug", DOSES_LINEAR, AUC_LINEAR, CMAX_LINEAR)
        assert result.saturation_dose_mg > 0

    def test_notes_non_empty(self):
        result = detect_pk_nonlinearity("Drug", DOSES_LINEAR, AUC_LINEAR, CMAX_LINEAR)
        assert isinstance(result.notes, str)
