"""Tests for pk_report_generator — Phase 470."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.analysis.pk_report_generator import (
    PKReportResult,
    assess_pk_quality,
    compare_studies,
    format_nca_table,
    generate_pk_report,
)


# ---------------------------------------------------------------------------
# Helpers — generate mono-exponential IV profiles
# ---------------------------------------------------------------------------


def _mono_exp_profile(
    dose_mg: float,
    vd_L: float,
    ke: float,   # 1/h
    times: list[float],
) -> list[float]:
    """C(t) = (Dose/Vd) * exp(-ke * t)."""
    c0 = dose_mg / vd_L
    return [c0 * math.exp(-ke * t) for t in times]


TIMES_DENSE = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
DOSE_MG = 100.0
VD_L = 50.0
KE = 0.1  # t½ = ln2/0.1 ≈ 6.93 h

CONCS_DENSE = _mono_exp_profile(DOSE_MG, VD_L, KE, TIMES_DENSE)

TIMES_SPARSE = [1.0, 2.0, 4.0, 8.0, 12.0]
CONCS_SPARSE = _mono_exp_profile(DOSE_MG, VD_L, KE, TIMES_SPARSE)


def _generate_simple_report(
    ke: float = KE,
    dose: float = DOSE_MG,
    vd: float = VD_L,
    times: list[float] | None = None,
) -> PKReportResult:
    if times is None:
        times = TIMES_DENSE
    concs = _mono_exp_profile(dose, vd, ke, times)
    return generate_pk_report("TestDrug", times, concs, dose, route="iv")


# ---------------------------------------------------------------------------
# generate_pk_report — basic NCA tests
# ---------------------------------------------------------------------------


class TestGeneratePkReport:
    def test_returns_pk_report_result(self):
        report = _generate_simple_report()
        assert isinstance(report, PKReportResult)

    def test_t_half_within_5pct_of_true_value(self):
        true_t_half = math.log(2) / KE
        report = _generate_simple_report()
        assert abs(report.t_half_h - true_t_half) / true_t_half < 0.05

    def test_lambda_z_within_5pct_of_ke(self):
        report = _generate_simple_report()
        assert abs(report.lambda_z_per_h - KE) / KE < 0.05

    def test_auc_inf_greater_than_auc_last(self):
        report = _generate_simple_report()
        assert report.auc_inf_mg_h_L > report.auc_last_mg_h_L

    def test_cl_equals_dose_over_auc_inf(self):
        report = _generate_simple_report()
        expected_cl = DOSE_MG / report.auc_inf_mg_h_L
        assert abs(report.cl_L_per_h - expected_cl) / expected_cl < 1e-6

    def test_cmax_is_positive(self):
        report = _generate_simple_report()
        assert report.cmax_mg_L > 0

    def test_tmax_is_first_timepoint_for_iv(self):
        """For pure IV mono-exp starting from t=0.25h, Cmax is at t[0]."""
        report = _generate_simple_report()
        assert report.tmax_h == TIMES_DENSE[0]

    def test_r_squared_terminal_in_0_1(self):
        report = _generate_simple_report()
        assert 0.0 <= report.r_squared_terminal <= 1.0

    def test_mrt_positive(self):
        report = _generate_simple_report()
        assert report.mrt_h > 0

    def test_vd_positive(self):
        report = _generate_simple_report()
        assert report.vd_L > 0

    def test_n_timepoints_correct(self):
        report = _generate_simple_report()
        assert report.n_timepoints == len(TIMES_DENSE)

    def test_drug_name_preserved(self):
        report = generate_pk_report(
            "Midazolam", TIMES_DENSE, CONCS_DENSE, DOSE_MG, route="iv"
        )
        assert report.drug_name == "Midazolam"

    def test_route_preserved(self):
        report = generate_pk_report(
            "DrugX", TIMES_DENSE, CONCS_DENSE, DOSE_MG, route="oral"
        )
        assert report.route == "oral"

    def test_notes_is_string(self):
        report = _generate_simple_report()
        assert isinstance(report.notes, str)

    def test_mismatched_array_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            generate_pk_report("X", [1.0, 2.0], [1.0], 100.0)

    def test_negative_concentration_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            generate_pk_report("X", TIMES_DENSE, [-1.0] + CONCS_DENSE[1:], DOSE_MG)

    def test_non_increasing_times_raises(self):
        bad_times = [1.0, 0.5, 2.0] + TIMES_DENSE[3:]
        concs = _mono_exp_profile(DOSE_MG, VD_L, KE, bad_times)
        with pytest.raises(ValueError, match="strictly increasing"):
            generate_pk_report("X", bad_times, concs, DOSE_MG)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            generate_pk_report("X", TIMES_DENSE, CONCS_DENSE, 0.0)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            generate_pk_report("X", TIMES_DENSE, CONCS_DENSE, DOSE_MG, route="im")

    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            generate_pk_report("", TIMES_DENSE, CONCS_DENSE, DOSE_MG)

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="At least"):
            generate_pk_report("X", [1.0, 2.0], [1.0, 0.5], 100.0)

    def test_high_ke_gives_shorter_half_life(self):
        report_fast = _generate_simple_report(ke=0.5)
        report_slow = _generate_simple_report(ke=0.05)
        assert report_fast.t_half_h < report_slow.t_half_h

    def test_auc_scales_with_dose(self):
        report_low = _generate_simple_report(dose=50.0)
        report_high = _generate_simple_report(dose=200.0)
        ratio = report_high.auc_inf_mg_h_L / report_low.auc_inf_mg_h_L
        assert abs(ratio - 4.0) < 0.5  # rough proportionality


# ---------------------------------------------------------------------------
# format_nca_table
# ---------------------------------------------------------------------------


class TestFormatNcaTable:
    def test_returns_non_empty_string(self):
        report = _generate_simple_report()
        table = format_nca_table(report)
        assert isinstance(table, str)
        assert len(table) > 0

    def test_table_contains_drug_name(self):
        report = _generate_simple_report()
        table = format_nca_table(report)
        assert "TestDrug" in table

    def test_table_contains_cmax(self):
        report = _generate_simple_report()
        table = format_nca_table(report)
        assert "Cmax" in table

    def test_non_report_raises(self):
        with pytest.raises(TypeError):
            format_nca_table("not a report")  # type: ignore[arg-type]

    def test_table_has_multiple_lines(self):
        report = _generate_simple_report()
        table = format_nca_table(report)
        assert table.count("\n") >= 5


# ---------------------------------------------------------------------------
# assess_pk_quality
# ---------------------------------------------------------------------------


class TestAssessPkQuality:
    def test_returns_dict_with_required_keys(self):
        report = _generate_simple_report()
        result = assess_pk_quality(report)
        assert "data_quality" in result
        assert "nca_reliability" in result
        assert "flags" in result

    def test_good_quality_for_dense_mono_exp(self):
        report = _generate_simple_report()
        result = assess_pk_quality(report)
        assert result["data_quality"] in {"good", "acceptable"}

    def test_flags_is_list(self):
        report = _generate_simple_report()
        result = assess_pk_quality(report)
        assert isinstance(result["flags"], list)

    def test_high_reliability_for_good_r_squared(self):
        report = _generate_simple_report()
        result = assess_pk_quality(report)
        if report.r_squared_terminal >= 0.80:
            assert result["nca_reliability"] in {"high", "medium"}

    def test_non_report_raises(self):
        with pytest.raises(TypeError):
            assess_pk_quality({"not": "a report"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compare_studies
# ---------------------------------------------------------------------------


class TestCompareStudies:
    def test_identical_reports_gmr_1(self):
        report = _generate_simple_report()
        result = compare_studies([report, report])
        assert result["gmr_cmax"] == pytest.approx(1.0, rel=1e-5)
        assert result["gmr_auc"] == pytest.approx(1.0, rel=1e-5)

    def test_identical_reports_bioequivalent(self):
        report = _generate_simple_report()
        result = compare_studies([report, report])
        assert result["bioequivalent"] is True

    def test_different_dose_reports_not_be(self):
        report_low = _generate_simple_report(dose=50.0)
        report_high = _generate_simple_report(dose=400.0)
        result = compare_studies([report_low, report_high])
        assert result["bioequivalent"] is False

    def test_gmr_cmax_in_result(self):
        r1 = _generate_simple_report()
        r2 = _generate_simple_report()
        result = compare_studies([r1, r2])
        assert "gmr_cmax" in result

    def test_gmr_auc_in_result(self):
        r1 = _generate_simple_report()
        r2 = _generate_simple_report()
        result = compare_studies([r1, r2])
        assert "gmr_auc" in result

    def test_cv_pct_cmax_is_float(self):
        r1 = _generate_simple_report()
        r2 = _generate_simple_report(dose=120.0)
        result = compare_studies([r1, r2])
        assert isinstance(result["cv_pct_cmax"], float)

    def test_n_studies_correct(self):
        reports = [_generate_simple_report() for _ in range(3)]
        result = compare_studies(reports)
        assert result["n_studies"] == 3

    def test_single_report_raises(self):
        report = _generate_simple_report()
        with pytest.raises(ValueError, match="At least 2"):
            compare_studies([report])

    def test_non_list_raises(self):
        with pytest.raises(TypeError):
            compare_studies("not a list")  # type: ignore[arg-type]

    def test_notes_is_string(self):
        reports = [_generate_simple_report(), _generate_simple_report()]
        result = compare_studies(reports)
        assert isinstance(result["notes"], str)
