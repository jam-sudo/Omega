"""Tests for Phase 671 — pk_report_generator structured summary report."""

from __future__ import annotations

import pytest

from omega_pbpk.analysis.pk_report_generator import (
    PKReport,
    compare_to_benchmarks,
    format_pk_table,
    generate_pk_summary_report,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

GOOD_PK = {
    "cmax": 2.5,
    "auc": 50.0,
    "tmax": 1.5,
    "t_half": 8.0,
    "cl": 10.0,
    "vd": 100.0,
    "f_oral": 0.75,
}

GOOD_SAFETY = {
    "ti": 15.0,
    "herg_risk": False,
    "mec": 0.5,
    "mtc": 10.0,
}

BAD_SAFETY_LOW_TI = {
    "ti": 1.5,
    "herg_risk": False,
}

CONCERNING_SAFETY = {
    "ti": 2.5,
    "herg_risk": True,
}

GOOD_ADMET = {
    "logP": 2.0,
    "mw": 350.0,
    "psa": 60.0,
    "fup": 0.1,
    "clint": 20.0,
}


def _make_report(**kwargs) -> PKReport:
    return generate_pk_summary_report(
        drug_name=kwargs.get("drug_name", "TestDrug"),
        pk_params=kwargs.get("pk_params", GOOD_PK),
        safety_params=kwargs.get("safety_params", None),
        admet_params=kwargs.get("admet_params", None),
        regulatory_context=kwargs.get("regulatory_context", "IND"),
    )


# ---------------------------------------------------------------------------
# generate_pk_summary_report basic tests
# ---------------------------------------------------------------------------


class TestGeneratePkSummaryReport:
    def test_returns_pk_report(self):
        report = _make_report()
        assert isinstance(report, PKReport)

    def test_executive_summary_nonempty(self):
        report = _make_report()
        assert isinstance(report.executive_summary, str)
        assert len(report.executive_summary) > 0

    def test_sections_at_least_one(self):
        report = _make_report()
        assert len(report.sections) >= 1

    def test_overall_assessment_valid(self):
        report = _make_report()
        assert report.overall_assessment in {"favorable", "acceptable", "concerning", "unfavorable"}

    def test_go_nogo_valid(self):
        report = _make_report()
        assert report.go_nogo_recommendation in {"go", "nogo", "needs_optimization"}

    def test_key_findings_nonempty_list(self):
        report = _make_report()
        assert isinstance(report.key_findings, list)
        assert len(report.key_findings) >= 1

    def test_notes_nonempty(self):
        report = _make_report()
        assert isinstance(report.notes, str)
        assert len(report.notes) > 0

    def test_low_ti_concerning_or_worse(self):
        report = _make_report(safety_params={"ti": 2.5, "herg_risk": False})
        assert report.overall_assessment in {"concerning", "unfavorable"}

    def test_very_low_ti_unfavorable(self):
        report = _make_report(safety_params=BAD_SAFETY_LOW_TI)
        assert report.overall_assessment == "unfavorable"

    def test_high_ti_no_herg_favorable(self):
        report = _make_report(safety_params=GOOD_SAFETY)
        assert report.overall_assessment == "favorable"

    def test_go_nogo_go_for_favorable(self):
        report = _make_report(safety_params=GOOD_SAFETY)
        assert report.go_nogo_recommendation == "go"

    def test_go_nogo_nogo_for_unfavorable(self):
        report = _make_report(safety_params=BAD_SAFETY_LOW_TI)
        assert report.go_nogo_recommendation == "nogo"

    def test_ind_context_executive_summary_mentions_ind_or_preclinical(self):
        report = _make_report(regulatory_context="IND")
        summary_lower = report.executive_summary.lower()
        assert "preclinical" in summary_lower or "ind" in summary_lower

    def test_sections_content_nonempty(self):
        report = _make_report(safety_params=GOOD_SAFETY, admet_params=GOOD_ADMET)
        for section in report.sections:
            assert isinstance(section.content, str)
            assert len(section.content) > 0

    def test_sections_key_metrics_dict(self):
        report = _make_report(safety_params=GOOD_SAFETY, admet_params=GOOD_ADMET)
        for section in report.sections:
            assert isinstance(section.key_metrics, dict)

    def test_validation_empty_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            generate_pk_summary_report("", GOOD_PK)

    def test_validation_missing_cmax_raises(self):
        bad_params = {"auc": 50.0}
        with pytest.raises(ValueError, match="cmax"):
            generate_pk_summary_report("TestDrug", bad_params)

    def test_validation_missing_auc_raises(self):
        bad_params = {"cmax": 2.5}
        with pytest.raises(ValueError, match="auc"):
            generate_pk_summary_report("TestDrug", bad_params)

    def test_herg_risk_causes_concerning(self):
        report = _make_report(safety_params={"ti": 10.0, "herg_risk": True})
        assert report.overall_assessment in {"concerning", "unfavorable"}

    def test_go_nogo_needs_optimization_for_concerning(self):
        report = _make_report(safety_params=CONCERNING_SAFETY)
        assert report.go_nogo_recommendation == "needs_optimization"

    def test_drug_name_preserved(self):
        report = _make_report(drug_name="Aspirin")
        assert report.drug_name == "Aspirin"

    def test_generated_at_nonempty(self):
        report = _make_report()
        assert isinstance(report.generated_at, str)
        assert len(report.generated_at) > 0


# ---------------------------------------------------------------------------
# format_pk_table
# ---------------------------------------------------------------------------


class TestFormatPkTable:
    def test_returns_nonempty_string(self):
        table = format_pk_table(GOOD_PK)
        assert isinstance(table, str)
        assert len(table) > 0

    def test_contains_metric_names(self):
        table = format_pk_table(GOOD_PK)
        assert "cmax" in table
        assert "auc" in table

    def test_multiple_lines(self):
        table = format_pk_table(GOOD_PK)
        assert "\n" in table


# ---------------------------------------------------------------------------
# compare_to_benchmarks
# ---------------------------------------------------------------------------


class TestCompareToBenchmarks:
    def test_returns_dict(self):
        result = compare_to_benchmarks("TestDrug", GOOD_PK, "oncology")
        assert isinstance(result, dict)

    def test_dict_has_pass_fail_keys(self):
        result = compare_to_benchmarks("TestDrug", GOOD_PK, "oncology")
        # should have at least 'drug_name', 'therapeutic_area', 'overall_pass'
        assert "drug_name" in result
        assert "therapeutic_area" in result
        assert "overall_pass" in result

    def test_cns_high_logp_fails_logp(self):
        pk = {**GOOD_PK, "logP": 5.5}
        result = compare_to_benchmarks("TestDrug", pk, "cns")
        # logP > 3 should fail for CNS
        assert "logP_range" in result
        assert result["logP_range"]["pass"] is False

    def test_oncology_low_ti_fails(self):
        pk = {**GOOD_PK, "ti": 1.0}
        result = compare_to_benchmarks("TestDrug", pk, "oncology")
        assert "ti_min" in result
        assert result["ti_min"]["pass"] is False

    def test_cardiovascular_short_t_half_fails(self):
        pk = {**GOOD_PK, "t_half": 1.0}
        result = compare_to_benchmarks("TestDrug", pk, "cardiovascular")
        assert "t_half_range" in result
        assert result["t_half_range"]["pass"] is False

    def test_overall_pass_true_for_good_drug_oncology(self):
        pk = {**GOOD_PK, "ti": 5.0, "fup": 0.05}
        result = compare_to_benchmarks("TestDrug", pk, "oncology")
        assert result["overall_pass"] is True
