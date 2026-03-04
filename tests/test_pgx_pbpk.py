"""Tests for the PGx-stratified PBPK integration module.

Covers:
  1. run_pgx_pbpk returns PGxPBPKResult
  2. All four phenotypes (PM, IM, NM, UM) present in results
  3. NM AUC ratio is 1.0
  4. PM AUC is higher than NM AUC (lower CLint → higher exposure)
  5. UM AUC is lower than NM AUC (higher CLint → lower exposure)
  6. Gene sensitivity index is positive
  7. plot_pgx_forest runs without error and returns bytes or a path
  8. Symbols exported from omega_pbpk.clinical
  9. pgx_report_html returns a non-empty HTML string
  10. NM CLint scaling produces correct relative AUC
"""

from __future__ import annotations

import os
import tempfile

import pytest

from omega_pbpk.drugs.drug import Drug

# ---------------------------------------------------------------------------
# Shared test fixture — a simple CYP2D6 substrate drug with moderate CLint
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cyp2d6_drug() -> Drug:
    """A simple Drug whose hepatic clearance is predominantly via CYP2D6."""
    return Drug(
        name="TestSubstrateCYP2D6",
        mw=310.0,
        logP=2.5,
        fup=0.3,
        rbp=1.2,
        pka=[8.5],
        drug_type="monoprotic_base",
        clint_hepatic_L_per_h=20.0,  # moderate hepatic CLint
        clr_L_per_h=0.5,
        peff=1.5,
    )


@pytest.fixture(scope="module")
def pgx_result(cyp2d6_drug: Drug):
    """Run once and reuse across multiple tests."""
    from omega_pbpk.clinical.pgx_pbpk import run_pgx_pbpk

    return run_pgx_pbpk(
        drug=cyp2d6_drug,
        gene="CYP2D6",
        dose_mg=100.0,
        route="oral",
        t_end_h=24.0,
        body_weight=70.0,
    )


# ---------------------------------------------------------------------------
# Test 1 — return type
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRunPgxPbpkReturnsResult:
    def test_pgx_pbpk_returns_result(self, pgx_result) -> None:
        """run_pgx_pbpk must return a PGxPBPKResult instance."""
        from omega_pbpk.clinical.pgx_pbpk import PGxPBPKResult

        assert isinstance(pgx_result, PGxPBPKResult)


# ---------------------------------------------------------------------------
# Test 2 — all four phenotypes present
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFourPhenotypesPresent:
    def test_four_phenotypes_present(self, pgx_result) -> None:
        """PM, IM, NM, UM must all appear in phenotype_results."""
        phenotypes = {rec["phenotype"] for rec in pgx_result.phenotype_results}
        assert phenotypes == {"PM", "IM", "NM", "UM"}

    def test_exactly_four_phenotype_records(self, pgx_result) -> None:
        """phenotype_results must contain exactly 4 entries."""
        assert len(pgx_result.phenotype_results) == 4


# ---------------------------------------------------------------------------
# Test 3 — NM ratio is 1.0
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestNmRatioIsOne:
    def test_nm_ratio_is_one(self, pgx_result) -> None:
        """NM AUC ratio vs NM must equal 1.0 by definition."""
        nm_rec = next(r for r in pgx_result.phenotype_results if r["phenotype"] == "NM")
        assert nm_rec["auc_ratio_vs_nm"] == pytest.approx(1.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Test 4 — PM AUC higher than NM (lower CLint → more exposure)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPmAucHigherThanNm:
    def test_pm_auc_higher_than_nm(self, pgx_result) -> None:
        """For a CYP2D6 substrate, PM phenotype should produce a higher AUC than NM."""
        pm_rec = next(r for r in pgx_result.phenotype_results if r["phenotype"] == "PM")
        nm_rec = next(r for r in pgx_result.phenotype_results if r["phenotype"] == "NM")
        assert pm_rec["auc_mg_h_L"] >= nm_rec["auc_mg_h_L"], (
            f"PM AUC ({pm_rec['auc_mg_h_L']}) should be >= NM AUC ({nm_rec['auc_mg_h_L']})"
        )

    def test_pm_ratio_greater_than_one(self, pgx_result) -> None:
        """PM AUC ratio vs NM must be >= 1.0."""
        pm_rec = next(r for r in pgx_result.phenotype_results if r["phenotype"] == "PM")
        assert pm_rec["auc_ratio_vs_nm"] >= 1.0


# ---------------------------------------------------------------------------
# Test 5 — UM AUC lower than NM (higher CLint → less exposure)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestUmAucLowerThanNm:
    def test_um_auc_lower_than_nm(self, pgx_result) -> None:
        """For a CYP2D6 substrate, UM phenotype should produce a lower AUC than NM."""
        um_rec = next(r for r in pgx_result.phenotype_results if r["phenotype"] == "UM")
        nm_rec = next(r for r in pgx_result.phenotype_results if r["phenotype"] == "NM")
        assert um_rec["auc_mg_h_L"] <= nm_rec["auc_mg_h_L"], (
            f"UM AUC ({um_rec['auc_mg_h_L']}) should be <= NM AUC ({nm_rec['auc_mg_h_L']})"
        )

    def test_um_ratio_less_than_or_equal_one(self, pgx_result) -> None:
        """UM AUC ratio vs NM must be <= 1.0."""
        um_rec = next(r for r in pgx_result.phenotype_results if r["phenotype"] == "UM")
        assert um_rec["auc_ratio_vs_nm"] <= 1.0


# ---------------------------------------------------------------------------
# Test 6 — gene sensitivity index is positive
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGeneSensitivityIndex:
    def test_gene_sensitivity_index_positive(self, pgx_result) -> None:
        """Gene sensitivity index must be strictly positive."""
        assert pgx_result.gene_sensitivity_index > 0.0

    def test_gene_sensitivity_index_equals_max_ratio(self, pgx_result) -> None:
        """Gene sensitivity index must equal the maximum AUC ratio across phenotypes."""
        max_ratio = max(r["auc_ratio_vs_nm"] for r in pgx_result.phenotype_results)
        assert pgx_result.gene_sensitivity_index == pytest.approx(max_ratio, rel=1e-4)


# ---------------------------------------------------------------------------
# Test 7 — plot_pgx_forest runs without error
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPlotPgxForest:
    def test_plot_returns_base64_string_when_no_out_path(self, pgx_result) -> None:
        """plot_pgx_forest without out_path returns a non-empty base-64 string."""
        from omega_pbpk.clinical.pgx_pbpk import plot_pgx_forest

        output = plot_pgx_forest(pgx_result, out_path=None)
        assert isinstance(output, str)
        assert output.startswith("data:image/png;base64,")
        assert len(output) > 100

    def test_plot_saves_file_when_out_path_given(self, pgx_result) -> None:
        """plot_pgx_forest with an out_path saves a PNG file and returns the path."""
        from omega_pbpk.clinical.pgx_pbpk import plot_pgx_forest

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        try:
            returned = plot_pgx_forest(pgx_result, out_path=tmp_path)
            assert returned == tmp_path
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 1000  # a real PNG is at least 1 KB
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_plot_returns_bytes_or_path(self, pgx_result) -> None:
        """Convenience alias test: plot runs without raising an exception."""
        from omega_pbpk.clinical.pgx_pbpk import plot_pgx_forest

        output = plot_pgx_forest(pgx_result)
        assert output  # truthy — either a non-empty string (base64) or file path


# ---------------------------------------------------------------------------
# Test 8 — public import from omega_pbpk.clinical
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestImportFromClinical:
    def test_import_run_pgx_pbpk(self) -> None:
        """run_pgx_pbpk must be importable from omega_pbpk.clinical."""
        from omega_pbpk.clinical import run_pgx_pbpk  # noqa: F401

        assert callable(run_pgx_pbpk)

    def test_import_pgx_pbpk_result(self) -> None:
        """PGxPBPKResult must be importable from omega_pbpk.clinical."""
        from omega_pbpk.clinical import PGxPBPKResult  # noqa: F401

        assert PGxPBPKResult is not None

    def test_import_plot_pgx_forest(self) -> None:
        """plot_pgx_forest must be importable from omega_pbpk.clinical."""
        from omega_pbpk.clinical import plot_pgx_forest  # noqa: F401

        assert callable(plot_pgx_forest)


# ---------------------------------------------------------------------------
# Test 9 — pgx_report_html returns valid HTML
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPgxReportHtml:
    def test_html_report_is_non_empty_string(self, pgx_result) -> None:
        """pgx_report_html must return a non-empty string."""
        from omega_pbpk.clinical.pgx_pbpk import pgx_report_html

        html = pgx_report_html(pgx_result, drug_name="TestDrug")
        assert isinstance(html, str)
        assert len(html) > 500

    def test_html_contains_phenotype_labels(self, pgx_result) -> None:
        """HTML report must include all four phenotype labels."""
        from omega_pbpk.clinical.pgx_pbpk import pgx_report_html

        html = pgx_report_html(pgx_result, drug_name="TestDrug")
        for pheno in ("PM", "IM", "NM", "UM"):
            assert pheno in html, f"Phenotype {pheno} not found in HTML report"

    def test_html_contains_forest_plot(self, pgx_result) -> None:
        """HTML report must contain an embedded base-64 PNG image."""
        from omega_pbpk.clinical.pgx_pbpk import pgx_report_html

        html = pgx_report_html(pgx_result, drug_name="TestDrug")
        assert "data:image/png;base64," in html

    def test_html_contains_gene_name(self, pgx_result) -> None:
        """HTML report must mention the gene name."""
        from omega_pbpk.clinical.pgx_pbpk import pgx_report_html

        html = pgx_report_html(pgx_result, drug_name="TestDrug")
        assert "CYP2D6" in html


# ---------------------------------------------------------------------------
# Test 10 — result fields are well-formed
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestResultFieldsSanity:
    def test_nm_auc_is_positive(self, pgx_result) -> None:
        """nm_auc must be strictly positive for a drug with hepatic clearance."""
        assert pgx_result.nm_auc > 0.0

    def test_nm_cmax_is_positive(self, pgx_result) -> None:
        """nm_cmax must be strictly positive."""
        assert pgx_result.nm_cmax > 0.0

    def test_gene_field_matches_input(self, pgx_result) -> None:
        """gene field in result must match the gene passed to run_pgx_pbpk."""
        assert pgx_result.gene == "CYP2D6"

    def test_population_frequencies_are_non_negative(self, pgx_result) -> None:
        """Population frequencies in phenotype_results must be >= 0."""
        for rec in pgx_result.phenotype_results:
            assert rec["population_frequency"] >= 0.0

    def test_auc_ratios_are_non_negative(self, pgx_result) -> None:
        """AUC ratios must all be non-negative."""
        for rec in pgx_result.phenotype_results:
            assert rec["auc_ratio_vs_nm"] >= 0.0

    def test_result_is_frozen(self, pgx_result) -> None:
        """PGxPBPKResult must be frozen (immutable)."""
        with pytest.raises((AttributeError, TypeError)):
            pgx_result.gene = "CYP999"  # type: ignore[misc]

    def test_cyp2c19_gene_works(self, cyp2d6_drug: Drug) -> None:
        """run_pgx_pbpk also works for CYP2C19 (different gene)."""
        from omega_pbpk.clinical.pgx_pbpk import run_pgx_pbpk

        result = run_pgx_pbpk(
            drug=cyp2d6_drug,
            gene="CYP2C19",
            dose_mg=50.0,
            route="oral",
            t_end_h=12.0,
            body_weight=70.0,
        )
        assert result.gene == "CYP2C19"
        phenotypes = {r["phenotype"] for r in result.phenotype_results}
        assert phenotypes == {"PM", "IM", "NM", "UM"}

    def test_iv_route_works(self, cyp2d6_drug: Drug) -> None:
        """run_pgx_pbpk must also work with intravenous route."""
        from omega_pbpk.clinical.pgx_pbpk import run_pgx_pbpk

        result = run_pgx_pbpk(
            drug=cyp2d6_drug,
            gene="CYP2D6",
            dose_mg=10.0,
            route="iv",
            t_end_h=12.0,
            body_weight=70.0,
        )
        assert result.nm_auc > 0.0
