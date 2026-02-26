"""Tests for HTML report generator and VPC plots."""
import numpy as np
import pytest
from pathlib import Path


class TestGenerateReport:
    def test_generates_html_file(self, tmp_path):
        from omega_pbpk.clinical.report import ReportInput, generate_report
        inp = ReportInput(drug_name="TestDrug", smiles="C", dose_mg=100.0)
        out = str(tmp_path / "report.html")
        path = generate_report(inp, out)
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "<html" in content
        assert "TestDrug" in content

    def test_nca_section_present(self, tmp_path):
        from omega_pbpk.clinical.report import ReportInput, generate_report
        from omega_pbpk.clinical.nca import run_nca
        t = np.linspace(0, 24, 50)
        cp = 2.0 * np.exp(-0.3 * t)
        nca = run_nca(t, cp, dose_mg=100.0)
        inp = ReportInput(drug_name="Test", nca_result=nca)
        out = str(tmp_path / "r.html")
        generate_report(inp, out)
        assert "Non-Compartmental" in Path(out).read_text()

    def test_ddi_section_present(self, tmp_path):
        from omega_pbpk.clinical.report import ReportInput, generate_report
        from omega_pbpk.clinical.ddi_report import DDIInhibitor, assess_ddi_risk
        inh = DDIInhibitor(name="test", cmax_uM=1.0, ki_3a4_uM=0.1)
        ddi = assess_ddi_risk(inh)
        inp = ReportInput(drug_name="Test", ddi_report=ddi)
        out = str(tmp_path / "r.html")
        generate_report(inp, out)
        assert "DDI Risk" in Path(out).read_text()

    def test_warnings_section(self, tmp_path):
        from omega_pbpk.clinical.report import ReportInput, generate_report
        inp = ReportInput(drug_name="Test", warnings=["Test warning"])
        out = str(tmp_path / "r.html")
        generate_report(inp, out)
        assert "Test warning" in Path(out).read_text()

    def test_empty_report_valid_html(self, tmp_path):
        from omega_pbpk.clinical.report import ReportInput, generate_report
        inp = ReportInput(drug_name="Empty")
        out = str(tmp_path / "r.html")
        generate_report(inp, out)
        content = Path(out).read_text()
        assert "</html>" in content


class TestVPC:
    def _make_pop_data(self, n=10, n_t=50):
        t = np.linspace(0, 24, n_t)
        cp = np.array([2.0 * np.exp(-0.2 * t) * (0.8 + 0.4 * np.random.rand()) for _ in range(n)])
        return t, cp

    def test_vpc_saves_file(self, tmp_path):
        from omega_pbpk.visualization.vpc import plot_vpc
        t, cp = self._make_pop_data()
        out = str(tmp_path / "vpc.png")
        plot_vpc(t, cp, output_path=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 1000

    def test_forest_saves_file(self, tmp_path):
        from omega_pbpk.visualization.vpc import plot_forest
        out = str(tmp_path / "forest.png")
        plot_forest("Cmax", ["Group A", "Group B"], [2.0, 3.5],
                    [1.5, 2.8], [2.5, 4.2], output_path=out)
        assert Path(out).exists()
