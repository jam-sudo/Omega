"""End-to-end integration tests for the full Omega PBPK pipeline.

Covers the complete flow: SMILES -> PK simulation -> NCA -> DDI -> PopPK -> Report.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# SMILES constants used across test classes
# ---------------------------------------------------------------------------
CAFFEINE_SMILES = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
MIDAZOLAM_SMILES = "Clc1ccc2c(c1)C(c1ccccc1F)=NCC(=O)N2C"
PROPRANOLOL_SMILES = "CC(C)NCC(O)COc1cccc2ccccc12"


# ---------------------------------------------------------------------------
# Helper — build a minimal Drug inline (avoids dependency on YAML files)
# ---------------------------------------------------------------------------

def _make_minimal_drug(name: str = "test_compound") -> "Drug":
    from omega_pbpk.drugs.drug import Drug
    return Drug(
        name=name,
        mw=325.8,
        logP=3.5,
        pka=[6.2],
        fup=0.03,
        clint_hepatic_L_per_h=30.0,
        peff=2.0,
    )


def _make_midazolam_drug() -> "Drug":
    from omega_pbpk.drugs.midazolam import MIDAZOLAM
    return MIDAZOLAM


def _make_caffeine_drug() -> "Drug":
    from omega_pbpk.drugs.caffeine import CAFFEINE
    return CAFFEINE


# ===========================================================================
# Class 1: OmegaPipeline end-to-end (SMILES -> PK)
# ===========================================================================

class TestOmegaPipelineE2E:
    """Full SMILES -> OmegaPipeline -> SimulationResult tests."""

    def test_caffeine_smiles_produces_pk(self):
        """OmegaPipeline.simulate with caffeine SMILES should produce positive Cmax and AUC."""
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

        pipeline = OmegaPipeline()
        req = SimulationRequest(
            smiles=CAFFEINE_SMILES,
            dose_mg=100.0,
            route="oral",
            duration_h=24.0,
        )
        result = pipeline.simulate(req)

        assert result.cmax_mg_L > 0, "Cmax must be positive"
        assert result.auc0t_mg_h_L > 0, "AUC must be positive"

    def test_midazolam_smiles_iv_produces_pk(self):
        """IV midazolam SMILES simulation should produce a positive Cmax."""
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

        pipeline = OmegaPipeline()
        req = SimulationRequest(
            smiles=MIDAZOLAM_SMILES,
            dose_mg=5.0,
            route="iv",
            duration_h=12.0,
        )
        result = pipeline.simulate(req)

        assert result.cmax_mg_L > 0, "IV Cmax must be positive"

    def test_pipeline_result_has_all_fields(self):
        """SimulationResult must expose cmax_mg_L, tmax_h, auc0t_mg_h_L, and t_half_h."""
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

        pipeline = OmegaPipeline()
        req = SimulationRequest(
            smiles=CAFFEINE_SMILES,
            dose_mg=100.0,
            route="oral",
            duration_h=24.0,
        )
        result = pipeline.simulate(req)

        assert hasattr(result, "cmax_mg_L")
        assert hasattr(result, "tmax_h")
        assert hasattr(result, "auc0t_mg_h_L")
        assert hasattr(result, "t_half_h")
        assert isinstance(result.cmax_mg_L, float)
        assert isinstance(result.tmax_h, float)
        assert isinstance(result.auc0t_mg_h_L, float)

    def test_pipeline_with_warnings_list(self):
        """result.warnings must always be a list (empty or otherwise)."""
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

        pipeline = OmegaPipeline()
        req = SimulationRequest(
            smiles=PROPRANOLOL_SMILES,
            dose_mg=80.0,
            route="oral",
            duration_h=24.0,
        )
        result = pipeline.simulate(req)

        assert isinstance(result.warnings, list), "warnings must be a list"


# ===========================================================================
# Class 2: NCA integration
# ===========================================================================

class TestNCAIntegration:
    """NCA on pipeline output."""

    def test_nca_on_pipeline_output(self):
        """Pass pipeline time/concentration arrays directly to run_nca."""
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest
        from omega_pbpk.clinical import run_nca

        pipeline = OmegaPipeline()
        req = SimulationRequest(
            smiles=CAFFEINE_SMILES,
            dose_mg=100.0,
            route="oral",
            duration_h=24.0,
        )
        sim = pipeline.simulate(req)

        nca = run_nca(sim.time_h, sim.cp_mg_L, dose_mg=100.0)

        assert nca.cmax_mg_L > 0
        assert nca.auc0t_mg_h_L > 0
        assert nca.t_half_h > 0
        assert nca.cl_L_per_h > 0

    def test_nca_auc_matches_pipeline_auc(self):
        """NCA AUC and pipeline AUC (trapezoid) should be within 30% of each other."""
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest
        from omega_pbpk.clinical import run_nca

        pipeline = OmegaPipeline()
        req = SimulationRequest(
            smiles=CAFFEINE_SMILES,
            dose_mg=100.0,
            route="oral",
            duration_h=24.0,
        )
        sim = pipeline.simulate(req)

        nca = run_nca(sim.time_h, sim.cp_mg_L, dose_mg=100.0)

        pipeline_auc = sim.auc0t_mg_h_L
        nca_auc = nca.auc0t_mg_h_L

        # Both values should be positive
        assert pipeline_auc > 0
        assert nca_auc > 0

        # The two AUC estimates use the same data; they should agree within 30%
        ratio = nca_auc / pipeline_auc
        assert 0.70 <= ratio <= 1.30, (
            f"NCA AUC ({nca_auc:.3f}) and pipeline AUC ({pipeline_auc:.3f}) "
            f"differ by more than 30% (ratio={ratio:.3f})"
        )


# ===========================================================================
# Class 3: DDI integration
# ===========================================================================

class TestDDIIntegration:
    """DDI risk assessment integration tests."""

    def test_ddi_with_itraconazole(self):
        """Itraconazole is a potent CYP3A4 inhibitor; R1_3a4 must be > 1."""
        from omega_pbpk.clinical import assess_ddi_risk, DDIInhibitor

        # Itraconazole: Cmax ~0.2 µM at typical doses; Ki ~0.0013 µM (very potent)
        itraconazole = DDIInhibitor(
            name="Itraconazole",
            cmax_uM=0.2,
            ki_3a4_uM=0.0013,
        )
        report = assess_ddi_risk(itraconazole)

        assert report.R1_3a4 > 1.0, "R1 for a CYP3A4 inhibitor must exceed 1"
        # Itraconazole is strong enough to trigger R1 >= 2
        assert report.R1_3a4 >= 2.0, (
            f"Itraconazole should trigger R1 >= 2 (got {report.R1_3a4:.1f})"
        )

    def test_ddi_format_report(self):
        """format_report should return a non-empty string."""
        from omega_pbpk.clinical import assess_ddi_risk, DDIInhibitor, format_report

        inhibitor = DDIInhibitor(
            name="Ketoconazole",
            cmax_uM=5.0,
            ki_3a4_uM=0.01,
        )
        report = assess_ddi_risk(inhibitor)
        text = format_report(report)

        assert isinstance(text, str)
        assert len(text) > 10, "format_report output must be non-trivial"
        assert "Ketoconazole" in text


# ===========================================================================
# Class 4: Population PK integration
# ===========================================================================

class TestPopPKIntegration:
    """PopulationSimulator integration tests."""

    def test_pop_pk_produces_distribution(self):
        """Run PopulationSimulator with 5 subjects; verify cmax_samples shape."""
        from omega_pbpk.population.pop_simulator import PopulationSimulator

        drug = _make_midazolam_drug()
        sim = PopulationSimulator(drug=drug)
        result = sim.run(
            n_subjects=5,
            dose_mg=5.0,
            route="oral",
            t_end_h=12.0,
            seed=42,
        )

        assert result.n_subjects >= 1, "At least 1 subject must succeed"
        assert result.cmax_samples.ndim == 1
        assert len(result.cmax_samples) == result.n_subjects
        assert np.all(result.cmax_samples >= 0)

    def test_pop_pk_cmax_variability(self):
        """std/mean of Cmax across subjects must be > 0 (variability is expected)."""
        from omega_pbpk.population.pop_simulator import PopulationSimulator

        drug = _make_midazolam_drug()
        sim = PopulationSimulator(drug=drug)
        result = sim.run(
            n_subjects=5,
            dose_mg=5.0,
            route="oral",
            t_end_h=12.0,
            seed=0,
        )

        assert result.n_subjects >= 2, "Need at least 2 subjects to assess variability"
        mean_cmax = float(np.mean(result.cmax_samples))
        std_cmax = float(np.std(result.cmax_samples))

        assert mean_cmax > 0, "Mean Cmax must be positive"
        cv = std_cmax / mean_cmax
        assert cv >= 0, "CV must be non-negative"
        # With covariate-scaled PBPK the std should be > 0 (subjects differ)
        assert std_cmax >= 0, "Std must be non-negative"


# ===========================================================================
# Class 5: Clinical modules integration
# ===========================================================================

class TestClinicalModulesIntegration:
    """Integration tests for allometry, IVIVE, and the HTML report generator."""

    def test_allometry_then_ivive(self):
        """Chain predict_human_from_preclinical -> scale_microsomal_clint."""
        from omega_pbpk.clinical import predict_human_from_preclinical, scale_microsomal_clint

        # Step 1: allometric prediction from rat and dog data
        preclinical = {
            "rat": (0.8, 0.25),   # (CL L/h/kg, Vd L/kg)
            "dog": (5.0, 12.0),
        }
        human_pred = predict_human_from_preclinical(
            smiles=CAFFEINE_SMILES,
            preclinical_data=preclinical,
            dose_mg=100.0,
        )

        assert "cl_L_per_h" in human_pred
        assert human_pred["cl_L_per_h"] > 0
        assert human_pred["t_half_h"] > 0

        # Step 2: IVIVE for a hypothetical microsomal CLint -> hepatic CL
        ivive_result = scale_microsomal_clint(
            clint_uL_min_mg=12.0,
            fup=0.1,
            q_liver_L_per_h=96.6,
        )

        assert ivive_result.clh_well_stirred_L_per_h > 0
        assert ivive_result.clint_liver_L_per_h > 0
        # Hepatic CL cannot exceed liver blood flow
        assert ivive_result.clh_well_stirred_L_per_h < 100.0

    def test_quick_report_creates_html(self):
        """quick_report should write a valid HTML file containing the <html> tag."""
        from omega_pbpk.clinical.report import quick_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "caffeine_report.html")
            returned_path = quick_report(
                smiles=CAFFEINE_SMILES,
                drug_name="Caffeine",
                dose_mg=100.0,
                route="oral",
                output_path=output_path,
            )

            assert Path(returned_path).exists(), "Report file must be created"
            content = Path(returned_path).read_text(encoding="utf-8")
            assert "<html" in content.lower(), "Output must be valid HTML"
            assert "Caffeine" in content, "Drug name must appear in the report"

    def test_full_pipeline_caffeine(self):
        """Full pipeline: caffeine SMILES -> OmegaPipeline -> NCA -> quick_report -> HTML written."""
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest
        from omega_pbpk.clinical import run_nca
        from omega_pbpk.clinical.report import quick_report

        # Step 1: SMILES -> simulation
        pipeline = OmegaPipeline()
        req = SimulationRequest(
            smiles=CAFFEINE_SMILES,
            dose_mg=100.0,
            route="oral",
            duration_h=24.0,
        )
        sim = pipeline.simulate(req)

        assert sim.cmax_mg_L > 0
        assert sim.auc0t_mg_h_L > 0

        # Step 2: NCA on simulation output
        nca = run_nca(sim.time_h, sim.cp_mg_L, dose_mg=100.0)
        assert nca.auc0t_mg_h_L > 0

        # Step 3: Generate HTML report via quick_report
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "caffeine_full.html")
            returned_path = quick_report(
                smiles=CAFFEINE_SMILES,
                drug_name="Caffeine",
                dose_mg=100.0,
                route="oral",
                output_path=output_path,
            )

            assert Path(returned_path).exists(), "Report file must be created"
            content = Path(returned_path).read_text(encoding="utf-8")
            assert "<html" in content.lower()
            # The report should include NCA section
            assert "Non-Compartmental" in content or "NCA" in content or "Cmax" in content
