"""Tests for Phase 30 metabolic reaction phenotyping — CYP contribution analysis.

Covers:
  - TestChemicalInhibition (6): inhibition-based fm calculation
  - TestRecombinantCorrelation (3): rCYP ISEF/RAF scaling
  - TestCombinedPhenotyping (2): weighted consensus and end-to-end workflow
  - TestVulnerabilityAssessment (1): DDI and PGx classification
  - TestPhenotypingAPI (2): POST /phenotyping endpoint contract
"""

from __future__ import annotations

import math

from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.clinical.reaction_phenotyping import (
    DEFAULT_ABUNDANCE,
    DEFAULT_ISEF,
    InhibitionExperiment,
    PhenotypingResult,
    RecombinantCYPData,
    assess_ddi_vulnerability,
    assess_pgx_sensitivity,
    phenotype_by_inhibition,
    phenotype_by_recombinant,
    phenotype_combined,
    run_reaction_phenotyping,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# TestChemicalInhibition (6 tests)
# ---------------------------------------------------------------------------


class TestChemicalInhibition:
    def test_single_enzyme_dominant(self):
        """CYP3A4 accounts for ~80% of metabolism → fm ≈ 0.8."""
        experiments = [
            InhibitionExperiment(
                enzyme="CYP3A4",
                inhibitor_name="ketoconazole",
                control_clint=100.0,
                inhibited_clint=20.0,
            ),
        ]
        contributions = phenotype_by_inhibition(experiments)
        assert len(contributions) == 1
        c = contributions[0]
        assert c.enzyme == "CYP3A4"
        assert math.isclose(c.fm, 0.8, abs_tol=0.01)
        assert c.method == "inhibition"
        assert c.confidence == "high"

    def test_multiple_enzyme_contributions(self):
        """Three CYP enzymes with fractions summing to ≈ 1.0."""
        experiments = [
            InhibitionExperiment(
                enzyme="CYP3A4",
                inhibitor_name="ketoconazole",
                control_clint=100.0,
                inhibited_clint=50.0,
            ),
            InhibitionExperiment(
                enzyme="CYP2D6",
                inhibitor_name="quinidine",
                control_clint=100.0,
                inhibited_clint=70.0,
            ),
            InhibitionExperiment(
                enzyme="CYP2C9",
                inhibitor_name="sulfaphenazole",
                control_clint=100.0,
                inhibited_clint=80.0,
            ),
        ]
        contributions = phenotype_by_inhibition(experiments)
        assert len(contributions) == 3
        fm_sum = sum(c.fm for c in contributions)
        assert math.isclose(fm_sum, 1.0, abs_tol=0.05)

    def test_normalization_when_sum_exceeds_one(self):
        """When raw fm values sum > 1.0, they are normalized to 1.0."""
        experiments = [
            InhibitionExperiment(
                enzyme="CYP3A4",
                inhibitor_name="ketoconazole",
                control_clint=100.0,
                inhibited_clint=10.0,  # fm_raw = 0.9
            ),
            InhibitionExperiment(
                enzyme="CYP2D6",
                inhibitor_name="quinidine",
                control_clint=100.0,
                inhibited_clint=30.0,  # fm_raw = 0.7
            ),
        ]
        contributions = phenotype_by_inhibition(experiments)
        fm_sum = sum(c.fm for c in contributions)
        assert math.isclose(fm_sum, 1.0, abs_tol=0.001)
        # CYP3A4 should still dominate
        fm_dict = {c.enzyme: c.fm for c in contributions}
        assert fm_dict["CYP3A4"] > fm_dict["CYP2D6"]

    def test_ki_correction(self):
        """Sub-saturating inhibitor is corrected via Ki adjustment."""
        # Without Ki correction: fm_apparent = (100 - 50) / 100 = 0.5
        # With Ki correction: fractional_inhibition = 1.0 / (1.0 + 0.5) = 0.667
        #   fm_corrected = 0.5 / 0.667 = 0.75
        exp_no_ki = InhibitionExperiment(
            enzyme="CYP3A4",
            inhibitor_name="ketoconazole",
            control_clint=100.0,
            inhibited_clint=50.0,
        )
        exp_with_ki = InhibitionExperiment(
            enzyme="CYP3A4",
            inhibitor_name="ketoconazole",
            control_clint=100.0,
            inhibited_clint=50.0,
            inhibitor_conc_uM=1.0,
            ki_uM=0.5,
        )
        result_no_ki = phenotype_by_inhibition([exp_no_ki])
        result_with_ki = phenotype_by_inhibition([exp_with_ki])
        # Ki-corrected fm should be higher (adjusting for incomplete inhibition)
        assert result_with_ki[0].fm > result_no_ki[0].fm
        assert math.isclose(result_no_ki[0].fm, 0.5, abs_tol=0.01)
        assert math.isclose(result_with_ki[0].fm, 0.75, abs_tol=0.01)

    def test_zero_control_clint_warning(self):
        """Edge case: near-zero control CLint should not crash (division-safe)."""
        experiments = [
            InhibitionExperiment(
                enzyme="CYP3A4",
                inhibitor_name="ketoconazole",
                control_clint=0.0,
                inhibited_clint=0.0,
            ),
        ]
        contributions = phenotype_by_inhibition(experiments)
        assert len(contributions) == 1
        # Should handle gracefully — fm = 0 (0/tiny_number → 0 after clamp)
        assert contributions[0].fm >= 0.0
        assert contributions[0].fm <= 1.0

    def test_unaccounted_fraction_flagged(self):
        """When fm sum < 0.7, warning about non-CYP pathways is raised."""
        experiments = [
            InhibitionExperiment(
                enzyme="CYP3A4",
                inhibitor_name="ketoconazole",
                control_clint=100.0,
                inhibited_clint=80.0,  # fm = 0.2
            ),
            InhibitionExperiment(
                enzyme="CYP2D6",
                inhibitor_name="quinidine",
                control_clint=100.0,
                inhibited_clint=85.0,  # fm = 0.15
            ),
        ]
        # Run full phenotyping to get warnings
        result = run_reaction_phenotyping(
            drug_name="TestDrug",
            inhibition_experiments=experiments,
        )
        assert any("accounted" in w.lower() for w in result.warnings)
        fm_sum = sum(result.fm_dict.values())
        assert fm_sum < 0.7


# ---------------------------------------------------------------------------
# TestRecombinantCorrelation (3 tests)
# ---------------------------------------------------------------------------


class TestRecombinantCorrelation:
    def test_isef_method(self):
        """ISEF method: CLint = velocity × abundance × ISEF → normalized fm."""
        data = [
            RecombinantCYPData(
                enzyme="CYP3A4",
                velocity_pmol_min_pmol=10.0,
                isef=1.0,
                abundance_pmol_mg=137.0,
            ),
            RecombinantCYPData(
                enzyme="CYP2D6",
                velocity_pmol_min_pmol=5.0,
                isef=4.8,
                abundance_pmol_mg=8.0,
            ),
        ]
        contributions = phenotype_by_recombinant(data)
        assert len(contributions) == 2
        fm_dict = {c.enzyme: c.fm for c in contributions}
        # CYP3A4: 10 * 137 * 1.0 = 1370
        # CYP2D6: 5 * 8 * 4.8 = 192
        # total = 1562
        # fm_3A4 = 1370/1562 ≈ 0.877
        assert fm_dict["CYP3A4"] > fm_dict["CYP2D6"]
        assert math.isclose(fm_dict["CYP3A4"], 1370.0 / 1562.0, abs_tol=0.01)
        fm_sum = sum(fm_dict.values())
        assert math.isclose(fm_sum, 1.0, abs_tol=0.001)
        assert all(c.method == "recombinant_isef" for c in contributions)

    def test_raf_method(self):
        """RAF method: CLint = velocity × RAF → normalized fm."""
        data = [
            RecombinantCYPData(
                enzyme="CYP3A4",
                velocity_pmol_min_pmol=10.0,
                raf=5.0,
            ),
            RecombinantCYPData(
                enzyme="CYP2C9",
                velocity_pmol_min_pmol=8.0,
                raf=3.0,
            ),
        ]
        contributions = phenotype_by_recombinant(data)
        fm_dict = {c.enzyme: c.fm for c in contributions}
        # CYP3A4: 10 * 5 = 50, CYP2C9: 8 * 3 = 24, total = 74
        assert math.isclose(fm_dict["CYP3A4"], 50.0 / 74.0, abs_tol=0.01)
        assert math.isclose(fm_dict["CYP2C9"], 24.0 / 74.0, abs_tol=0.01)
        assert all(c.method == "recombinant_raf" for c in contributions)

    def test_custom_abundance(self):
        """User-supplied abundance overrides the default table value."""
        custom_abundance = 200.0  # much higher than default 137.0 for CYP3A4
        data_custom = [
            RecombinantCYPData(
                enzyme="CYP3A4",
                velocity_pmol_min_pmol=10.0,
                abundance_pmol_mg=custom_abundance,
            ),
        ]
        data_default = [
            RecombinantCYPData(
                enzyme="CYP3A4",
                velocity_pmol_min_pmol=10.0,
            ),
        ]
        result_custom = phenotype_by_recombinant(data_custom)
        result_default = phenotype_by_recombinant(data_default)
        # Both have fm=1.0 (single enzyme), but clint_enzyme should differ
        expected_custom = 10.0 * custom_abundance * DEFAULT_ISEF["CYP3A4"]
        expected_default = 10.0 * DEFAULT_ABUNDANCE["CYP3A4"] * DEFAULT_ISEF["CYP3A4"]
        assert math.isclose(result_custom[0].clint_enzyme_uL_min_mg, expected_custom, rel_tol=0.01)
        assert math.isclose(
            result_default[0].clint_enzyme_uL_min_mg, expected_default, rel_tol=0.01
        )
        assert result_custom[0].clint_enzyme_uL_min_mg != result_default[0].clint_enzyme_uL_min_mg


# ---------------------------------------------------------------------------
# TestCombinedPhenotyping (2 tests)
# ---------------------------------------------------------------------------


class TestCombinedPhenotyping:
    def test_combined_inhibition_priority(self):
        """Inhibition data is weighted higher (default 70%) in combined method."""
        inhibition = [
            InhibitionExperiment(
                enzyme="CYP3A4",
                inhibitor_name="ketoconazole",
                control_clint=100.0,
                inhibited_clint=20.0,  # fm_inh = 0.8
            ),
        ]
        recombinant = [
            RecombinantCYPData(
                enzyme="CYP3A4",
                velocity_pmol_min_pmol=10.0,
                raf=5.0,
            ),
        ]
        combined = phenotype_combined(inhibition, recombinant)

        fm_combined = combined[0].fm
        assert combined[0].method == "combined"
        # For single enzyme: all fm = 1.0, so check it's still 1.0
        assert math.isclose(fm_combined, 1.0, abs_tol=0.01)

    def test_full_workflow(self):
        """End-to-end run_reaction_phenotyping returns complete result."""
        inhibition = [
            InhibitionExperiment(
                enzyme="CYP3A4",
                inhibitor_name="ketoconazole",
                control_clint=100.0,
                inhibited_clint=15.0,
            ),
            InhibitionExperiment(
                enzyme="CYP2D6",
                inhibitor_name="quinidine",
                control_clint=100.0,
                inhibited_clint=90.0,
            ),
        ]
        recombinant = [
            RecombinantCYPData(
                enzyme="CYP3A4",
                velocity_pmol_min_pmol=10.0,
                isef=1.0,
                abundance_pmol_mg=137.0,
            ),
            RecombinantCYPData(
                enzyme="CYP2D6",
                velocity_pmol_min_pmol=5.0,
                isef=4.8,
                abundance_pmol_mg=8.0,
            ),
        ]
        result = run_reaction_phenotyping(
            drug_name="TestCompound",
            inhibition_experiments=inhibition,
            recombinant_data=recombinant,
            method="auto",
        )
        assert isinstance(result, PhenotypingResult)
        assert result.drug_name == "TestCompound"
        assert result.method == "combined"
        assert result.primary_enzyme == "CYP3A4"
        assert result.fm_primary > 0.5
        assert len(result.contributions) >= 2
        assert result.total_clint_uL_min_mg > 0
        assert result.ddi_vulnerability in ("low", "moderate", "high")
        assert result.pgx_sensitivity in ("low", "moderate", "high")
        fm_sum = sum(result.fm_dict.values())
        assert math.isclose(fm_sum, 1.0, abs_tol=0.05)
        assert isinstance(result.warnings, list)
        assert isinstance(result.recommendation, str)


# ---------------------------------------------------------------------------
# TestVulnerabilityAssessment (1 test)
# ---------------------------------------------------------------------------


class TestVulnerabilityAssessment:
    def test_ddi_and_pgx_classification(self):
        """High DDI risk (fm_3A4 = 0.85) and low PGx sensitivity."""
        fm_dict = {"CYP3A4": 0.85, "CYP2D6": 0.10, "CYP2C9": 0.05}

        ddi = assess_ddi_vulnerability(fm_dict)
        pgx = assess_pgx_sensitivity(fm_dict)

        # CYP3A4 fm = 0.85 ≥ 0.8 → high DDI
        assert ddi == "high"
        # CYP2D6 = 0.10 < 0.25, CYP2C9 = 0.05 < 0.25 → low PGx
        assert pgx == "low"


# ---------------------------------------------------------------------------
# TestPhenotypingAPI (2 tests)
# ---------------------------------------------------------------------------


class TestPhenotypingAPI:
    def test_endpoint_200(self):
        """POST /phenotyping returns 200 with valid inhibition data."""
        payload = {
            "drug_name": "APIDrug",
            "inhibition_experiments": [
                {
                    "enzyme": "CYP3A4",
                    "inhibitor_name": "ketoconazole",
                    "control_clint": 100.0,
                    "inhibited_clint": 20.0,
                },
                {
                    "enzyme": "CYP2D6",
                    "inhibitor_name": "quinidine",
                    "control_clint": 100.0,
                    "inhibited_clint": 70.0,
                },
            ],
            "method": "auto",
        }
        r = client.post("/phenotyping", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["drug_name"] == "APIDrug"
        assert "contributions" in body
        assert "fm_dict" in body
        assert "primary_enzyme" in body
        assert "ddi_vulnerability" in body
        assert "pgx_sensitivity" in body
        assert body["method"] == "inhibition"
        assert len(body["contributions"]) == 2
        assert isinstance(body["warnings"], list)

    def test_endpoint_empty_data(self):
        """No experiments provided → warning in response."""
        payload = {
            "drug_name": "EmptyDrug",
        }
        r = client.post("/phenotyping", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["drug_name"] == "EmptyDrug"
        assert len(body["warnings"]) > 0
        assert any("no experimental data" in w.lower() for w in body["warnings"])
        assert body["method"] == "none"
        assert len(body["contributions"]) == 0
