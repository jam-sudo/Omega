"""Tests for Phase 35 MIST (Metabolites in Safety Testing) assessment.

Covers:
  - TestMISTUnit (6): core metabolite AUC estimation & classification logic
  - TestMISTIntegration (6): full run_mist_assessment pipeline & format_mist_table
  - TestMISTAPI (2): POST /mist endpoint contract
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.drugs.drug import Drug
from omega_pbpk.risk.mist_assessment import (
    MetaboliteSpec,
    MISTReport,
    classify_metabolite,
    estimate_metabolite_auc,
    format_mist_table,
    run_mist_assessment,
)

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PARENT = Drug(
    name="TestParent",
    mw=300.0,
    logP=2.0,
    fup=0.5,
    rbp=1.0,
    drug_type="neutral",
    clint_hepatic_L_per_h=5.0,
    peff=1.0,
)

_MET_LOW = MetaboliteSpec(
    name="M1-low",
    mw=280.0,
    fm_parent=0.02,
    clint_met_L_per_h=10.0,
)

_MET_HIGH = MetaboliteSpec(
    name="M2-high",
    mw=310.0,
    fm_parent=0.40,
    clint_met_L_per_h=2.0,
    pharmacologically_active=True,
)

_MET_UNIQUE = MetaboliteSpec(
    name="M3-unique",
    mw=290.0,
    fm_parent=0.05,
    clint_met_L_per_h=8.0,
    unique_human=True,
)

_MET_NEAR_ZERO_CL = MetaboliteSpec(
    name="M4-zeroCL",
    mw=300.0,
    fm_parent=0.10,
    clint_met_L_per_h=0.0,
)


# ---------------------------------------------------------------------------
# TestMISTUnit — core metabolite AUC estimation & classification
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMISTUnit:
    def test_estimate_metabolite_auc_basic(self):
        """AUC_met = fm * AUC_parent * (CL_parent / CL_met) * (MW_met / MW_parent)."""
        parent_auc = 20.0  # mg·h/L
        parent_mw = 300.0
        parent_cl = 5.0
        auc_met, warnings = estimate_metabolite_auc(
            parent_auc=parent_auc,
            parent_mw=parent_mw,
            met_spec=_MET_LOW,
            parent_cl=parent_cl,
        )
        expected = 0.02 * 20.0 * (5.0 / 10.0) * (280.0 / 300.0)
        assert abs(auc_met - expected) < 1e-9
        assert warnings == []

    def test_estimate_metabolite_auc_near_zero_cl_clamps(self):
        """Near-zero metabolite CL should be clamped and produce a warning."""
        auc_met, warnings = estimate_metabolite_auc(
            parent_auc=10.0,
            parent_mw=300.0,
            met_spec=_MET_NEAR_ZERO_CL,
            parent_cl=5.0,
        )
        assert auc_met > 0
        assert len(warnings) == 1
        assert "clamped" in warnings[0].lower()

    def test_classify_covered(self):
        """Ratio <= 10% → covered / low."""
        classification, risk, rec = classify_metabolite(
            ratio=0.05, active=False, unique_human=False
        )
        assert classification == "covered"
        assert risk == "low"

    def test_classify_disproportionate_inactive(self):
        """Ratio > 10%, inactive → disproportionate / moderate."""
        classification, risk, rec = classify_metabolite(
            ratio=0.20, active=False, unique_human=False
        )
        assert classification == "disproportionate"
        assert risk == "moderate"

    def test_classify_disproportionate_active(self):
        """Ratio > 10%, active → disproportionate / high."""
        classification, risk, rec = classify_metabolite(ratio=0.25, active=True, unique_human=False)
        assert classification == "disproportionate"
        assert risk == "high"

    def test_classify_unique_human_always_high(self):
        """Unique human metabolite → high risk regardless of ratio."""
        classification, risk, rec = classify_metabolite(ratio=0.01, active=False, unique_human=True)
        assert classification == "unique_human"
        assert risk == "high"
        assert "dedicated safety studies" in rec.lower()


# ---------------------------------------------------------------------------
# TestMISTIntegration — full assessment pipeline
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMISTIntegration:
    def test_single_low_metabolite_overall_low(self):
        """One covered metabolite → overall low risk, 0 flagged."""
        report = run_mist_assessment(
            drug=_PARENT,
            metabolites=[_MET_LOW],
            dose_mg=100.0,
            route="oral",
        )
        assert isinstance(report, MISTReport)
        assert report.overall_mist_risk == "low"
        assert report.n_flagged == 0
        assert len(report.metabolite_results) == 1
        assert report.metabolite_results[0].classification == "covered"

    def test_high_metabolite_flagged(self):
        """Active disproportionate metabolite → flagged, overall high."""
        report = run_mist_assessment(
            drug=_PARENT,
            metabolites=[_MET_HIGH],
            dose_mg=100.0,
            route="oral",
        )
        assert report.n_flagged == 1
        assert report.overall_mist_risk == "high"
        mr = report.metabolite_results[0]
        assert mr.exceeds_10pct is True
        assert mr.classification == "disproportionate"

    def test_unique_human_flagged_regardless_of_ratio(self):
        """Unique human metabolite counted as flagged even if ratio < 10%."""
        report = run_mist_assessment(
            drug=_PARENT,
            metabolites=[_MET_UNIQUE],
            dose_mg=100.0,
            route="oral",
        )
        assert report.n_flagged == 1
        assert report.overall_mist_risk == "high"
        assert report.metabolite_results[0].classification == "unique_human"

    def test_iv_route_bioavailability_one(self):
        """IV route uses F=1; AUC = Dose / CL, Cmax = Dose / Vd."""
        report = run_mist_assessment(
            drug=_PARENT,
            metabolites=[_MET_LOW],
            dose_mg=100.0,
            route="iv",
        )
        cl = _PARENT.clint_scaled_L_per_h + _PARENT.clr_L_per_h
        expected_auc = 100.0 / cl
        assert abs(report.parent_auc_mg_h_L - round(expected_auc, 6)) < 1e-4

    def test_multiple_metabolites_mixed_risk(self):
        """Mixed metabolites: low + high + unique → overall high, 2 flagged."""
        report = run_mist_assessment(
            drug=_PARENT,
            metabolites=[_MET_LOW, _MET_HIGH, _MET_UNIQUE],
            dose_mg=100.0,
            route="oral",
        )
        assert len(report.metabolite_results) == 3
        assert report.n_flagged == 2
        assert report.overall_mist_risk == "high"
        assert len(report.recommendations) >= 2

    def test_format_mist_table_contains_key_info(self):
        """format_mist_table output includes drug name, metabolites, and risk."""
        report = run_mist_assessment(
            drug=_PARENT,
            metabolites=[_MET_LOW, _MET_HIGH],
            dose_mg=100.0,
            route="oral",
        )
        table = format_mist_table(report)
        assert isinstance(table, str)
        assert "TestParent" in table
        assert "M1-low" in table
        assert "M2-high" in table
        assert "Overall MIST risk" in table


# ---------------------------------------------------------------------------
# TestMISTAPI — POST /mist endpoint
# ---------------------------------------------------------------------------

_API_DRUG = {
    "name": "APIDrug",
    "mw": 300.0,
    "logP": 2.0,
    "fup": 0.5,
    "rbp": 1.0,
    "drug_type": "neutral",
    "clint_hepatic_L_per_h": 5.0,
    "peff": 1.0,
    "vdss_L_per_kg": 0.6,
}

_API_MET = {
    "name": "ApiMet1",
    "mw": 280.0,
    "fm_parent": 0.30,
    "clint_met_L_per_h": 3.0,
}

_VALID_MIST_PAYLOAD = {
    "drug": _API_DRUG,
    "metabolites": [_API_MET],
    "dose_mg": 100.0,
    "route": "oral",
    "body_weight": 70.0,
}


@pytest.mark.unit
class TestMISTAPI:
    def test_api_mist_200(self):
        r = client.post("/mist", json=_VALID_MIST_PAYLOAD)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["drug_name"] == "APIDrug"
        assert body["parent_auc_mg_h_L"] > 0
        assert body["parent_cmax_mg_L"] > 0
        assert len(body["metabolite_results"]) == 1
        mr = body["metabolite_results"][0]
        assert "classification" in mr
        assert mr["classification"] in ("covered", "disproportionate", "unique_human")
        assert body["overall_mist_risk"] in ("low", "moderate", "high")
        assert isinstance(body["n_flagged"], int)
        assert isinstance(body["recommendations"], list)
        assert isinstance(body["warnings"], list)

    def test_api_mist_422_invalid_route(self):
        bad = {**_VALID_MIST_PAYLOAD, "route": "inhaled"}
        r = client.post("/mist", json=bad)
        assert r.status_code == 422
