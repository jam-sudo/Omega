"""Tests for Phase 29 organ-specific toxicity risk scoring.

Covers:
  - TestOrganExposure (5): extract_organ_exposures and OrganExposure fields
  - TestOrganToxicityScoring (6): individual organ scoring functions
  - TestOrganToxicityReport (1): end-to-end run_organ_toxicity_assessment
  - TestToxicityAPI (2): POST /toxicity endpoint contract
"""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.core.body import SimulationResult
from omega_pbpk.core.organ import Organ
from omega_pbpk.drugs.drug import Drug
from omega_pbpk.risk.organ_toxicity import (
    OrganExposure,
    OrganToxicityReport,
    OrganToxicityScore,
    extract_organ_exposures,
    run_organ_toxicity_assessment,
    score_cardiac_risk,
    score_hepatotoxicity,
    score_nephrotoxicity,
    score_pulmonary_risk,
)

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers — build a lightweight SimulationResult without running the ODE
# ---------------------------------------------------------------------------

_SAFE_DRUG = Drug(
    name="SafeDrugX",
    mw=300.0,
    logP=1.5,
    fup=0.5,
    rbp=1.0,
    dose_mg=50.0,
    route="iv",
)

_HIGH_RISK_DRUG = Drug(
    name="RiskyDrugY",
    mw=300.0,
    logP=4.0,
    fup=0.05,
    rbp=1.0,
    dose_mg=500.0,
    route="iv",
)


def _make_sim_result(drug: Drug, peak_cp: float = 5.0) -> SimulationResult:
    """Build a synthetic SimulationResult with a simple PK curve."""
    time = np.linspace(0.0, 24.0, 241)
    # Simple biexponential: Cp(t) = peak * (exp(-0.1t) - exp(-2t))
    cp = peak_cp * (np.exp(-0.1 * time) - np.exp(-2.0 * time))
    cp = np.maximum(cp, 0.0)

    # amounts[:, 0] = venous blood amount = Cp * V_ven * rbp
    v_ven = 3.5
    amounts = np.zeros((len(time), 35))
    amounts[:, 0] = cp * v_ven * drug.rbp

    # Minimal organs dict with venous_blood for plasma_concentration()
    organs: dict[str, Organ] = {
        "venous_blood": Organ(name="venous_blood", volume_L=v_ven, blood_flow_L_per_h=0.0, kp=1.0),
    }

    return SimulationResult(
        time_h=time,
        amounts=amounts,
        drug=drug,
        organs=organs,
    )


# ---------------------------------------------------------------------------
# TestOrganExposure (5 tests)
# ---------------------------------------------------------------------------


class TestOrganExposure:
    def test_extract_organ_exposures_returns_list(self):
        sim = _make_sim_result(_SAFE_DRUG)
        exposures = extract_organ_exposures(sim, fup=_SAFE_DRUG.fup, mw=_SAFE_DRUG.mw)
        assert isinstance(exposures, list)
        assert len(exposures) > 0
        assert all(isinstance(e, OrganExposure) for e in exposures)

    def test_liver_cmax_positive(self):
        sim = _make_sim_result(_SAFE_DRUG)
        exposures = extract_organ_exposures(sim, fup=_SAFE_DRUG.fup, mw=_SAFE_DRUG.mw)
        liver = [e for e in exposures if e.organ == "liver"]
        assert len(liver) == 1
        assert liver[0].cmax_mg_L > 0

    def test_kidney_exposure_includes_auc(self):
        sim = _make_sim_result(_SAFE_DRUG)
        exposures = extract_organ_exposures(sim, fup=_SAFE_DRUG.fup, mw=_SAFE_DRUG.mw)
        kidney = [e for e in exposures if e.organ == "kidney"]
        assert len(kidney) == 1
        assert kidney[0].auc_mg_h_L > 0

    def test_unbound_concentration_conversion(self):
        sim = _make_sim_result(_SAFE_DRUG, peak_cp=10.0)
        exposures = extract_organ_exposures(sim, fup=_SAFE_DRUG.fup, mw=_SAFE_DRUG.mw)
        liver = [e for e in exposures if e.organ == "liver"][0]
        # cmax_unbound_uM = cmax_mg_L * fup * 1000 / MW
        expected = liver.cmax_mg_L * _SAFE_DRUG.fup * 1000.0 / _SAFE_DRUG.mw
        assert abs(liver.cmax_unbound_uM - expected) < 0.01

    def test_organ_exposure_fields(self):
        sim = _make_sim_result(_SAFE_DRUG)
        exposures = extract_organ_exposures(sim, fup=_SAFE_DRUG.fup, mw=_SAFE_DRUG.mw)
        for exp in exposures:
            assert isinstance(exp.organ, str) and len(exp.organ) > 0
            assert isinstance(exp.cmax_mg_L, float) and exp.cmax_mg_L >= 0
            assert isinstance(exp.auc_mg_h_L, float) and exp.auc_mg_h_L >= 0
            assert isinstance(exp.tmax_h, float) and exp.tmax_h >= 0
            assert isinstance(exp.cmax_unbound_uM, float) and exp.cmax_unbound_uM >= 0
            assert isinstance(exp.cave_mg_L, float) and exp.cave_mg_L >= 0


# ---------------------------------------------------------------------------
# TestOrganToxicityScoring (6 tests)
# ---------------------------------------------------------------------------


class TestOrganToxicityScoring:
    def test_hepatotoxicity_low_risk(self):
        """Low dose + high fup + low logP → low DILI risk."""
        exp = OrganExposure(
            organ="liver",
            cmax_mg_L=1.0,
            auc_mg_h_L=10.0,
            tmax_h=2.0,
            cmax_unbound_uM=5.0,
            cave_mg_L=0.5,
        )
        score = score_hepatotoxicity(exp, dose_mg=50.0, fup=0.5, logP=1.5)
        assert isinstance(score, OrganToxicityScore)
        assert score.risk_level == "low"

    def test_hepatotoxicity_high_risk(self):
        """High dose + low fup triggers high DILI risk."""
        exp = OrganExposure(
            organ="liver",
            cmax_mg_L=10.0,
            auc_mg_h_L=100.0,
            tmax_h=2.0,
            cmax_unbound_uM=5.0,
            cave_mg_L=5.0,
        )
        score = score_hepatotoxicity(exp, dose_mg=500.0, fup=0.05, logP=4.0)
        assert score.risk_level == "high"
        assert len(score.warning) > 0

    def test_nephrotoxicity_scoring(self):
        """Low kidney cmax_unbound → low risk."""
        exp = OrganExposure(
            organ="kidney",
            cmax_mg_L=2.0,
            auc_mg_h_L=20.0,
            tmax_h=3.0,
            cmax_unbound_uM=10.0,
            cave_mg_L=1.0,
        )
        score = score_nephrotoxicity(exp)
        assert score.risk_level == "low"
        assert score.organ == "kidney"

    def test_cardiac_risk_safe_margin(self):
        """hERG margin > 30 → low cardiac risk."""
        exp = OrganExposure(
            organ="heart",
            cmax_mg_L=1.0,
            auc_mg_h_L=10.0,
            tmax_h=2.0,
            cmax_unbound_uM=1.0,
            cave_mg_L=0.5,
        )
        # IC50=100 µM, Cmax_ub=1.0 µM → margin=100× > 30
        score = score_cardiac_risk(exp, herg_ic50_uM=100.0, cmax_plasma_unbound_uM=1.0)
        assert score.risk_level == "low"

    def test_cardiac_risk_narrow_margin(self):
        """hERG margin < 10 → high cardiac risk."""
        exp = OrganExposure(
            organ="heart",
            cmax_mg_L=5.0,
            auc_mg_h_L=50.0,
            tmax_h=2.0,
            cmax_unbound_uM=10.0,
            cave_mg_L=2.5,
        )
        # IC50=5 µM, Cmax_ub=10.0 µM → margin=0.5× < 10
        score = score_cardiac_risk(exp, herg_ic50_uM=5.0, cmax_plasma_unbound_uM=10.0)
        assert score.risk_level == "high"
        assert len(score.warning) > 0

    def test_pulmonary_accumulation(self):
        """High lung/plasma ratio + high logP → high risk."""
        exp = OrganExposure(
            organ="lung",
            cmax_mg_L=60.0,
            auc_mg_h_L=500.0,
            tmax_h=2.0,
            cmax_unbound_uM=20.0,
            cave_mg_L=25.0,
        )
        # ratio = 60 / 5 = 12 > 10, logP=4.0 > 2.0 → high
        score = score_pulmonary_risk(exp, plasma_cmax=5.0, logP=4.0)
        assert score.risk_level == "high"
        assert "phospholipidosis" in score.warning.lower()


# ---------------------------------------------------------------------------
# TestOrganToxicityReport (1 test)
# ---------------------------------------------------------------------------


class TestOrganToxicityReport:
    def test_run_full_assessment(self):
        """End-to-end assessment produces report with all expected fields."""
        sim = _make_sim_result(_SAFE_DRUG, peak_cp=5.0)
        report = run_organ_toxicity_assessment(
            result=sim,
            drug=_SAFE_DRUG,
            dose_mg=50.0,
            herg_ic50_uM=100.0,
        )
        assert isinstance(report, OrganToxicityReport)
        assert report.drug_name == "SafeDrugX"
        assert len(report.organ_exposures) == 4  # liver, kidney, heart, lung
        assert len(report.toxicity_scores) == 4
        assert report.overall_risk in ("low", "moderate", "high")
        assert isinstance(report.risk_count, int) and report.risk_count >= 0
        assert isinstance(report.max_concern_organ, str)
        assert "n_organs_assessed" in report.summary
        assert isinstance(report.warnings, list)


# ---------------------------------------------------------------------------
# TestToxicityAPI (2 tests)
# ---------------------------------------------------------------------------


class TestToxicityAPI:
    def test_toxicity_endpoint_200(self):
        payload = {
            "drug": {
                "name": "TestDrug",
                "mw": 300.0,
                "logP": 2.0,
                "fup": 0.5,
                "rbp": 1.0,
                "clint_hepatic_L_per_h": 5.0,
                "clr_L_per_h": 0.0,
                "peff": 1.0,
            },
            "dose_mg": 100.0,
            "route": "iv",
            "duration_h": 24.0,
            "herg_ic50_uM": 50.0,
        }
        r = client.post("/toxicity", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert "drug_name" in body
        assert "organ_exposures" in body
        assert "toxicity_scores" in body
        assert "overall_risk" in body
        assert body["overall_risk"] in ("low", "moderate", "high")
        assert isinstance(body["organ_exposures"], list)
        assert isinstance(body["toxicity_scores"], list)
        assert len(body["organ_exposures"]) > 0
        assert len(body["toxicity_scores"]) > 0

    def test_toxicity_endpoint_422(self):
        # Invalid route should trigger 422
        payload = {
            "dose_mg": 100.0,
            "route": "nasal",
        }
        r = client.post("/toxicity", json=payload)
        assert r.status_code == 422
