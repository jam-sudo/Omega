"""Tests for Phase 33 Virtual Crossover Trial Simulation.

Covers:
  - TestCrossoverUnit (10): core crossover trial simulation behaviour
  - TestCrossoverStatistics (2): statistical estimation checks
  - TestCrossoverAPI (2): POST /trial/crossover endpoint contract
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.clinical.crossover_trial import (
    CrossoverArm,
    CrossoverDesign,
    CrossoverTrialResult,
    run_crossover_trial,
)
from omega_pbpk.drugs.drug import Drug

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DRUG_A = Drug(
    name="DrugA",
    mw=300.0,
    logP=2.0,
    fup=0.5,
    rbp=1.0,
    drug_type="neutral",
    clint_hepatic_L_per_h=5.0,
    peff=1.0,
)

_DRUG_B = Drug(
    name="DrugB",
    mw=300.0,
    logP=2.0,
    fup=0.5,
    rbp=1.0,
    drug_type="neutral",
    clint_hepatic_L_per_h=5.0,
    peff=1.0,
)


def _make_design(
    drug_test: Drug = _DRUG_A,
    drug_ref: Drug = _DRUG_B,
    n_subjects: int = 12,
    **kwargs,
) -> CrossoverDesign:
    """Build a CrossoverDesign with sensible defaults."""
    arms = (
        CrossoverArm(name="Test", drug=drug_test, dose_mg=100.0),
        CrossoverArm(name="Reference", drug=drug_ref, dose_mg=100.0),
    )
    defaults = dict(
        arms=arms,
        design="2x2",
        n_subjects=n_subjects,
        washout_sufficient=True,
        wsv_cv_cl=0.15,
        wsv_cv_vd=0.10,
        wsv_cv_ka=0.20,
        period_effect=0.0,
        seed=42,
    )
    defaults.update(kwargs)
    return CrossoverDesign(**defaults)


def _run(**kwargs) -> CrossoverTrialResult:
    return run_crossover_trial(_make_design(**kwargs))


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCrossoverUnit:
    def test_run_crossover_returns_result(self):
        result = _run()
        assert isinstance(result, CrossoverTrialResult)

    def test_identical_formulations_be_passes(self):
        result = _run(drug_test=_DRUG_A, drug_ref=_DRUG_A)
        assert result.be_conclusion_overall == "BE"

    def test_gmr_near_unity_for_identical(self):
        result = _run(drug_test=_DRUG_A, drug_ref=_DRUG_A)
        assert 0.85 <= result.gmr_auc <= 1.15
        assert 0.85 <= result.gmr_cmax <= 1.15

    def test_different_clearance_shifts_gmr(self):
        drug_fast = Drug(
            name="FastCL",
            mw=300.0,
            logP=2.0,
            fup=0.5,
            rbp=1.0,
            drug_type="neutral",
            clint_hepatic_L_per_h=10.0,
            peff=1.0,
        )
        result = _run(drug_test=drug_fast, drug_ref=_DRUG_A, n_subjects=12)
        assert result.gmr_auc < 0.80
        assert result.be_conclusion_overall == "NOT BE"

    def test_sequence_assignment_balanced(self):
        result = _run(n_subjects=24)
        seq_counts = {}
        for sr in result.subject_results:
            if sr.period == 1:
                seq_counts[sr.sequence] = seq_counts.get(sr.sequence, 0) + 1
        assert seq_counts.get(0, 0) == 12
        assert seq_counts.get(1, 0) == 12

    def test_wsv_increases_cv(self):
        low = _run(wsv_cv_cl=0.05, wsv_cv_vd=0.05, wsv_cv_ka=0.05, seed=99)
        high = _run(wsv_cv_cl=0.40, wsv_cv_vd=0.30, wsv_cv_ka=0.50, seed=99)
        assert high.cv_within_auc > low.cv_within_auc

    def test_period_effect_in_anova(self):
        result = _run(period_effect=0.20, n_subjects=12)
        has_nonzero = (
            result.anova_summary["mean_diff_auc"] != 0.0
            or result.anova_summary["mean_diff_cmax"] != 0.0
        )
        assert has_nonzero

    def test_tost_ci_valid(self):
        result = _run()
        lo, hi = result.ci90_auc
        assert lo > 0
        assert hi > 0
        assert lo < result.gmr_auc < hi
        lo_c, hi_c = result.ci90_cmax
        assert lo_c > 0
        assert hi_c > 0
        assert lo_c < result.gmr_cmax < hi_c

    def test_nca_parameters_positive(self):
        result = _run()
        for sr in result.subject_results:
            assert sr.cmax_mg_L > 0
            assert sr.auc0inf_mg_h_L > 0
            assert sr.auc0t_mg_h_L > 0

    def test_n_completed_valid(self):
        result = _run(n_subjects=8)
        assert 0 < result.n_completed <= result.n_subjects


# ---------------------------------------------------------------------------
# Statistics tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCrossoverStatistics:
    def test_within_subject_cv_estimation(self):
        result = _run(wsv_cv_cl=0.15, n_subjects=12)
        assert result.cv_within_auc >= 0.0
        assert result.cv_within_cmax >= 0.0

    def test_power_between_0_and_1(self):
        result = _run()
        assert 0.0 <= result.power_auc <= 1.0
        assert 0.0 <= result.power_cmax <= 1.0


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

_API_DRUG = {
    "name": "TestDrug",
    "mw": 300.0,
    "logP": 2.0,
    "fup": 0.5,
    "rbp": 1.0,
    "drug_type": "neutral",
    "clint_hepatic_L_per_h": 5.0,
    "peff": 1.0,
}

_VALID_PAYLOAD = {
    "arms": [
        {"name": "Test", "drug": _API_DRUG, "dose_mg": 100.0, "route": "oral"},
        {"name": "Reference", "drug": _API_DRUG, "dose_mg": 100.0, "route": "oral"},
    ],
    "design": "2x2",
    "n_subjects": 12,
    "seed": 42,
}


@pytest.mark.unit
class TestCrossoverAPI:
    def test_crossover_endpoint_200(self):
        r = client.post("/trial/crossover", json=_VALID_PAYLOAD)
        assert r.status_code == 200, r.text
        body = r.json()
        assert "gmr_auc" in body
        assert "gmr_cmax" in body
        assert "be_conclusion_overall" in body
        assert body["n_subjects"] == 12
        assert body["n_completed"] > 0

    def test_crossover_endpoint_422(self):
        invalid = {"arms": [{"name": "OnlyOne", "drug": _API_DRUG}]}
        r = client.post("/trial/crossover", json=invalid)
        assert r.status_code == 422
