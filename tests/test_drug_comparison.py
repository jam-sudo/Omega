"""Tests for Phase 34 Multi-candidate Drug Comparison Tool.

Covers:
  - TestComparisonUnit (6): core comparison logic & validation
  - TestComparisonRanking (6): ranking, scoring, and formatting
  - TestComparisonAPI (2): POST /compare endpoint contract
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.clinical.drug_comparison import (
    ComparisonCandidate,
    DrugComparisonResult,
    compare_drugs,
    format_comparison_table,
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
    mw=350.0,
    logP=2.5,
    fup=0.4,
    rbp=1.0,
    drug_type="neutral",
    clint_hepatic_L_per_h=8.0,
    peff=1.2,
)

_DRUG_RISKY = Drug(
    name="RiskyDrug",
    mw=500.0,
    logP=5.0,
    fup=0.05,
    rbp=1.0,
    drug_type="neutral",
    clint_hepatic_L_per_h=30.0,
    peff=0.5,
)


def _cand(drug: Drug, **kwargs) -> ComparisonCandidate:
    defaults = dict(dose_mg=100.0, route="oral")
    defaults.update(kwargs)
    return ComparisonCandidate(drug=drug, **defaults)


def _two() -> list[ComparisonCandidate]:
    return [_cand(_DRUG_A), _cand(_DRUG_B)]


# ---------------------------------------------------------------------------
# TestComparisonUnit
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComparisonUnit:
    def test_compare_two_drugs_returns_result(self):
        result = compare_drugs(_two())
        assert isinstance(result, DrugComparisonResult)
        assert result.n_candidates == 2

    def test_compare_five_drugs_max(self):
        drugs = [
            Drug(
                name=f"D{i}",
                mw=250 + i * 50,
                logP=1.5 + i * 0.3,
                fup=0.5,
                rbp=1.0,
                drug_type="neutral",
                clint_hepatic_L_per_h=5.0 + i,
                peff=1.0,
            )
            for i in range(5)
        ]
        candidates = [_cand(d) for d in drugs]
        result = compare_drugs(candidates)
        assert result.n_candidates == 5
        assert len(result.overall_ranking) == 5

    def test_compare_one_drug_raises(self):
        with pytest.raises(ValueError, match="2.*5"):
            compare_drugs([_cand(_DRUG_A)])

    def test_compare_six_drugs_raises(self):
        drugs = [
            Drug(
                name=f"X{i}",
                mw=300.0,
                logP=2.0,
                fup=0.5,
                rbp=1.0,
                drug_type="neutral",
                clint_hepatic_L_per_h=5.0,
                peff=1.0,
            )
            for i in range(6)
        ]
        with pytest.raises(ValueError, match="2.*5"):
            compare_drugs([_cand(d) for d in drugs])

    def test_pk_profiles_populated(self):
        result = compare_drugs(_two())
        for pk in result.pk_profiles:
            assert pk.cmax_mg_L > 0
            assert pk.auc0inf_mg_h_L > 0
            assert pk.t_half_h > 0

    def test_safety_profiles_populated(self):
        result = compare_drugs(_two())
        valid_levels = {"low", "moderate", "high"}
        for sp in result.safety_profiles:
            assert sp.overall_tox_risk in valid_levels
            assert sp.hepatotox_risk in valid_levels
            assert sp.nephrotox_risk in valid_levels
            assert sp.cardiac_risk in valid_levels
            assert sp.pulmonary_risk in valid_levels


# ---------------------------------------------------------------------------
# TestComparisonRanking
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComparisonRanking:
    def test_high_risk_drug_ranked_lower(self):
        safe = _cand(_DRUG_A)
        risky = _cand(_DRUG_RISKY, dose_mg=200.0)
        result = compare_drugs([safe, risky])
        assert result.overall_ranking[0] == safe.drug.name
        assert result.composite_scores[safe.drug.name] > result.composite_scores[risky.drug.name]

    def test_dimension_ranking_auc(self):
        result = compare_drugs(_two())
        auc_dim = next(d for d in result.dimension_rankings if d.dimension == "total_exposure")
        # The drug with higher AUC should be ranked first
        first = auc_dim.ranked_labels[0]
        assert auc_dim.values[first] >= auc_dim.values[auc_dim.ranked_labels[-1]]

    def test_dimension_ranking_tox(self):
        safe = _cand(_DRUG_A)
        risky = _cand(_DRUG_RISKY, dose_mg=200.0)
        result = compare_drugs([safe, risky])
        safety_dim = next(d for d in result.dimension_rankings if d.dimension == "safety")
        # Lower tox_risk_count is better; safe drug should rank first
        assert safety_dim.ranked_labels[0] == safe.drug.name

    def test_composite_score_range(self):
        result = compare_drugs(_two())
        for label, score in result.composite_scores.items():
            assert 0 <= score <= 100, f"{label} score {score} outside [0, 100]"

    def test_custom_weights(self):
        default_result = compare_drugs(_two())
        custom_result = compare_drugs(
            _two(),
            weights={
                "safety": 0.90,
                "ddi": 0.02,
                "auc_adequacy": 0.02,
                "pk_quality": 0.02,
                "clearance": 0.04,
            },
        )
        # With drastically different weights, at least one score should differ
        default_scores = list(default_result.composite_scores.values())
        custom_scores = list(custom_result.composite_scores.values())
        assert default_scores != custom_scores

    def test_format_comparison_table(self):
        result = compare_drugs(_two())
        table = format_comparison_table(result)
        assert isinstance(table, str)
        assert len(table) > 0
        # All candidate labels should appear in the table
        for label in result.candidate_labels:
            assert label in table


# ---------------------------------------------------------------------------
# TestComparisonAPI
# ---------------------------------------------------------------------------

_API_DRUG_A = {
    "name": "ApiDrugA",
    "mw": 300.0,
    "logP": 2.0,
    "fup": 0.5,
    "rbp": 1.0,
    "drug_type": "neutral",
    "clint_hepatic_L_per_h": 5.0,
    "peff": 1.0,
}

_API_DRUG_B = {
    "name": "ApiDrugB",
    "mw": 350.0,
    "logP": 2.5,
    "fup": 0.4,
    "rbp": 1.0,
    "drug_type": "neutral",
    "clint_hepatic_L_per_h": 8.0,
    "peff": 1.0,
}

_VALID_COMPARE_PAYLOAD = {
    "candidates": [
        {"drug": _API_DRUG_A, "dose_mg": 100.0, "route": "oral"},
        {"drug": _API_DRUG_B, "dose_mg": 100.0, "route": "oral"},
    ],
    "t_end_h": 24.0,
}


@pytest.mark.unit
class TestComparisonAPI:
    def test_api_compare_200(self):
        r = client.post("/compare", json=_VALID_COMPARE_PAYLOAD)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_candidates"] == 2
        assert len(body["overall_ranking"]) == 2
        assert body["best_candidate"] in body["candidate_labels"]
        assert "composite_scores" in body
        assert "pk_profiles" in body
        assert "safety_profiles" in body

    def test_api_compare_422(self):
        invalid = {
            "candidates": [
                {"drug": _API_DRUG_A, "dose_mg": 100.0, "route": "oral"},
            ],
        }
        r = client.post("/compare", json=invalid)
        assert r.status_code == 422
