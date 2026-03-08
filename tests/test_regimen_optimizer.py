"""Tests for Phase 26 Dosing Regimen Optimizer.

Covers:
  - TestRegimenUnit (13): optimize_regimen core behavior and result validation
  - TestRegimenAPI (1): POST /dose/optimize endpoint contract
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow
from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.clinical.regimen_optimizer import (
    RegimenResult,
    TherapeuticTarget,
    optimize_regimen,
)
from omega_pbpk.contracts.drug_spec import DrugSpec

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SPEC = DrugSpec(
    name="test_drug",
    mw=300.0,
    logP=2.0,
    fup=0.5,
    rbp=1.0,
    clint_hepatic_L_per_h=10.0,
    peff=1.0,
)

WIDE_TARGET = TherapeuticTarget(cmin_mg_L=0.001, cmax_mg_L=1000.0)
IMPOSSIBLE_TARGET = TherapeuticTarget(cmin_mg_L=999.0, cmax_mg_L=999.1)


def _run_optimize(**kwargs):
    """Helper to run optimize_regimen with defaults."""
    defaults = dict(
        spec=SPEC,
        target=WIDE_TARGET,
        dose_range_mg=(10.0, 500.0),
        intervals_h=(8, 12, 24),
        route="oral",
        n_doses=10,
        body_weight=70.0,
    )
    defaults.update(kwargs)
    return optimize_regimen(**defaults)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegimenUnit:
    def test_optimize_returns_result(self):
        result = _run_optimize()
        assert isinstance(result, RegimenResult)

    def test_result_has_required_fields(self):
        result = _run_optimize()
        for attr in (
            "optimal_dose_mg",
            "optimal_interval_h",
            "css_max",
            "css_min",
            "css_avg",
            "auc_ss",
            "feasible",
            "warnings",
            "all_regimens",
        ):
            assert hasattr(result, attr), f"Missing field: {attr}"

    def test_optimal_dose_in_range(self):
        lo, hi = 50.0, 300.0
        result = _run_optimize(dose_range_mg=(lo, hi))
        assert lo <= result.optimal_dose_mg <= hi

    def test_optimal_interval_in_candidates(self):
        intervals = (8, 12, 24)
        result = _run_optimize(intervals_h=intervals)
        assert result.optimal_interval_h in intervals

    def test_css_max_positive(self):
        result = _run_optimize()
        assert result.css_max > 0

    def test_css_min_non_negative(self):
        result = _run_optimize()
        assert result.css_min >= 0

    def test_auc_ss_positive(self):
        result = _run_optimize()
        assert result.auc_ss > 0

    def test_feasible_with_wide_target(self):
        result = _run_optimize(target=WIDE_TARGET)
        assert result.feasible is True

    def test_infeasible_with_impossible_target(self):
        result = _run_optimize(target=IMPOSSIBLE_TARGET)
        assert result.feasible is False

    def test_all_regimens_top3(self):
        result = _run_optimize()
        assert len(result.all_regimens) <= 3
        penalties = [r["penalty"] for r in result.all_regimens]
        assert penalties == sorted(penalties)

    def test_custom_intervals(self):
        intervals = (4, 6)
        result = _run_optimize(intervals_h=intervals)
        assert result.optimal_interval_h in intervals

    def test_custom_dose_range(self):
        lo, hi = 100.0, 200.0
        result = _run_optimize(dose_range_mg=(lo, hi))
        assert lo <= result.optimal_dose_mg <= hi

    def test_iv_route(self):
        result = _run_optimize(route="iv")
        assert isinstance(result, RegimenResult)
        assert result.css_max > 0


# ---------------------------------------------------------------------------
# API endpoint test
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegimenAPI:
    def test_api_endpoint(self):
        payload = {
            "drug": {
                "name": "test_drug",
                "mw": 300.0,
                "logP": 2.0,
                "fup": 0.5,
                "rbp": 1.0,
                "clint_hepatic_L_per_h": 10.0,
                "peff": 1.0,
            },
            "cmin_mg_L": 0.001,
            "cmax_mg_L": 1000.0,
            "dose_range_mg": [10.0, 500.0],
            "intervals_h": [8, 12, 24],
            "route": "oral",
            "n_doses": 10,
            "body_weight": 70.0,
        }
        r = client.post("/dose/optimize", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert "optimal_dose_mg" in body
        assert "optimal_interval_h" in body
        assert "feasible" in body
        assert "all_regimens" in body
        assert isinstance(body["all_regimens"], list)
