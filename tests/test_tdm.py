"""Tests for Phase 27 TDM Bayesian dose individualization.

Covers:
  - TestTDMUnit (12): run_tdm core behavior, parameter estimation, dose recommendation
  - TestTDMAPI (2): POST /tdm endpoint contract
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.clinical.tdm import (
    ObservedConcentration,
    PopulationPrior,
    TDMResult,
    run_tdm,
)

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PRIOR = PopulationPrior(cl_mean=5.0, cl_cv=0.40, vd_mean=42.0, vd_cv=0.30, ka_mean=1.0, ka_cv=0.50)

SINGLE_OBS = [
    ObservedConcentration(time_h=12.0, conc_mg_L=2.0, dose_number=5),
]

MULTI_OBS = [
    ObservedConcentration(time_h=2.0, conc_mg_L=8.0, dose_number=5),  # peak-ish
    ObservedConcentration(time_h=12.0, conc_mg_L=2.0, dose_number=5),  # trough
]


def _run(**kwargs):
    """Helper to run_tdm with sensible defaults."""
    defaults = dict(
        observations=MULTI_OBS,
        prior=PRIOR,
        dose_mg=100.0,
        interval_h=12.0,
        route="oral",
        target_trough=2.0,
        dose_range_mg=(10.0, 1000.0),
        forward_hours=48.0,
    )
    defaults.update(kwargs)
    return run_tdm(**defaults)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTDMUnit:
    def test_run_tdm_returns_result(self):
        result = _run()
        assert isinstance(result, TDMResult)

    def test_result_has_required_fields(self):
        result = _run()
        for attr in (
            "individual_cl",
            "individual_vd",
            "individual_ka",
            "cl_ratio",
            "vd_ratio",
            "ka_ratio",
            "predicted",
            "observed",
            "residuals",
            "objective_function_value",
            "recommended_dose_mg",
            "predicted_trough",
            "predicted_peak",
            "forward_time_h",
            "forward_conc_mg_L",
            "converged",
            "warnings",
        ):
            assert hasattr(result, attr), f"Missing field: {attr}"

    def test_single_observation_converges(self):
        result = _run(observations=SINGLE_OBS)
        assert result.converged is True

    def test_multiple_observations_converges(self):
        result = _run(observations=MULTI_OBS)
        assert result.converged is True

    def test_individual_cl_positive(self):
        result = _run()
        assert 0.1 <= result.individual_cl <= 200.0

    def test_individual_vd_positive(self):
        result = _run()
        assert 1.0 <= result.individual_vd <= 500.0

    def test_recommended_dose_in_range(self):
        lo, hi = 10.0, 1000.0
        result = _run(dose_range_mg=(lo, hi))
        assert lo <= result.recommended_dose_mg <= hi

    def test_predicted_trough_near_target(self):
        target = 2.0
        result = _run(target_trough=target)
        assert result.predicted_trough > target * 0.5
        assert result.predicted_trough < target * 1.5

    def test_high_observed_level_increases_cl_ratio(self):
        # High observed concentration → lower individual CL relative to pop
        high_obs = [
            ObservedConcentration(time_h=12.0, conc_mg_L=20.0, dose_number=5),
        ]
        result = _run(observations=high_obs)
        assert result.cl_ratio < 1.0

    def test_low_observed_level_decreases_cl_ratio(self):
        # Low observed concentration → higher individual CL relative to pop
        low_obs = [
            ObservedConcentration(time_h=12.0, conc_mg_L=0.1, dose_number=5),
        ]
        result = _run(observations=low_obs)
        assert result.cl_ratio > 1.0

    def test_residuals_computed(self):
        result = _run(observations=MULTI_OBS)
        assert len(result.residuals) == len(MULTI_OBS)

    def test_forward_prediction_length(self):
        result = _run()
        assert len(result.forward_time_h) > 0
        assert len(result.forward_conc_mg_L) > 0
        assert len(result.forward_time_h) == len(result.forward_conc_mg_L)


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTDMAPI:
    def test_tdm_endpoint_200(self):
        payload = {
            "observations": [
                {"time_h": 2.0, "conc_mg_L": 8.0, "dose_number": 5},
                {"time_h": 12.0, "conc_mg_L": 2.0, "dose_number": 5},
            ],
            "dose_mg": 100.0,
            "interval_h": 12.0,
            "route": "oral",
            "target_trough": 2.0,
            "dose_range_mg": [10.0, 1000.0],
            "forward_hours": 48.0,
        }
        r = client.post("/tdm", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert "individual_cl" in body
        assert "recommended_dose_mg" in body
        assert "converged" in body
        assert "forward_time_h" in body
        assert isinstance(body["forward_time_h"], list)
        assert isinstance(body["forward_conc_mg_L"], list)

    def test_tdm_endpoint_422_invalid(self):
        # Missing required 'observations' field
        r = client.post("/tdm", json={})
        assert r.status_code == 422
