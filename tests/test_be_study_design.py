"""Tests for Phase 28 BE Study Design — sample size, power, and sensitivity.

Covers:
  - TestBEDesignUnit (12): core sample-size, power, sensitivity behaviour
  - TestBEDesignAPI (2): POST /be/design endpoint contract
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.clinical.be_study_design import (
    BEPowerResult,
    BEStudyDesign,
    SensitivityResult,
    compute_power,
    run_be_design,
    sensitivity_analysis,
)

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run(**kwargs) -> BEStudyDesign:
    """Run BE design with sensible defaults."""
    defaults = dict(
        cv_auc=0.20,
        cv_cmax=0.25,
        gmr=0.95,
        target_power=0.80,
        design="2x2",
        dropout_rate=0.10,
        is_nti=False,
    )
    defaults.update(kwargs)
    return run_be_design(**defaults)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBEDesignUnit:
    def test_run_be_design_returns_result(self):
        result = _run()
        assert isinstance(result, BEStudyDesign)

    def test_result_has_required_fields(self):
        result = _run()
        for attr in (
            "design",
            "n_subjects",
            "n_per_sequence",
            "power_achieved",
            "alpha",
            "be_limits",
            "gmr_assumed",
            "cv_within",
            "cv_between",
            "endpoint",
            "dropout_adjusted_n",
            "dropout_rate",
            "power_curve",
            "warnings",
        ):
            assert hasattr(result, attr), f"Missing field: {attr}"

    def test_sample_size_increases_with_cv(self):
        low = _run(cv_auc=0.15, cv_cmax=0.15)
        high = _run(cv_auc=0.40, cv_cmax=0.40)
        assert high.n_subjects > low.n_subjects

    def test_sample_size_driven_by_cmax(self):
        # When Cmax CV >> AUC CV, the driving CV should be Cmax
        result = _run(cv_auc=0.10, cv_cmax=0.40)
        assert result.cv_within >= 0.40

    def test_nti_requires_more_subjects(self):
        normal = _run(cv_auc=0.20, cv_cmax=0.20, is_nti=False)
        nti = _run(cv_auc=0.20, cv_cmax=0.20, is_nti=True)
        assert nti.n_subjects > normal.n_subjects

    def test_power_at_computed_n_meets_target(self):
        result = _run()
        assert result.power_achieved >= 0.80

    def test_dropout_inflates_sample_size(self):
        result = _run(dropout_rate=0.15)
        assert result.dropout_adjusted_n >= result.n_subjects

    def test_compute_power_returns_result(self):
        result = compute_power(n_subjects=36, cv_auc=0.25, cv_cmax=0.30)
        assert isinstance(result, BEPowerResult)
        assert 0.0 <= result.power_auc <= 1.0
        assert 0.0 <= result.power_cmax <= 1.0
        assert 0.0 <= result.power_joint <= 1.0

    def test_power_increases_with_n(self):
        low_n = compute_power(n_subjects=12, cv_auc=0.25, cv_cmax=0.30)
        high_n = compute_power(n_subjects=48, cv_auc=0.25, cv_cmax=0.30)
        assert high_n.power_joint > low_n.power_joint

    def test_sensitivity_analysis_returns_matrix(self):
        n_cv, n_gmr = 5, 4
        result = sensitivity_analysis(
            cv_range=(0.15, 0.35),
            gmr_range=(0.90, 1.00),
            n_cv_steps=n_cv,
            n_gmr_steps=n_gmr,
        )
        assert isinstance(result, SensitivityResult)
        assert len(result.cv_values) == n_cv
        assert len(result.gmr_values) == n_gmr
        assert len(result.sample_size_matrix) == n_cv
        assert all(len(row) == n_gmr for row in result.sample_size_matrix)

    def test_known_sample_size_cv25(self):
        # CV=25%, GMR=0.95 → typically 20-40 subjects (published range)
        result = _run(cv_auc=0.25, cv_cmax=0.25, gmr=0.95)
        assert 20 <= result.n_subjects <= 40

    def test_power_curve_in_result(self):
        result = _run()
        assert len(result.power_curve) > 0
        for entry in result.power_curve:
            assert "n" in entry
            assert "power" in entry
            assert 0.0 <= entry["power"] <= 1.0


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBEDesignAPI:
    def test_be_design_endpoint_200(self):
        payload = {
            "cv_auc": 0.25,
            "cv_cmax": 0.30,
            "gmr": 0.95,
            "target_power": 0.80,
            "design": "2x2",
            "dropout_rate": 0.10,
            "is_nti": False,
        }
        r = client.post("/be/design", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert "n_subjects" in body
        assert "power_achieved" in body
        assert "power_curve" in body
        assert body["n_subjects"] > 0
        assert 0.0 <= body["power_achieved"] <= 1.0
        assert isinstance(body["power_curve"], list)

    def test_be_design_endpoint_422_invalid_cv(self):
        # CV must be > 0; negative should trigger 422
        payload = {
            "cv_auc": -0.10,
            "cv_cmax": 0.30,
        }
        r = client.post("/be/design", json=payload)
        assert r.status_code == 422
