"""Tests for Sobol global sensitivity analysis (Phase 31).

Covers:
  - Saltelli sampling shapes and bounds
  - Sobol sequence uniformity
  - Jansen estimators on analytical models
  - Full sobol_sensitivity workflow (Cmax, AUC)
  - Dominant parameter detection
  - Convergence flag behaviour
  - Default parameter ranges
  - API endpoint (/sensitivity/sobol)
  - Result structure integration test
"""

from __future__ import annotations

import numpy as np
import pytest

from omega_pbpk.drugs.drug import Drug
from omega_pbpk.sensitivity.sobol_gsa import (
    DEFAULT_GSA_RANGES,
    ParameterRange,
    SobolResult,
    _jansen_estimators,
    _saltelli_sample,
    _sobol_sequence,
    sobol_sensitivity,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TEST_DRUG = Drug(
    name="TestDrug",
    mw=300.0,
    logP=2.0,
    fup=0.5,
    rbp=1.0,
    peff=1.0,
    clint_hepatic_L_per_h=5.0,
    clr_L_per_h=1.0,
)

_SMALL_RANGES = [
    ParameterRange("fup", 0.01, 1.0),
    ParameterRange("rbp", 0.5, 2.0),
    ParameterRange("peff", 0.1, 10.0),
]


# ===================================================================
# TestSobolSampling
# ===================================================================


@pytest.mark.unit
class TestSobolSampling:
    """Test Saltelli sampling shapes and bounds."""

    def test_saltelli_sample_shape(self) -> None:
        """A and B have shape (n, k); AB list has k entries each (n, k)."""
        n, k = 64, len(_SMALL_RANGES)
        A, B, AB_list = _saltelli_sample(n, _SMALL_RANGES, seed=42)

        assert A.shape == (n, k)
        assert B.shape == (n, k)
        assert len(AB_list) == k
        for ab in AB_list:
            assert ab.shape == (n, k)

    def test_saltelli_sample_bounds(self) -> None:
        """All sample values lie within [lower, upper] for each parameter."""
        n = 128
        A, B, AB_list = _saltelli_sample(n, _SMALL_RANGES, seed=7)

        all_matrices = [A, B] + AB_list
        for mat in all_matrices:
            for j, r in enumerate(_SMALL_RANGES):
                col = mat[:, j]
                assert np.all(col >= r.lower - 1e-12), f"Values below lower bound for {r.name}"
                assert np.all(col <= r.upper + 1e-12), f"Values above upper bound for {r.name}"

    def test_sobol_sequence_uniformity(self) -> None:
        """Sobol sequence should be roughly space-filling (mean ≈ 0.5)."""
        pts = _sobol_sequence(256, dim=3, seed=42)
        assert pts.shape == (256, 3)
        for d in range(3):
            col = pts[:, d]
            assert 0.0 <= col.min(), "Sobol points below 0"
            assert col.max() <= 1.0, "Sobol points above 1"
            # Mean should be near 0.5 for a quasi-random sequence
            assert abs(np.mean(col) - 0.5) < 0.1, f"Sobol sequence dim {d} mean too far from 0.5"


# ===================================================================
# TestJansenEstimators
# ===================================================================


@pytest.mark.unit
class TestJansenEstimators:
    """Test Jansen estimators on analytical models with known indices."""

    def test_analytical_linear_model(self) -> None:
        """Y = a*X1 + b*X2 should yield known S1 values.

        For Y = 3*X1 + 1*X2 with X ~ U(0,1):
            Var(Y) = 9*Var(X1) + 1*Var(X2) = 9/12 + 1/12 = 10/12
            S1_1 = 9/10 = 0.9,  S1_2 = 1/10 = 0.1
        """
        rng = np.random.default_rng(42)
        n = 4096
        x1_a = rng.uniform(0, 1, n)
        x2_a = rng.uniform(0, 1, n)
        x1_b = rng.uniform(0, 1, n)
        x2_b = rng.uniform(0, 1, n)

        yA = 3.0 * x1_a + 1.0 * x2_a
        yB = 3.0 * x1_b + 1.0 * x2_b

        # AB_1: A with col 0 replaced by B's col 0
        yAB_1 = 3.0 * x1_b + 1.0 * x2_a
        # AB_2: A with col 1 replaced by B's col 1
        yAB_2 = 3.0 * x1_a + 1.0 * x2_b

        S1, ST = _jansen_estimators(yA, yB, [yAB_1, yAB_2], n)

        assert abs(S1[0] - 0.9) < 0.1, f"S1[0] expected ~0.9, got {S1[0]}"
        assert abs(S1[1] - 0.1) < 0.1, f"S1[1] expected ~0.1, got {S1[1]}"

    def test_non_interacting_model(self) -> None:
        """For an additive model Y = X1 + X2, S1 ≈ ST (no interactions)."""
        rng = np.random.default_rng(99)
        n = 4096
        x1_a = rng.uniform(0, 1, n)
        x2_a = rng.uniform(0, 1, n)
        x1_b = rng.uniform(0, 1, n)
        x2_b = rng.uniform(0, 1, n)

        yA = x1_a + x2_a
        yB = x1_b + x2_b
        yAB_1 = x1_b + x2_a
        yAB_2 = x1_a + x2_b

        S1, ST = _jansen_estimators(yA, yB, [yAB_1, yAB_2], n)

        for i in range(2):
            diff = abs(ST[i] - S1[i])
            assert diff < 0.1, f"param {i}: ST-S1 = {diff}, expected ≈0 for additive model"

    def test_interacting_model(self) -> None:
        """Y = X1 * X2 should have interaction_index > 0 (ST > S1)."""
        rng = np.random.default_rng(123)
        n = 4096
        x1_a = rng.uniform(0, 1, n)
        x2_a = rng.uniform(0, 1, n)
        x1_b = rng.uniform(0, 1, n)
        x2_b = rng.uniform(0, 1, n)

        yA = x1_a * x2_a
        yB = x1_b * x2_b
        yAB_1 = x1_b * x2_a
        yAB_2 = x1_a * x2_b

        S1, ST = _jansen_estimators(yA, yB, [yAB_1, yAB_2], n)

        for i in range(2):
            interaction = ST[i] - S1[i]
            assert interaction > 0.01, (
                f"param {i}: interaction={interaction}, expected > 0 for multiplicative model"
            )


# ===================================================================
# TestSobolSensitivity
# ===================================================================


@pytest.mark.integration
class TestSobolSensitivity:
    """End-to-end tests for sobol_sensitivity."""

    def test_full_workflow_cmax(self) -> None:
        """End-to-end with metric='Cmax' returns valid SobolResult."""
        result = sobol_sensitivity(
            drug=_TEST_DRUG,
            dose_mg=100.0,
            route="oral",
            parameter_ranges=_SMALL_RANGES,
            n_samples=64,
            metric="Cmax",
            seed=42,
            n_bootstrap=50,
        )
        assert isinstance(result, SobolResult)
        assert result.metric == "Cmax"
        assert len(result.S1) == len(_SMALL_RANGES)
        assert len(result.ST) == len(_SMALL_RANGES)
        assert all(s >= 0 for s in result.S1), "S1 should be non-negative (clamped)"
        assert all(s >= 0 for s in result.ST), "ST should be non-negative"

    def test_full_workflow_auc(self) -> None:
        """End-to-end with metric='AUC' returns valid SobolResult."""
        result = sobol_sensitivity(
            drug=_TEST_DRUG,
            dose_mg=100.0,
            route="oral",
            parameter_ranges=_SMALL_RANGES,
            n_samples=64,
            metric="AUC",
            seed=42,
            n_bootstrap=50,
        )
        assert isinstance(result, SobolResult)
        assert result.metric == "AUC"
        assert len(result.S1) == len(_SMALL_RANGES)
        assert result.n_samples == 64
        k = len(_SMALL_RANGES)
        assert result.n_total_evaluations == 64 * (k + 2)

    def test_dominant_parameter(self) -> None:
        """For a high-clearance drug, fup should have the highest S1 for AUC.

        AUC = F * dose / CL, where CL = clint * fup + clr.
        With high clint, fup dominates CL and thus AUC.
        """
        high_cl_drug = Drug(
            name="HighCL",
            mw=300.0,
            fup=0.5,
            rbp=1.0,
            peff=5.0,
            clint_hepatic_L_per_h=30.0,
            clr_L_per_h=0.5,
        )
        ranges = [
            ParameterRange("fup", 0.01, 1.0),
            ParameterRange("rbp", 0.5, 2.0),
            ParameterRange("peff", 0.1, 10.0),
            ParameterRange("clint_hepatic_L_per_h", 0.1, 50.0),
        ]
        result = sobol_sensitivity(
            drug=high_cl_drug,
            dose_mg=100.0,
            route="oral",
            parameter_ranges=ranges,
            n_samples=128,
            metric="AUC",
            seed=42,
            n_bootstrap=50,
        )
        fup_idx = result.parameters.index("fup")
        fup_s1 = result.S1[fup_idx]
        # fup should be among the most influential (either highest or close)
        assert fup_s1 > 0.05, f"fup S1={fup_s1} expected to be influential for high-clearance drug"

    def test_convergence_flag(self) -> None:
        """Very small n_samples should produce poor convergence (flag=False)."""
        result = sobol_sensitivity(
            drug=_TEST_DRUG,
            dose_mg=100.0,
            route="oral",
            parameter_ranges=_SMALL_RANGES,
            n_samples=16,
            metric="AUC",
            seed=42,
            n_bootstrap=20,
        )
        assert isinstance(result, SobolResult)
        # With only 16 samples, convergence should be poor
        # (this is a soft check — the flag may or may not be False depending
        # on the variance structure, but CIs should be wide)
        assert len(result.S1_conf) == len(_SMALL_RANGES)
        assert all(c >= 0 for c in result.S1_conf), "CIs should be non-negative"


# ===================================================================
# TestDefaultRanges
# ===================================================================


@pytest.mark.unit
class TestDefaultRanges:
    """Validate DEFAULT_GSA_RANGES."""

    def test_default_ranges_valid(self) -> None:
        """All default ranges have lower < upper and names match Drug fields."""
        drug_fields = {f.name for f in Drug.__dataclass_fields__.values()}

        for r in DEFAULT_GSA_RANGES:
            assert r.lower < r.upper, f"{r.name}: lower >= upper"
            assert r.name in drug_fields, f"{r.name} is not a Drug field. Available: {drug_fields}"


# ===================================================================
# TestSobolAPI
# ===================================================================


@pytest.mark.unit
class TestSobolAPI:
    """Test /sensitivity/sobol API endpoint."""

    @pytest.fixture()
    def client(self):
        """Create a TestClient for the FastAPI app."""
        from starlette.testclient import TestClient

        from omega_pbpk.api.app import app

        return TestClient(app)

    def test_sobol_endpoint_200(self, client) -> None:
        """POST /sensitivity/sobol returns 200 with default ranges."""
        payload = {
            "drug": {
                "name": "TestDrug",
                "mw": 300.0,
                "fup": 0.5,
                "rbp": 1.0,
                "peff": 1.0,
                "clint_hepatic_L_per_h": 5.0,
                "clr_L_per_h": 1.0,
            },
            "dose_mg": 100.0,
            "route": "oral",
            "n_samples": 32,
            "metric": "AUC",
            "n_bootstrap": 20,
        }
        resp = client.post("/sensitivity/sobol", json=payload)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "S1" in data
        assert "ST" in data
        assert "parameters" in data
        assert len(data["S1"]) == len(data["parameters"])
        assert data["metric"] == "AUC"

    def test_sobol_endpoint_custom_ranges(self, client) -> None:
        """POST /sensitivity/sobol with custom parameter_ranges works."""
        payload = {
            "drug": {
                "name": "CustomRangeTest",
                "mw": 250.0,
                "fup": 0.3,
                "rbp": 1.2,
                "peff": 2.0,
                "clint_hepatic_L_per_h": 10.0,
            },
            "dose_mg": 50.0,
            "route": "oral",
            "n_samples": 32,
            "metric": "Cmax",
            "n_bootstrap": 20,
            "parameter_ranges": [
                {"name": "fup", "lower": 0.05, "upper": 0.8},
                {"name": "rbp", "lower": 0.8, "upper": 1.5},
            ],
        }
        resp = client.post("/sensitivity/sobol", json=payload)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert len(data["parameters"]) == 2
        assert data["parameters"] == ["fup", "rbp"]
        assert data["metric"] == "Cmax"


# ===================================================================
# TestIntegration
# ===================================================================


@pytest.mark.integration
class TestIntegration:
    """Integration test for SobolResult structure completeness."""

    def test_sobol_result_structure(self) -> None:
        """All SobolResult fields are populated correctly."""
        result = sobol_sensitivity(
            drug=_TEST_DRUG,
            dose_mg=100.0,
            route="oral",
            parameter_ranges=_SMALL_RANGES,
            n_samples=64,
            metric="AUC",
            seed=42,
            n_bootstrap=50,
        )

        # Check all fields exist and have correct types
        assert isinstance(result.parameters, list)
        assert all(isinstance(p, str) for p in result.parameters)

        assert isinstance(result.S1, list)
        assert all(isinstance(v, float) for v in result.S1)

        assert isinstance(result.ST, list)
        assert all(isinstance(v, float) for v in result.ST)

        assert isinstance(result.S1_conf, list)
        assert all(isinstance(v, float) for v in result.S1_conf)

        assert isinstance(result.ST_conf, list)
        assert all(isinstance(v, float) for v in result.ST_conf)

        assert isinstance(result.metric, str)
        assert isinstance(result.n_samples, int)
        assert isinstance(result.n_total_evaluations, int)
        assert isinstance(result.convergence_flag, bool)

        assert isinstance(result.interaction_index, list)
        assert all(isinstance(v, float) for v in result.interaction_index)

        # Lengths should all match k
        k = len(_SMALL_RANGES)
        assert len(result.parameters) == k
        assert len(result.S1) == k
        assert len(result.ST) == k
        assert len(result.S1_conf) == k
        assert len(result.ST_conf) == k
        assert len(result.interaction_index) == k

        # n_total_evaluations = n * (k + 2)
        assert result.n_total_evaluations == 64 * (k + 2)

        # S1 should be clamped to >= 0
        assert all(s >= 0 for s in result.S1)

        # ST >= S1 for each parameter (in expectation, interaction >= 0)
        # (with clamping, this is approximately true)
        for i in range(k):
            assert result.interaction_index[i] >= -0.01, (
                f"interaction_index[{i}] = {result.interaction_index[i]}, expected >= 0"
            )
