"""Tests for Phase 20: Conformal UQ propagation."""

from __future__ import annotations

import pytest

from omega_pbpk.uncertainty.conformal_uq import (
    ConformalUQResult,
    ParameterBounds,
    build_bounds_from_adme,
    propagate_conformal_intervals,
)

BOUNDS = ParameterBounds(
    fup_lo=0.05,
    fup_hi=0.20,
    clint_lo=0.5,
    clint_hi=5.0,
    peff_lo=1e-5,
    peff_hi=5e-4,
    rbp_lo=0.6,
    rbp_hi=1.2,
)


@pytest.mark.integration
class TestParameterBounds:
    def test_construction(self):
        b = ParameterBounds(0.05, 0.20, 0.5, 5.0, 1e-5, 5e-4, 0.6, 1.2)
        assert b.fup_lo == 0.05
        assert b.fup_hi == 0.20

    def test_lo_le_hi(self):
        assert BOUNDS.fup_lo <= BOUNDS.fup_hi
        assert BOUNDS.clint_lo <= BOUNDS.clint_hi
        assert BOUNDS.peff_lo <= BOUNDS.peff_hi
        assert BOUNDS.rbp_lo <= BOUNDS.rbp_hi


@pytest.mark.integration
class TestBuildBoundsFromADME:
    def test_build_from_adme_props(self):
        from omega_pbpk.prediction.adme_predictor import ADMEPredictor

        predictor = ADMEPredictor()
        props = predictor.predict_from_dict({"mw": 300.0, "logP": 2.0})
        bounds = build_bounds_from_adme(props)
        assert isinstance(bounds, ParameterBounds)
        assert bounds.fup_lo <= bounds.fup_hi
        assert bounds.clint_lo <= bounds.clint_hi

    def test_build_from_smiles(self):
        from omega_pbpk.prediction.adme_predictor import ADMEPredictor

        predictor = ADMEPredictor()
        props = predictor.predict("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        bounds = build_bounds_from_adme(props)
        assert bounds.fup_lo >= 0
        assert bounds.fup_hi > 0


@pytest.mark.integration
class TestPropagateOral:
    def setup_method(self):
        self.result = propagate_conformal_intervals(
            drug_name="test", dose_mg=100.0, route="oral", bounds=BOUNDS, n_samples=50, seed=42
        )

    def test_returns_conformal_uq_result(self):
        assert isinstance(self.result, ConformalUQResult)

    def test_n_samples_matches(self):
        assert self.result.n_samples == 50

    def test_cmax_ordering(self):
        assert self.result.cmax_p5 <= self.result.cmax_p50 <= self.result.cmax_p95

    def test_auc_ordering(self):
        assert self.result.auc_p5 <= self.result.auc_p50 <= self.result.auc_p95

    def test_tmax_ordering(self):
        assert self.result.tmax_p5 <= self.result.tmax_p50 <= self.result.tmax_p95

    def test_t_half_ordering(self):
        assert self.result.t_half_p5 <= self.result.t_half_p50 <= self.result.t_half_p95

    def test_all_values_positive(self):
        r = self.result
        for val in [
            r.cmax_p5,
            r.cmax_p95,
            r.auc_p5,
            r.auc_p95,
            r.tmax_p5,
            r.tmax_p95,
            r.t_half_p5,
            r.t_half_p95,
        ]:
            assert val > 0

    def test_uncertainty_spread(self):
        # p95/p5 ratio should be > 1 (some spread)
        assert self.result.cmax_p95 / self.result.cmax_p5 > 1.0


@pytest.mark.integration
class TestPropagateIV:
    def setup_method(self):
        self.result = propagate_conformal_intervals(
            drug_name="test_iv", dose_mg=50.0, route="iv", bounds=BOUNDS, n_samples=50, seed=0
        )

    def test_iv_returns_valid(self):
        assert isinstance(self.result, ConformalUQResult)

    def test_iv_tmax_small(self):
        # IV bolus: tmax ~ 0.0833 h (5 min)
        assert self.result.tmax_p50 < 0.5

    def test_iv_cmax_positive(self):
        assert self.result.cmax_p50 > 0


@pytest.mark.integration
class TestEdgeCases:
    def test_small_n_samples(self):
        result = propagate_conformal_intervals(
            drug_name="x", dose_mg=10.0, route="oral", bounds=BOUNDS, n_samples=10, seed=1
        )
        assert result.n_samples == 10
        assert result.cmax_p5 <= result.cmax_p95

    def test_degenerate_bounds(self):
        degen = ParameterBounds(
            fup_lo=0.1,
            fup_hi=0.1,
            clint_lo=1.0,
            clint_hi=1.0,
            peff_lo=1e-4,
            peff_hi=1e-4,
            rbp_lo=0.8,
            rbp_hi=0.8,
        )
        result = propagate_conformal_intervals(
            drug_name="degen", dose_mg=100.0, route="oral", bounds=degen, n_samples=20, seed=42
        )
        # With degenerate bounds, p5 ≈ p50 ≈ p95
        assert abs(result.cmax_p95 - result.cmax_p5) / max(result.cmax_p50, 1e-12) < 0.01

    def test_warnings_field_is_tuple(self):
        result = propagate_conformal_intervals(
            drug_name="x", dose_mg=100.0, route="oral", bounds=BOUNDS, n_samples=10, seed=42
        )
        assert isinstance(result.warnings, tuple)


@pytest.mark.integration
class TestAPIUncertaintyEndpoint:
    def setup_method(self):
        from fastapi.testclient import TestClient

        from omega_pbpk.api.app import app

        self.client = TestClient(app)

    def test_endpoint_exists_with_bounds(self):
        resp = self.client.post(
            "/simulate/uncertainty",
            json={
                "drug_name": "aspirin",
                "dose_mg": 500.0,
                "route": "oral",
                "fup_lo": 0.05,
                "fup_hi": 0.20,
                "clint_lo": 0.1,
                "clint_hi": 2.0,
                "peff_lo": 1e-5,
                "peff_hi": 5e-4,
                "rbp_lo": 0.6,
                "rbp_hi": 1.2,
                "n_samples": 20,
            },
        )
        assert resp.status_code == 200

    def test_response_has_all_keys(self):
        resp = self.client.post(
            "/simulate/uncertainty",
            json={
                "drug_name": "test",
                "dose_mg": 100.0,
                "route": "iv",
                "fup_lo": 0.1,
                "fup_hi": 0.3,
                "clint_lo": 0.5,
                "clint_hi": 5.0,
                "peff_lo": 1e-5,
                "peff_hi": 5e-4,
                "rbp_lo": 0.7,
                "rbp_hi": 1.1,
                "n_samples": 10,
            },
        )
        data = resp.json()
        for key in [
            "cmax_p5",
            "cmax_p50",
            "cmax_p95",
            "auc_p5",
            "auc_p50",
            "auc_p95",
            "tmax_p5",
            "tmax_p50",
            "tmax_p95",
            "t_half_p5",
            "t_half_p50",
            "t_half_p95",
            "n_samples",
            "warnings",
        ]:
            assert key in data

    def test_invalid_route_returns_error(self):
        resp = self.client.post(
            "/simulate/uncertainty",
            json={
                "drug_name": "test",
                "dose_mg": 100.0,
                "route": "subcutaneous",
                "fup_lo": 0.1,
                "fup_hi": 0.3,
                "clint_lo": 0.5,
                "clint_hi": 5.0,
                "peff_lo": 1e-5,
                "peff_hi": 5e-4,
                "rbp_lo": 0.7,
                "rbp_hi": 1.1,
            },
        )
        assert resp.status_code in (400, 422)

    def test_cmax_ordering_in_response(self):
        resp = self.client.post(
            "/simulate/uncertainty",
            json={
                "drug_name": "drug",
                "dose_mg": 200.0,
                "route": "oral",
                "fup_lo": 0.05,
                "fup_hi": 0.30,
                "clint_lo": 1.0,
                "clint_hi": 10.0,
                "peff_lo": 1e-5,
                "peff_hi": 1e-3,
                "rbp_lo": 0.6,
                "rbp_hi": 1.5,
                "n_samples": 50,
            },
        )
        data = resp.json()
        assert data["cmax_p5"] <= data["cmax_p50"] <= data["cmax_p95"]
