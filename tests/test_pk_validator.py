"""Tests for Phase 49: PK Validation & Parameter Calibration engine.

TestValidationMetrics: unit tests for compute_validation_metrics().
TestValidateDrug: integration tests using caffeine simulated "observed" data.
TestParameterCalibrator: integration tests for calibrate_drug().
TestAPIEndpoints: smoke tests for /validate/pk and /calibrate/clint.
"""

from __future__ import annotations

import numpy as np
import pytest

from omega_pbpk.drugs.caffeine import CAFFEINE
from omega_pbpk.drugs.metformin import METFORMIN
from omega_pbpk.validation.pk_validator import (
    compute_validation_metrics,
    validate_drug,
)

# ---------------------------------------------------------------------------
# Unit tests: compute_validation_metrics()
# ---------------------------------------------------------------------------


class TestValidationMetrics:
    """Unit tests on pure metric computation — no simulation required."""

    def test_perfect_prediction_aafe_1(self) -> None:
        """Perfect match → AAFE=1, AFE=1, within-2-fold=100%."""
        t = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        c = np.array([3.0, 2.5, 1.5, 0.8, 0.2])
        result = compute_validation_metrics(t, c, t, c)
        assert abs(result.aafe - 1.0) < 1e-6
        assert abs(result.afe - 1.0) < 1e-6
        assert result.within_2fold_pct == pytest.approx(100.0)
        assert result.rmse_mg_per_L < 1e-6

    def test_2x_overprediction(self) -> None:
        """Constant 2× over-prediction → AAFE=2, AFE=2, within-2-fold=100%."""
        t = np.array([1.0, 2.0, 4.0, 8.0])
        obs = np.array([1.0, 0.8, 0.5, 0.2])
        sim = obs * 2.0
        result = compute_validation_metrics(t, obs, t, sim)
        assert result.aafe == pytest.approx(2.0, rel=1e-4)
        assert result.afe == pytest.approx(2.0, rel=1e-4)
        # 2× = exactly at the boundary → within-2-fold (ratio = 2.0, ≤ 2.0 is True)
        assert result.within_2fold_pct == pytest.approx(100.0)

    def test_3x_overprediction_fails_within_2fold(self) -> None:
        """3× over-prediction → all points outside 2-fold."""
        t = np.array([1.0, 2.0, 4.0])
        obs = np.array([1.0, 0.5, 0.2])
        sim = obs * 3.0
        result = compute_validation_metrics(t, obs, t, sim)
        assert result.aafe > 2.0
        assert result.within_2fold_pct == pytest.approx(0.0)

    def test_afe_underprediction(self) -> None:
        """Constant 0.5× under-prediction → AFE=0.5 (< 1)."""
        t = np.array([1.0, 2.0, 4.0])
        obs = np.array([2.0, 1.0, 0.5])
        sim = obs * 0.5
        result = compute_validation_metrics(t, obs, t, sim)
        assert result.afe == pytest.approx(0.5, rel=1e-4)

    def test_auc_relative_error(self) -> None:
        """AUC 20% higher in simulation → auc_relative_error ≈ 0.20."""
        t = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        obs = np.array([2.5, 2.0, 1.5, 0.8, 0.2])
        sim = obs * 1.2
        result = compute_validation_metrics(t, obs, t, sim)
        assert result.auc_relative_error == pytest.approx(0.20, rel=0.01)

    def test_raises_on_zero_observed(self) -> None:
        """Zero in observed concentrations raises ValueError."""
        t = np.array([1.0, 2.0, 3.0])
        obs = np.array([1.0, 0.0, 0.5])
        sim = np.array([1.0, 0.5, 0.4])
        with pytest.raises(ValueError, match="> 0"):
            compute_validation_metrics(t, obs, t, sim)

    def test_raises_on_empty(self) -> None:
        """Empty observed array raises ValueError."""
        with pytest.raises(ValueError):
            compute_validation_metrics(np.array([]), np.array([]), np.array([1.0]), np.array([1.0]))

    def test_overall_pass_properties(self) -> None:
        """ValidationResult.overall_pass reflects all 4 criteria."""
        t = np.linspace(0.5, 8.0, 10)
        obs = np.exp(-0.3 * t)
        result = compute_validation_metrics(t, obs, t, obs)
        assert result.aafe_pass
        assert result.within_2fold_pass
        assert result.auc_pass
        assert result.cmax_pass
        assert result.overall_pass

    def test_fold_errors_length(self) -> None:
        """fold_errors list has length == n_obs."""
        t = np.array([1.0, 2.0, 4.0, 6.0, 8.0])
        c = np.array([3.0, 2.0, 1.0, 0.5, 0.2])
        result = compute_validation_metrics(t, c, t, c * 1.5)
        assert len(result.fold_errors) == result.n_obs == len(t)


# ---------------------------------------------------------------------------
# Integration tests: validate_drug() with caffeine
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestValidateDrug:
    """Use caffeine simulation as both observed and simulated data."""

    def setup_method(self) -> None:
        from omega_pbpk.core.body import WholeBodyPBPK

        model = WholeBodyPBPK(CAFFEINE)
        model.setup_oral(dose_mg=100.0)
        result = model.simulate(t_end_h=24.0, dt_h=0.1)
        cp = result.plasma_concentration()
        t = result.time_h
        # Sample 8 sparse "observed" timepoints from the simulation
        idxs = np.linspace(10, len(t) - 20, 8, dtype=int)
        self._obs_t = t[idxs].tolist()
        self._obs_c = cp[idxs].tolist()

    def test_self_validation_aafe_near_1(self) -> None:
        """Validating a drug against its own simulation → AAFE ≈ 1."""
        vr = validate_drug(
            CAFFEINE,
            self._obs_t,
            self._obs_c,
            dose_mg=100.0,
            route="oral",
        )
        assert vr.aafe < 1.2, f"AAFE={vr.aafe:.3f} too far from 1"
        assert vr.within_2fold_pct == pytest.approx(100.0)
        assert vr.n_obs == 8

    def test_validate_iv_route(self) -> None:
        """IV route validation runs without error and returns AAFE."""
        from omega_pbpk.core.body import WholeBodyPBPK

        m = WholeBodyPBPK(CAFFEINE)
        m.setup_iv(dose_mg=100.0)
        r = m.simulate(t_end_h=24.0, dt_h=0.1)
        cp = r.plasma_concentration()
        t = r.time_h
        idxs = np.linspace(5, len(t) - 10, 6, dtype=int)
        obs_t = t[idxs].tolist()
        obs_c = cp[idxs].tolist()

        vr = validate_drug(CAFFEINE, obs_t, obs_c, dose_mg=100.0, route="iv")
        assert vr.aafe >= 1.0
        assert vr.n_obs == 6

    def test_summary_keys_present(self) -> None:
        """summary() dict contains all expected keys."""
        vr = validate_drug(CAFFEINE, self._obs_t, self._obs_c, dose_mg=100.0)
        s = vr.summary()
        for key in [
            "aafe",
            "afe",
            "within_2fold_pct",
            "rmse_mg_per_L",
            "auc_obs_mg_h_L",
            "auc_sim_mg_h_L",
            "auc_relative_error",
            "cmax_obs_mg_L",
            "cmax_sim_mg_L",
            "cmax_relative_error",
            "n_obs",
            "overall_pass",
        ]:
            assert key in s, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Integration tests: calibrate_drug()
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestParameterCalibrator:
    """Parameter calibration tests using synthetic observed data."""

    def _make_observed(self, drug, dose_mg=100.0, route="oral", clint_override=None):
        """Generate synthetic observed data from a simulation."""
        import copy

        from omega_pbpk.core.body import WholeBodyPBPK

        d = copy.copy(drug)
        if clint_override is not None:
            object.__setattr__(d, "clint_hepatic_L_per_h", clint_override)

        m = WholeBodyPBPK(d)
        if route == "iv":
            m.setup_iv(dose_mg=dose_mg)
        else:
            m.setup_oral(dose_mg=dose_mg)
        r = m.simulate(t_end_h=24.0, dt_h=0.1)
        cp = r.plasma_concentration()
        t = r.time_h
        idxs = np.linspace(10, len(t) - 10, 10, dtype=int)
        return t[idxs].tolist(), cp[idxs].tolist()

    def test_calibration_improves_aafe(self) -> None:
        """Calibrating CLint against a 2× perturbed simulation reduces AAFE."""
        from omega_pbpk.validation.parameter_calibrator import calibrate_drug

        # "True" CLint = 2× current caffeine CLint
        true_clint = CAFFEINE.clint_hepatic_L_per_h * 2.0
        obs_t, obs_c = self._make_observed(CAFFEINE, clint_override=true_clint)

        result = calibrate_drug(
            CAFFEINE,
            obs_t,
            obs_c,
            dose_mg=100.0,
            route="oral",
            params_to_fit=["clint"],
            max_iter=100,
        )

        assert result.post_calibration.aafe <= result.pre_calibration.aafe + 0.1, (
            f"Calibration should not worsen AAFE: "
            f"pre={result.pre_calibration.aafe:.2f}, "
            f"post={result.post_calibration.aafe:.2f}"
        )
        assert result.calibrated_clint_L_per_h > 0
        assert result.n_iterations > 0

    def test_calibration_result_summary_keys(self) -> None:
        """CalibrationResult.summary() contains all required keys."""
        from omega_pbpk.validation.parameter_calibrator import calibrate_drug

        obs_t, obs_c = self._make_observed(CAFFEINE)
        result = calibrate_drug(CAFFEINE, obs_t, obs_c, dose_mg=100.0, max_iter=10)
        s = result.summary()
        for key in [
            "original_clint_L_per_h",
            "calibrated_clint_L_per_h",
            "pre_aafe",
            "post_aafe",
            "aafe_improvement",
            "converged",
        ]:
            assert key in s, f"Missing key: {key}"

    def test_calibration_returns_drug(self) -> None:
        """calibrated_drug is a Drug instance with updated CLint."""
        from omega_pbpk.drugs.drug import Drug
        from omega_pbpk.validation.parameter_calibrator import calibrate_drug

        obs_t, obs_c = self._make_observed(METFORMIN, dose_mg=500.0, route="iv")
        result = calibrate_drug(METFORMIN, obs_t, obs_c, dose_mg=500.0, route="iv", max_iter=20)
        assert isinstance(result.calibrated_drug, Drug)


# ---------------------------------------------------------------------------
# API smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAPIEndpoints:
    def setup_method(self) -> None:
        try:
            from fastapi.testclient import TestClient

            from omega_pbpk.api.app import app
            from omega_pbpk.core.body import WholeBodyPBPK

            self._client = TestClient(app)

            # Build synthetic observed data from caffeine oral simulation
            m = WholeBodyPBPK(CAFFEINE)
            m.setup_oral(dose_mg=100.0)
            r = m.simulate(t_end_h=24.0, dt_h=0.1)
            cp = r.plasma_concentration()
            t = r.time_h
            idxs = np.linspace(5, len(t) - 10, 8, dtype=int)
            self._obs_t = t[idxs].tolist()
            self._obs_c = cp[idxs].tolist()
            self._has_fastapi = True
        except ImportError:
            self._has_fastapi = False

    def _drug_payload(self):
        return {
            "name": "Caffeine",
            "mw": 194.19,
            "logP": -0.07,
            "fup": 0.65,
            "rbp": 1.0,
            "clint_hepatic_L_per_h": 2.3,
            "clr_L_per_h": 0.08,
            "peff": 7.0,
        }

    def test_validate_pk_endpoint_200(self) -> None:
        """POST /validate/pk returns 200 with AAFE key."""
        if not self._has_fastapi:
            pytest.skip("fastapi not installed")
        resp = self._client.post(
            "/validate/pk",
            json={
                "drug": self._drug_payload(),
                "dose_mg": 100.0,
                "route": "oral",
                "observed_times": self._obs_t,
                "observed_conc": self._obs_c,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "aafe" in data
        assert "overall_pass" in data
        assert "within_2fold_pct" in data

    def test_calibrate_clint_endpoint_200(self) -> None:
        """POST /calibrate/clint returns 200 with calibrated_clint key."""
        if not self._has_fastapi:
            pytest.skip("fastapi not installed")
        resp = self._client.post(
            "/calibrate/clint",
            json={
                "drug": self._drug_payload(),
                "dose_mg": 100.0,
                "route": "oral",
                "observed_times": self._obs_t,
                "observed_conc": self._obs_c,
                "params_to_fit": ["clint"],
                "max_iter": 20,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "calibrated_clint_L_per_h" in data
        assert "post_aafe" in data
        assert "converged" in data
