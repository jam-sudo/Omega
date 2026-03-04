"""Tests for Phase 40 Neural PK Surrogate (NumpyMLP ensemble).

Covers:
  - TestDataClasses (2): PKSurrogateInput / PKSurrogateOutput creation
  - TestNumpyMLP (3): forward shape, determinism, get/set params roundtrip
  - TestTrainingData (2): generate_analytical_pk_data shapes and sign
  - TestAnalyticalPK (2): oral and IV formulas within 1%
  - TestEnsemble (3): train+predict, confidence, save/load roundtrip
  - TestIntegration (2): singleton convenience fn, API endpoint
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.ml_models.pk_surrogate import (
    NumpyMLP,
    PKSurrogateEnsemble,
    PKSurrogateInput,
    PKSurrogateOutput,
    analytical_pk_1cpt,
    generate_analytical_pk_data,
    predict_pk_surrogate,
)

client = TestClient(app)

_LN2 = 0.6931471805599453

# ---------------------------------------------------------------------------
# Shared small training set (100 samples, reused across ensemble tests)
# ---------------------------------------------------------------------------

_X_SMALL, _Y_SMALL = generate_analytical_pk_data(n_samples=100, seed=7)


# ---------------------------------------------------------------------------
# TestDataClasses
# ---------------------------------------------------------------------------


class TestDataClasses:
    def test_pk_surrogate_input_creation(self):
        """PKSurrogateInput stores all fields correctly."""
        inp = PKSurrogateInput(
            dose_mg=100.0,
            cl_L_per_h=5.0,
            vd_L=70.0,
            ka_per_h=1.2,
            f_bioavail=0.8,
            route="oral",
        )
        assert inp.dose_mg == 100.0
        assert inp.cl_L_per_h == 5.0
        assert inp.vd_L == 70.0
        assert inp.ka_per_h == 1.2
        assert inp.f_bioavail == 0.8
        assert inp.route == "oral"

    def test_pk_surrogate_output_fields(self):
        """PKSurrogateOutput exposes confidence and ensemble std fields."""
        out = PKSurrogateOutput(
            auc_mg_h_per_L=12.0,
            cmax_mg_per_L=3.5,
            tmax_h=1.0,
            t_half_h=8.0,
            confidence="high",
            auc_std=0.1,
            cmax_std=0.05,
            tmax_std=0.02,
            t_half_std=0.3,
        )
        assert out.confidence == "high"
        assert out.auc_std == pytest.approx(0.1)
        assert out.cmax_std == pytest.approx(0.05)
        assert out.tmax_std == pytest.approx(0.02)
        assert out.t_half_std == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# TestNumpyMLP
# ---------------------------------------------------------------------------


class TestNumpyMLP:
    def test_numpy_mlp_forward_shape(self):
        """NumpyMLP (10, 6) input → (10, 4) output."""
        mlp = NumpyMLP(layer_sizes=(6, 32, 32, 4), seed=0)
        rng = np.random.default_rng(1)
        X = rng.standard_normal((10, 6)).astype(np.float32)
        out = mlp.forward(X)
        assert out.shape == (10, 4)

    def test_numpy_mlp_deterministic(self):
        """Same seed produces identical forward outputs."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 6)).astype(np.float32)
        mlp_a = NumpyMLP(layer_sizes=(6, 32, 32, 4), seed=99)
        mlp_b = NumpyMLP(layer_sizes=(6, 32, 32, 4), seed=99)
        np.testing.assert_array_equal(mlp_a.forward(X), mlp_b.forward(X))

    def test_numpy_mlp_get_set_params(self):
        """get_params / set_params roundtrip preserves weights exactly."""
        mlp = NumpyMLP(layer_sizes=(6, 32, 32, 4), seed=3)
        params_orig = [p.copy() for p in mlp.get_params()]
        # Corrupt weights
        for p in mlp.get_params():
            p[:] = 0.0
        # Restore
        mlp.set_params(params_orig)
        for orig, restored in zip(params_orig, mlp.get_params(), strict=True):
            np.testing.assert_array_equal(orig, restored)


# ---------------------------------------------------------------------------
# TestTrainingData
# ---------------------------------------------------------------------------


class TestTrainingData:
    def test_generate_analytical_pk_data_shapes(self):
        """generate_analytical_pk_data(500) → X(500,6), y(500,4)."""
        X, y = generate_analytical_pk_data(n_samples=500, seed=0)
        assert X.shape == (500, 6)
        assert y.shape == (500, 4)
        assert X.dtype == np.float32
        assert y.dtype == np.float32

    def test_analytical_pk_data_positive_outputs(self):
        """AUC, Cmax, and t_half must be > 0; tmax >= 0."""
        X, y = generate_analytical_pk_data(n_samples=200, seed=1)
        # Columns: AUC, Cmax, tmax, t_half
        assert np.all(y[:, 0] > 0), "AUC must be positive"
        assert np.all(y[:, 1] >= 0), "Cmax must be non-negative"
        assert np.all(y[:, 2] >= 0), "tmax must be non-negative"
        assert np.all(y[:, 3] > 0), "t_half must be positive"


# ---------------------------------------------------------------------------
# TestAnalyticalPK
# ---------------------------------------------------------------------------


class TestAnalyticalPK:
    def test_analytical_pk_1cpt_oral(self):
        """Oral 1-cpt: AUC = F*dose/CL and t_half = ln2*Vd/CL within 1%."""
        inp = PKSurrogateInput(
            dose_mg=100.0,
            cl_L_per_h=5.0,
            vd_L=70.0,
            ka_per_h=1.5,
            f_bioavail=0.9,
            route="oral",
        )
        out = analytical_pk_1cpt(inp)
        assert out.confidence == "analytical"

        expected_auc = inp.f_bioavail * inp.dose_mg / inp.cl_L_per_h  # 18 mg·h/L
        expected_thalf = _LN2 * inp.vd_L / inp.cl_L_per_h  # ~9.7 h

        assert out.auc_mg_h_per_L == pytest.approx(expected_auc, rel=0.01)
        assert out.t_half_h == pytest.approx(expected_thalf, rel=0.01)
        assert out.cmax_mg_per_L > 0
        assert out.tmax_h >= 0

    def test_analytical_pk_1cpt_iv(self):
        """IV bolus: Cmax = dose/Vd, tmax = 0, AUC = dose/CL within 1%."""
        inp = PKSurrogateInput(
            dose_mg=50.0,
            cl_L_per_h=10.0,
            vd_L=40.0,
            ka_per_h=1.0,
            f_bioavail=1.0,
            route="iv",
        )
        out = analytical_pk_1cpt(inp)
        assert out.confidence == "analytical"

        expected_cmax = inp.dose_mg / inp.vd_L  # 1.25 mg/L
        expected_auc = inp.dose_mg / inp.cl_L_per_h  # 5 mg·h/L

        assert out.cmax_mg_per_L == pytest.approx(expected_cmax, rel=0.01)
        assert out.auc_mg_h_per_L == pytest.approx(expected_auc, rel=0.01)
        assert out.tmax_h == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# TestEnsemble
# ---------------------------------------------------------------------------


class TestEnsemble:
    def _trained_ensemble(self) -> PKSurrogateEnsemble:
        """Return a small trained ensemble (n_samples=100, epochs=10)."""
        ens = PKSurrogateEnsemble(n_models=3, seed=42)
        ens.train(_X_SMALL, _Y_SMALL, epochs=10, lr=0.005, batch_size=32)
        return ens

    def test_ensemble_train_and_predict(self):
        """Trained ensemble returns a PKSurrogateOutput with positive values."""
        ens = self._trained_ensemble()
        inp = PKSurrogateInput(
            dose_mg=100.0,
            cl_L_per_h=5.0,
            vd_L=70.0,
            ka_per_h=1.2,
            f_bioavail=0.8,
            route="oral",
        )
        out = ens.predict(inp)
        assert isinstance(out, PKSurrogateOutput)
        assert out.auc_mg_h_per_L >= 0.0
        assert out.cmax_mg_per_L >= 0.0
        assert out.tmax_h >= 0.0
        assert out.t_half_h >= 0.0
        assert out.confidence in ("high", "medium", "low")

    def test_ensemble_confidence_reasonable(self):
        """Typical pharmacokinetic inputs yield 'high' or 'medium' confidence."""
        ens = PKSurrogateEnsemble(n_models=5, seed=42)
        ens.train(_X_SMALL, _Y_SMALL, epochs=10, lr=0.005, batch_size=32)
        inp = PKSurrogateInput(
            dose_mg=200.0,
            cl_L_per_h=8.0,
            vd_L=50.0,
            ka_per_h=1.0,
            f_bioavail=0.75,
            route="oral",
        )
        out = ens.predict(inp)
        assert out.confidence in ("high", "medium", "low")

    def test_ensemble_save_load_roundtrip(self):
        """save() then load() reproduces identical predictions."""
        ens = self._trained_ensemble()
        inp = PKSurrogateInput(
            dose_mg=150.0,
            cl_L_per_h=6.0,
            vd_L=60.0,
            ka_per_h=0.8,
            f_bioavail=0.9,
            route="oral",
        )
        out_before = ens.predict(inp)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = f"{tmpdir}/ens_test"
            ens.save(save_path)

            ens2 = PKSurrogateEnsemble(n_models=3, seed=42)
            ens2.load(f"{save_path}.npz")
            out_after = ens2.predict(inp)

        assert out_after.auc_mg_h_per_L == pytest.approx(out_before.auc_mg_h_per_L, rel=1e-5)
        assert out_after.cmax_mg_per_L == pytest.approx(out_before.cmax_mg_per_L, rel=1e-5)
        assert out_after.t_half_h == pytest.approx(out_before.t_half_h, rel=1e-5)


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_predict_pk_surrogate_convenience(self):
        """predict_pk_surrogate returns a PKSurrogateOutput (mocked singleton)."""
        from unittest.mock import patch

        fake_out = PKSurrogateOutput(
            auc_mg_h_per_L=15.0,
            cmax_mg_per_L=4.0,
            tmax_h=1.5,
            t_half_h=9.6,
            confidence="high",
        )
        inp = PKSurrogateInput(
            dose_mg=100.0,
            cl_L_per_h=5.0,
            vd_L=70.0,
            ka_per_h=1.2,
            f_bioavail=0.8,
            route="oral",
        )
        with patch(
            "omega_pbpk.ml_models.pk_surrogate.PKSurrogateEnsemble.predict",
            return_value=fake_out,
        ):
            import omega_pbpk.ml_models.pk_surrogate as _mod

            _mod._SINGLETON = PKSurrogateEnsemble(n_models=2, seed=0)
            _mod._SINGLETON._trained = True
            result = predict_pk_surrogate(inp)

        assert isinstance(result, PKSurrogateOutput)
        assert result.auc_mg_h_per_L == pytest.approx(15.0)
        assert result.confidence == "high"

    def test_pk_surrogate_endpoint(self):
        """POST /predict/pk_surrogate returns 200 with auc field."""
        from unittest.mock import patch

        fake_out = PKSurrogateOutput(
            auc_mg_h_per_L=18.0,
            cmax_mg_per_L=5.0,
            tmax_h=2.0,
            t_half_h=9.6,
            confidence="high",
        )
        with patch(
            "omega_pbpk.ml_models.pk_surrogate.predict_pk_surrogate",
            return_value=fake_out,
        ):
            response = client.post(
                "/predict/pk_surrogate",
                json={
                    "dose_mg": 100.0,
                    "cl_L_per_h": 5.0,
                    "vd_L": 70.0,
                    "ka_per_h": 1.2,
                    "f_bioavail": 0.8,
                    "route": "oral",
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert "auc_mg_h_per_L" in data
        assert data["auc_mg_h_per_L"] == pytest.approx(18.0)
        assert "confidence" in data
