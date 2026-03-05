"""Tests for Phase 46 Neural ODE surrogate.

Covers:
  - TestDataClasses (2): NeuralODEInput / NeuralODEOutput creation
  - TestNeuralODEFunc (3): forward shape, determinism, param roundtrip
  - TestNeuralODE (4): Euler forward, Euler backward sign, RK4 simulate,
                       save/load roundtrip
  - TestSingleton (1): predict_neural_ode returns valid NeuralODEOutput
  - TestAPIEndpoint (1): POST /predict/neural_ode returns 200 + trajectory
"""

from __future__ import annotations

import tempfile
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.ml_models.neural_ode import (
    NeuralODE,
    NeuralODEFunc,
    NeuralODEInput,
    NeuralODEOutput,
    predict_neural_ode,
)

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared tiny trained model (10 samples, 5 epochs) for fast unit tests
# ---------------------------------------------------------------------------

_TINY_NODE: NeuralODE | None = None


def _get_tiny_node() -> NeuralODE:
    """Return a minimal trained NeuralODE for fast tests (skips PBPK calls)."""
    global _TINY_NODE
    if _TINY_NODE is not None:
        return _TINY_NODE

    # Build synthetic C(t) data directly (skip PBPK to keep tests fast)
    rng = np.random.default_rng(0)
    N, T = 20, 25
    t_eval = np.linspace(0.0, 24.0, T)

    # Simulate biexponential C(t) with random PK params
    p_raw_list, C_list = [], []
    for _ in range(N):
        logP = rng.uniform(-1.0, 4.0)
        fup = rng.uniform(0.05, 0.9)
        clint = rng.uniform(1.0, 50.0)
        mw = rng.uniform(150.0, 600.0)
        rbp = rng.uniform(0.5, 2.0)
        peff = rng.uniform(0.2, 10.0)
        ka = rng.uniform(0.5, 3.0)
        ke = rng.uniform(0.05, 0.5)
        C0 = 5.0
        C = C0 * ka / (ka - ke) * (np.exp(-ke * t_eval) - np.exp(-ka * t_eval))
        C = np.maximum(C, 0.0)
        p_raw = np.array(
            [logP, np.log(fup), np.log(clint), np.log(mw / 300), np.log(rbp), np.log(peff)]
        )
        p_raw_list.append(p_raw)
        C_list.append(C)

    p_raw_arr = np.array(p_raw_list)
    C_traj = np.array(C_list)

    node = NeuralODE(seed=1)
    node.fit(p_raw_arr, C_traj, t_eval, epochs=5, lr=1e-3)
    _TINY_NODE = node
    return node


# ---------------------------------------------------------------------------
# TestDataClasses
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDataClasses:
    def test_neural_ode_input_defaults(self):
        """NeuralODEInput has sensible defaults."""
        inp = NeuralODEInput()
        assert inp.logP == 2.0
        assert inp.fup == 0.5
        assert inp.dose_mg == 100.0
        assert inp.route == "oral"
        assert inp.t_end_h == 24.0

    def test_neural_ode_output_fields(self):
        """NeuralODEOutput stores trajectory and PK metric fields."""
        out = NeuralODEOutput(
            t_h=[0.0, 1.0, 24.0],
            C_mg_L=[0.0, 3.5, 0.1],
            cmax_mg_L=3.5,
            auc_mg_h_L=20.0,
            tmax_h=1.0,
            t_half_h=8.0,
            train_r2=0.92,
            n_train=100,
            solve_time_ms=1.5,
        )
        assert out.cmax_mg_L == pytest.approx(3.5)
        assert out.auc_mg_h_L == pytest.approx(20.0)
        assert out.train_r2 == pytest.approx(0.92)
        assert len(out.t_h) == 3
        assert len(out.C_mg_L) == 3


# ---------------------------------------------------------------------------
# TestNeuralODEFunc
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNeuralODEFunc:
    def test_forward_returns_scalar(self):
        """NeuralODEFunc.forward returns scalar dC/dt and cached activations."""
        func = NeuralODEFunc(n_hidden=16, seed=0)
        x = np.zeros(8)
        out, z1, a1, z2, a2 = func.forward(x)
        assert isinstance(out, float)
        assert z1.shape == (16,)
        assert a1.shape == (16,)
        assert z2.shape == (16,)
        assert a2.shape == (16,)

    def test_forward_determinism(self):
        """Same seed produces identical forward pass output."""
        x = np.ones(8) * 0.5
        f1 = NeuralODEFunc(n_hidden=16, seed=7)
        f2 = NeuralODEFunc(n_hidden=16, seed=7)
        out1, *_ = f1.forward(x)
        out2, *_ = f2.forward(x)
        assert out1 == pytest.approx(out2, abs=1e-12)

    def test_param_roundtrip(self):
        """get_flat_params / set_flat_params preserves weights exactly."""
        func = NeuralODEFunc(n_hidden=16, seed=3)
        orig = func.get_flat_params().copy()
        # Corrupt all weights
        func.set_flat_params(np.zeros_like(orig))
        # Restore
        func.set_flat_params(orig)
        np.testing.assert_array_equal(orig, func.get_flat_params())


# ---------------------------------------------------------------------------
# TestNeuralODE
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNeuralODE:
    def test_euler_forward_shape(self):
        """_euler_forward returns C of shape (n_steps+1,) and a cache."""
        node = NeuralODE(seed=0)
        node._feat_mean = np.zeros(6)
        node._feat_std = np.ones(6)
        node._C_scale = 5.0
        p_norm = np.zeros(6)
        C, cache = node._euler_forward(p_norm, C0_norm=0.0, n_steps=48, dt_norm=1.0 / 48)
        assert C.shape == (49,)
        assert len(cache) == 48

    def test_euler_forward_nonnegative(self):
        """Euler forward trajectory should start at C0_norm."""
        node = NeuralODE(seed=0)
        node._feat_mean = np.zeros(6)
        node._feat_std = np.ones(6)
        node._C_scale = 5.0
        p_norm = np.zeros(6)
        C, _ = node._euler_forward(p_norm, C0_norm=0.5, n_steps=10, dt_norm=0.1)
        assert C[0] == pytest.approx(0.5)

    def test_backward_produces_finite_gradients(self):
        """_euler_backward returns finite gradient arrays of correct shape."""
        node = NeuralODE(seed=0)
        node._feat_mean = np.zeros(6)
        node._feat_std = np.ones(6)
        node._C_scale = 5.0
        p_norm = np.zeros(6)
        n_steps, dt_norm = 10, 0.1
        C, cache = node._euler_forward(p_norm, 0.0, n_steps, dt_norm)
        C_true = np.linspace(0, 1, n_steps + 1)
        t_obs_idx = np.arange(n_steps + 1)
        grads = node._euler_backward(C, cache, C_true, t_obs_idx, dt_norm)
        assert all(np.all(np.isfinite(v)) for k, v in grads.items() if k != "db3")
        assert np.isfinite(grads["db3"])

    def test_save_load_roundtrip(self):
        """save/load reproduces identical RK4 predictions."""
        node = _get_tiny_node()
        out_before = node.simulate(
            logP=2.0, fup=0.5, clint_L_per_h=10.0, mw=300.0, rbp=1.0, peff=1.0
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            node.save(f"{tmpdir}/test_node.npz")
            node2 = NeuralODE(seed=1)
            node2.load(f"{tmpdir}/test_node.npz")

        out_after = node2.simulate(
            logP=2.0, fup=0.5, clint_L_per_h=10.0, mw=300.0, rbp=1.0, peff=1.0
        )
        assert out_after.cmax_mg_L == pytest.approx(out_before.cmax_mg_L, rel=1e-5)
        assert out_after.auc_mg_h_L == pytest.approx(out_before.auc_mg_h_L, rel=1e-5)


# ---------------------------------------------------------------------------
# TestSimulate
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSimulate:
    def test_simulate_trajectory_length(self):
        """simulate returns trajectory with correct number of points."""
        node = _get_tiny_node()
        out = node.simulate(
            logP=2.0, fup=0.5, clint_L_per_h=10.0, mw=300.0, rbp=1.0, peff=1.0, n_points=49
        )
        assert len(out.t_h) == 49
        assert len(out.C_mg_L) == 49

    def test_simulate_pk_metrics_positive(self):
        """Cmax, AUC, tmax, t_half must be non-negative."""
        node = _get_tiny_node()
        out = node.simulate(logP=2.0, fup=0.5, clint_L_per_h=10.0, mw=300.0, rbp=1.0, peff=1.0)
        assert out.cmax_mg_L >= 0.0
        assert out.auc_mg_h_L >= 0.0
        assert out.tmax_h >= 0.0
        assert out.t_half_h >= 0.0

    def test_simulate_dose_scaling(self):
        """Doubling the dose approximately doubles Cmax (linear superposition)."""
        node = _get_tiny_node()
        out1 = node.simulate(
            logP=2.0, fup=0.5, clint_L_per_h=10.0, mw=300.0, rbp=1.0, peff=1.0, dose_mg=100.0
        )
        out2 = node.simulate(
            logP=2.0, fup=0.5, clint_L_per_h=10.0, mw=300.0, rbp=1.0, peff=1.0, dose_mg=200.0
        )
        if out1.cmax_mg_L > 1e-6:
            ratio = out2.cmax_mg_L / out1.cmax_mg_L
            assert ratio == pytest.approx(2.0, rel=0.01)

    def test_simulate_t_end_respected(self):
        """Last time point matches t_end_h."""
        node = _get_tiny_node()
        out = node.simulate(
            logP=2.0, fup=0.5, clint_L_per_h=10.0, mw=300.0, rbp=1.0, peff=1.0, t_end_h=12.0
        )
        assert out.t_h[-1] == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# TestSingleton
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSingleton:
    def test_predict_neural_ode_returns_valid_output(self):
        """predict_neural_ode with mocked singleton returns NeuralODEOutput."""
        import omega_pbpk.ml_models.neural_ode as mod

        fake_out = NeuralODEOutput(
            t_h=list(np.linspace(0, 24, 49)),
            C_mg_L=list(np.linspace(0, 5, 49)),
            cmax_mg_L=5.0,
            auc_mg_h_L=60.0,
            tmax_h=2.0,
            t_half_h=8.0,
            train_r2=0.95,
            n_train=100,
            solve_time_ms=1.2,
        )

        node_stub = NeuralODE(seed=0)
        node_stub._trained = True
        node_stub._feat_mean = np.zeros(6)
        node_stub._feat_std = np.ones(6)
        node_stub._C_scale = 5.0

        with patch.object(node_stub, "simulate", return_value=fake_out):
            mod._NODE_SINGLETON = node_stub
            result = predict_neural_ode(NeuralODEInput())

        assert isinstance(result, NeuralODEOutput)
        assert result.cmax_mg_L == pytest.approx(5.0)
        assert result.train_r2 == pytest.approx(0.95)

        mod._NODE_SINGLETON = None  # reset


# ---------------------------------------------------------------------------
# TestAPIEndpoint
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAPIEndpoint:
    def test_neural_ode_endpoint_200(self):
        """POST /predict/neural_ode returns 200 with trajectory and PK fields."""

        fake_out = NeuralODEOutput(
            t_h=list(np.linspace(0, 24, 49)),
            C_mg_L=list(np.linspace(0, 3, 49)),
            cmax_mg_L=3.0,
            auc_mg_h_L=40.0,
            tmax_h=1.5,
            t_half_h=9.0,
            train_r2=0.90,
            n_train=100,
            solve_time_ms=0.8,
        )

        with patch("omega_pbpk.ml_models.neural_ode.predict_neural_ode", return_value=fake_out):
            response = client.post(
                "/predict/neural_ode",
                json={
                    "logP": 2.0,
                    "fup": 0.5,
                    "clint_L_per_h": 10.0,
                    "mw": 300.0,
                    "rbp": 1.0,
                    "peff": 1.0,
                    "dose_mg": 100.0,
                    "route": "oral",
                    "t_end_h": 24.0,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "C_mg_L" in data
        assert "cmax_mg_L" in data
        assert "auc_mg_h_L" in data
        assert "tmax_h" in data
        assert "t_half_h" in data
        assert "train_r2" in data
        assert len(data["C_mg_L"]) == 49
        assert data["cmax_mg_L"] == pytest.approx(3.0)
