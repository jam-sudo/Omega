"""Tests for synthetic PK data generation (Tasks A.1, A.2, A.3)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from omega_pbpk.ml.data.synthetic import (
    ONECOMP_PARAM_RANGES,
    PBPK_PARAM_RANGES,
    CurveDataset,
    _onecomp_oral_curve,
    generate_1cpt_data,
    generate_pbpk_data,
)

# ---------------------------------------------------------------------------
# 1-cpt analytical formula correctness
# ---------------------------------------------------------------------------


class TestOneCptFormula:
    """Test the 1-compartment oral PK analytical formula."""

    def test_known_solution(self):
        """Verify C(t) against hand-calculated values for specific params."""
        t = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 24.0])
        ka, ke, Vd, F, dose = 1.0, 0.1, 100.0, 1.0, 1000.0

        cp = _onecomp_oral_curve(t, ka, ke, Vd, F, dose)

        # C(t) = (F*dose*ka/(Vd*(ka-ke))) * (exp(-ke*t) - exp(-ka*t))
        coeff = F * dose * ka / (Vd * (ka - ke))
        expected = coeff * (np.exp(-ke * t) - np.exp(-ka * t))

        np.testing.assert_allclose(cp, expected, rtol=1e-10)

    def test_zero_at_t0(self):
        """Concentration should be 0 at t=0 for oral dosing."""
        t = np.array([0.0])
        cp = _onecomp_oral_curve(t, ka=2.0, ke=0.5, Vd=50.0, F=0.8, dose=500.0)
        assert cp[0] == pytest.approx(0.0, abs=1e-15)

    def test_positive_concentrations(self):
        """All concentrations should be non-negative for valid params."""
        t = np.linspace(0, 24, 241)
        cp = _onecomp_oral_curve(t, ka=1.5, ke=0.3, Vd=100.0, F=0.7, dose=200.0)
        assert np.all(cp >= -1e-15)

    def test_ka_equals_ke_limiting_case(self):
        """When ka == ke, the limiting form should be used."""
        t = np.linspace(0, 24, 241)
        ka = 0.5
        ke = 0.5
        Vd = 100.0
        F = 1.0
        dose = 500.0

        cp = _onecomp_oral_curve(t, ka, ke, Vd, F, dose)

        # Limiting form: (F*dose/Vd) * t * ka * exp(-ke*t)
        expected = (F * dose / Vd) * t * ka * np.exp(-ke * t)
        np.testing.assert_allclose(cp, expected, rtol=1e-10)

    def test_dose_linearity(self):
        """Doubling the dose should double concentrations."""
        t = np.linspace(0, 24, 241)
        cp1 = _onecomp_oral_curve(t, ka=1.0, ke=0.2, Vd=80.0, F=0.9, dose=100.0)
        cp2 = _onecomp_oral_curve(t, ka=1.0, ke=0.2, Vd=80.0, F=0.9, dose=200.0)
        np.testing.assert_allclose(cp2, 2.0 * cp1, rtol=1e-10)

    def test_cmax_at_expected_time(self):
        """Tmax should match analytical expression: ln(ka/ke) / (ka - ke)."""
        ka, ke = 2.0, 0.3
        t = np.linspace(0, 24, 10000)  # Fine grid for precision
        cp = _onecomp_oral_curve(t, ka, ke, Vd=100.0, F=1.0, dose=100.0)

        tmax_analytical = np.log(ka / ke) / (ka - ke)
        tmax_numerical = t[np.argmax(cp)]

        assert tmax_numerical == pytest.approx(tmax_analytical, abs=0.01)


# ---------------------------------------------------------------------------
# 1-cpt data generator
# ---------------------------------------------------------------------------


class TestGenerate1CptData:
    """Test the 1-compartment data generator."""

    def test_output_format(self):
        """Generated dataset should have correct shapes and types."""
        ds = generate_1cpt_data(n_samples=100, seed=99)

        assert isinstance(ds, CurveDataset)
        assert ds.params.ndim == 2
        assert ds.curves.ndim == 2
        assert ds.metrics.ndim == 2
        assert ds.n_params == 5  # ka, ke, Vd, F, dose
        assert ds.curves.shape[1] == 241  # timepoints
        assert ds.metrics.shape[1] == 4  # Cmax, AUC, Tmax, t1/2

    def test_valid_sample_count(self):
        """Most samples should be valid (near 100% for analytical)."""
        ds = generate_1cpt_data(n_samples=1000, seed=42)
        # Analytical model should have very high success rate
        assert ds.n_samples >= 990

    def test_parameter_ranges(self):
        """Generated parameters should respect specified ranges."""
        ds = generate_1cpt_data(n_samples=5000, seed=42)

        for j, name in enumerate(ds.param_names):
            lo, hi = ONECOMP_PARAM_RANGES[name]
            col = ds.params[:, j]
            assert np.all(col >= lo * 0.99), f"{name} has values below {lo}"
            assert np.all(col <= hi * 1.01), f"{name} has values above {hi}"

    def test_curves_non_negative(self):
        """All concentration curves should be non-negative."""
        ds = generate_1cpt_data(n_samples=500, seed=42)
        assert np.all(ds.curves >= -1e-12)

    def test_metrics_positive(self):
        """All PK metrics should be positive."""
        ds = generate_1cpt_data(n_samples=500, seed=42)
        assert np.all(ds.metrics > 0)

    def test_reproducibility(self):
        """Same seed should produce identical results."""
        ds1 = generate_1cpt_data(n_samples=50, seed=123)
        ds2 = generate_1cpt_data(n_samples=50, seed=123)
        np.testing.assert_array_equal(ds1.params, ds2.params)
        np.testing.assert_array_equal(ds1.curves, ds2.curves)


# ---------------------------------------------------------------------------
# HDF5 save/load
# ---------------------------------------------------------------------------


class TestHDF5IO:
    """Test HDF5 serialization."""

    def test_roundtrip(self):
        pytest.importorskip("h5py", reason="h5py not installed")
        """Save and load should produce identical dataset."""
        ds = generate_1cpt_data(n_samples=50, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_data.h5"
            ds.save_hdf5(path)

            assert path.exists()

            ds_loaded = CurveDataset.load_hdf5(path)

            np.testing.assert_array_equal(ds.params, ds_loaded.params)
            np.testing.assert_array_equal(ds.curves, ds_loaded.curves)
            np.testing.assert_array_equal(ds.metrics, ds_loaded.metrics)
            np.testing.assert_array_equal(ds.time_h, ds_loaded.time_h)
            assert ds.param_names == ds_loaded.param_names
            assert ds.metric_names == ds_loaded.metric_names


# ---------------------------------------------------------------------------
# PBPK data generator (lightweight test — only 5 samples)
# ---------------------------------------------------------------------------


class TestGeneratePBPKData:
    """Test the PBPK ODE data generator with a small batch."""

    @pytest.mark.slow
    def test_output_format_small(self):
        """PBPK generator should produce valid output for a small batch."""
        ds = generate_pbpk_data(n_samples=5, seed=42, n_workers=1)

        assert isinstance(ds, CurveDataset)
        assert ds.params.ndim == 2
        assert ds.curves.ndim == 2
        assert ds.n_params == 6  # logP, fup, clint_L_h, mw, rbp, peff

        # At least some samples should succeed
        assert ds.n_samples >= 1, "No valid PBPK samples generated"

        # Curves should have correct number of timepoints
        assert ds.curves.shape[1] == 241

    @pytest.mark.slow
    def test_parameter_ranges_respected(self):
        """PBPK parameters should stay within specified ranges."""
        ds = generate_pbpk_data(n_samples=10, seed=42, n_workers=1)

        if ds.n_samples == 0:
            pytest.skip("No valid PBPK samples generated")

        for j, name in enumerate(ds.param_names):
            lo, hi = PBPK_PARAM_RANGES[name]
            col = ds.params[:, j]
            # Allow small numerical tolerance
            assert np.all(col >= lo * 0.99), f"{name} below range"
            assert np.all(col <= hi * 1.01), f"{name} above range"


# ---------------------------------------------------------------------------
# Differentiable ODE surrogate
# ---------------------------------------------------------------------------


class TestDifferentiableODESurrogate:
    """Test the neural surrogate model."""

    def test_forward_pass_shape(self):
        """Model forward pass should produce correct output shape."""
        torch = pytest.importorskip("torch")
        from omega_pbpk.ml.models.surrogate.differentiable_ode import (
            DifferentiableODESurrogate,
        )

        model = DifferentiableODESurrogate(n_input=6, n_output=241)

        x = torch.randn(32, 6)
        y = model(x)

        assert y.shape == (32, 241)

    def test_single_sample(self):
        """Model should handle single-sample input."""
        torch = pytest.importorskip("torch")
        from omega_pbpk.ml.models.surrogate.differentiable_ode import (
            DifferentiableODESurrogate,
        )

        model = DifferentiableODESurrogate(n_input=6, n_output=241)

        x = torch.randn(1, 6)
        y = model(x)

        assert y.shape == (1, 241)

    def test_predict_numpy(self):
        """predict() should accept numpy and return numpy."""
        pytest.importorskip("torch")
        from omega_pbpk.ml.models.surrogate.differentiable_ode import (
            DifferentiableODESurrogate,
        )

        model = DifferentiableODESurrogate(n_input=6, n_output=241)

        params = np.random.randn(10, 6)
        result = model.predict(params)

        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 241)

    def test_predict_1d_input(self):
        """predict() should handle 1D input (single sample)."""
        pytest.importorskip("torch")
        from omega_pbpk.ml.models.surrogate.differentiable_ode import (
            DifferentiableODESurrogate,
        )

        model = DifferentiableODESurrogate(n_input=6, n_output=241)

        params = np.random.randn(6)
        result = model.predict(params)

        assert isinstance(result, np.ndarray)
        assert result.shape == (241,)

    def test_custom_dimensions(self):
        """Model should accept custom input/output dimensions."""
        torch = pytest.importorskip("torch")
        from omega_pbpk.ml.models.surrogate.differentiable_ode import (
            DifferentiableODESurrogate,
        )

        model = DifferentiableODESurrogate(n_input=10, n_output=100, hidden_dim=128)

        x = torch.randn(16, 10)
        y = model(x)

        assert y.shape == (16, 100)

    def test_gradients_flow(self):
        """Gradients should flow through the model for backprop."""
        torch = pytest.importorskip("torch")
        from omega_pbpk.ml.models.surrogate.differentiable_ode import (
            DifferentiableODESurrogate,
        )

        model = DifferentiableODESurrogate(n_input=6, n_output=241)

        x = torch.randn(8, 6, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (8, 6)
