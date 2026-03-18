"""Pre-ODE ADME Corrector tests."""

import numpy as np
import pytest


def test_pre_ode_corrector_finite_diff_gradient():
    """Finite-difference gradient computation should work."""
    from omega_pbpk.ml.corrections.pre_ode_corrector import _compute_fd_gradient

    # Mock: simple quadratic loss
    def mock_ode_cmax(fup, clint):
        return 1.0 / (fup * clint + 0.01)

    grad_fup, grad_clint = _compute_fd_gradient(mock_ode_cmax, fup=0.5, clint=10.0, epsilon=0.01)
    assert np.isfinite(grad_fup)
    assert np.isfinite(grad_clint)
    assert grad_fup != 0.0


def test_pre_ode_corrector_untrained_returns_zeros():
    """When not trained, corrector should return zero corrections."""
    from omega_pbpk.ml.corrections.pre_ode_corrector import PreODECorrector

    corrector = PreODECorrector()
    if corrector.is_trained:
        pytest.skip("Model already trained")

    delta = corrector.predict("c1ccccc1")  # benzene
    assert delta["delta_log_fup"] == 0.0
    assert delta["delta_log_clint"] == 0.0
    assert delta["delta_log_peff"] == 0.0


def test_pre_ode_corrector_corrections_bounded():
    """Corrections must be bounded to [-0.5, 0.5] log units."""
    from omega_pbpk.ml.corrections.pre_ode_corrector import PreODECorrector

    corrector = PreODECorrector()
    if not corrector.is_trained:
        pytest.skip("Model not yet trained")

    delta = corrector.predict("c1ccc2c(c1)c(=O)c1c(n2C)cc(Cl)c(F)c1")  # midazolam
    assert -0.5 <= delta["delta_log_fup"] <= 0.5
    assert -0.5 <= delta["delta_log_clint"] <= 0.5
    assert -0.5 <= delta["delta_log_peff"] <= 0.5
