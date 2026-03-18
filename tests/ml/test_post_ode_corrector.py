"""Post-ODE Residual Corrector tests."""

import numpy as np
import pytest


def test_post_ode_corrector_untrained_returns_zeros():
    """When not trained, corrector should return zero correction."""
    from omega_pbpk.ml.corrections.post_ode_corrector import PostODECorrector

    corrector = PostODECorrector()
    if corrector.is_trained:
        pytest.skip("Model already trained")

    correction = corrector.predict(
        smiles="c1ccccc1",
        ode_cmax=1.0,
        ode_auc=10.0,
        adme_pred={"fup": 0.5, "clint": 5.0, "logP": 2.0, "peff": 1.0},
    )
    assert correction["cmax_correction_log"] == 0.0
    assert correction["auc_correction_log"] == 0.0


def test_post_ode_corrector_corrections_bounded():
    """Corrections must be bounded to [-1.0, 1.0] log units."""
    from omega_pbpk.ml.corrections.post_ode_corrector import PostODECorrector

    corrector = PostODECorrector()
    if not corrector.is_trained:
        pytest.skip("Model not yet trained")

    correction = corrector.predict(
        smiles="c1ccc2c(c1)c(=O)c1c(n2C)cc(Cl)c(F)c1",  # midazolam-like
        ode_cmax=0.05,
        ode_auc=1.0,
        adme_pred={"fup": 0.04, "clint": 5.0, "logP": 3.0, "peff": 5.0},
    )
    assert -1.0 <= correction["cmax_correction_log"] <= 1.0
    assert -1.0 <= correction["auc_correction_log"] <= 1.0


def test_post_ode_corrector_interface():
    """Corrector should accept ODE output + features and return correction."""
    from omega_pbpk.ml.corrections.post_ode_corrector import PostODECorrector

    corrector = PostODECorrector()
    correction = corrector.predict(
        smiles="CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        ode_cmax=1.0,
        ode_auc=10.0,
        adme_pred={"fup": 0.5, "clint": 5.0, "logP": 2.0, "peff": 1.0},
    )
    assert "cmax_correction_log" in correction
    assert "auc_correction_log" in correction


def test_post_ode_corrector_missing_adme_keys():
    """Corrector should handle missing keys in adme_pred gracefully."""
    from omega_pbpk.ml.corrections.post_ode_corrector import PostODECorrector

    corrector = PostODECorrector()
    # Minimal adme_pred — all optional keys missing
    correction = corrector.predict(
        smiles="c1ccccc1",
        ode_cmax=1.0,
        ode_auc=10.0,
        adme_pred={},
    )
    assert "cmax_correction_log" in correction
    assert "auc_correction_log" in correction


def test_post_ode_corrector_invalid_smiles():
    """Invalid SMILES should return zero corrections."""
    from omega_pbpk.ml.corrections.post_ode_corrector import PostODECorrector

    corrector = PostODECorrector()
    correction = corrector.predict(
        smiles="NOT_A_VALID_SMILES",
        ode_cmax=1.0,
        ode_auc=10.0,
        adme_pred={"fup": 0.5, "clint": 5.0, "logP": 2.0, "peff": 1.0},
    )
    assert correction["cmax_correction_log"] == 0.0
    assert correction["auc_correction_log"] == 0.0


def test_post_ode_corrector_feature_extraction():
    """Feature extraction should produce correct dimensionality."""
    from omega_pbpk.ml.corrections.post_ode_corrector import (
        N_FEATURES,
        PostODECorrector,
    )

    corrector = PostODECorrector()
    features = corrector._extract_features(
        smiles="CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        ode_cmax=1.0,
        ode_auc=10.0,
        adme_pred={"fup": 0.5, "clint": 5.0, "logP": 2.0, "peff": 1.0},
    )
    assert features is not None
    assert features.shape == (N_FEATURES,)
    # All values should be finite
    assert np.all(np.isfinite(features))


def test_post_ode_corrector_zero_ode_values():
    """Zero or negative ODE values should be handled gracefully."""
    from omega_pbpk.ml.corrections.post_ode_corrector import PostODECorrector

    corrector = PostODECorrector()
    correction = corrector.predict(
        smiles="c1ccccc1",
        ode_cmax=0.0,
        ode_auc=0.0,
        adme_pred={"fup": 0.5, "clint": 5.0, "logP": 2.0, "peff": 1.0},
    )
    assert "cmax_correction_log" in correction
    assert "auc_correction_log" in correction
    # Should not crash or return NaN
    assert np.isfinite(correction["cmax_correction_log"])
    assert np.isfinite(correction["auc_correction_log"])
