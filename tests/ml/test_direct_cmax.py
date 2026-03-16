"""Tests for DirectCmaxPredictor and feature extraction."""

from __future__ import annotations

import numpy as np

from omega_pbpk.ml.models.direct_pk.xgboost_cmax import (
    DirectCmaxPredictor,
    smiles_to_features,
)

CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


def test_features_extraction():
    """smiles_to_features returns a 2057-length array."""
    feat = smiles_to_features(CAFFEINE)
    assert feat is not None
    assert feat.shape == (2057,)
    assert feat.dtype == np.float32


def test_features_invalid_smiles():
    """Invalid SMILES returns None."""
    assert smiles_to_features("INVALID_SMILES_XYZ") is None


def test_predict_returns_positive():
    """Caffeine prediction should be > 0."""
    predictor = DirectCmaxPredictor()
    cmax = predictor.predict(CAFFEINE, dose_mg=100.0)
    assert cmax > 0


def test_batch_predict():
    """Batch prediction should return positive values for all drugs."""
    predictor = DirectCmaxPredictor()
    results = predictor.predict_batch([CAFFEINE, IBUPROFEN], dose_mg=100.0)
    assert len(results) == 2
    assert all(r > 0 for r in results)
