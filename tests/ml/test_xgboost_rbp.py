"""Tests for XGBoost RBP predictor -- training, prediction, save/load."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory with a small reference CSV."""
    csv_path = tmp_path / "adme_reference.csv"
    rows = [
        {"name": "aspirin", "mw": "180.16", "logP": "1.2", "fup": "0.5", "rbp": "0.7", "clint_3a4_uL_min_pmol": "0.5", "peff_cm_s": "1.5e-4"},
        {"name": "caffeine", "mw": "194.2", "logP": "-0.07", "fup": "0.65", "rbp": "0.72", "clint_3a4_uL_min_pmol": "0.5", "peff_cm_s": "0.9e-4"},
        {"name": "midazolam", "mw": "325.8", "logP": "3.89", "fup": "0.035", "rbp": "0.55", "clint_3a4_uL_min_pmol": "11.2", "peff_cm_s": "2.8e-4"},
        {"name": "warfarin", "mw": "308.3", "logP": "2.70", "fup": "0.005", "rbp": "0.72", "clint_3a4_uL_min_pmol": "0.9", "peff_cm_s": "1.5e-4"},
        {"name": "propranolol", "mw": "259.3", "logP": "3.48", "fup": "0.13", "rbp": "0.83", "clint_3a4_uL_min_pmol": "11.0", "peff_cm_s": "3.2e-4"},
        {"name": "metformin", "mw": "129.2", "logP": "-1.43", "fup": "1.00", "rbp": "0.95", "clint_3a4_uL_min_pmol": "0.01", "peff_cm_s": "0.06e-4"},
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return tmp_path, csv_path


# ---------------------------------------------------------------------------
# Tests that don't require xgboost/rdkit
# ---------------------------------------------------------------------------


class TestDependencyChecks:
    """Test that missing dependencies are handled gracefully."""

    def test_import_error_without_xgboost(self):
        """Should raise ImportError if xgboost is not installed."""
        try:
            import xgboost  # noqa: F401

            pytest.skip("xgboost is installed")
        except ImportError:
            pass

        with pytest.raises(ImportError, match="XGBoost"):
            from omega_pbpk.ml.models.adme.xgboost_adme import XGBoostRBPPredictor

            XGBoostRBPPredictor(auto_load=False)


class TestNameToSmiles:
    """Test the drug name -> SMILES lookup."""

    def test_known_drugs(self):
        from omega_pbpk.ml.models.adme.xgboost_adme import _name_to_smiles

        assert _name_to_smiles("aspirin") != ""
        assert _name_to_smiles("caffeine") != ""
        assert _name_to_smiles("midazolam") != ""

    def test_unknown_drug_returns_empty(self):
        from omega_pbpk.ml.models.adme.xgboost_adme import _name_to_smiles

        assert _name_to_smiles("unknowndrugxyz") == ""


class TestFingerprintFunction:
    """Test Morgan fingerprint computation."""

    def test_valid_smiles(self):
        try:
            from rdkit import Chem  # noqa: F401
        except ImportError:
            pytest.skip("RDKit not installed")

        from omega_pbpk.ml.models.adme.xgboost_adme import _smiles_to_fingerprint

        fp = _smiles_to_fingerprint("CCO")
        assert fp is not None
        assert fp.shape == (2048,)
        assert fp.dtype == np.float32
        assert np.all((fp == 0) | (fp == 1))

    def test_invalid_smiles_returns_none(self):
        try:
            from rdkit import Chem  # noqa: F401
        except ImportError:
            pytest.skip("RDKit not installed")

        from omega_pbpk.ml.models.adme.xgboost_adme import _smiles_to_fingerprint

        assert _smiles_to_fingerprint("INVALID_SMILES") is None


class TestXGBoostRBPPredictor:
    """Tests for the XGBoost RBP predictor (require xgboost + rdkit)."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        """Skip all tests in this class if xgboost or rdkit not installed."""
        try:
            import xgboost  # noqa: F401
            from rdkit import Chem  # noqa: F401
        except ImportError:
            pytest.skip("xgboost and/or rdkit not installed")

    def test_train_and_predict(self, temp_data_dir):
        """Train on small data and make a prediction."""
        from omega_pbpk.ml.models.adme.xgboost_adme import XGBoostRBPPredictor

        tmp_path, csv_path = temp_data_dir
        model_dir = tmp_path / "models" / "xgboost_rbp"

        predictor = XGBoostRBPPredictor(
            data_path=csv_path,
            model_dir=model_dir,
            auto_load=False,
        )
        metrics = predictor.train(n_folds=2)

        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["n_compounds"] == 6

        # Predict for aspirin
        rbp = predictor.predict_rbp("CC(=O)Oc1ccccc1C(=O)O")
        assert 0.5 <= rbp <= 3.0

    def test_model_save_load(self, temp_data_dir):
        """Model should be saveable and loadable."""
        from omega_pbpk.ml.models.adme.xgboost_adme import XGBoostRBPPredictor

        tmp_path, csv_path = temp_data_dir
        model_dir = tmp_path / "models" / "xgboost_rbp"

        # Train and save
        predictor1 = XGBoostRBPPredictor(
            data_path=csv_path,
            model_dir=model_dir,
            auto_load=False,
        )
        predictor1.train()

        # Verify model file exists
        assert (model_dir / "model.json").exists()
        assert (model_dir / "meta.json").exists()

        # Load and predict
        predictor2 = XGBoostRBPPredictor(
            data_path=csv_path,
            model_dir=model_dir,
            auto_load=True,  # Should load cached model
        )
        rbp = predictor2.predict_rbp("CC(=O)Oc1ccccc1C(=O)O")
        assert 0.5 <= rbp <= 3.0

    def test_invalid_smiles_raises(self, temp_data_dir):
        """Invalid SMILES should raise ValueError."""
        from omega_pbpk.ml.models.adme.xgboost_adme import XGBoostRBPPredictor

        tmp_path, csv_path = temp_data_dir
        model_dir = tmp_path / "models" / "xgboost_rbp"

        predictor = XGBoostRBPPredictor(
            data_path=csv_path,
            model_dir=model_dir,
            auto_load=False,
        )
        predictor.train()

        with pytest.raises(ValueError, match="Invalid SMILES"):
            predictor.predict_rbp("NOT_A_SMILES")

    def test_predict_without_training_raises(self, temp_data_dir):
        """Predicting without training should raise RuntimeError."""
        from omega_pbpk.ml.models.adme.xgboost_adme import XGBoostRBPPredictor

        tmp_path, csv_path = temp_data_dir
        model_dir = tmp_path / "models_empty"

        predictor = XGBoostRBPPredictor(
            data_path=csv_path,
            model_dir=model_dir,
            auto_load=False,
        )

        with pytest.raises(RuntimeError, match="not trained"):
            predictor.predict_rbp("CCO")

    def test_batch_prediction(self, temp_data_dir):
        """Batch prediction should work for multiple SMILES."""
        from omega_pbpk.ml.models.adme.xgboost_adme import XGBoostRBPPredictor

        tmp_path, csv_path = temp_data_dir
        model_dir = tmp_path / "models" / "xgboost_rbp"

        predictor = XGBoostRBPPredictor(
            data_path=csv_path,
            model_dir=model_dir,
            auto_load=False,
        )
        predictor.train()

        results = predictor.predict_rbp_batch(["CCO", "c1ccccc1"])
        assert len(results) == 2
        assert all(0.5 <= r <= 3.0 for r in results)

    def test_cv_metrics_property(self, temp_data_dir):
        """cv_metrics property should return dict after training."""
        from omega_pbpk.ml.models.adme.xgboost_adme import XGBoostRBPPredictor

        tmp_path, csv_path = temp_data_dir
        model_dir = tmp_path / "models" / "xgboost_rbp"

        predictor = XGBoostRBPPredictor(
            data_path=csv_path,
            model_dir=model_dir,
            auto_load=False,
        )
        predictor.train()

        metrics = predictor.cv_metrics
        assert "mae" in metrics
        assert "r2" in metrics
