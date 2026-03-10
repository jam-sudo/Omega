"""Tests for XGBoost fup (fraction unbound) predictor."""

from __future__ import annotations

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost", reason="XGBoost not installed")


@pytest.fixture(scope="module")
def fup_predictor():
    """Create and train the XGBoost fup predictor once for all tests."""
    from omega_pbpk.ml.models.adme.xgboost_fup import XGBoostFupPredictor

    return XGBoostFupPredictor()


class TestXGBoostFupPredictor:
    """Tests for XGBoostFupPredictor."""

    def test_predict_returns_float_in_range(self, fup_predictor):
        fup = fup_predictor.predict_fup("CC(=O)Nc1ccc(O)cc1")  # acetaminophen
        assert isinstance(fup, float)
        assert 0.001 <= fup <= 1.0

    def test_predict_with_interval(self, fup_predictor):
        fup, lo, hi = fup_predictor.predict_fup_with_interval("CC(=O)Nc1ccc(O)cc1")
        assert lo <= fup <= hi
        assert lo >= 0.001
        assert hi <= 1.0

    def test_highly_bound_drug(self, fup_predictor):
        """Ibuprofen: actual fup ≈ 0.005. Should predict < 0.05."""
        fup = fup_predictor.predict_fup("CC(C)Cc1ccc(C(C)C(=O)O)cc1")
        assert fup < 0.05, f"Ibuprofen fup={fup}, expected < 0.05 (actual ~0.005)"

    def test_unbound_drug(self, fup_predictor):
        """Metformin: actual fup ≈ 1.0. Should predict > 0.3."""
        fup = fup_predictor.predict_fup("CN(C)C(=N)NC(=N)N")
        assert fup > 0.3, f"Metformin fup={fup}, expected > 0.3 (actual ~1.0)"

    def test_moderate_binding(self, fup_predictor):
        """Propranolol: actual fup ≈ 0.13. Should predict in [0.02, 0.8]."""
        fup = fup_predictor.predict_fup("CC(C)NCC(O)COc1cccc2ccccc12")
        assert 0.02 < fup < 0.8, f"Propranolol fup={fup}, expected ~0.13"

    def test_invalid_smiles_raises(self, fup_predictor):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            fup_predictor.predict_fup("not_a_molecule")

    def test_cv_metrics_populated(self, fup_predictor):
        metrics = fup_predictor.cv_metrics
        assert "mae_log10" in metrics
        assert "r2" in metrics
        assert "n_compounds" in metrics
        assert metrics["n_compounds"] > 100

    def test_conformal_interval_covers_known_drugs(self, fup_predictor):
        """At 90% confidence, conformal interval should cover most known drugs."""
        test_cases = [
            ("CC(C)Cc1ccc(C(C)C(=O)O)cc1", 0.005),  # ibuprofen
            ("CC(=O)Nc1ccc(O)cc1", 0.83),  # acetaminophen
            ("Cn1c(=O)c2[nH]cnc2n(C)c1=O", 0.60),  # theophylline
            ("CN(C)C(=N)NC(=N)N", 1.0),  # metformin
        ]
        covered = 0
        for smi, actual_fup in test_cases:
            _, lo, hi = fup_predictor.predict_fup_with_interval(smi, confidence=0.9)
            if lo <= actual_fup <= hi:
                covered += 1
        # At least 3/4 should be covered at 90% confidence
        assert covered >= 3, f"Only {covered}/4 drugs covered by 90% CI"

    def test_log_space_accuracy(self, fup_predictor):
        """Average fold error across test drugs should be < 5x."""
        test_cases = [
            ("CC(C)Cc1ccc(C(C)C(=O)O)cc1", 0.005),  # ibuprofen
            ("CC(=O)Nc1ccc(O)cc1", 0.83),  # acetaminophen
            ("Cn1c(=O)c2[nH]cnc2n(C)c1=O", 0.60),  # theophylline
            ("CC(C)NCC(O)COc1cccc2ccccc12", 0.13),  # propranolol
            ("CN(C)C(=N)NC(=N)N", 1.0),  # metformin
        ]
        fold_errors = []
        for smi, actual in test_cases:
            pred = fup_predictor.predict_fup(smi)
            fe = max(pred / actual, actual / pred)
            fold_errors.append(fe)

        aafe = float(np.exp(np.mean(np.log(fold_errors))))
        assert aafe < 5.0, f"AAFE={aafe:.2f}, expected < 5.0"
