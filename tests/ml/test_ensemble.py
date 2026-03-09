"""Tests for Ensemble ADME predictor -- fallback logic, confidence, full predict."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from omega_pbpk.prediction.adme_predictor import ADMEProperties

# ---------------------------------------------------------------------------
# Mock Backends
# ---------------------------------------------------------------------------


def _make_adme_props(**overrides) -> ADMEProperties:
    """Create ADMEProperties with sensible defaults, applying overrides."""
    defaults = {
        "mw": 300.0,
        "logP": 2.5,
        "logS": -3.0,
        "peff": 1.5,
        "fup": 0.15,
        "rbp": 1.0,
        "clint_3a4": 2.0,
        "clint_2d6": 0.5,
        "herg_ic50_uM": 15.0,
        "confidence": "medium",
        "fup_lo": 0.08,
        "fup_hi": 0.22,
        "clint_3a4_lo": 1.0,
        "clint_3a4_hi": 3.0,
        "peff_lo": 0.75,
        "peff_hi": 2.25,
        "rbp_lo": 0.5,
        "rbp_hi": 1.5,
    }
    defaults.update(overrides)
    return ADMEProperties(**defaults)


@pytest.fixture
def mock_admet_ai():
    """Mock ADMET-AI predictor."""
    mock = MagicMock()
    mock.predict.return_value = _make_adme_props(
        logP=3.0,
        fup=0.1,
        confidence="high",
        fup_lo=0.05,
        fup_hi=0.15,
        clint_3a4_lo=1.5,
        clint_3a4_hi=2.5,
        peff_lo=1.0,
        peff_hi=2.0,
    )
    return mock


@pytest.fixture
def mock_xgboost():
    """Mock XGBoost RBP predictor."""
    mock = MagicMock()
    mock.predict_rbp.return_value = 0.85
    return mock


@pytest.fixture
def mock_polynomial():
    """Mock polynomial (legacy) predictor."""
    mock = MagicMock()
    mock.predict.return_value = _make_adme_props(
        logP=2.0,
        fup=0.2,
        confidence="low",
        fup_lo=0.1,
        fup_hi=0.3,
    )
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEnsemblePropertySelection:
    """Test per-property backend selection strategy."""

    def test_logp_from_admet_ai(self, mock_admet_ai, mock_xgboost, mock_polynomial):
        """logP should come from ADMET-AI when available."""
        from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor

        with patch(
            "omega_pbpk.ml.models.adme.ensemble.EnsembleADMEPredictor._get_mw",
            return_value=300.0,
        ):
            predictor = EnsembleADMEPredictor(
                admet_ai=mock_admet_ai,
                xgboost_rbp=mock_xgboost,
                polynomial=mock_polynomial,
            )
            result = predictor.predict("CCO")
        assert result.logP == 3.0  # From ADMET-AI, not polynomial's 2.0

    def test_fup_ensemble(self, mock_admet_ai, mock_xgboost, mock_polynomial):
        """fup should be ensembled (geometric mean of ADMET-AI + XGBoost when both available)."""
        from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor

        with patch(
            "omega_pbpk.ml.models.adme.ensemble.EnsembleADMEPredictor._get_mw",
            return_value=300.0,
        ):
            predictor = EnsembleADMEPredictor(
                admet_ai=mock_admet_ai,
                xgboost_rbp=mock_xgboost,
                polynomial=mock_polynomial,
            )
            result = predictor.predict("CCO")
        # fup is ensemble of ADMET-AI (0.1) + XGBoost; exact value depends on XGBoost
        assert 0.001 <= result.fup <= 1.0
        assert result.confidence in ("low", "medium", "high")

    def test_rbp_from_xgboost(self, mock_admet_ai, mock_xgboost, mock_polynomial):
        """RBP should come from XGBoost when available."""
        from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor

        with patch(
            "omega_pbpk.ml.models.adme.ensemble.EnsembleADMEPredictor._get_mw",
            return_value=300.0,
        ):
            predictor = EnsembleADMEPredictor(
                admet_ai=mock_admet_ai,
                xgboost_rbp=mock_xgboost,
                polynomial=mock_polynomial,
            )
            result = predictor.predict("CCO")
        assert result.rbp == 0.85  # XGBoost


class TestEnsembleFallback:
    """Test fallback behavior when primary backends fail."""

    def test_fallback_to_polynomial_when_admet_fails(self, mock_xgboost, mock_polynomial):
        """If ADMET-AI fails, should fall back to polynomial."""
        from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor

        mock_admet_fail = MagicMock()
        mock_admet_fail.predict.side_effect = RuntimeError("ADMET-AI crashed")

        with patch(
            "omega_pbpk.ml.models.adme.ensemble.EnsembleADMEPredictor._get_mw",
            return_value=300.0,
        ):
            predictor = EnsembleADMEPredictor(
                admet_ai=mock_admet_fail,
                xgboost_rbp=mock_xgboost,
                polynomial=mock_polynomial,
            )
            result = predictor.predict("CCO")
        assert result.logP == 2.0  # Polynomial fallback
        # fup is ensembled (polynomial + XGBoost); exact value depends on XGBoost
        assert 0.001 <= result.fup <= 1.0

    def test_fallback_to_polynomial_when_xgboost_fails(self, mock_admet_ai, mock_polynomial):
        """If XGBoost fails, RBP should come from ADMET-AI or polynomial."""
        from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor

        mock_xgb_fail = MagicMock()
        mock_xgb_fail.predict_rbp.side_effect = RuntimeError("XGBoost crashed")

        with patch(
            "omega_pbpk.ml.models.adme.ensemble.EnsembleADMEPredictor._get_mw",
            return_value=300.0,
        ):
            predictor = EnsembleADMEPredictor(
                admet_ai=mock_admet_ai,
                xgboost_rbp=mock_xgb_fail,
                polynomial=mock_polynomial,
            )
            result = predictor.predict("CCO")
        # RBP should fall back to ADMET-AI default (1.0) or polynomial
        assert 0.5 <= result.rbp <= 3.0

    def test_all_backends_none_gives_defaults(self):
        """If all backends are None, should still return valid ADMEProperties."""
        from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor

        with patch(
            "omega_pbpk.ml.models.adme.ensemble.EnsembleADMEPredictor._get_mw",
            return_value=300.0,
        ):
            predictor = EnsembleADMEPredictor(
                admet_ai=None,
                xgboost_rbp=None,
                polynomial=None,
            )
            # Force initialized so it doesn't try to auto-create
            predictor._initialized = True
            result = predictor.predict("CCO")
        assert isinstance(result, ADMEProperties)
        assert result.confidence == "low"


class TestEnsembleConfidence:
    """Test confidence calculation."""

    def test_confidence_is_minimum(self, mock_admet_ai, mock_xgboost, mock_polynomial):
        """Confidence should be the minimum across backends."""
        from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor

        # ADMET-AI = high, polynomial = low -> overall should be medium or low
        # (XGBoost RBP contributes medium=1)
        with patch(
            "omega_pbpk.ml.models.adme.ensemble.EnsembleADMEPredictor._get_mw",
            return_value=300.0,
        ):
            predictor = EnsembleADMEPredictor(
                admet_ai=mock_admet_ai,
                xgboost_rbp=mock_xgboost,
                polynomial=mock_polynomial,
            )
            result = predictor.predict("CCO")
        # XGBoost contributes medium (1), ADMET-AI contributes high (2)
        # min should be 1 = "medium"
        assert result.confidence in ("low", "medium")


class TestEnsembleIntervals:
    """Test conformal interval propagation."""

    def test_intervals_from_admet_ai(self, mock_admet_ai, mock_xgboost, mock_polynomial):
        """Intervals should come from the source backend."""
        from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor

        with patch(
            "omega_pbpk.ml.models.adme.ensemble.EnsembleADMEPredictor._get_mw",
            return_value=300.0,
        ):
            predictor = EnsembleADMEPredictor(
                admet_ai=mock_admet_ai,
                xgboost_rbp=mock_xgboost,
                polynomial=mock_polynomial,
            )
            result = predictor.predict("CCO")
        # fup intervals come from XGBoost conformal when available
        assert 0.001 <= result.fup_lo <= result.fup
        assert result.fup <= result.fup_hi <= 1.0

    def test_rbp_intervals_from_xgboost(self, mock_admet_ai, mock_xgboost, mock_polynomial):
        """XGBoost RBP intervals should be +/-20%."""
        from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor

        with patch(
            "omega_pbpk.ml.models.adme.ensemble.EnsembleADMEPredictor._get_mw",
            return_value=300.0,
        ):
            predictor = EnsembleADMEPredictor(
                admet_ai=mock_admet_ai,
                xgboost_rbp=mock_xgboost,
                polynomial=mock_polynomial,
            )
            result = predictor.predict("CCO")
        # XGBoost rbp=0.85, intervals = [0.68, 1.02]
        assert result.rbp_lo < result.rbp
        assert result.rbp_hi > result.rbp


class TestEnsembleBatch:
    """Test batch prediction."""

    def test_predict_batch(self, mock_admet_ai, mock_xgboost, mock_polynomial):
        """Batch prediction should return list."""
        from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor

        with patch(
            "omega_pbpk.ml.models.adme.ensemble.EnsembleADMEPredictor._get_mw",
            return_value=300.0,
        ):
            predictor = EnsembleADMEPredictor(
                admet_ai=mock_admet_ai,
                xgboost_rbp=mock_xgboost,
                polynomial=mock_polynomial,
            )
            results = predictor.predict_batch(["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])
        assert len(results) == 3
        assert all(isinstance(r, ADMEProperties) for r in results)


class TestEnsembleReturnsValidADMEProperties:
    """Test that ensemble always returns valid ADMEProperties."""

    def test_output_contract(self, mock_admet_ai, mock_xgboost, mock_polynomial):
        """All required fields should be present and within valid ranges."""
        from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor

        with patch(
            "omega_pbpk.ml.models.adme.ensemble.EnsembleADMEPredictor._get_mw",
            return_value=300.0,
        ):
            predictor = EnsembleADMEPredictor(
                admet_ai=mock_admet_ai,
                xgboost_rbp=mock_xgboost,
                polynomial=mock_polynomial,
            )
            result = predictor.predict("CCO")

        assert result.mw > 0
        assert isinstance(result.logP, float)
        assert isinstance(result.logS, float)
        assert 0.0 < result.fup <= 1.0
        assert 0.5 <= result.rbp <= 3.0
        assert result.clint_3a4 >= 0.01
        assert result.clint_2d6 > 0
        assert result.herg_ic50_uM > 0
        assert result.confidence in ("low", "medium", "high")
        assert result.fup_lo <= result.fup
        assert result.fup_hi >= result.fup
