"""Tests for ADMET-AI wrapper -- unit conversions, error handling, mocking.

All ADMET-AI calls are mocked so tests run without admet-ai installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from omega_pbpk.prediction.adme_predictor import ADMEProperties

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_admet_model():
    """Create a mock ADMETModel that returns plausible predictions."""
    import pandas as pd

    mock_model = MagicMock()

    def _predict(**kwargs):
        """Return a DataFrame mimicking ADMET-AI output."""
        return pd.DataFrame(
            [
                {
                    "Lipophilicity_AstraZeneca": 2.5,  # logP
                    "PPBR_AZ": 90.0,  # 90% bound -> fup = 0.10
                    "Clearance_Hepatocyte_AZ": 54.8,  # uL/min/10^6 cells
                    "Caco2_Wang": -5.0,  # log10(Papp) in cm/s
                    "Solubility_AqSolDB": -3.5,  # log mol/L
                    "hERG": 0.3,  # probability of blocking
                    "CYP2D6_Substrate": 1.0,  # is substrate
                }
            ]
        )

    mock_model.predict = _predict
    return mock_model


@pytest.fixture
def mock_admet_model_non_substrate():
    """Mock with CYP2D6 non-substrate."""
    import pandas as pd

    mock_model = MagicMock()

    def _predict(**kwargs):
        return pd.DataFrame(
            [
                {
                    "Lipophilicity_AstraZeneca": 1.0,
                    "PPBR_AZ": 50.0,  # 50% bound -> fup = 0.50
                    "Clearance_Hepatocyte_AZ": 10.0,
                    "Caco2_Wang": -4.5,
                    "Solubility_AqSolDB": -2.0,
                    "hERG": 0.1,
                    "CYP2D6_Substrate": 0.0,
                }
            ]
        )

    mock_model.predict = _predict
    return mock_model


# ---------------------------------------------------------------------------
# Import helper: patch admet_ai import
# ---------------------------------------------------------------------------


def _create_predictor(mock_model):
    """Create ADMETAIPredictor with mocked admet-ai and rdkit dependencies."""
    import importlib
    import sys

    # Create a fake admet_ai module
    fake_admet_ai = MagicMock()
    fake_admet_ai.ADMETModel = MagicMock(return_value=mock_model)

    # We need to patch admet_ai AND _compute_mw_from_smiles at module level
    with patch.dict(sys.modules, {"admet_ai": fake_admet_ai}):
        import omega_pbpk.ml.models.adme.admet_ai_wrapper as wrapper_mod

        importlib.reload(wrapper_mod)
        predictor = wrapper_mod.ADMETAIPredictor()

    return predictor


def _predict_with_mock_mw(predictor, smiles, mw=180.16):
    """Call predictor.predict with _compute_mw_from_smiles mocked.

    We monkey-patch the module's global dict that the predictor's methods
    use, which is the same dict returned by the reloaded module.
    """
    # Access the actual module globals through the predictor's method
    method_globals = predictor._predict_single.__func__.__globals__
    orig = method_globals["_compute_mw_from_smiles"]
    method_globals["_compute_mw_from_smiles"] = lambda s: mw
    try:
        return predictor.predict(smiles)
    finally:
        method_globals["_compute_mw_from_smiles"] = orig


def _create_mock_with_predict(data_dict):
    """Create a mock model whose .predict(smiles=...) returns a DataFrame."""
    import pandas as pd

    mock = MagicMock()

    def _predict(**kwargs):
        return pd.DataFrame([data_dict])

    mock.predict = _predict
    return mock


# ---------------------------------------------------------------------------
# Tests: Unit Conversions
# ---------------------------------------------------------------------------


class TestUnitConversions:
    """Test all ADMET-AI -> ADMEProperties unit conversions."""

    def test_ppbr_to_fup(self, mock_admet_model):
        """PPBR 90% bound -> fup = 1 - 90/100 = 0.10."""
        predictor = _create_predictor(mock_admet_model)
        # Mock RDKit MW
        result = _predict_with_mock_mw(predictor, "CC(=O)Oc1ccccc1C(=O)O", 180.16)
        assert abs(result.fup - 0.10) < 0.01

    def test_ppbr_to_fup_zero_bound(self):
        """PPBR 0% bound -> fup = 1.0."""
        mock = _create_mock_with_predict(
            {
                "PPBR_AZ": 0.0,
                "Lipophilicity_AstraZeneca": 1.0,
                "Clearance_Hepatocyte_AZ": 10.0,
                "Caco2_Wang": -5.0,
                "Solubility_AqSolDB": -2.0,
                "hERG": 0.1,
                "CYP2D6_Substrate": 0.0,
            }
        )
        predictor = _create_predictor(mock)
        result = _predict_with_mock_mw(predictor, "CCO", 100.0)
        assert abs(result.fup - 1.0) < 0.01

    def test_ppbr_to_fup_full_bound(self):
        """PPBR 100% bound -> fup ~ 0.001 (clamped)."""
        mock = _create_mock_with_predict(
            {
                "PPBR_AZ": 100.0,
                "Lipophilicity_AstraZeneca": 4.0,
                "Clearance_Hepatocyte_AZ": 50.0,
                "Caco2_Wang": -5.0,
                "Solubility_AqSolDB": -4.0,
                "hERG": 0.5,
                "CYP2D6_Substrate": 1.0,
            }
        )
        predictor = _create_predictor(mock)
        result = _predict_with_mock_mw(predictor, "c1ccccc1", 400.0)
        assert result.fup == 0.001  # clamped minimum

    def test_hepatocyte_clearance_to_pmol(self, mock_admet_model):
        """Clearance_Hepatocyte 54.8 uL/min/10^6 cells -> ~0.01 uL/min/pmol CYP3A4."""
        from omega_pbpk.ml.models.adme.admet_ai_wrapper import HEPATOCYTE_TO_PMOL_CYP3A4

        predictor = _create_predictor(mock_admet_model)
        result = _predict_with_mock_mw(predictor, "CC(=O)Oc1ccccc1C(=O)O", 180.16)

        expected = 54.8 / HEPATOCYTE_TO_PMOL_CYP3A4
        assert abs(result.clint_3a4 - max(0.01, expected)) < 0.001

    def test_caco2_to_peff(self, mock_admet_model):
        """Caco2_Wang log(-5.0) -> Papp = 10^-5 cm/s -> peff = 10^-5 * 10^4 = 0.1 x10^-4 cm/s."""
        predictor = _create_predictor(mock_admet_model)
        result = _predict_with_mock_mw(predictor, "CC(=O)Oc1ccccc1C(=O)O", 180.16)

        # 10^(-5) * 10^4 = 0.1
        assert abs(result.peff - 0.1) < 0.05

    def test_logp_direct(self, mock_admet_model):
        """Lipophilicity maps directly to logP."""
        predictor = _create_predictor(mock_admet_model)
        result = _predict_with_mock_mw(predictor, "CC(=O)Oc1ccccc1C(=O)O", 180.16)
        assert abs(result.logP - 2.5) < 0.01

    def test_logs_direct(self, mock_admet_model):
        """Solubility_AqSolDB maps directly to logS."""
        predictor = _create_predictor(mock_admet_model)
        result = _predict_with_mock_mw(predictor, "CC(=O)Oc1ccccc1C(=O)O", 180.16)
        assert abs(result.logS - (-3.5)) < 0.01

    def test_herg_probability_to_ic50(self, mock_admet_model):
        """hERG probability 0.3 -> IC50 = 10^(2 - 3*0.3) = 10^1.1 ~ 12.6 uM."""
        predictor = _create_predictor(mock_admet_model)
        result = _predict_with_mock_mw(predictor, "CC(=O)Oc1ccccc1C(=O)O", 180.16)
        expected = 10.0 ** (2.0 - 3.0 * 0.3)
        assert abs(result.herg_ic50_uM - expected) < 1.0

    def test_cyp2d6_substrate(self, mock_admet_model):
        """CYP2D6_Substrate = 1 -> clint_2d6 = 5.0."""
        predictor = _create_predictor(mock_admet_model)
        result = _predict_with_mock_mw(predictor, "CC(=O)Oc1ccccc1C(=O)O", 180.16)
        assert result.clint_2d6 == 5.0

    def test_cyp2d6_non_substrate(self, mock_admet_model_non_substrate):
        """CYP2D6_Substrate = 0 -> clint_2d6 = 0.5."""
        predictor = _create_predictor(mock_admet_model_non_substrate)
        result = _predict_with_mock_mw(predictor, "CCO", 100.0)
        assert result.clint_2d6 == 0.5

    def test_rbp_default(self, mock_admet_model):
        """RBP should default to 1.0 (overridden by XGBoost in ensemble)."""
        predictor = _create_predictor(mock_admet_model)
        result = _predict_with_mock_mw(predictor, "CC(=O)Oc1ccccc1C(=O)O", 180.16)
        assert result.rbp == 1.0


class TestConfidenceAndIntervals:
    """Test confidence determination and conformal intervals."""

    def test_default_confidence_medium(self, mock_admet_model):
        """Without AD score, confidence should be 'medium'."""
        predictor = _create_predictor(mock_admet_model)
        result = _predict_with_mock_mw(predictor, "CC(=O)Oc1ccccc1C(=O)O", 180.16)
        assert result.confidence == "medium"

    def test_ad_score_high(self):
        """AD score > 0.8 -> confidence = 'high'."""
        mock = _create_mock_with_predict(
            {
                "Lipophilicity_AstraZeneca": 2.0,
                "PPBR_AZ": 50.0,
                "Clearance_Hepatocyte_AZ": 10.0,
                "Caco2_Wang": -5.0,
                "Solubility_AqSolDB": -2.0,
                "hERG": 0.1,
                "CYP2D6_Substrate": 0.0,
                "applicability_domain": 0.9,
            }
        )
        predictor = _create_predictor(mock)
        result = _predict_with_mock_mw(predictor, "c1ccccc1", 200.0)
        assert result.confidence == "high"

    def test_intervals_default_50pct(self, mock_admet_model):
        """Without ADMET-AI intervals, use +/-50% default."""
        predictor = _create_predictor(mock_admet_model)
        result = _predict_with_mock_mw(predictor, "CC(=O)Oc1ccccc1C(=O)O", 180.16)
        # fup interval should be around [fup*0.5, fup*1.5]
        assert result.fup_lo < result.fup
        assert result.fup_hi > result.fup

    def test_returns_adme_properties(self, mock_admet_model):
        """Result should be an ADMEProperties instance with all required fields."""
        predictor = _create_predictor(mock_admet_model)
        result = _predict_with_mock_mw(predictor, "CC(=O)Oc1ccccc1C(=O)O", 180.16)
        assert isinstance(result, ADMEProperties)
        assert result.mw > 0
        assert result.logP is not None
        assert 0.0 <= result.fup <= 1.0
        assert 0.5 <= result.rbp <= 3.0
        assert result.clint_3a4 > 0
        assert result.clint_2d6 > 0
        assert result.herg_ic50_uM > 0


class TestErrorHandling:
    """Test error handling for invalid inputs and missing deps."""

    def test_import_error_without_admet_ai(self):
        """Should raise ImportError if admet-ai is not installed."""
        import sys

        # Remove any cached admet_ai module
        mods_to_remove = [k for k in sys.modules if k.startswith("admet_ai")]
        for k in mods_to_remove:
            del sys.modules[k]

        with pytest.raises(ImportError, match="admet-ai"):
            from omega_pbpk.ml.models.adme.admet_ai_wrapper import ADMETAIPredictor

            ADMETAIPredictor()

    def test_invalid_smiles_raises_value_error(self, mock_admet_model):
        """Invalid SMILES should raise ValueError from MW computation."""
        predictor = _create_predictor(mock_admet_model)
        with patch(
            "omega_pbpk.ml.models.adme.admet_ai_wrapper._compute_mw_from_smiles",
            side_effect=ValueError("Invalid SMILES: INVALID"),
        ):
            with pytest.raises(ValueError, match="Invalid SMILES"):
                predictor.predict("INVALID")  # noqa: SIM117

    def test_admet_ai_failure_raises(self, mock_admet_model):
        """If ADMET-AI predict() fails, should propagate as ValueError."""
        mock_admet_model.predict = MagicMock(side_effect=RuntimeError("ADMET-AI crash"))
        predictor = _create_predictor(mock_admet_model)
        with patch(
            "omega_pbpk.ml.models.adme.admet_ai_wrapper._compute_mw_from_smiles",
            return_value=180.0,
        ):
            with pytest.raises(ValueError, match="ADMET-AI prediction failed"):
                predictor.predict("CC(=O)Oc1ccccc1C(=O)O")  # noqa: SIM117


class TestBatchPrediction:
    """Test batch prediction."""

    def test_predict_batch(self, mock_admet_model):
        """predict_batch should return list of ADMEProperties."""
        predictor = _create_predictor(mock_admet_model)
        with patch(
            "omega_pbpk.ml.models.adme.admet_ai_wrapper._compute_mw_from_smiles",
            return_value=180.16,
        ):
            results = predictor.predict_batch(["CCO", "c1ccccc1"])  # noqa: SIM117
        assert len(results) == 2
        assert all(isinstance(r, ADMEProperties) for r in results)


class TestMissingProperties:
    """Test handling of missing properties in ADMET-AI output."""

    def test_missing_lipophilicity_defaults_to_2(self):
        """Missing Lipophilicity -> logP defaults to 2.0."""
        mock = _create_mock_with_predict(
            {
                "PPBR_AZ": 50.0,
                "Clearance_Hepatocyte_AZ": 10.0,
                "Caco2_Wang": -5.0,
                "Solubility_AqSolDB": -2.0,
                "hERG": 0.1,
                "CYP2D6_Substrate": 0.0,
            }
        )
        predictor = _create_predictor(mock)
        result = _predict_with_mock_mw(predictor, "CCO", 100.0)
        assert result.logP == 2.0

    def test_missing_ppbr_defaults_fup(self):
        """Missing PPBR -> fup defaults to 0.1."""
        mock = _create_mock_with_predict(
            {
                "Lipophilicity_AstraZeneca": 2.0,
                "Clearance_Hepatocyte_AZ": 10.0,
                "Caco2_Wang": -5.0,
                "Solubility_AqSolDB": -2.0,
                "hERG": 0.1,
                "CYP2D6_Substrate": 0.0,
            }
        )
        predictor = _create_predictor(mock)
        result = _predict_with_mock_mw(predictor, "CCO", 100.0)
        assert result.fup == 0.1
