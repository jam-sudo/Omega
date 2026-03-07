"""Tests for Phase 361 — Tissue Binding Predictor (suffix _p361)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.tissue_binding_predictor_p361 import (
    TissueBindingResult,
    predict_tissue_binding,
)

_TISSUES = {"muscle", "adipose", "liver", "kidney", "brain", "lung", "heart"}

# ---------------------------------------------------------------------------
# Helper compound fixtures
# ---------------------------------------------------------------------------


def _lipophilic_base() -> TissueBindingResult:
    """High logP basic compound — lipid-loving."""
    return predict_tissue_binding(
        drug_name="LipBase",
        logP=4.5,
        pKa=9.0,
        drug_type="base",
        fu_plasma=0.1,
        mw_Da=380.0,
    )


def _hydrophilic_acid() -> TissueBindingResult:
    """Low logP acidic compound — plasma-bound."""
    return predict_tissue_binding(
        drug_name="HydroAcid",
        logP=-1.0,
        pKa=4.0,
        drug_type="acid",
        fu_plasma=0.05,
        mw_Da=250.0,
    )


def _neutral() -> TissueBindingResult:
    """Neutral drug, moderate logP."""
    return predict_tissue_binding(
        drug_name="Neutral",
        logP=2.0,
        pKa=7.0,
        drug_type="neutral",
        fu_plasma=0.5,
        mw_Da=300.0,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_tissue_binding_result(self):
        result = _lipophilic_base()
        assert isinstance(result, TissueBindingResult)

    def test_drug_name_preserved(self):
        result = predict_tissue_binding("TestDrug", 2.0, 7.0, "neutral", 0.5, 300.0)
        assert result.drug_name == "TestDrug"

    def test_logp_preserved(self):
        result = _neutral()
        assert result.logP == pytest.approx(2.0)

    def test_fu_plasma_preserved(self):
        result = _neutral()
        assert result.fu_plasma == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# kp_tissues dict structure
# ---------------------------------------------------------------------------


class TestKpTissuesDict:
    def test_kp_tissues_has_seven_keys(self):
        result = _neutral()
        assert len(result.kp_tissues) == 7

    def test_kp_tissues_has_correct_tissue_names(self):
        result = _neutral()
        assert set(result.kp_tissues.keys()) == _TISSUES

    def test_all_kp_values_positive(self):
        result = _lipophilic_base()
        for tissue, kp in result.kp_tissues.items():
            assert kp > 0.0, f"Kp for {tissue} should be positive, got {kp}"

    def test_all_kp_values_positive_acid(self):
        result = _hydrophilic_acid()
        for tissue, kp in result.kp_tissues.items():
            assert kp > 0.0, f"Kp for {tissue} should be positive, got {kp}"


# ---------------------------------------------------------------------------
# Lipophilicity effect
# ---------------------------------------------------------------------------


class TestLipophilicityEffect:
    def test_high_logp_gives_higher_adipose_kp(self):
        high = predict_tissue_binding("H", 5.0, 7.0, "neutral", 0.5, 300.0)
        low = predict_tissue_binding("L", 0.0, 7.0, "neutral", 0.5, 300.0)
        assert high.kp_tissues["adipose"] > low.kp_tissues["adipose"]

    def test_high_logp_adipose_accumulation(self):
        result = predict_tissue_binding("HighLogP", 5.5, 7.0, "neutral", 0.3, 400.0)
        assert result.adipose_accumulation is True

    def test_low_logp_no_adipose_accumulation(self):
        result = predict_tissue_binding("LowLogP", -2.0, 7.0, "neutral", 0.8, 200.0)
        assert result.adipose_accumulation is False


# ---------------------------------------------------------------------------
# fu_plasma effect
# ---------------------------------------------------------------------------


class TestFuPlasmaEffect:
    def test_lower_fu_increases_distribution_for_neutral(self):
        high_fu = predict_tissue_binding("HF", 2.0, 7.0, "neutral", 0.9, 300.0)
        low_fu = predict_tissue_binding("LF", 2.0, 7.0, "neutral", 0.1, 300.0)
        # Lower fu → larger protein binding term → different Kp
        assert high_fu.kp_tissues != low_fu.kp_tissues

    def test_fu_validation_zero_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_tissue_binding("X", 2.0, 7.0, "neutral", 0.0, 300.0)

    def test_fu_validation_above_one_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_tissue_binding("X", 2.0, 7.0, "neutral", 1.1, 300.0)


# ---------------------------------------------------------------------------
# Vss
# ---------------------------------------------------------------------------


class TestVss:
    def test_vss_positive(self):
        result = _neutral()
        assert result.vss_predicted_L > 0.0

    def test_vss_includes_plasma_volume(self):
        # Vss must be at least plasma volume (3.0 L)
        result = _neutral()
        assert result.vss_predicted_L >= 3.0

    def test_high_logp_gives_higher_vss(self):
        high = predict_tissue_binding("H", 5.0, 7.0, "neutral", 0.5, 300.0)
        low = predict_tissue_binding("L", 0.0, 7.0, "neutral", 0.5, 300.0)
        assert high.vss_predicted_L > low.vss_predicted_L


# ---------------------------------------------------------------------------
# highest / lowest tissue
# ---------------------------------------------------------------------------


class TestHighestLowestTissue:
    def test_highest_kp_tissue_is_valid_tissue_name(self):
        result = _lipophilic_base()
        assert result.highest_kp_tissue in _TISSUES

    def test_lowest_kp_tissue_is_valid_tissue_name(self):
        result = _lipophilic_base()
        assert result.lowest_kp_tissue in _TISSUES

    def test_highest_kp_tissue_has_max_value(self):
        result = _neutral()
        expected_max = max(result.kp_tissues.values())
        assert result.kp_tissues[result.highest_kp_tissue] == pytest.approx(expected_max)

    def test_lowest_kp_tissue_has_min_value(self):
        result = _neutral()
        expected_min = min(result.kp_tissues.values())
        assert result.kp_tissues[result.lowest_kp_tissue] == pytest.approx(expected_min)


# ---------------------------------------------------------------------------
# Brain penetration index
# ---------------------------------------------------------------------------


class TestBrainPenetration:
    def test_brain_penetration_index_positive(self):
        result = _lipophilic_base()
        assert result.brain_penetration_index > 0.0

    def test_brain_penetration_index_matches_kp_brain(self):
        result = _neutral()
        assert result.brain_penetration_index == pytest.approx(result.kp_tissues["brain"], rel=1e-4)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_logp_too_low_raises(self):
        with pytest.raises(ValueError, match="logP"):
            predict_tissue_binding("X", -6.0, 7.0, "neutral", 0.5, 300.0)

    def test_logp_too_high_raises(self):
        with pytest.raises(ValueError, match="logP"):
            predict_tissue_binding("X", 11.0, 7.0, "neutral", 0.5, 300.0)

    def test_mw_zero_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_tissue_binding("X", 2.0, 7.0, "neutral", 0.5, 0.0)

    def test_mw_negative_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_tissue_binding("X", 2.0, 7.0, "neutral", 0.5, -100.0)

    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError, match="drug_type"):
            predict_tissue_binding("X", 2.0, 7.0, "zwitterion", 0.5, 300.0)
