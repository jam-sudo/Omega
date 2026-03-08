"""Tests for Phase 352 — Drug-Induced Phospholipidosis Predictor."""

from __future__ import annotations

import pytest

from omega_pbpk.risk.phospholipidosis_predictor import (
    PhospholipidosisResult,
    predict_phospholipidosis,
)

# Shared baseline kwargs
_BASE = dict(
    drug_name="TestDrug",
    logP=2.0,
    pKa=8.5,
    drug_type="base",
    mw_Da=350.0,
    n_rings=2,
    n_positive_charges=1,
    has_long_alkyl_chain=False,
    lysosomal_trapping_pct=40.0,
)


def _predict(**kwargs):
    params = {**_BASE, **kwargs}
    return predict_phospholipidosis(**params)


# ── Return type ────────────────────────────────────────────────────────────────


class TestReturnType:
    def test_returns_dataclass(self):
        result = _predict()
        assert isinstance(result, PhospholipidosisResult)

    def test_plain_dataclass_mutable(self):
        """Plain @dataclass (not frozen) should be mutable."""
        result = _predict()
        result.drug_name = "Changed"
        assert result.drug_name == "Changed"

    def test_structural_alerts_is_list(self):
        result = _predict()
        assert isinstance(result.structural_alerts, list)


# ── Score and Risk Class ──────────────────────────────────────────────────────


class TestScoreAndRiskClass:
    def test_dipl_risk_score_in_range(self):
        result = _predict()
        assert 0.0 <= result.dipl_risk_score <= 100.0

    def test_dipl_risk_class_valid(self):
        valid = {"low", "moderate", "high"}
        for drug_type in ("acid", "base", "neutral", "zwitterion"):
            result = _predict(drug_type=drug_type)
            assert result.dipl_risk_class in valid

    def test_high_logp_basic_drug_high_risk(self):
        """High logP + basic drug with pKa > 7.4 → high DIPL risk."""
        result = _predict(
            logP=4.0,
            pKa=9.0,
            drug_type="base",
            mw_Da=400.0,
            n_rings=3,
            n_positive_charges=2,
            has_long_alkyl_chain=True,
        )
        assert result.dipl_risk_class == "high"
        assert result.dipl_risk_score > 60.0

    def test_neutral_low_logp_low_risk(self):
        """Neutral drug with low logP → low DIPL risk."""
        result = _predict(
            logP=0.5,
            pKa=5.0,
            drug_type="neutral",
            mw_Da=200.0,
            n_rings=0,
            n_positive_charges=0,
            has_long_alkyl_chain=False,
            lysosomal_trapping_pct=5.0,
        )
        assert result.dipl_risk_class == "low"
        assert result.dipl_risk_score < 25.0

    def test_score_capped_at_100(self):
        """DIPL risk score should never exceed 100."""
        result = _predict(
            logP=9.0,
            pKa=10.0,
            drug_type="base",
            mw_Da=500.0,
            n_rings=5,
            n_positive_charges=3,
            has_long_alkyl_chain=True,
        )
        assert result.dipl_risk_score <= 100.0

    def test_cad_score_raw_can_exceed_100(self):
        """Raw cad_score can exceed 100 before normalization."""
        result = _predict(
            logP=9.0,
            pKa=10.0,
            drug_type="base",
            mw_Da=500.0,
            n_rings=5,
            n_positive_charges=3,
            has_long_alkyl_chain=True,
        )
        # raw score may exceed 100, but dipl_risk_score capped at 100
        assert result.cad_score >= result.dipl_risk_score


# ── is_cad_drug ───────────────────────────────────────────────────────────────


class TestIsCadDrug:
    def test_is_cad_true_when_score_above_25(self):
        result = _predict(
            logP=4.0,
            pKa=9.0,
            drug_type="base",
            mw_Da=400.0,
            n_rings=2,
            n_positive_charges=1,
        )
        assert result.dipl_risk_score > 25.0
        assert result.is_cad_drug is True

    def test_is_cad_false_when_score_below_25(self):
        result = _predict(
            logP=0.5,
            pKa=5.0,
            drug_type="neutral",
            mw_Da=200.0,
            n_rings=0,
            n_positive_charges=0,
            has_long_alkyl_chain=False,
            lysosomal_trapping_pct=5.0,
        )
        assert result.is_cad_drug is False


# ── Structural Alerts ─────────────────────────────────────────────────────────


class TestStructuralAlerts:
    def test_alerts_empty_for_zero_contributors(self):
        """Drug with logP <= 0, neutral, low MW, no rings/charges/chain → empty alerts."""
        result = _predict(
            logP=-1.0,
            pKa=5.0,
            drug_type="neutral",
            mw_Da=200.0,
            n_rings=0,
            n_positive_charges=0,
            has_long_alkyl_chain=False,
            lysosomal_trapping_pct=0.0,
        )
        assert result.structural_alerts == []

    def test_alerts_non_empty_for_cad_drug(self):
        result = _predict(
            logP=3.0,
            pKa=9.0,
            drug_type="base",
            mw_Da=400.0,
            n_rings=2,
            n_positive_charges=1,
        )
        assert len(result.structural_alerts) > 0

    def test_alkyl_chain_alert_present_when_flagged(self):
        result = _predict(has_long_alkyl_chain=True)
        alert_text = " ".join(result.structural_alerts)
        assert "alkyl" in alert_text.lower() or "chain" in alert_text.lower()


# ── Lysosomal Accumulation Index ─────────────────────────────────────────────


class TestLysosomalAccumulation:
    def test_lai_gte_trapping_when_logp_positive(self):
        result = _predict(logP=2.0, lysosomal_trapping_pct=50.0)
        assert result.lysosomal_accumulation_index >= 50.0

    def test_lai_equals_trapping_when_logp_zero_or_negative(self):
        result = _predict(logP=0.0, lysosomal_trapping_pct=40.0)
        assert result.lysosomal_accumulation_index == pytest.approx(40.0, abs=1e-6)

    def test_lai_clamped_at_200(self):
        result = _predict(logP=10.0, lysosomal_trapping_pct=100.0)
        assert result.lysosomal_accumulation_index <= 200.0

    def test_lai_non_negative(self):
        result = _predict(logP=-5.0, lysosomal_trapping_pct=10.0)
        assert result.lysosomal_accumulation_index >= 0.0


# ── Validation Errors ─────────────────────────────────────────────────────────


class TestValidationErrors:
    def test_logp_too_low_raises(self):
        with pytest.raises(ValueError, match="logP"):
            _predict(logP=-6.0)

    def test_logp_too_high_raises(self):
        with pytest.raises(ValueError, match="logP"):
            _predict(logP=11.0)

    def test_mw_zero_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _predict(mw_Da=0.0)

    def test_mw_negative_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _predict(mw_Da=-100.0)

    def test_n_rings_negative_raises(self):
        with pytest.raises(ValueError, match="n_rings"):
            _predict(n_rings=-1)

    def test_n_positive_charges_negative_raises(self):
        with pytest.raises(ValueError, match="n_positive_charges"):
            _predict(n_positive_charges=-1)

    def test_lysosomal_trapping_above_100_raises(self):
        with pytest.raises(ValueError, match="lysosomal_trapping_pct"):
            _predict(lysosomal_trapping_pct=101.0)

    def test_lysosomal_trapping_below_0_raises(self):
        with pytest.raises(ValueError, match="lysosomal_trapping_pct"):
            _predict(lysosomal_trapping_pct=-1.0)
