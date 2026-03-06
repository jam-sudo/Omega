"""Tests for genotoxicity prediction module."""

from __future__ import annotations

import pytest

from omega_pbpk.risk.genotoxicity import (
    GenotoxResult,
    assess_genotoxicity,
    get_genotoxic_alerts_summary,
    screen_library,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_compound() -> GenotoxResult:
    """Simple alkane – no alerts expected."""
    return assess_genotoxicity("hexane", "CCCCCC", mw=86.18, logP=3.9)


def _nitro_compound() -> GenotoxResult:
    """Nitro-containing compound – strong genotoxic alert."""
    return assess_genotoxicity("nitro_cpd", "CC([N+](=O)[O-])C", mw=89.09, logP=0.2)


# ---------------------------------------------------------------------------
# 1-3: Basic result structure
# ---------------------------------------------------------------------------


class TestBasicResult:
    def test_returns_genotox_result(self):
        result = _safe_compound()
        assert isinstance(result, GenotoxResult)

    def test_n_alerts_non_negative(self):
        result = _safe_compound()
        assert result.n_alerts >= 0

    def test_alert_score_non_negative(self):
        result = _safe_compound()
        assert result.alert_score >= 0

    def test_ames_probability_in_range(self):
        result = _nitro_compound()
        assert 0 <= result.ames_probability <= 1


# ---------------------------------------------------------------------------
# 4-5: Ames prediction
# ---------------------------------------------------------------------------


class TestAmesPrediction:
    def test_nitro_compound_positive(self):
        result = _nitro_compound()
        assert result.ames_prediction == "positive"

    def test_simple_alkane_negative(self):
        result = _safe_compound()
        assert result.ames_prediction == "negative"


# ---------------------------------------------------------------------------
# 6-8: Individual alert triggers
# ---------------------------------------------------------------------------


class TestAlertTriggers:
    def test_epoxide_alert(self):
        result = assess_genotoxicity("epoxide_cpd", "C1OC1CC", mw=58.08, logP=0.5)
        triggered = {a.name for a in result.alerts if a.triggered}
        assert "epoxide" in triggered

    def test_low_mw_reactive(self):
        result = assess_genotoxicity("small_mol", "CCO", mw=80.0, logP=0.5)
        triggered = {a.name for a in result.alerts if a.triggered}
        assert "low_mw_reactive" in triggered

    def test_high_logP_lipophilic(self):
        result = assess_genotoxicity("lipophilic", "CCCCCCCCCC", mw=142.28, logP=6.0)
        triggered = {a.name for a in result.alerts if a.triggered}
        assert "high_logP_lipophilic" in triggered


# ---------------------------------------------------------------------------
# 9-12: ICH S2(R1) and micronucleus
# ---------------------------------------------------------------------------


class TestICHAndMicronucleus:
    def test_ichs2r1_concern_for_genotoxic(self):
        result = _nitro_compound()
        assert result.ichs2r1_flag == "concern"

    def test_ichs2r1_no_concern_for_safe(self):
        result = _safe_compound()
        assert result.ichs2r1_flag == "no_concern"

    def test_micronucleus_high(self):
        # Nitro compound with MW > 150
        result = assess_genotoxicity("heavy_nitro", "CCCCCC([N+](=O)[O-])CCCC", mw=200.0, logP=2.0)
        assert result.micronucleus_risk == "high"

    def test_micronucleus_low(self):
        result = _safe_compound()
        assert result.micronucleus_risk == "low"


# ---------------------------------------------------------------------------
# 13-14: Alert counting and scoring
# ---------------------------------------------------------------------------


class TestAlertScoring:
    def test_n_alerts_equals_triggered_count(self):
        result = _nitro_compound()
        triggered_count = sum(1 for a in result.alerts if a.triggered)
        assert result.n_alerts == triggered_count

    def test_alert_score_equals_weight_sum(self):
        result = _nitro_compound()
        weight_sum = sum(a.weight for a in result.alerts if a.triggered)
        assert abs(result.alert_score - weight_sum) < 1e-9


# ---------------------------------------------------------------------------
# 15-16: screen_library
# ---------------------------------------------------------------------------


class TestScreenLibrary:
    def test_returns_same_length(self):
        compounds = [
            {"name": "a", "smiles": "CCCC", "mw": 58.12, "logP": 2.0, "tpsa": 0.0},
            {"name": "b", "smiles": "CC([N+](=O)[O-])C", "mw": 89.0, "logP": 0.2, "tpsa": 50.0},
        ]
        results = screen_library(compounds)
        assert len(results) == len(compounds)

    def test_sorted_by_alert_score_descending(self):
        compounds = [
            {"name": "safe", "smiles": "CCCC", "mw": 58.12, "logP": 2.0, "tpsa": 0.0},
            {"name": "risky", "smiles": "CC([N+](=O)[O-])C", "mw": 89.0, "logP": 0.2, "tpsa": 50.0},
        ]
        results = screen_library(compounds)
        scores = [r.alert_score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 17: get_genotoxic_alerts_summary
# ---------------------------------------------------------------------------


class TestAlertsSummary:
    def test_returns_at_least_5_keys(self):
        summary = get_genotoxic_alerts_summary()
        assert isinstance(summary, dict)
        assert len(summary) >= 5


# ---------------------------------------------------------------------------
# 18-20: Enum-like field validation
# ---------------------------------------------------------------------------


class TestFieldValues:
    def test_structural_class_valid(self):
        for smiles, mw in [("CCCC", 58.12), ("CC([N+](=O)[O-])C", 89.0)]:
            result = assess_genotoxicity("test", smiles, mw=mw, logP=1.0)
            assert result.structural_class in {"genotoxic", "non-genotoxic", "borderline"}

    def test_ames_prediction_valid(self):
        result = _safe_compound()
        assert result.ames_prediction in {"positive", "negative", "inconclusive"}

    def test_compound_name_preserved(self):
        result = assess_genotoxicity("my_drug_123", "CCCC", mw=58.0, logP=1.0)
        assert result.compound_name == "my_drug_123"


# ---------------------------------------------------------------------------
# 21-22: Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_smiles_raises(self):
        with pytest.raises(ValueError, match="SMILES"):
            assess_genotoxicity("bad", "", mw=100.0, logP=1.0)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="Molecular weight"):
            assess_genotoxicity("bad", "CCCC", mw=-10.0, logP=1.0)


# ---------------------------------------------------------------------------
# 23-25: Multiple alerts, nitroso, borderline micronucleus
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_multiple_alerts_sum(self):
        # Compound with epoxide + low MW → both triggered, scores sum
        result = assess_genotoxicity("multi", "C1OC1C", mw=58.0, logP=0.5)
        triggered = [a for a in result.alerts if a.triggered]
        assert len(triggered) >= 2
        assert abs(result.alert_score - sum(a.weight for a in triggered)) < 1e-9

    def test_nitroso_alert(self):
        # N=O not part of nitro group
        result = assess_genotoxicity("nitroso_cpd", "CCN=OCC", mw=87.0, logP=0.5)
        triggered_names = {a.name for a in result.alerts if a.triggered}
        assert "nitroso" in triggered_names

    def test_micronucleus_moderate(self):
        # Borderline: alert_score >= 1.0 but < 2.0 or mw <= 150
        # Use an epoxide (weight=1.5) with small MW
        result = assess_genotoxicity("border", "C1OC1CC", mw=58.08, logP=0.5)
        assert result.micronucleus_risk == "moderate"
