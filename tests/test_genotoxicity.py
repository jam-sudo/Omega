"""Tests for genotoxicity structural alert screening — Phase 259."""

from __future__ import annotations

import pytest

from omega_pbpk.risk.genotoxicity import (
    GenotoxAlert,
    GenotoxResult,
    assess_genotoxicity,
    screen_genotoxicity,
)

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_smiles_raises(self):
        with pytest.raises(ValueError, match="smiles"):
            assess_genotoxicity("bad", "")

    def test_whitespace_smiles_raises(self):
        with pytest.raises(ValueError, match="smiles"):
            assess_genotoxicity("bad", "   ")

    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="daily_dose_mg"):
            assess_genotoxicity("bad", "CCCC", daily_dose_mg=0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="daily_dose_mg"):
            assess_genotoxicity("bad", "CCCC", daily_dose_mg=-1.0)


# ---------------------------------------------------------------------------
# Clean compound (simple alkane)
# ---------------------------------------------------------------------------


class TestCleanCompound:
    def setup_method(self):
        self.result = assess_genotoxicity("butane", "CCCC")

    def test_n_alerts_zero(self):
        assert self.result.n_alerts == 0

    def test_ames_negative(self):
        assert self.result.ames_prediction == "negative"

    def test_total_weight_zero(self):
        assert self.result.total_weight == 0.0

    def test_icm_class_5(self):
        assert self.result.icm_m7_class == "class 5"

    def test_daily_ai_ng_high(self):
        # Non-genotoxic → 120000 ng/day
        assert self.result.daily_acceptable_intake_ng == pytest.approx(120000.0)

    def test_compound_name_stored(self):
        assert self.result.compound_name == "butane"

    def test_smiles_stored(self):
        assert self.result.smiles == "CCCC"


# ---------------------------------------------------------------------------
# Nitroaromatic compound
# ---------------------------------------------------------------------------


class TestNitroaromatic:
    def setup_method(self):
        self.result = assess_genotoxicity("nitrobenzene", "[N+](=O)[O-]c1ccccc1")

    def test_nitro_aromatic_triggered(self):
        triggered = {a.name for a in self.result.alerts if a.triggered}
        assert "nitro_aromatic" in triggered

    def test_ames_positive(self):
        assert self.result.ames_prediction == "positive"

    def test_total_weight_positive(self):
        assert self.result.total_weight >= 2.0

    def test_daily_ai_ng_ttc(self):
        # Genotoxic → TTC 1500 ng/day
        assert self.result.daily_acceptable_intake_ng == pytest.approx(1500.0)


# ---------------------------------------------------------------------------
# Hydrazine
# ---------------------------------------------------------------------------


class TestHydrazine:
    def test_hydrazine_triggered(self):
        result = assess_genotoxicity("hydrazine", "NN")
        triggered = {a.name for a in result.alerts if a.triggered}
        assert "hydrazine" in triggered


# ---------------------------------------------------------------------------
# Aziridine
# ---------------------------------------------------------------------------


class TestAziridine:
    def test_aziridine_triggered(self):
        result = assess_genotoxicity("aziridine", "C1CN1")
        triggered = {a.name for a in result.alerts if a.triggered}
        assert "aziridine" in triggered

    def test_aziridine_class_1(self):
        result = assess_genotoxicity("aziridine", "C1CN1")
        assert result.icm_m7_class == "class 1"


# ---------------------------------------------------------------------------
# General structural checks
# ---------------------------------------------------------------------------


class TestGeneralStructure:
    def test_n_alerts_equals_triggered_count(self):
        result = assess_genotoxicity("nitrobenzene", "[N+](=O)[O-]c1ccccc1")
        triggered_count = sum(1 for a in result.alerts if a.triggered)
        assert result.n_alerts == triggered_count

    def test_ames_prediction_valid_values(self):
        for smiles in ["CCCC", "[N+](=O)[O-]c1ccccc1", "C1OC1"]:
            result = assess_genotoxicity("test", smiles)
            assert result.ames_prediction in {"positive", "negative", "equivocal"}

    def test_icm_class_starts_with_class(self):
        for smiles in ["CCCC", "[N+](=O)[O-]c1ccccc1", "C1CN1"]:
            result = assess_genotoxicity("test", smiles)
            assert result.icm_m7_class.startswith("class ")

    def test_daily_acceptable_intake_positive(self):
        result = assess_genotoxicity("test", "CCCC")
        assert result.daily_acceptable_intake_ng > 0

    def test_mitigation_strategy_non_empty_list(self):
        for smiles in ["CCCC", "[N+](=O)[O-]c1ccccc1"]:
            result = assess_genotoxicity("test", smiles)
            assert isinstance(result.mitigation_strategy, list)
            assert len(result.mitigation_strategy) > 0

    def test_alerts_list_length(self):
        result = assess_genotoxicity("test", "CCCC")
        # Should have entries for all defined alert types
        assert len(result.alerts) > 0

    def test_alert_objects_are_genotox_alert(self):
        result = assess_genotoxicity("test", "CCCC")
        for a in result.alerts:
            assert isinstance(a, GenotoxAlert)


# ---------------------------------------------------------------------------
# screen_genotoxicity
# ---------------------------------------------------------------------------


class TestScreenGenotoxicity:
    def test_empty_list_returns_empty(self):
        assert screen_genotoxicity([]) == []

    def test_sorted_by_total_weight_descending(self):
        compounds = [
            {"name": "safe", "smiles": "CCCC"},
            {"name": "nitrobenzene", "smiles": "[N+](=O)[O-]c1ccccc1"},
            {"name": "aziridine", "smiles": "C1CN1"},
        ]
        results = screen_genotoxicity(compounds)
        weights = [r.total_weight for r in results]
        assert weights == sorted(weights, reverse=True)

    def test_returns_all_compounds(self):
        compounds = [
            {"name": "a", "smiles": "CCCC"},
            {"name": "b", "smiles": "CCCCC"},
        ]
        results = screen_genotoxicity(compounds)
        assert len(results) == 2

    def test_returns_genotox_result_objects(self):
        compounds = [{"name": "test", "smiles": "CCCC"}]
        results = screen_genotoxicity(compounds)
        assert isinstance(results[0], GenotoxResult)

    def test_daily_dose_mg_passed_through(self):
        # No error when optional key is missing
        compounds = [{"name": "test", "smiles": "CCCC", "daily_dose_mg": 10.0}]
        results = screen_genotoxicity(compounds)
        assert len(results) == 1
