"""Tests for mitochondrial drug accumulation model (Phase 881)."""

import math
import pytest

from omega_pbpk.core.mitochondrial_accumulation import (
    MitochondrialAccumulationResult,
    predict_mitochondrial_accumulation,
    screen_mito_toxicity,
)

_F = 96485.0
_R = 8.314


class TestPredictMitochondrialAccumulation:
    def test_basic_cation_returns_result(self):
        result = predict_mitochondrial_accumulation("TestDrug", dose_mg=100.0)
        assert isinstance(result, MitochondrialAccumulationResult)

    def test_drug_name_preserved(self):
        result = predict_mitochondrial_accumulation("Doxorubicin", dose_mg=50.0)
        assert result.drug_name == "Doxorubicin"

    def test_dose_preserved(self):
        result = predict_mitochondrial_accumulation("TestDrug", dose_mg=200.0)
        assert result.dose_mg == 200.0

    def test_delta_psi_preserved(self):
        result = predict_mitochondrial_accumulation("TestDrug", dose_mg=100.0, delta_psi_mV=-180.0)
        assert result.delta_psi_mV == -180.0

    def test_charge_preserved(self):
        result = predict_mitochondrial_accumulation("TestDrug", dose_mg=100.0, charge=2)
        assert result.charge == 2

    def test_logp_preserved(self):
        result = predict_mitochondrial_accumulation("TestDrug", dose_mg=100.0, logp=3.5)
        assert result.logp == 3.5

    def test_nernst_ratio_positive(self):
        result = predict_mitochondrial_accumulation("TestDrug", dose_mg=100.0, charge=1)
        assert result.nernst_ratio > 1.0

    def test_nernst_ratio_calculation(self):
        """Verify Nernst ratio matches manual calculation."""
        charge = 1
        delta_psi_mV = -180.0
        temp_K = 310.0
        expected = math.exp(charge * _F * abs(delta_psi_mV) * 1e-3 / (_R * temp_K))
        result = predict_mitochondrial_accumulation(
            "TestDrug", dose_mg=100.0, charge=charge, delta_psi_mV=delta_psi_mV, temp_K=temp_K
        )
        assert abs(result.nernst_ratio - expected) < 1e-3

    def test_higher_charge_higher_nernst(self):
        r1 = predict_mitochondrial_accumulation("Drug", dose_mg=100.0, charge=1, delta_psi_mV=-180.0)
        r2 = predict_mitochondrial_accumulation("Drug", dose_mg=100.0, charge=2, delta_psi_mV=-180.0)
        assert r2.nernst_ratio > r1.nernst_ratio

    def test_logp_increases_conc_factor(self):
        r_low = predict_mitochondrial_accumulation("Drug", dose_mg=100.0, logp=0.0)
        r_high = predict_mitochondrial_accumulation("Drug", dose_mg=100.0, logp=5.0)
        assert r_high.mitochondrial_conc_factor >= r_low.mitochondrial_conc_factor

    def test_negative_logp_no_enhancement(self):
        """Negative logP: conc factor equals nernst ratio (no multiplication)."""
        result = predict_mitochondrial_accumulation("Drug", dose_mg=100.0, logp=-2.0)
        assert abs(result.mitochondrial_conc_factor - result.nernst_ratio) < 1.0

    def test_conc_factor_clamped_min(self):
        """Very low charge or neutral -> factor >= 1.0."""
        result = predict_mitochondrial_accumulation("Drug", dose_mg=100.0, charge=0, logp=-5.0)
        assert result.mitochondrial_conc_factor >= 1.0

    def test_conc_factor_clamped_max(self):
        """Very high charge -> factor <= 1e6."""
        result = predict_mitochondrial_accumulation("Drug", dose_mg=100.0, charge=5, logp=10.0)
        assert result.mitochondrial_conc_factor <= 1e6

    def test_dili_risk_score_range(self):
        result = predict_mitochondrial_accumulation("Drug", dose_mg=100.0)
        assert 0.0 <= result.dili_risk_score <= 100.0

    def test_dili_risk_high(self):
        """High logP + high charge -> high DILI risk."""
        result = predict_mitochondrial_accumulation("Drug", dose_mg=100.0, charge=3, logp=8.0)
        assert result.dili_risk_level == "high"

    def test_dili_risk_low(self):
        """Neutral molecule -> low DILI risk."""
        result = predict_mitochondrial_accumulation("Drug", dose_mg=100.0, charge=0, logp=0.0)
        assert result.dili_risk_level == "low"

    def test_dili_risk_levels_consistent(self):
        result = predict_mitochondrial_accumulation("Drug", dose_mg=100.0, charge=1, logp=2.0)
        if result.dili_risk_score > 60:
            assert result.dili_risk_level == "high"
        elif result.dili_risk_score > 30:
            assert result.dili_risk_level == "moderate"
        else:
            assert result.dili_risk_level == "low"

    def test_notes_high_risk(self):
        result = predict_mitochondrial_accumulation("Drug", dose_mg=100.0, charge=3, logp=8.0)
        if result.dili_risk_level == "high":
            assert "DILI risk" in result.notes

    def test_notes_low_risk(self):
        result = predict_mitochondrial_accumulation("Drug", dose_mg=100.0, charge=0, logp=0.0)
        if result.dili_risk_level == "low":
            assert "Low" in result.notes

    def test_auc_ratio_positive(self):
        result = predict_mitochondrial_accumulation("Drug", dose_mg=100.0)
        assert result.predicted_mitochondrial_auc_ratio > 0.0

    def test_auc_ratio_is_10pct_of_conc_factor(self):
        result = predict_mitochondrial_accumulation("Drug", dose_mg=100.0)
        expected = result.mitochondrial_conc_factor * 0.1
        assert abs(result.predicted_mitochondrial_auc_ratio - expected) < 1e-9

    def test_validation_positive_delta_psi(self):
        with pytest.raises(ValueError, match="negative"):
            predict_mitochondrial_accumulation("Drug", dose_mg=100.0, delta_psi_mV=10.0)

    def test_validation_zero_delta_psi(self):
        with pytest.raises(ValueError, match="negative"):
            predict_mitochondrial_accumulation("Drug", dose_mg=100.0, delta_psi_mV=0.0)


class TestScreenMitoToxicity:
    def test_returns_list(self):
        candidates = [
            {"name": "DrugA", "dose_mg": 100.0, "charge": 1, "logp": 2.0},
            {"name": "DrugB", "dose_mg": 50.0, "charge": 2, "logp": 4.0},
        ]
        results = screen_mito_toxicity(candidates)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_sorted_by_dili_score_descending(self):
        candidates = [
            {"name": "Safe", "dose_mg": 100.0, "charge": 0, "logp": 0.0},
            {"name": "Toxic", "dose_mg": 100.0, "charge": 3, "logp": 8.0},
            {"name": "Medium", "dose_mg": 100.0, "charge": 1, "logp": 3.0},
        ]
        results = screen_mito_toxicity(candidates)
        scores = [r.dili_risk_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_default_charge_and_logp(self):
        candidates = [{"name": "DrugX", "dose_mg": 100.0}]
        results = screen_mito_toxicity(candidates)
        assert results[0].charge == 1
        assert results[0].logp == 1.0

    def test_empty_list(self):
        results = screen_mito_toxicity([])
        assert results == []
