"""Tests for biopharmaceutics/biopharma_risk.py — Phase 506."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.biopharma_risk import (
    BiopharmRiskResult,
    assess_biopharmaceutics_risk,
    solubility_risk_score,
    permeability_risk_score,
    absorption_risk_index,
    formulation_strategy_from_risk,
)


# ---------------------------------------------------------------------------
# solubility_risk_score
# ---------------------------------------------------------------------------

class TestSolubilityRiskScore:
    def test_dn_less_than_1_low_risk(self):
        """Dn = 0.4 < 1 → low risk score."""
        # dose=100, sol=1.0, vol=250 → Dn = 0.4
        score = solubility_risk_score(solubility_mg_mL=1.0, dose_mg=100.0)
        assert score < 20.0

    def test_dn_exactly_1_boundary(self):
        """Dn=1 is the boundary between low and moderate risk."""
        # dose=250, sol=1.0, vol=250 → Dn = 1.0
        score = solubility_risk_score(solubility_mg_mL=1.0, dose_mg=250.0)
        assert abs(score - 20.0) < 1e-6

    def test_dn_5_high_risk_score(self):
        """Dn=5 → score = 60 (boundary of high risk)."""
        # dose=1250, sol=1.0, vol=250 → Dn = 5.0
        score = solubility_risk_score(solubility_mg_mL=1.0, dose_mg=1250.0)
        assert abs(score - 60.0) < 1e-6

    def test_high_dn_capped_at_100(self):
        """Very high Dn → score capped at 100."""
        score = solubility_risk_score(solubility_mg_mL=0.001, dose_mg=10000.0)
        assert score <= 100.0

    def test_high_solubility_low_dose_low_score(self):
        """High solubility, low dose → near-zero score."""
        score = solubility_risk_score(solubility_mg_mL=10.0, dose_mg=10.0)
        assert score < 10.0

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError, match="solubility"):
            solubility_risk_score(0.0, 100.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose"):
            solubility_risk_score(1.0, 0.0)

    def test_custom_gi_volume(self):
        """Using a smaller GI volume increases the dose number (higher risk)."""
        s1 = solubility_risk_score(1.0, 100.0, gi_volume_mL=250.0)
        s2 = solubility_risk_score(1.0, 100.0, gi_volume_mL=100.0)
        assert s2 > s1


# ---------------------------------------------------------------------------
# permeability_risk_score
# ---------------------------------------------------------------------------

class TestPermeabilityRiskScore:
    def test_high_papp_low_score(self):
        """Papp > 2e-5 → low risk score."""
        score = permeability_risk_score(papp_cm_s=4e-5)
        assert score < 20.0

    def test_papp_at_2e5_boundary(self):
        """Papp exactly 2e-5 → score = 20."""
        score = permeability_risk_score(papp_cm_s=2e-5)
        assert abs(score - 20.0) < 1e-6

    def test_moderate_papp_moderate_score(self):
        """Papp in moderate range → score between 20 and 60."""
        score = permeability_risk_score(papp_cm_s=1e-5)
        assert 20.0 < score < 60.0

    def test_low_papp_high_score(self):
        """Papp < 0.5e-5 → score >= 60."""
        score = permeability_risk_score(papp_cm_s=0.1e-5)
        assert score >= 60.0

    def test_zero_papp_returns_100(self):
        score = permeability_risk_score(papp_cm_s=0.0)
        assert score == 100.0

    def test_negative_papp_raises(self):
        with pytest.raises(ValueError, match="papp"):
            permeability_risk_score(-1e-5)


# ---------------------------------------------------------------------------
# absorption_risk_index
# ---------------------------------------------------------------------------

class TestAbsorptionRiskIndex:
    def test_weighted_average(self):
        """Weighted average: 0.6*40 + 0.4*60 = 48."""
        result = absorption_risk_index(40.0, 60.0, weights=(0.6, 0.4))
        assert abs(result - 48.0) < 1e-9

    def test_equal_weights(self):
        result = absorption_risk_index(30.0, 70.0, weights=(0.5, 0.5))
        assert abs(result - 50.0) < 1e-9

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError, match="weights"):
            absorption_risk_index(50.0, 50.0, weights=(0.3, 0.5))

    def test_zero_scores_gives_zero(self):
        assert absorption_risk_index(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# formulation_strategy_from_risk
# ---------------------------------------------------------------------------

class TestFormulationStrategyFromRisk:
    def test_low_risk_standard_formulation(self):
        strategy = formulation_strategy_from_risk(15.0, 1.0, 3e-5, 2.0)
        assert "Standard" in strategy or "standard" in strategy.lower()

    def test_bcs_ii_lipid_strategy(self):
        """BCS II (high logP, low solubility, high risk) → lipid-based strategy."""
        strategy = formulation_strategy_from_risk(70.0, 4.0, 3e-5, 0.05)
        assert any(kw in strategy.lower() for kw in ["lipid", "solid dispersion", "amorphous"])

    def test_very_high_risk_comprehensive(self):
        """Very high risk → comprehensive re-engineering mentioned."""
        strategy = formulation_strategy_from_risk(90.0, 2.0, 0.1e-5, 0.01)
        assert len(strategy) > 50  # Should be a detailed recommendation


# ---------------------------------------------------------------------------
# assess_biopharmaceutics_risk (integration tests)
# ---------------------------------------------------------------------------

class TestAssessBiopharmaceuticsRisk:
    def test_bcs_i_low_risk(self):
        """BCS I drug (high sol, high perm) → low overall risk."""
        result = assess_biopharmaceutics_risk(
            drug_name="Caffeine",
            logP=-0.1,
            mw=194.0,
            solubility_mg_mL=20.0,   # Dn << 1
            papp_cm_s=5e-5,           # high perm
            dose_mg=100.0,
        )
        assert result.bcs_class == "I"
        assert result.overall_risk == "low"

    def test_bcs_iv_very_high_risk(self):
        """BCS IV drug (low sol, low perm) → very_high risk."""
        result = assess_biopharmaceutics_risk(
            drug_name="BCS_IV_Drug",
            logP=5.0,
            mw=500.0,
            solubility_mg_mL=0.001,  # extremely low
            papp_cm_s=0.1e-5,         # very low perm
            dose_mg=500.0,
        )
        assert result.bcs_class == "IV"
        assert result.overall_risk == "very_high"

    def test_bcs_ii_classification(self):
        """BCS II: low solubility, high permeability."""
        result = assess_biopharmaceutics_risk(
            drug_name="BCS_II_Drug",
            logP=3.0,
            mw=300.0,
            solubility_mg_mL=0.05,  # Dn >> 1
            papp_cm_s=4e-5,          # high perm
            dose_mg=200.0,
        )
        assert result.bcs_class == "II"

    def test_bcs_iii_classification(self):
        """BCS III: high solubility, low permeability."""
        result = assess_biopharmaceutics_risk(
            drug_name="BCS_III_Drug",
            logP=-1.0,
            mw=150.0,
            solubility_mg_mL=50.0,  # Dn << 1
            papp_cm_s=0.3e-5,         # low perm
            dose_mg=100.0,
        )
        assert result.bcs_class == "III"

    def test_dose_number_calculation(self):
        """Dn = dose / (sol * 250): dose=100, sol=0.4 mg/mL → Dn=1.0."""
        result = assess_biopharmaceutics_risk(
            drug_name="TestDrug",
            logP=1.0,
            mw=200.0,
            solubility_mg_mL=0.4,
            papp_cm_s=3e-5,
            dose_mg=100.0,
        )
        assert abs(result.dose_number - 1.0) < 1e-9

    def test_result_fields_populated(self):
        result = assess_biopharmaceutics_risk(
            "Drug", logP=2.0, mw=300.0, solubility_mg_mL=1.0,
            papp_cm_s=2e-5, dose_mg=100.0
        )
        assert result.drug_name == "Drug"
        assert result.bcs_class in ("I", "II", "III", "IV")
        assert 0 <= result.solubility_risk_score <= 100
        assert 0 <= result.permeability_risk_score <= 100
        assert 0 <= result.absorption_risk_index <= 100
        assert result.overall_risk in ("low", "moderate", "high", "very_high")
        assert isinstance(result.formulation_strategy, str)
        assert isinstance(result.clinical_relevance, str)
        assert isinstance(result.notes, str)

    def test_pka_included_in_notes(self):
        result = assess_biopharmaceutics_risk(
            "Drug", logP=1.5, mw=250.0, pka=7.4, drug_type="base",
            solubility_mg_mL=1.0, papp_cm_s=2e-5, dose_mg=100.0
        )
        assert "pKa" in result.notes

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            assess_biopharmaceutics_risk("D", 1.0, 200.0, dose_mg=0.0)

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError, match="solubility"):
            assess_biopharmaceutics_risk("D", 1.0, 200.0, solubility_mg_mL=0.0)

    def test_negative_papp_raises(self):
        with pytest.raises(ValueError, match="papp"):
            assess_biopharmaceutics_risk("D", 1.0, 200.0, papp_cm_s=-1e-5)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            assess_biopharmaceutics_risk("D", 1.0, 0.0)

    def test_result_is_frozen(self):
        """BiopharmRiskResult is a frozen dataclass."""
        result = assess_biopharmaceutics_risk(
            "Drug", logP=1.0, mw=200.0, solubility_mg_mL=1.0,
            papp_cm_s=2e-5, dose_mg=100.0
        )
        with pytest.raises((AttributeError, TypeError)):
            result.bcs_class = "X"  # type: ignore[misc]

    def test_high_dn_triggers_threshold_note(self):
        """When Dn > target threshold, notes mention solubility enhancement."""
        result = assess_biopharmaceutics_risk(
            "Drug", logP=3.0, mw=400.0,
            solubility_mg_mL=0.01,  # Dn will be >> 1
            papp_cm_s=4e-5,
            dose_mg=500.0,
            target_dose_number_max=1.0,
        )
        assert "solubility" in result.notes.lower()

    def test_bcs_ii_formulation_mentions_dissolution(self):
        """BCS II strategy should address dissolution/solubility."""
        result = assess_biopharmaceutics_risk(
            "BCS_II", logP=4.0, mw=380.0,
            solubility_mg_mL=0.002,
            papp_cm_s=5e-5,
            dose_mg=400.0,
        )
        assert result.bcs_class == "II"
        strategy_lower = result.formulation_strategy.lower()
        assert any(
            kw in strategy_lower
            for kw in ["lipid", "dispersion", "amorphous", "crystal", "nano", "salt"]
        )
