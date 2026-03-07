"""Tests for Phase 350 — Drug Oral Absorption Enhancement Prediction.

Coverage:
  - BCS classification correctness (all 4 classes)
  - Baseline fa values per BCS class
  - fa values in [0, 1] for all strategies
  - best_strategy is a valid strategy name
  - fold_improvement >= 1.0
  - BCS I drug: formulation strategies have minimal impact (fold close to 1)
  - BCS II drug: ASD or lipid gives significant improvement
  - screen_formulation_strategies returns sorted list
  - Frozen dataclass field access
  - recommendation and notes non-empty
  - Input validation
"""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.absorption_enhancement import (
    AbsorptionEnhancementResult,
    predict_absorption_enhancement,
    screen_formulation_strategies,
)

# ---------------------------------------------------------------------------
# Reference drugs
# ---------------------------------------------------------------------------
# BCS I: high Peff (> 2e-4), high solubility (Dn < 1)
_BCS1 = dict(
    drug_name="BCS1Drug",
    logP=1.0,
    mw=200.0,
    peff_cm_per_s=5e-4,
    solubility_mg_mL=10.0,
    dose_mg=10.0,  # Dn = 10 / (10 * 250) = 0.004 < 1
)

# BCS II: high Peff, low solubility
_BCS2 = dict(
    drug_name="BCS2Drug",
    logP=4.0,
    mw=350.0,
    peff_cm_per_s=5e-4,
    solubility_mg_mL=0.01,
    dose_mg=100.0,  # Dn = 100 / (0.01 * 250) = 40 >> 1
)

# BCS III: low Peff, high solubility
_BCS3 = dict(
    drug_name="BCS3Drug",
    logP=-1.0,
    mw=150.0,
    peff_cm_per_s=1e-5,
    solubility_mg_mL=20.0,
    dose_mg=5.0,  # Dn = 5 / (20 * 250) = 0.001 < 1
)

# BCS IV: low Peff, low solubility
_BCS4 = dict(
    drug_name="BCS4Drug",
    logP=3.0,
    mw=400.0,
    peff_cm_per_s=1e-5,
    solubility_mg_mL=0.005,
    dose_mg=200.0,  # Dn = 200 / (0.005 * 250) = 160 >> 1
)

_VALID_STRATEGIES = {
    "nanosizing",
    "solid_dispersion",
    "lipid_formulation",
    "cyclodextrin",
    "cocrystal",
}


# ---------------------------------------------------------------------------
# BCS Classification tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBCSClassification:
    def test_bcs_class_i(self):
        result = predict_absorption_enhancement(**_BCS1)
        assert result.bcs_class == "I"

    def test_bcs_class_ii(self):
        result = predict_absorption_enhancement(**_BCS2)
        assert result.bcs_class == "II"

    def test_bcs_class_iii(self):
        result = predict_absorption_enhancement(**_BCS3)
        assert result.bcs_class == "III"

    def test_bcs_class_iv(self):
        result = predict_absorption_enhancement(**_BCS4)
        assert result.bcs_class == "IV"

    def test_high_dn_low_peff_gives_bcs_iv(self):
        """Dn > 1 AND low Peff → BCS IV."""
        result = predict_absorption_enhancement(
            drug_name="X",
            logP=2.0,
            mw=300.0,
            peff_cm_per_s=5e-6,
            solubility_mg_mL=0.001,
            dose_mg=500.0,  # Dn = 500 / (0.001*250) = 2000 >> 1
        )
        assert result.bcs_class == "IV"


# ---------------------------------------------------------------------------
# Baseline fa tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBaselineFa:
    def test_bcs_i_baseline_fa(self):
        result = predict_absorption_enhancement(**_BCS1)
        assert abs(result.baseline_fa - 0.90) < 1e-9

    def test_bcs_iii_baseline_fa(self):
        result = predict_absorption_enhancement(**_BCS3)
        assert abs(result.baseline_fa - 0.40) < 1e-9

    def test_bcs_ii_baseline_fa_in_range(self):
        result = predict_absorption_enhancement(**_BCS2)
        assert 0.05 <= result.baseline_fa <= 0.85

    def test_bcs_iv_baseline_fa_in_range(self):
        result = predict_absorption_enhancement(**_BCS4)
        assert 0.03 <= result.baseline_fa <= 0.30


# ---------------------------------------------------------------------------
# Strategy fa values in [0, 1]
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStrategyFaValues:
    @pytest.mark.parametrize("drug_params", [_BCS1, _BCS2, _BCS3, _BCS4])
    def test_all_fa_in_unit_interval(self, drug_params):
        result = predict_absorption_enhancement(**drug_params)
        for fa in [
            result.fa_nanosizing,
            result.fa_solid_dispersion,
            result.fa_lipid_formulation,
            result.fa_cyclodextrin,
            result.fa_cocrystal,
            result.best_fa,
            result.baseline_fa,
        ]:
            assert 0.0 <= fa <= 1.0, f"fa={fa} out of [0,1] for {drug_params['drug_name']}"


# ---------------------------------------------------------------------------
# Best strategy tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBestStrategy:
    @pytest.mark.parametrize("drug_params", [_BCS1, _BCS2, _BCS3, _BCS4])
    def test_best_strategy_is_valid(self, drug_params):
        result = predict_absorption_enhancement(**drug_params)
        assert result.best_strategy in _VALID_STRATEGIES

    @pytest.mark.parametrize("drug_params", [_BCS1, _BCS2, _BCS3, _BCS4])
    def test_best_fa_matches_best_strategy(self, drug_params):
        result = predict_absorption_enhancement(**drug_params)
        strategy_fas = {
            "nanosizing": result.fa_nanosizing,
            "solid_dispersion": result.fa_solid_dispersion,
            "lipid_formulation": result.fa_lipid_formulation,
            "cyclodextrin": result.fa_cyclodextrin,
            "cocrystal": result.fa_cocrystal,
        }
        expected_best_fa = max(strategy_fas.values())
        assert abs(result.best_fa - expected_best_fa) < 1e-9

    @pytest.mark.parametrize("drug_params", [_BCS1, _BCS2, _BCS3, _BCS4])
    def test_fold_improvement_ge_one(self, drug_params):
        result = predict_absorption_enhancement(**drug_params)
        assert result.fold_improvement >= 1.0


# ---------------------------------------------------------------------------
# BCS-specific enhancement effectiveness tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnhancementEffectiveness:
    def test_bcs_i_fold_improvement_near_one(self):
        """BCS I: strategies offer minimal benefit (permeability not limited by solubility)."""
        result = predict_absorption_enhancement(**_BCS1)
        # For BCS I, only lipid gives +0.05 so fold ~= (0.9+0.05)/0.9 = 1.056
        assert result.fold_improvement < 1.2

    def test_bcs_ii_significant_improvement(self):
        """BCS II: ASD or lipid should give fold > 2."""
        result = predict_absorption_enhancement(**_BCS2)
        assert result.fold_improvement > 2.0

    def test_bcs_ii_asd_improves_fa(self):
        """BCS II: solid dispersion should give higher fa than baseline."""
        result = predict_absorption_enhancement(**_BCS2)
        assert result.fa_solid_dispersion > result.baseline_fa

    def test_bcs_iv_nanosizing_improves_fa(self):
        """BCS IV: nanosizing should improve fa."""
        result = predict_absorption_enhancement(**_BCS4)
        assert result.fa_nanosizing > result.baseline_fa

    def test_bcs_iii_nanosizing_no_benefit(self):
        """BCS III: permeability-limited, nanosizing doesn't help."""
        result = predict_absorption_enhancement(**_BCS3)
        assert result.fa_nanosizing == result.baseline_fa


# ---------------------------------------------------------------------------
# Result content tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResultContent:
    def test_frozen_dataclass(self):
        result = predict_absorption_enhancement(**_BCS2)
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "other"  # type: ignore[misc]

    def test_drug_name_stored(self):
        result = predict_absorption_enhancement(**_BCS2)
        assert result.drug_name == "BCS2Drug"

    def test_recommendation_non_empty(self):
        result = predict_absorption_enhancement(**_BCS2)
        assert len(result.recommendation) > 0

    def test_notes_non_empty(self):
        result = predict_absorption_enhancement(**_BCS2)
        assert len(result.notes) > 0

    def test_notes_contains_drug_name(self):
        result = predict_absorption_enhancement(**_BCS2)
        assert "BCS2Drug" in result.notes

    def test_sol_enhancement_nanosizing_positive(self):
        result = predict_absorption_enhancement(**_BCS2)
        assert result.solubility_enhancement_nanosizing > 0

    def test_sol_enhancement_asd_positive(self):
        result = predict_absorption_enhancement(**_BCS2)
        assert result.solubility_enhancement_asd > 0


# ---------------------------------------------------------------------------
# screen_formulation_strategies tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScreenFormulationStrategies:
    def test_returns_list(self):
        compounds = [
            {"name": "A", "logP": 4.0, "mw": 350.0, "peff": 5e-4, "solubility_mg_mL": 0.01, "dose_mg": 100.0},
            {"name": "B", "logP": 1.0, "mw": 200.0, "peff": 5e-4, "solubility_mg_mL": 10.0, "dose_mg": 10.0},
        ]
        results = screen_formulation_strategies(compounds)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_sorted_by_fold_improvement_descending(self):
        compounds = [
            # BCS I: low fold improvement
            {"name": "BCS1", "logP": 1.0, "mw": 200.0, "peff": 5e-4, "solubility_mg_mL": 10.0, "dose_mg": 10.0},
            # BCS II: high fold improvement
            {"name": "BCS2", "logP": 4.0, "mw": 350.0, "peff": 5e-4, "solubility_mg_mL": 0.01, "dose_mg": 100.0},
            # BCS IV: moderate improvement
            {"name": "BCS4", "logP": 3.0, "mw": 400.0, "peff": 1e-5, "solubility_mg_mL": 0.005, "dose_mg": 200.0},
        ]
        results = screen_formulation_strategies(compounds)
        folds = [r.fold_improvement for r in results]
        assert folds == sorted(folds, reverse=True)

    def test_single_compound(self):
        compounds = [
            {"name": "Solo", "logP": 3.0, "mw": 300.0, "peff": 5e-4, "solubility_mg_mL": 0.05, "dose_mg": 50.0},
        ]
        results = screen_formulation_strategies(compounds)
        assert len(results) == 1
        assert isinstance(results[0], AbsorptionEnhancementResult)

    def test_empty_list(self):
        results = screen_formulation_strategies([])
        assert results == []


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInputValidation:
    def test_invalid_mw(self):
        with pytest.raises(ValueError, match="mw"):
            predict_absorption_enhancement(
                drug_name="X", logP=2.0, mw=0.0, peff_cm_per_s=1e-4,
                solubility_mg_mL=1.0, dose_mg=100.0,
            )

    def test_invalid_peff(self):
        with pytest.raises(ValueError, match="peff"):
            predict_absorption_enhancement(
                drug_name="X", logP=2.0, mw=300.0, peff_cm_per_s=0.0,
                solubility_mg_mL=1.0, dose_mg=100.0,
            )

    def test_invalid_solubility(self):
        with pytest.raises(ValueError, match="solubility"):
            predict_absorption_enhancement(
                drug_name="X", logP=2.0, mw=300.0, peff_cm_per_s=1e-4,
                solubility_mg_mL=0.0, dose_mg=100.0,
            )

    def test_invalid_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            predict_absorption_enhancement(
                drug_name="X", logP=2.0, mw=300.0, peff_cm_per_s=1e-4,
                solubility_mg_mL=1.0, dose_mg=-5.0,
            )

    def test_negative_logp_accepted(self):
        """logP can be negative (hydrophilic drug)."""
        result = predict_absorption_enhancement(
            drug_name="HydrophilicDrug",
            logP=-2.0,
            mw=150.0,
            peff_cm_per_s=1e-4,
            solubility_mg_mL=50.0,
            dose_mg=10.0,
        )
        assert isinstance(result, AbsorptionEnhancementResult)
