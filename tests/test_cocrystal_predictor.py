"""Tests for Phase 810 — cocrystal_predictor module."""

import dataclasses

import pytest

from omega_pbpk.prediction.cocrystal_predictor import (
    CocrystalResult,
    predict_cocrystal_enhancement,
)

# ---------------------------------------------------------------------------
# Return type and frozen dataclass
# ---------------------------------------------------------------------------


class TestReturnTypeAndFrozen:
    def test_returns_correct_type(self):
        result = predict_cocrystal_enhancement(
            drug_name="DrugA",
            logp=2.5,
            mw_da=300.0,
        )
        assert isinstance(result, CocrystalResult)

    def test_result_is_frozen(self):
        result = predict_cocrystal_enhancement(
            drug_name="DrugA",
            logp=2.5,
            mw_da=300.0,
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            result.drug_name = "Changed"  # type: ignore

    def test_all_fields_present(self):
        result = predict_cocrystal_enhancement(
            drug_name="DrugA",
            logp=2.5,
            mw_da=300.0,
        )
        for field in dataclasses.fields(result):
            assert hasattr(result, field.name)


# ---------------------------------------------------------------------------
# Fold increase > 1 for all strategies
# ---------------------------------------------------------------------------


class TestFoldIncreasePositive:
    @pytest.mark.parametrize("strategy", ["cocrystal", "salt", "asd", "cyclodextrin"])
    def test_fold_increase_above_1(self, strategy):
        result = predict_cocrystal_enhancement(
            drug_name="Drug",
            logp=2.5,
            mw_da=300.0,
            strategy=strategy,
        )
        assert result.predicted_solubility_fold_increase > 1.0

    @pytest.mark.parametrize("strategy", ["cocrystal", "salt", "asd", "cyclodextrin"])
    def test_dissolution_fold_positive(self, strategy):
        result = predict_cocrystal_enhancement(
            drug_name="Drug",
            logp=2.5,
            mw_da=300.0,
            strategy=strategy,
        )
        assert result.predicted_dissolution_rate_fold > 0.0


# ---------------------------------------------------------------------------
# Strategy in expected set
# ---------------------------------------------------------------------------


class TestStrategyField:
    @pytest.mark.parametrize("strategy", ["cocrystal", "salt", "asd", "cyclodextrin"])
    def test_strategy_field_matches(self, strategy):
        result = predict_cocrystal_enhancement(
            drug_name="Drug",
            logp=2.5,
            mw_da=300.0,
            strategy=strategy,
        )
        assert result.strategy == strategy

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            predict_cocrystal_enhancement(
                drug_name="Drug",
                logp=2.5,
                mw_da=300.0,
                strategy="nanoparticle",
            )


# ---------------------------------------------------------------------------
# ASD gives highest fold but shorter parachute
# ---------------------------------------------------------------------------


class TestASDCharacteristics:
    def test_asd_highest_fold(self):
        asd = predict_cocrystal_enhancement("Drug", logp=2.5, mw_da=300.0, strategy="asd")
        cocrystal = predict_cocrystal_enhancement(
            "Drug", logp=2.5, mw_da=300.0, strategy="cocrystal"
        )
        cyclodextrin = predict_cocrystal_enhancement(
            "Drug", logp=2.5, mw_da=300.0, strategy="cyclodextrin"
        )
        assert asd.predicted_solubility_fold_increase > cocrystal.predicted_solubility_fold_increase
        assert (
            asd.predicted_solubility_fold_increase > cyclodextrin.predicted_solubility_fold_increase
        )

    def test_asd_shorter_parachute_than_cocrystal(self):
        asd = predict_cocrystal_enhancement("Drug", logp=2.5, mw_da=300.0, strategy="asd")
        cocrystal = predict_cocrystal_enhancement(
            "Drug", logp=2.5, mw_da=300.0, strategy="cocrystal"
        )
        assert asd.parachute_duration_h < cocrystal.parachute_duration_h

    def test_asd_stability_conditional(self):
        result = predict_cocrystal_enhancement("Drug", logp=2.5, mw_da=300.0, strategy="asd")
        assert result.predicted_stability == "conditional"


# ---------------------------------------------------------------------------
# Salt strategy for ionizable drug → high fold
# ---------------------------------------------------------------------------


class TestSaltStrategy:
    def test_salt_basic_drug_high_fold(self):
        """Basic drug (pKa=9) with acid coformer (pKa=4) → 50x fold."""
        result = predict_cocrystal_enhancement(
            drug_name="BasicDrug",
            logp=2.0,
            mw_da=250.0,
            pka=9.0,
            ionization_type="base",
            strategy="salt",
            coformer_name="HCl",
            coformer_pka=4.0,
        )
        assert result.predicted_solubility_fold_increase >= 20.0

    def test_salt_higher_than_cocrystal_for_ionizable(self):
        """Salt should outperform neutral cocrystal for ionizable drug."""
        salt_result = predict_cocrystal_enhancement(
            drug_name="Drug",
            logp=2.0,
            mw_da=250.0,
            pka=9.0,
            strategy="salt",
            coformer_pka=4.0,
        )
        cocrystal_result = predict_cocrystal_enhancement(
            drug_name="Drug",
            logp=2.0,
            mw_da=250.0,
            pka=9.0,
            strategy="cocrystal",
            coformer_pka=4.0,
        )
        assert (
            salt_result.predicted_solubility_fold_increase
            > cocrystal_result.predicted_solubility_fold_increase
        )


# ---------------------------------------------------------------------------
# Cocrystal pKa-dependent enhancement
# ---------------------------------------------------------------------------


class TestCocrystalPKaDependence:
    def test_large_pka_diff_gives_higher_fold(self):
        """DeltapKa > 2 → salt-like 8x vs neutral 3x."""
        salt_like = predict_cocrystal_enhancement(
            drug_name="Drug",
            logp=2.5,
            mw_da=300.0,
            pka=9.0,
            strategy="cocrystal",
            coformer_pka=4.0,  # diff = 5 > 2
        )
        neutral = predict_cocrystal_enhancement(
            drug_name="Drug",
            logp=2.5,
            mw_da=300.0,
            pka=7.0,
            strategy="cocrystal",
            coformer_pka=7.0,  # diff = 0
        )
        assert (
            salt_like.predicted_solubility_fold_increase
            > neutral.predicted_solubility_fold_increase
        )


# ---------------------------------------------------------------------------
# Cyclodextrin logP dependence
# ---------------------------------------------------------------------------


class TestCyclodextrin:
    def test_optimal_logp_range_higher_fold(self):
        """logP in [2,4] → 10x; logP=5 → 5x."""
        optimal = predict_cocrystal_enhancement(
            "Drug", logp=3.0, mw_da=300.0, strategy="cyclodextrin"
        )
        suboptimal = predict_cocrystal_enhancement(
            "Drug", logp=5.0, mw_da=300.0, strategy="cyclodextrin"
        )
        assert (
            optimal.predicted_solubility_fold_increase
            > suboptimal.predicted_solubility_fold_increase
        )


# ---------------------------------------------------------------------------
# Bioavailability prediction label
# ---------------------------------------------------------------------------


class TestBioavailabilityLabel:
    def test_asd_significantly_improved(self):
        result = predict_cocrystal_enhancement("Drug", logp=2.5, mw_da=300.0, strategy="asd")
        assert result.bioavailability_prediction == "significantly_improved"

    def test_neutral_cocrystal_moderately_or_marginal(self):
        result = predict_cocrystal_enhancement(
            "Drug", logp=2.5, mw_da=300.0, pka=7.0, strategy="cocrystal", coformer_pka=7.0
        )
        assert result.bioavailability_prediction in ("moderately_improved", "marginal")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError):
            predict_cocrystal_enhancement(
                "Drug", logp=2.5, mw_da=300.0, intrinsic_solubility_mg_L=0.0
            )

    def test_negative_solubility_raises(self):
        with pytest.raises(ValueError):
            predict_cocrystal_enhancement(
                "Drug", logp=2.5, mw_da=300.0, intrinsic_solubility_mg_L=-5.0
            )


# ---------------------------------------------------------------------------
# Spring factor sanity
# ---------------------------------------------------------------------------


class TestSpringFactor:
    @pytest.mark.parametrize("strategy", ["cocrystal", "salt", "asd", "cyclodextrin"])
    def test_spring_factor_positive(self, strategy):
        result = predict_cocrystal_enhancement("Drug", logp=2.5, mw_da=300.0, strategy=strategy)
        assert result.spring_factor > 0.0
