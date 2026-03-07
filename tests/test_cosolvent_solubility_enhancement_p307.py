"""Tests for Phase 307 — Drug Solubility Enhancement by Cosolvents."""

import pytest

from omega_pbpk.prediction.cosolvent_solubility_enhancement import (
    CosolventSolubilityResult,
    compare_cosolvents,
    optimize_cosolvent_fraction,
    predict_cosolvent_solubility,
)

# ---------------------------------------------------------------------------
# Basic return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass_instance(self):
        result = predict_cosolvent_solubility("DrugA", 2.5, 0.01, "PEG400", 0.3)
        assert isinstance(result, CosolventSolubilityResult)

    def test_drug_name_preserved(self):
        result = predict_cosolvent_solubility("TestDrug", 2.0, 0.05, "ethanol", 0.1)
        assert result.drug_name == "TestDrug"

    def test_logp_preserved(self):
        result = predict_cosolvent_solubility("DrugA", 3.0, 0.1, "PEG400", 0.2)
        assert result.logp == pytest.approx(3.0)

    def test_intrinsic_solubility_preserved(self):
        result = predict_cosolvent_solubility("DrugA", 2.0, 0.05, "glycerin", 0.4)
        assert result.intrinsic_solubility_mg_mL == pytest.approx(0.05)

    def test_cosolvent_type_preserved(self):
        result = predict_cosolvent_solubility("DrugA", 2.0, 0.1, "propylene_glycol", 0.2)
        assert result.cosolvent_type == "propylene_glycol"

    def test_cosolvent_fraction_preserved(self):
        result = predict_cosolvent_solubility("DrugA", 2.0, 0.1, "PEG400", 0.25)
        assert result.cosolvent_fraction == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Enhanced solubility >= intrinsic solubility
# ---------------------------------------------------------------------------


class TestSolubilityEnhancement:
    def test_enhanced_ge_intrinsic_lipophilic(self):
        result = predict_cosolvent_solubility("Drug1", 3.0, 0.01, "PEG400", 0.3)
        assert result.enhanced_solubility_mg_mL >= result.intrinsic_solubility_mg_mL

    def test_enhanced_ge_intrinsic_hydrophilic(self):
        result = predict_cosolvent_solubility("Drug2", -1.0, 10.0, "ethanol", 0.1)
        assert result.enhanced_solubility_mg_mL >= result.intrinsic_solubility_mg_mL

    def test_zero_fraction_equals_intrinsic(self):
        result = predict_cosolvent_solubility("Drug3", 2.0, 0.5, "PEG400", 0.0)
        assert result.enhanced_solubility_mg_mL == pytest.approx(result.intrinsic_solubility_mg_mL)

    def test_enhancement_ratio_ge_one(self):
        result = predict_cosolvent_solubility("Drug4", 2.5, 0.01, "ethanol", 0.08)
        assert result.enhancement_ratio >= 1.0


# ---------------------------------------------------------------------------
# Higher logp => higher enhancement_ratio (for positive logp)
# ---------------------------------------------------------------------------


class TestLogpEffect:
    def test_higher_logp_higher_enhancement(self):
        low = predict_cosolvent_solubility("Drug", 1.0, 0.01, "PEG400", 0.3)
        high = predict_cosolvent_solubility("Drug", 4.0, 0.01, "PEG400", 0.3)
        assert high.enhancement_ratio > low.enhancement_ratio

    def test_hydrophilic_minimal_enhancement(self):
        result = predict_cosolvent_solubility("Drug", -2.0, 5.0, "PEG400", 0.3)
        # effective sigma = 1.8 * 0.1 = 0.18, f=0.3 => 10^0.054 ~ 1.13
        assert result.enhancement_ratio == pytest.approx(10 ** (1.8 * 0.1 * 0.3), rel=1e-4)


# ---------------------------------------------------------------------------
# Higher cosolvent fraction => higher enhancement
# ---------------------------------------------------------------------------


class TestFractionEffect:
    def test_higher_fraction_higher_solubility(self):
        low = predict_cosolvent_solubility("Drug", 2.0, 0.01, "PEG400", 0.1)
        high = predict_cosolvent_solubility("Drug", 2.0, 0.01, "PEG400", 0.4)
        assert high.enhanced_solubility_mg_mL > low.enhanced_solubility_mg_mL


# ---------------------------------------------------------------------------
# PEG400 > glycerin for same fraction (higher sigma)
# ---------------------------------------------------------------------------


class TestCosolventComparison:
    def test_peg400_better_than_glycerin(self):
        peg = predict_cosolvent_solubility("Drug", 3.0, 0.01, "PEG400", 0.3)
        gly = predict_cosolvent_solubility("Drug", 3.0, 0.01, "glycerin", 0.3)
        assert peg.enhanced_solubility_mg_mL > gly.enhanced_solubility_mg_mL


# ---------------------------------------------------------------------------
# compare_cosolvents
# ---------------------------------------------------------------------------


class TestCompareCosolvents:
    def test_returns_four_results(self):
        results = compare_cosolvents("Drug", 2.0, 0.01, 0.3)
        assert len(results) == 4

    def test_all_cosolvent_types_present(self):
        results = compare_cosolvents("Drug", 2.0, 0.01, 0.3)
        types = {r.cosolvent_type for r in results}
        assert types == {"PEG400", "ethanol", "propylene_glycol", "glycerin"}

    def test_sorted_descending(self):
        results = compare_cosolvents("Drug", 3.0, 0.01, 0.3)
        solubilities = [r.enhanced_solubility_mg_mL for r in results]
        assert solubilities == sorted(solubilities, reverse=True)

    def test_all_results_are_dataclass(self):
        results = compare_cosolvents("Drug", 2.0, 0.05, 0.2)
        for r in results:
            assert isinstance(r, CosolventSolubilityResult)


# ---------------------------------------------------------------------------
# optimize_cosolvent_fraction
# ---------------------------------------------------------------------------


class TestOptimizeCosolventFraction:
    def test_returns_dataclass(self):
        result = optimize_cosolvent_fraction("Drug", 3.0, 0.01, "PEG400", 1.0)
        assert isinstance(result, CosolventSolubilityResult)

    def test_achieves_target_solubility(self):
        target = 0.5
        result = optimize_cosolvent_fraction("Drug", 3.0, 0.001, "PEG400", target)
        assert result.enhanced_solubility_mg_mL >= target * 0.999  # within numeric tolerance

    def test_does_not_exceed_max_fraction(self):
        result = optimize_cosolvent_fraction("Drug", 3.0, 0.001, "PEG400", 100.0, max_fraction=0.5)
        assert result.cosolvent_fraction <= 0.5 + 1e-10

    def test_caps_at_max_fraction_when_unreachable(self):
        # Set a very high target that requires fraction > max_fraction=0.05
        result = optimize_cosolvent_fraction(
            "Drug", 1.0, 0.001, "glycerin", 1000.0, max_fraction=0.05
        )
        assert result.cosolvent_fraction == pytest.approx(0.05)

    def test_validation_target_le_intrinsic(self):
        with pytest.raises(ValueError):
            optimize_cosolvent_fraction("Drug", 2.0, 1.0, "PEG400", 0.5)

    def test_validation_max_fraction_zero(self):
        with pytest.raises(ValueError):
            optimize_cosolvent_fraction("Drug", 2.0, 0.01, "PEG400", 1.0, max_fraction=0.0)


# ---------------------------------------------------------------------------
# formulation_feasible
# ---------------------------------------------------------------------------


class TestFormulationFeasible:
    def test_feasible_within_ethanol_limit(self):
        result = predict_cosolvent_solubility("Drug", 2.0, 0.01, "ethanol", 0.05)
        assert result.formulation_feasible is True

    def test_not_feasible_above_ethanol_limit(self):
        result = predict_cosolvent_solubility("Drug", 2.0, 0.01, "ethanol", 0.15)
        assert result.formulation_feasible is False

    def test_feasible_peg400_at_limit(self):
        result = predict_cosolvent_solubility("Drug", 2.0, 0.01, "PEG400", 0.50)
        assert result.formulation_feasible is True

    def test_toxicity_concern_fraction_correct(self):
        result = predict_cosolvent_solubility("Drug", 2.0, 0.01, "propylene_glycol", 0.1)
        assert result.toxicity_concern_fraction == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# Solubility class
# ---------------------------------------------------------------------------


class TestSolubilityClass:
    def test_excellent(self):
        result = predict_cosolvent_solubility("Drug", 0.5, 100.0, "PEG400", 0.3)
        assert result.solubility_class == "excellent"

    def test_poor(self):
        result = predict_cosolvent_solubility("Drug", 0.5, 0.001, "glycerin", 0.0)
        assert result.solubility_class == "poor"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_logp_too_low(self):
        with pytest.raises(ValueError):
            predict_cosolvent_solubility("Drug", -6.0, 0.1, "PEG400", 0.2)

    def test_logp_too_high(self):
        with pytest.raises(ValueError):
            predict_cosolvent_solubility("Drug", 11.0, 0.1, "PEG400", 0.2)

    def test_negative_solubility(self):
        with pytest.raises(ValueError):
            predict_cosolvent_solubility("Drug", 2.0, -0.1, "PEG400", 0.2)

    def test_zero_solubility(self):
        with pytest.raises(ValueError):
            predict_cosolvent_solubility("Drug", 2.0, 0.0, "PEG400", 0.2)

    def test_fraction_above_one(self):
        with pytest.raises(ValueError):
            predict_cosolvent_solubility("Drug", 2.0, 0.1, "PEG400", 1.5)

    def test_fraction_below_zero(self):
        with pytest.raises(ValueError):
            predict_cosolvent_solubility("Drug", 2.0, 0.1, "PEG400", -0.1)

    def test_invalid_cosolvent(self):
        with pytest.raises(ValueError):
            predict_cosolvent_solubility("Drug", 2.0, 0.1, "acetone", 0.2)
