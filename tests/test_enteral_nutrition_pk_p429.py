"""Tests for Phase 429 — Enteral Nutrition PK interactions."""

import pytest

from omega_pbpk.clinical.enteral_nutrition_pk_p429 import (
    FORMULAS,
    EnteralNutritionPKResult,
    assess_enteral_nutrition_pk,
    screen_formulas,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base(**kwargs):
    defaults = dict(
        drug_name="phenytoin",
        formula_type="standard",
        feeding_method="continuous",
        baseline_dose_mg=300.0,
        baseline_bioavailability=0.9,
        drug_pka=8.3,
        ionization_type="acid",
        drug_mw_Da=252.0,
        logp=2.5,
    )
    defaults.update(kwargs)
    return assess_enteral_nutrition_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_return_type(self):
        result = _base()
        assert isinstance(result, EnteralNutritionPKResult)


# ---------------------------------------------------------------------------
# Field preservation
# ---------------------------------------------------------------------------


class TestFieldPreservation:
    def test_drug_name_preserved(self):
        result = _base(drug_name="ciprofloxacin")
        assert result.drug_name == "ciprofloxacin"

    def test_formula_type_preserved(self):
        result = _base(formula_type="high_protein")
        assert result.formula_type == "high_protein"

    def test_feeding_method_preserved(self):
        result = _base(feeding_method="bolus")
        assert result.feeding_method == "bolus"

    def test_baseline_dose_preserved(self):
        result = _base(baseline_dose_mg=200.0)
        assert result.baseline_dose_mg == 200.0

    def test_baseline_bioavailability_preserved(self):
        result = _base(baseline_bioavailability=0.5)
        assert result.bioavailability_baseline == 0.5


# ---------------------------------------------------------------------------
# Bioavailability values
# ---------------------------------------------------------------------------


class TestBioavailability:
    def test_bioavailability_with_en_le_baseline(self):
        """EN interaction should reduce or maintain bioavailability."""
        result = _base()
        assert result.bioavailability_with_en <= result.bioavailability_baseline + 1e-9

    def test_bioavailability_ratio_between_0_and_1(self):
        result = _base()
        assert 0.0 < result.bioavailability_ratio <= 1.0

    def test_binding_loss_pct_nonnegative(self):
        result = _base()
        assert result.binding_loss_pct >= 0.0

    def test_gastric_emptying_factor_positive(self):
        result = _base()
        assert result.gastric_emptying_factor > 0.0


# ---------------------------------------------------------------------------
# Formula differences
# ---------------------------------------------------------------------------


class TestFormulaDifferences:
    def test_high_protein_more_binding_than_elemental(self):
        high_protein = _base(
            formula_type="high_protein", ionization_type="neutral", drug_mw_Da=300.0
        )
        elemental = _base(formula_type="elemental", ionization_type="neutral", drug_mw_Da=300.0)
        assert high_protein.binding_loss_pct > elemental.binding_loss_pct

    def test_elemental_lower_binding_than_standard(self):
        standard = _base(formula_type="standard", ionization_type="neutral", drug_mw_Da=300.0)
        elemental = _base(formula_type="elemental", ionization_type="neutral", drug_mw_Da=300.0)
        assert elemental.binding_loss_pct < standard.binding_loss_pct


# ---------------------------------------------------------------------------
# Clinical significance
# ---------------------------------------------------------------------------


class TestClinicalSignificance:
    def test_major_when_ratio_below_half(self):
        # Use high_protein formula, high_mw drug, acid ionization to push ratio down
        result = assess_enteral_nutrition_pk(
            drug_name="drug",
            formula_type="high_protein",
            feeding_method="continuous",
            baseline_dose_mg=100.0,
            baseline_bioavailability=0.5,
            drug_pka=7.0,
            ionization_type="acid",
            drug_mw_Da=600.0,  # > 500 Da
            logp=2.0,
        )
        if result.bioavailability_ratio < 0.5:
            assert result.clinical_significance == "major"

    def test_none_when_ratio_above_09(self):
        # Elemental formula, no ionization, low MW, bolus
        result = assess_enteral_nutrition_pk(
            drug_name="drug",
            formula_type="elemental",
            feeding_method="bolus",
            baseline_dose_mg=100.0,
            baseline_bioavailability=0.9,
            drug_pka=7.0,
            ionization_type="neutral",
            drug_mw_Da=200.0,
            logp=1.0,
        )
        if result.bioavailability_ratio > 0.9:
            assert result.clinical_significance == "none"


# ---------------------------------------------------------------------------
# Adjusted dose
# ---------------------------------------------------------------------------


class TestAdjustedDose:
    def test_adjusted_dose_ge_baseline_when_reduced(self):
        """If bioavailability reduced, adjusted dose should be >= baseline."""
        result = _base()
        if result.bioavailability_ratio < 1.0:
            assert result.adjusted_dose_mg >= result.baseline_dose_mg - 1e-9

    def test_adjusted_dose_capped_at_3x(self):
        result = _base(baseline_dose_mg=100.0)
        assert result.adjusted_dose_mg <= 100.0 * 3.0 + 1e-9


# ---------------------------------------------------------------------------
# Management strategy
# ---------------------------------------------------------------------------


class TestManagementStrategy:
    def test_management_strategy_non_empty(self):
        result = _base()
        assert len(result.management_strategy) > 0

    def test_notes_non_empty(self):
        result = _base()
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Screen formulas
# ---------------------------------------------------------------------------


class TestScreenFormulas:
    def test_screen_returns_4_results(self):
        results = screen_formulas(
            drug_name="phenytoin",
            baseline_dose_mg=300.0,
            baseline_bioavailability=0.9,
        )
        assert len(results) == 4

    def test_screen_sorted_descending(self):
        results = screen_formulas(
            drug_name="phenytoin",
            baseline_dose_mg=300.0,
            baseline_bioavailability=0.9,
        )
        ratios = [r.bioavailability_ratio for r in results]
        assert ratios == sorted(ratios, reverse=True)

    def test_screen_all_formulas_represented(self):
        results = screen_formulas(
            drug_name="drug",
            baseline_dose_mg=100.0,
            baseline_bioavailability=0.8,
        )
        formula_types = {r.formula_type for r in results}
        assert formula_types == set(FORMULAS.keys())


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_formula_raises(self):
        with pytest.raises(ValueError):
            _base(formula_type="soy_based")

    def test_invalid_feeding_method_raises(self):
        with pytest.raises(ValueError):
            _base(feeding_method="drip")

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            _base(baseline_dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            _base(baseline_dose_mg=-10.0)

    def test_zero_bioavailability_raises(self):
        with pytest.raises(ValueError):
            _base(baseline_bioavailability=0.0)

    def test_bioavailability_above_1_raises(self):
        with pytest.raises(ValueError):
            _base(baseline_bioavailability=1.1)
