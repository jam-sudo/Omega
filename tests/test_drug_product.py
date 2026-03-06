"""Tests for omega_pbpk.biopharmaceutics.drug_product module."""

import pytest

from omega_pbpk.biopharmaceutics.drug_product import (
    DrugProductResult,
    classify_formulation,
    formulation_comparison,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dissolution_number(ka_per_h: float, dissolution_time_h: float) -> float:
    """Dn = ka * t_diss  (reference formula used in the module)."""
    return ka_per_h * dissolution_time_h


def _absorption_number(ka_per_h: float, Peff_cm_s: float) -> float:
    """An depends on Peff; proxy check — we test sign/direction rather than
    exact formula to stay implementation-agnostic."""
    return Peff_cm_s  # placeholder for directional reasoning


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_drug_product_result_instance(self):
        result = classify_formulation("Aspirin", 500, 3.0)
        assert isinstance(result, DrugProductResult)

    def test_dataclass_fields_present(self):
        result = classify_formulation("Aspirin", 500, 3.0)
        for field in (
            "drug_name",
            "formulation_type",
            "dose_mg",
            "dose_number",
            "dissolution_number",
            "absorption_number",
            "bcs_class",
            "solubility_limited",
            "permeability_limited",
            "recommended_strategy",
        ):
            assert hasattr(result, field), f"Missing field: {field}"


# ---------------------------------------------------------------------------
# 2. BCS class membership
# ---------------------------------------------------------------------------


class TestBCSClassMembership:
    def test_bcs_class_is_valid_value(self):
        result = classify_formulation("Drug", 100, 1.0)
        assert result.bcs_class in {"I", "II", "III", "IV"}

    def test_bcs_class_i_high_sol_high_perm(self):
        # Do < 1: dose=50 mg, solubility=5 mg/mL → Do = 50/(5*250)=0.04 < 1
        # An > 1: high Peff
        result = classify_formulation("BCS_I_drug", 50, 5.0, Peff_cm_s=2e-3, ka_per_h=5.0)
        assert result.bcs_class == "I"

    def test_bcs_class_ii_low_sol_high_perm(self):
        # Do >= 1: dose=500 mg, solubility=0.1 mg/mL → Do = 500/(0.1*250)=20 >= 1
        # An > 1: high Peff
        result = classify_formulation("BCS_II_drug", 500, 0.1, Peff_cm_s=2e-3, ka_per_h=5.0)
        assert result.bcs_class == "II"

    def test_bcs_class_iii_high_sol_low_perm(self):
        # Do < 1: dose=10 mg, solubility=5 mg/mL → Do=0.008 < 1
        # An <= 1: very low Peff
        result = classify_formulation("BCS_III_drug", 10, 5.0, Peff_cm_s=1e-8, ka_per_h=0.1)
        assert result.bcs_class == "III"

    def test_bcs_class_iv_low_sol_low_perm(self):
        # Do >= 1: dose=500 mg, solubility=0.05 mg/mL → Do=40 >= 1
        # An <= 1: very low Peff
        result = classify_formulation("BCS_IV_drug", 500, 0.05, Peff_cm_s=1e-8, ka_per_h=0.1)
        assert result.bcs_class == "IV"


# ---------------------------------------------------------------------------
# 3. Dimensionless numbers must be positive
# ---------------------------------------------------------------------------


class TestDimensionlessNumbers:
    def test_dose_number_positive(self):
        result = classify_formulation("Drug", 100, 1.0)
        assert result.dose_number > 0

    def test_dissolution_number_positive(self):
        result = classify_formulation("Drug", 100, 1.0)
        assert result.dissolution_number > 0

    def test_absorption_number_positive(self):
        result = classify_formulation("Drug", 100, 1.0)
        assert result.absorption_number > 0

    def test_dose_number_scales_with_dose(self):
        r_low = classify_formulation("Drug", 50, 1.0)
        r_high = classify_formulation("Drug", 200, 1.0)
        assert r_high.dose_number > r_low.dose_number

    def test_dose_number_inversely_scales_with_solubility(self):
        r_low_sol = classify_formulation("Drug", 100, 0.5)
        r_high_sol = classify_formulation("Drug", 100, 5.0)
        assert r_low_sol.dose_number > r_high_sol.dose_number


# ---------------------------------------------------------------------------
# 4. solubility_limited flag
# ---------------------------------------------------------------------------


class TestSolubilityLimited:
    def test_solubility_limited_true_when_do_ge_1(self):
        # Do = 500/(0.1*250) = 20 >= 1
        result = classify_formulation("Drug", 500, 0.1)
        assert result.dose_number >= 1
        assert result.solubility_limited is True

    def test_solubility_limited_false_when_do_lt_1(self):
        # Do = 10/(5*250) = 0.008 < 1
        result = classify_formulation("Drug", 10, 5.0, Peff_cm_s=3e-4)
        assert result.dose_number < 1
        assert result.solubility_limited is False

    def test_solubility_limited_consistent_with_dose_number(self):
        result = classify_formulation("Drug", 250, 1.5)
        assert result.solubility_limited == (result.dose_number >= 1)


# ---------------------------------------------------------------------------
# 5. permeability_limited flag
# ---------------------------------------------------------------------------


class TestPermeabilityLimited:
    def test_permeability_limited_true_when_an_le_1(self):
        result = classify_formulation("Drug", 10, 5.0, Peff_cm_s=1e-8, ka_per_h=0.1)
        assert result.absorption_number <= 1
        assert result.permeability_limited is True

    def test_permeability_limited_false_when_an_gt_1(self):
        result = classify_formulation("Drug", 10, 5.0, Peff_cm_s=2e-3, ka_per_h=5.0)
        assert result.permeability_limited is False  # with Peff=1e-3, ka=5 -> An>1

    def test_permeability_limited_consistent_with_absorption_number(self):
        result = classify_formulation("Drug", 100, 1.0, Peff_cm_s=1e-5, ka_per_h=0.5)
        assert result.permeability_limited == (result.absorption_number <= 1)


# ---------------------------------------------------------------------------
# 6. High-solubility drug → BCS I or III (Do < 1)
# ---------------------------------------------------------------------------


class TestHighSolubilityDrug:
    def test_high_solubility_gives_bcs_i_or_iii(self):
        # Very high solubility ensures Do < 1 regardless of dose
        result = classify_formulation("HighSol", 100, 100.0, Peff_cm_s=1e-4, ka_per_h=1.0)
        assert result.bcs_class in {"I", "III"}
        assert result.dose_number < 1

    def test_high_solubility_not_solubility_limited(self):
        result = classify_formulation("HighSol", 50, 50.0)
        assert result.solubility_limited is False


# ---------------------------------------------------------------------------
# 7. ValueError for invalid inputs
# ---------------------------------------------------------------------------


class TestValueErrors:
    def test_raises_on_zero_dose(self):
        with pytest.raises(ValueError):
            classify_formulation("Drug", 0, 1.0)

    def test_raises_on_negative_dose(self):
        with pytest.raises(ValueError):
            classify_formulation("Drug", -100, 1.0)

    def test_raises_on_zero_solubility(self):
        with pytest.raises(ValueError):
            classify_formulation("Drug", 100, 0.0)

    def test_raises_on_negative_solubility(self):
        with pytest.raises(ValueError):
            classify_formulation("Drug", 100, -1.0)


# ---------------------------------------------------------------------------
# 8. recommended_strategy non-empty
# ---------------------------------------------------------------------------


class TestRecommendedStrategy:
    @pytest.mark.parametrize(
        "bcs_params",
        [
            # (dose_mg, sol, Peff, ka) targeting BCS I, II, III, IV
            (10, 5.0, 1e-3, 5.0),  # BCS I
            (500, 0.1, 1e-3, 5.0),  # BCS II
            (10, 5.0, 1e-8, 0.1),  # BCS III
            (500, 0.05, 1e-8, 0.1),  # BCS IV
        ],
    )
    def test_recommended_strategy_non_empty(self, bcs_params):
        dose, sol, peff, ka = bcs_params
        result = classify_formulation("Drug", dose, sol, Peff_cm_s=peff, ka_per_h=ka)
        assert isinstance(result.recommended_strategy, str)
        assert len(result.recommended_strategy.strip()) > 0


# ---------------------------------------------------------------------------
# 9. formulation_comparison
# ---------------------------------------------------------------------------


class TestFormulationComparison:
    def test_returns_list(self):
        types = ["tablet", "capsule", "solution"]
        results = formulation_comparison("Drug", 100, 1.0, types)
        assert isinstance(results, list)

    def test_length_matches_formulation_types(self):
        types = ["tablet", "capsule", "solution", "suspension"]
        results = formulation_comparison("Drug", 100, 1.0, types)
        assert len(results) == len(types)

    def test_single_formulation_type(self):
        results = formulation_comparison("Drug", 100, 1.0, ["tablet"])
        assert len(results) == 1

    def test_all_results_are_drug_product_result(self):
        types = ["tablet", "solution"]
        results = formulation_comparison("Drug", 100, 1.0, types)
        for r in results:
            assert isinstance(r, DrugProductResult)

    def test_formulation_type_preserved_in_each_result(self):
        types = ["tablet", "capsule", "solution"]
        results = formulation_comparison("Drug", 100, 1.0, types)
        for expected_type, result in zip(types, results, strict=True):
            assert result.formulation_type == expected_type


# ---------------------------------------------------------------------------
# 10. Field preservation
# ---------------------------------------------------------------------------


class TestFieldPreservation:
    def test_drug_name_preserved(self):
        result = classify_formulation("Ibuprofen", 400, 0.21)
        assert result.drug_name == "Ibuprofen"

    def test_formulation_type_preserved_default(self):
        result = classify_formulation("Drug", 100, 1.0)
        assert result.formulation_type == "tablet"

    def test_formulation_type_preserved_custom(self):
        result = classify_formulation("Drug", 100, 1.0, formulation_type="capsule")
        assert result.formulation_type == "capsule"

    def test_dose_mg_preserved(self):
        result = classify_formulation("Drug", 250, 1.0)
        assert result.dose_mg == 250

    def test_drug_name_preserved_in_comparison(self):
        results = formulation_comparison("Metformin", 500, 300.0, ["tablet", "solution"])
        for r in results:
            assert r.drug_name == "Metformin"
