"""Tests for Phase 189: solubility_ph_profile_p189."""

import pytest

from omega_pbpk.prediction.solubility_ph_profile_p189 import (
    SolubilityPhProfileResult,
    compare_ionization_types,
    generate_ph_profile,
)

# ---------------------------------------------------------------------------
# Return type and basic field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass_instance(self):
        result = generate_ph_profile("DrugA", 0.1, 4.0, "acid")
        assert isinstance(result, SolubilityPhProfileResult)

    def test_drug_name_preserved(self):
        result = generate_ph_profile("Ibuprofen", 0.05, 4.5, "acid")
        assert result.drug_name == "Ibuprofen"

    def test_ionization_type_preserved(self):
        result = generate_ph_profile("Atenolol", 13.5, 9.6, "base")
        assert result.ionization_type == "base"

    def test_pka_preserved(self):
        result = generate_ph_profile("DrugX", 1.0, 6.0, "neutral")
        assert result.pka == 6.0

    def test_intrinsic_solubility_preserved(self):
        result = generate_ph_profile("DrugX", 2.5, 5.0, "acid")
        assert result.intrinsic_solubility_mg_mL == 2.5

    def test_ph_values_list_type(self):
        result = generate_ph_profile("DrugX", 1.0, 5.0, "base")
        assert isinstance(result.ph_values, list)
        assert len(result.ph_values) > 0

    def test_solubility_list_type(self):
        result = generate_ph_profile("DrugX", 1.0, 5.0, "base")
        assert isinstance(result.solubility_mg_mL, list)
        assert len(result.solubility_mg_mL) == len(result.ph_values)


# ---------------------------------------------------------------------------
# Acid: solubility increases with pH
# ---------------------------------------------------------------------------


class TestAcidProfile:
    def test_acid_increases_with_ph(self):
        result = generate_ph_profile("AcidDrug", 0.1, 5.0, "acid")
        sol = result.solubility_mg_mL
        for i in range(len(sol) - 1):
            assert sol[i] <= sol[i + 1], (
                f"Acid solubility should be non-decreasing with pH at index {i}"
            )

    def test_acid_max_solubility_at_high_ph(self):
        result = generate_ph_profile("AcidDrug", 0.1, 5.0, "acid")
        assert result.ph_at_max_solubility == max(result.ph_values)

    def test_acid_min_solubility_at_low_ph(self):
        result = generate_ph_profile("AcidDrug", 0.1, 5.0, "acid")
        assert result.ph_at_min_solubility == min(result.ph_values)

    def test_acid_at_pka_solubility_is_2x_s0(self):
        s0 = 0.1
        pka = 5.0
        result = generate_ph_profile("AcidDrug", s0, pka, "acid", ph_range=[pka])
        # At pH = pKa: S = S0 * (1 + 10^0) = 2 * S0
        assert abs(result.solubility_mg_mL[0] - 2 * s0) < 1e-9

    def test_acid_fold_increase_ge_1(self):
        result = generate_ph_profile("AcidDrug", 0.1, 5.0, "acid")
        assert result.fold_increase >= 1.0


# ---------------------------------------------------------------------------
# Base: solubility increases with lower pH (decreases as pH increases)
# ---------------------------------------------------------------------------


class TestBaseProfile:
    def test_base_decreases_with_ph(self):
        result = generate_ph_profile("BaseDrug", 0.1, 8.0, "base")
        sol = result.solubility_mg_mL
        for i in range(len(sol) - 1):
            assert sol[i] >= sol[i + 1], (
                f"Base solubility should be non-increasing with pH at index {i}"
            )

    def test_base_max_solubility_at_low_ph(self):
        result = generate_ph_profile("BaseDrug", 0.1, 8.0, "base")
        assert result.ph_at_max_solubility == min(result.ph_values)

    def test_base_min_solubility_at_high_ph(self):
        result = generate_ph_profile("BaseDrug", 0.1, 8.0, "base")
        assert result.ph_at_min_solubility == max(result.ph_values)

    def test_base_at_pka_solubility_is_2x_s0(self):
        s0 = 0.2
        pka = 7.0
        result = generate_ph_profile("BaseDrug", s0, pka, "base", ph_range=[pka])
        assert abs(result.solubility_mg_mL[0] - 2 * s0) < 1e-9

    def test_base_fold_increase_ge_1(self):
        result = generate_ph_profile("BaseDrug", 0.1, 8.0, "base")
        assert result.fold_increase >= 1.0


# ---------------------------------------------------------------------------
# Neutral: constant solubility
# ---------------------------------------------------------------------------


class TestNeutralProfile:
    def test_neutral_constant_solubility(self):
        s0 = 1.5
        result = generate_ph_profile("NeutralDrug", s0, 7.0, "neutral")
        for sol in result.solubility_mg_mL:
            assert abs(sol - s0) < 1e-12

    def test_neutral_fold_increase_is_1(self):
        result = generate_ph_profile("NeutralDrug", 1.0, 7.0, "neutral")
        assert abs(result.fold_increase - 1.0) < 1e-12

    def test_neutral_max_equals_min(self):
        result = generate_ph_profile("NeutralDrug", 0.5, 7.0, "neutral")
        assert result.max_solubility_mg_mL == result.min_solubility_mg_mL


# ---------------------------------------------------------------------------
# fold_increase, min/max positivity
# ---------------------------------------------------------------------------


class TestFoldAndMinMax:
    def test_fold_increase_acid(self):
        result = generate_ph_profile("DrugA", 0.1, 4.0, "acid")
        expected = result.max_solubility_mg_mL / result.min_solubility_mg_mL
        assert abs(result.fold_increase - expected) < 1e-9

    def test_min_solubility_positive(self):
        result = generate_ph_profile("DrugB", 0.1, 5.0, "base")
        assert result.min_solubility_mg_mL > 0

    def test_max_solubility_positive(self):
        result = generate_ph_profile("DrugB", 0.1, 5.0, "base")
        assert result.max_solubility_mg_mL > 0

    def test_max_gte_min(self):
        result = generate_ph_profile("DrugA", 0.1, 5.0, "acid")
        assert result.max_solubility_mg_mL >= result.min_solubility_mg_mL


# ---------------------------------------------------------------------------
# compare_ionization_types
# ---------------------------------------------------------------------------


class TestCompareIonizationTypes:
    def test_returns_three_results(self):
        results = compare_ionization_types("TestDrug", 1.0, 5.0)
        assert len(results) == 3

    def test_all_are_dataclass_instances(self):
        results = compare_ionization_types("TestDrug", 1.0, 5.0)
        for r in results:
            assert isinstance(r, SolubilityPhProfileResult)

    def test_ionization_types_present(self):
        results = compare_ionization_types("TestDrug", 1.0, 5.0)
        types = {r.ionization_type for r in results}
        assert types == {"acid", "base", "neutral"}

    def test_same_drug_name_all_results(self):
        results = compare_ionization_types("Warfarin", 0.05, 5.0)
        for r in results:
            assert r.drug_name == "Warfarin"

    def test_same_s0_all_results(self):
        s0 = 2.5
        results = compare_ionization_types("DrugC", s0, 6.0)
        for r in results:
            assert r.intrinsic_solubility_mg_mL == s0


# ---------------------------------------------------------------------------
# Custom ph_range
# ---------------------------------------------------------------------------


class TestCustomPhRange:
    def test_custom_ph_range_used(self):
        custom = [2.0, 4.0, 6.0, 8.0]
        result = generate_ph_profile("DrugD", 1.0, 5.0, "acid", ph_range=custom)
        assert result.ph_values == custom
        assert len(result.solubility_mg_mL) == 4

    def test_single_ph_value(self):
        result = generate_ph_profile("DrugD", 1.0, 5.0, "acid", ph_range=[7.4])
        assert len(result.ph_values) == 1
        # At pH=7.4, acid: S = S0*(1 + 10^(7.4-5)) = S0*(1+10^2.4)
        expected = 1.0 * (1.0 + 10.0 ** (7.4 - 5.0))
        assert abs(result.solubility_mg_mL[0] - expected) < 1e-9


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_s0_zero_raises(self):
        with pytest.raises(ValueError):
            generate_ph_profile("DrugX", 0.0, 5.0, "acid")

    def test_s0_negative_raises(self):
        with pytest.raises(ValueError):
            generate_ph_profile("DrugX", -1.0, 5.0, "acid")

    def test_pka_below_0_raises(self):
        with pytest.raises(ValueError):
            generate_ph_profile("DrugX", 1.0, -1.0, "acid")

    def test_pka_above_14_raises(self):
        with pytest.raises(ValueError):
            generate_ph_profile("DrugX", 1.0, 15.0, "base")

    def test_invalid_ionization_type_raises(self):
        with pytest.raises(ValueError):
            generate_ph_profile("DrugX", 1.0, 5.0, "zwitterion")

    def test_valid_boundary_pka_0(self):
        result = generate_ph_profile("DrugX", 1.0, 0.0, "acid")
        assert isinstance(result, SolubilityPhProfileResult)

    def test_valid_boundary_pka_14(self):
        result = generate_ph_profile("DrugX", 1.0, 14.0, "base")
        assert isinstance(result, SolubilityPhProfileResult)
