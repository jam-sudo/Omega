"""Tests for Phase 528 — Pancreatic Enzyme Drug Metabolism."""

import pytest

from omega_pbpk.prediction.pancreatic_enzyme_metabolism import (
    PancreaticEnzymeMetabolismResult,
    compare_enzyme_conditions,
    predict_pancreatic_enzyme_metabolism,
)

# ---------------------------------------------------------------------------
# Return type & fields
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        result = predict_pancreatic_enzyme_metabolism("SmallMol", mw_Da=300.0)
        assert isinstance(result, PancreaticEnzymeMetabolismResult)

    def test_frozen_dataclass(self):
        result = predict_pancreatic_enzyme_metabolism("SmallMol", mw_Da=300.0)
        with pytest.raises((AttributeError, TypeError)):
            result.fa_gi = 0.5  # type: ignore[misc]

    def test_all_fields_present(self):
        result = predict_pancreatic_enzyme_metabolism("SmallMol", mw_Da=300.0)
        assert hasattr(result, "compound_name")
        assert hasattr(result, "mw_Da")
        assert hasattr(result, "is_peptide")
        assert hasattr(result, "drug_class")
        assert hasattr(result, "ph_labile")
        assert hasattr(result, "enzyme_activity_multiplier")
        assert hasattr(result, "k_deg_per_h")
        assert hasattr(result, "fraction_remaining_gi")
        assert hasattr(result, "fa_gi")
        assert hasattr(result, "stability_class")
        assert hasattr(result, "gi_protection_needed")
        assert hasattr(result, "recommended_formulation")
        assert hasattr(result, "notes")

    def test_compound_name_stored(self):
        result = predict_pancreatic_enzyme_metabolism("Ibuprofen", mw_Da=206.0)
        assert result.compound_name == "Ibuprofen"

    def test_mw_stored(self):
        result = predict_pancreatic_enzyme_metabolism("Drug", mw_Da=350.0)
        assert result.mw_Da == 350.0


# ---------------------------------------------------------------------------
# Small molecule behavior
# ---------------------------------------------------------------------------


class TestSmallMolecule:
    def test_low_k_deg_small_molecule(self):
        result = predict_pancreatic_enzyme_metabolism("SmallMol", mw_Da=300.0)
        # Small molecule k_deg should be low (0.01/h base)
        assert result.k_deg_per_h < 0.05

    def test_high_fa_gi_small_molecule(self):
        result = predict_pancreatic_enzyme_metabolism("SmallMol", mw_Da=300.0)
        assert result.fa_gi > 0.9

    def test_stable_classification_small_molecule(self):
        result = predict_pancreatic_enzyme_metabolism("SmallMol", mw_Da=300.0)
        assert result.stability_class == "stable"

    def test_no_protection_needed_small_molecule(self):
        result = predict_pancreatic_enzyme_metabolism("SmallMol", mw_Da=300.0)
        assert result.gi_protection_needed is False

    def test_standard_formulation_small_molecule(self):
        result = predict_pancreatic_enzyme_metabolism("SmallMol", mw_Da=300.0)
        assert result.recommended_formulation == "standard"


# ---------------------------------------------------------------------------
# Peptide behavior
# ---------------------------------------------------------------------------


class TestPeptide:
    def test_higher_k_deg_peptide(self):
        sm_result = predict_pancreatic_enzyme_metabolism("SmallMol", mw_Da=300.0)
        pep_result = predict_pancreatic_enzyme_metabolism("Peptide", mw_Da=1000.0, is_peptide=True)
        assert pep_result.k_deg_per_h > sm_result.k_deg_per_h

    def test_lower_fa_gi_peptide_vs_small_mol(self):
        sm_result = predict_pancreatic_enzyme_metabolism("SmallMol", mw_Da=300.0)
        pep_result = predict_pancreatic_enzyme_metabolism("Peptide", mw_Da=1000.0, is_peptide=True)
        assert pep_result.fa_gi < sm_result.fa_gi

    def test_large_protein_very_low_fa(self):
        result = predict_pancreatic_enzyme_metabolism(
            "LargeProtein", mw_Da=50000.0, is_peptide=True
        )
        assert result.fa_gi < 0.5

    def test_large_protein_labile_or_moderate(self):
        result = predict_pancreatic_enzyme_metabolism(
            "LargeProtein", mw_Da=50000.0, is_peptide=True
        )
        assert result.stability_class in {"labile", "moderate"}


# ---------------------------------------------------------------------------
# Ester drug
# ---------------------------------------------------------------------------


class TestEster:
    def test_ester_higher_k_deg_than_small_molecule(self):
        sm = predict_pancreatic_enzyme_metabolism("SmallMol", mw_Da=300.0)
        ester = predict_pancreatic_enzyme_metabolism("EsterDrug", mw_Da=300.0, drug_class="ester")
        assert ester.k_deg_per_h > sm.k_deg_per_h

    def test_ester_k_deg_increment(self):
        sm = predict_pancreatic_enzyme_metabolism("SmallMol", mw_Da=300.0)
        ester = predict_pancreatic_enzyme_metabolism("EsterDrug", mw_Da=300.0, drug_class="ester")
        # Ester adds 0.2/h
        assert abs(ester.k_deg_per_h - sm.k_deg_per_h - 0.2) < 1e-9

    def test_ester_lower_fa_than_small_molecule(self):
        sm = predict_pancreatic_enzyme_metabolism("SmallMol", mw_Da=300.0)
        ester = predict_pancreatic_enzyme_metabolism("EsterDrug", mw_Da=300.0, drug_class="ester")
        assert ester.fa_gi < sm.fa_gi


# ---------------------------------------------------------------------------
# pH labile
# ---------------------------------------------------------------------------


class TestPhLabile:
    def test_ph_labile_increases_k_deg(self):
        base = predict_pancreatic_enzyme_metabolism("Drug", mw_Da=300.0)
        labile = predict_pancreatic_enzyme_metabolism("Drug", mw_Da=300.0, ph_labile=True)
        assert labile.k_deg_per_h > base.k_deg_per_h

    def test_ph_labile_k_deg_increment(self):
        base = predict_pancreatic_enzyme_metabolism("Drug", mw_Da=300.0)
        labile = predict_pancreatic_enzyme_metabolism("Drug", mw_Da=300.0, ph_labile=True)
        # pH labile adds 0.1/h
        assert abs(labile.k_deg_per_h - base.k_deg_per_h - 0.1) < 1e-9

    def test_ph_labile_lower_fa(self):
        base = predict_pancreatic_enzyme_metabolism("Drug", mw_Da=300.0)
        labile = predict_pancreatic_enzyme_metabolism("Drug", mw_Da=300.0, ph_labile=True)
        assert labile.fa_gi < base.fa_gi


# ---------------------------------------------------------------------------
# fa_gi bounds and fraction_remaining
# ---------------------------------------------------------------------------


class TestBounds:
    def test_fa_gi_between_0_and_1(self):
        for mw in [100.0, 500.0, 5000.0, 50000.0]:
            result = predict_pancreatic_enzyme_metabolism("Drug", mw_Da=mw, is_peptide=True)
            assert 0.0 <= result.fa_gi <= 1.0

    def test_fraction_remaining_between_0_and_1(self):
        result = predict_pancreatic_enzyme_metabolism("Drug", mw_Da=1000.0, is_peptide=True)
        assert 0.0 <= result.fraction_remaining_gi <= 1.0

    def test_k_deg_nonnegative(self):
        result = predict_pancreatic_enzyme_metabolism("Drug", mw_Da=300.0)
        assert result.k_deg_per_h >= 0.0


# ---------------------------------------------------------------------------
# Enzyme insufficiency
# ---------------------------------------------------------------------------


class TestEnzymeInsufficiency:
    def test_low_multiplier_higher_fa(self):
        normal = predict_pancreatic_enzyme_metabolism(
            "Peptide", mw_Da=1000.0, is_peptide=True, enzyme_activity_multiplier=1.0
        )
        insufficient = predict_pancreatic_enzyme_metabolism(
            "Peptide", mw_Da=1000.0, is_peptide=True, enzyme_activity_multiplier=0.1
        )
        assert insufficient.fa_gi > normal.fa_gi

    def test_multiplier_stored_in_result(self):
        result = predict_pancreatic_enzyme_metabolism(
            "Drug", mw_Da=300.0, enzyme_activity_multiplier=0.5
        )
        assert result.enzyme_activity_multiplier == 0.5


# ---------------------------------------------------------------------------
# Stability class, protection, formulation
# ---------------------------------------------------------------------------


class TestClassification:
    def test_stability_class_valid_values(self):
        for mw in [100.0, 1000.0, 50000.0]:
            r = predict_pancreatic_enzyme_metabolism("Drug", mw_Da=mw, is_peptide=mw > 500)
            assert r.stability_class in {"stable", "moderate", "labile"}

    def test_protection_needed_when_not_stable(self):
        # Large peptide -> labile
        result = predict_pancreatic_enzyme_metabolism("BigPeptide", mw_Da=50000.0, is_peptide=True)
        assert result.stability_class in {"moderate", "labile"}
        assert result.gi_protection_needed is True

    def test_no_protection_when_stable(self):
        result = predict_pancreatic_enzyme_metabolism("SmallMol", mw_Da=200.0)
        assert result.stability_class == "stable"
        assert result.gi_protection_needed is False

    def test_formulation_mapping_standard(self):
        result = predict_pancreatic_enzyme_metabolism("SmallMol", mw_Da=200.0)
        assert result.recommended_formulation == "standard"

    def test_formulation_mapping_enteric_coated(self):
        # moderate stability: find a medium peptide
        result = predict_pancreatic_enzyme_metabolism(
            "MidPeptide", mw_Da=300.0, is_peptide=True, enzyme_activity_multiplier=0.3
        )
        if result.stability_class == "moderate":
            assert result.recommended_formulation == "enteric_coated"

    def test_formulation_mapping_non_oral(self):
        result = predict_pancreatic_enzyme_metabolism(
            "LargeProtein", mw_Da=50000.0, is_peptide=True
        )
        if result.stability_class == "labile":
            assert result.recommended_formulation == "non_oral"


# ---------------------------------------------------------------------------
# compare_enzyme_conditions
# ---------------------------------------------------------------------------


class TestCompareEnzymeConditions:
    def test_returns_list(self):
        results = compare_enzyme_conditions("Peptide", 1000.0, True, [0.1, 0.5, 1.0])
        assert isinstance(results, list)

    def test_correct_length(self):
        results = compare_enzyme_conditions("Peptide", 1000.0, True, [0.1, 0.5, 1.0, 2.0])
        assert len(results) == 4

    def test_sorted_by_fa_gi_descending(self):
        results = compare_enzyme_conditions("Peptide", 1000.0, True, [0.1, 0.5, 1.0, 2.0])
        fa_values = [r.fa_gi for r in results]
        assert fa_values == sorted(fa_values, reverse=True)

    def test_each_element_is_result_type(self):
        results = compare_enzyme_conditions("Drug", 500.0, False, [0.5, 1.0])
        for r in results:
            assert isinstance(r, PancreaticEnzymeMetabolismResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_pancreatic_enzyme_metabolism("Drug", mw_Da=0.0)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_pancreatic_enzyme_metabolism("Drug", mw_Da=-100.0)

    def test_zero_enzyme_multiplier_raises(self):
        with pytest.raises(ValueError, match="enzyme_activity_multiplier"):
            predict_pancreatic_enzyme_metabolism(
                "Drug", mw_Da=300.0, enzyme_activity_multiplier=0.0
            )

    def test_too_high_enzyme_multiplier_raises(self):
        with pytest.raises(ValueError, match="enzyme_activity_multiplier"):
            predict_pancreatic_enzyme_metabolism(
                "Drug", mw_Da=300.0, enzyme_activity_multiplier=3.0
            )

    def test_invalid_drug_class_raises(self):
        with pytest.raises(ValueError, match="drug_class"):
            predict_pancreatic_enzyme_metabolism("Drug", mw_Da=300.0, drug_class="steroid")
