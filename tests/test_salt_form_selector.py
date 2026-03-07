"""Tests for drug salt form selector — Phase 832."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.salt_form_selector import (
    SaltFormResult,
    predict_salt_properties,
    select_optimal_salt,
)

# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_predict_returns_salt_form_result(self):
        result = predict_salt_properties(
            "TestDrug", drug_pka=8.0, drug_logp=2.0, ionization_type="base", salt_form="HCl"
        )
        assert isinstance(result, SaltFormResult)

    def test_result_is_frozen(self):
        result = predict_salt_properties(
            "Drug", drug_pka=8.0, drug_logp=2.0, ionization_type="base", salt_form="HCl"
        )
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_select_optimal_returns_list(self):
        results = select_optimal_salt("Drug", drug_pka=8.0, drug_logp=2.0, ionization_type="base")
        assert isinstance(results, list)

    def test_select_optimal_nonempty(self):
        results = select_optimal_salt("Drug", drug_pka=8.0, drug_logp=2.0, ionization_type="base")
        assert len(results) > 0

    def test_select_optimal_all_salt_form_results(self):
        results = select_optimal_salt("Drug", drug_pka=8.0, drug_logp=2.0, ionization_type="base")
        for r in results:
            assert isinstance(r, SaltFormResult)


# ---------------------------------------------------------------------------
# Physicochemical property tests
# ---------------------------------------------------------------------------


class TestProperties:
    def test_solubility_fold_increase_at_least_one(self):
        for sf in ("HCl", "citrate", "mesylate", "free_base"):
            result = predict_salt_properties(
                "Drug", drug_pka=7.5, drug_logp=1.5, ionization_type="base", salt_form=sf
            )
            assert result.solubility_fold_increase >= 1.0, f"Failed for {sf}"

    def test_solubility_fold_increase_at_most_twenty(self):
        # high logP drug should still be clamped
        result = predict_salt_properties(
            "Drug", drug_pka=9.0, drug_logp=10.0, ionization_type="base", salt_form="HCl"
        )
        assert result.solubility_fold_increase <= 20.0

    def test_dissolution_rate_factor_sqrt_of_solubility(self):
        result = predict_salt_properties(
            "Drug", drug_pka=8.0, drug_logp=2.0, ionization_type="base", salt_form="HCl"
        )
        expected = math.sqrt(result.solubility_fold_increase)
        assert abs(result.dissolution_rate_factor - expected) < 1e-9

    def test_hygroscopicity_valid_value(self):
        result = predict_salt_properties(
            "Drug", drug_pka=8.0, drug_logp=2.0, ionization_type="base", salt_form="HCl"
        )
        assert result.hygroscopicity in ("low", "moderate", "high")

    def test_stability_score_in_range(self):
        result = predict_salt_properties(
            "Drug", drug_pka=8.0, drug_logp=2.0, ionization_type="base", salt_form="HCl"
        )
        assert 0.0 <= result.stability_score <= 100.0

    def test_drug_pka_preserved(self):
        result = predict_salt_properties(
            "Drug", drug_pka=7.2, drug_logp=1.5, ionization_type="base", salt_form="citrate"
        )
        assert result.drug_pka == 7.2


# ---------------------------------------------------------------------------
# Recommended logic
# ---------------------------------------------------------------------------


class TestRecommended:
    def test_recommended_true_when_conditions_met(self):
        # mesylate: stability=85, hygro=low, sol_fold will be >= 2 for logp=1
        result = predict_salt_properties(
            "Drug", drug_pka=8.0, drug_logp=1.0, ionization_type="base", salt_form="mesylate"
        )
        assert result.recommended is True

    def test_recommended_false_for_high_hygro(self):
        # H2SO4: hygro="high" => not recommended
        result = predict_salt_properties(
            "Drug", drug_pka=8.0, drug_logp=1.0, ionization_type="base", salt_form="H2SO4"
        )
        # high hygroscopicity => recommended=False regardless of stability
        assert result.recommended is False

    def test_recommended_false_for_free_form_low_sol(self):
        # free_base: sol=1.0 < 2.0 => not recommended
        result = predict_salt_properties(
            "Drug", drug_pka=8.0, drug_logp=1.0, ionization_type="base", salt_form="free_base"
        )
        assert result.recommended is False

    def test_recommended_false_for_low_stability(self):
        # H2SO4: stability=60 < 70 and hygro="high" => not recommended
        result = predict_salt_properties(
            "Drug", drug_pka=8.0, drug_logp=1.0, ionization_type="base", salt_form="H2SO4"
        )
        assert result.recommended is False


# ---------------------------------------------------------------------------
# Specific salt forms
# ---------------------------------------------------------------------------


class TestSpecificSalts:
    def test_hcl_salt_for_base_drug(self):
        result = predict_salt_properties(
            "BaseDrug", drug_pka=8.5, drug_logp=2.0, ionization_type="base", salt_form="HCl"
        )
        assert result.salt_form == "HCl"
        assert result.counterion == "Cl-"

    def test_na_salt_for_acid_drug(self):
        result = predict_salt_properties(
            "AcidDrug", drug_pka=4.0, drug_logp=1.5, ionization_type="acid", salt_form="Na"
        )
        assert result.salt_form == "Na"
        assert result.counterion == "Na+"

    def test_mesylate_higher_stability_than_h2so4_for_base(self):
        mesy = predict_salt_properties(
            "Drug", drug_pka=8.0, drug_logp=2.0, ionization_type="base", salt_form="mesylate"
        )
        sulf = predict_salt_properties(
            "Drug", drug_pka=8.0, drug_logp=2.0, ionization_type="base", salt_form="H2SO4"
        )
        assert mesy.stability_score > sulf.stability_score

    def test_ca_salt_for_acid_drug_low_hygro(self):
        result = predict_salt_properties(
            "AcidDrug", drug_pka=4.5, drug_logp=1.0, ionization_type="acid", salt_form="Ca"
        )
        assert result.hygroscopicity == "low"


# ---------------------------------------------------------------------------
# select_optimal_salt ordering
# ---------------------------------------------------------------------------


class TestSelectOptimalOrdering:
    def test_recommended_sorted_first(self):
        results = select_optimal_salt("Drug", drug_pka=8.0, drug_logp=2.0, ionization_type="base")
        # All recommended should appear before any non-recommended
        seen_non_recommended = False
        for r in results:
            if not r.recommended:
                seen_non_recommended = True
            if seen_non_recommended and r.recommended:
                pytest.fail("A recommended salt appeared after a non-recommended one")

    def test_recommended_group_sorted_by_stability_desc(self):
        results = select_optimal_salt("Drug", drug_pka=8.0, drug_logp=2.0, ionization_type="base")
        rec = [r for r in results if r.recommended]
        for i in range(len(rec) - 1):
            assert rec[i].stability_score >= rec[i + 1].stability_score

    def test_covers_all_base_salts(self):
        results = select_optimal_salt("Drug", drug_pka=8.0, drug_logp=2.0, ionization_type="base")
        expected = {
            "HCl",
            "H2SO4",
            "HBr",
            "citrate",
            "maleate",
            "tartrate",
            "mesylate",
            "phosphate",
            "free_base",
        }
        returned = {r.salt_form for r in results}
        assert expected == returned

    def test_covers_all_acid_salts(self):
        results = select_optimal_salt(
            "AcidDrug", drug_pka=4.0, drug_logp=1.5, ionization_type="acid"
        )
        expected = {"Na", "K", "Ca", "Mg", "lysine", "meglumine", "free_acid"}
        returned = {r.salt_form for r in results}
        assert expected == returned


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_pka_below_zero_raises_value_error(self):
        with pytest.raises(ValueError, match=r"\[0, 14\]"):
            predict_salt_properties(
                "Drug", drug_pka=-1.0, drug_logp=2.0, ionization_type="base", salt_form="HCl"
            )

    def test_pka_above_14_raises_value_error(self):
        with pytest.raises(ValueError, match=r"\[0, 14\]"):
            predict_salt_properties(
                "Drug", drug_pka=15.0, drug_logp=2.0, ionization_type="base", salt_form="HCl"
            )

    def test_invalid_ionization_type_raises_value_error(self):
        with pytest.raises(ValueError, match="ionization_type"):
            predict_salt_properties(
                "Drug", drug_pka=7.0, drug_logp=2.0, ionization_type="neutral", salt_form="HCl"
            )

    def test_select_optimal_invalid_ionization_raises(self):
        with pytest.raises(ValueError):
            select_optimal_salt("Drug", drug_pka=7.0, drug_logp=2.0, ionization_type="zwitterion")

    def test_unknown_salt_form_uses_fallback(self):
        # Should not raise; falls back to free_base for base drugs
        result = predict_salt_properties(
            "Drug", drug_pka=8.0, drug_logp=2.0, ionization_type="base", salt_form="unknown_salt"
        )
        assert isinstance(result, SaltFormResult)
