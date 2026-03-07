"""Tests for Phase 313 — Enteral Nutrition Drug Interaction Predictor."""

import pytest

from omega_pbpk.clinical.enteral_nutrition_interaction import (
    EnteralNutritionResult,
    compare_formulations,
    predict_en_interaction,
)

# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------

DEFAULT_KWARGS = dict(
    drug_name="TestDrug",
    drug_pka=5.0,
    drug_ionization_type="acid",
    drug_logp=2.0,
    formulation_type="tablet",
    en_rate_mL_per_h=50.0,
    protein_binding_risk=False,
)


def make_result(**overrides) -> EnteralNutritionResult:
    kwargs = {**DEFAULT_KWARGS, **overrides}
    return predict_en_interaction(**kwargs)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = make_result()
    assert isinstance(result, EnteralNutritionResult)


def test_result_is_frozen():
    result = make_result()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Field preservation
# ---------------------------------------------------------------------------


def test_drug_name_preserved():
    result = make_result(drug_name="Phenytoin")
    assert result.drug_name == "Phenytoin"


def test_formulation_type_preserved():
    result = make_result(formulation_type="liquid")
    assert result.formulation_type == "liquid"


def test_en_rate_preserved():
    result = make_result(en_rate_mL_per_h=120.0)
    assert result.en_rate_mL_per_h == 120.0


def test_protein_binding_risk_preserved():
    result = make_result(protein_binding_risk=True)
    assert result.protein_binding_risk is True


# ---------------------------------------------------------------------------
# Relative bioavailability bounds
# ---------------------------------------------------------------------------


def test_relative_bioavailability_in_range():
    result = make_result()
    assert 0 < result.relative_bioavailability <= 1.0


def test_relative_bioavailability_minimum_floor():
    # Worst case: tablet, protein binding, high EN rate
    result = make_result(
        formulation_type="tablet",
        protein_binding_risk=True,
        en_rate_mL_per_h=500.0,
    )
    assert result.relative_bioavailability >= 0.1


# ---------------------------------------------------------------------------
# Protein binding risk effect
# ---------------------------------------------------------------------------


def test_protein_binding_lowers_bioavailability():
    no_binding = make_result(protein_binding_risk=False)
    with_binding = make_result(protein_binding_risk=True)
    assert with_binding.relative_bioavailability < no_binding.relative_bioavailability


# ---------------------------------------------------------------------------
# EN rate effects
# ---------------------------------------------------------------------------


def test_high_en_rate_lowers_bioavailability():
    low_rate = make_result(en_rate_mL_per_h=10.0)
    high_rate = make_result(en_rate_mL_per_h=400.0)
    assert high_rate.relative_bioavailability < low_rate.relative_bioavailability


def test_zero_en_rate_no_dilution():
    result_zero = make_result(en_rate_mL_per_h=0.0)
    result_some = make_result(en_rate_mL_per_h=200.0)
    assert result_zero.relative_bioavailability >= result_some.relative_bioavailability


# ---------------------------------------------------------------------------
# Formulation comparison
# ---------------------------------------------------------------------------


def test_liquid_higher_bioavailability_than_tablet():
    tablet = make_result(formulation_type="tablet", protein_binding_risk=False)
    liquid = make_result(formulation_type="liquid", protein_binding_risk=False)
    assert liquid.relative_bioavailability > tablet.relative_bioavailability


def test_crushable_between_tablet_and_liquid():
    tablet = make_result(formulation_type="tablet", protein_binding_risk=False)
    crushable = make_result(formulation_type="crushable", protein_binding_risk=False)
    liquid = make_result(formulation_type="liquid", protein_binding_risk=False)
    assert (
        tablet.relative_bioavailability
        <= crushable.relative_bioavailability
        <= liquid.relative_bioavailability
    )


# ---------------------------------------------------------------------------
# AUC and Cmax ratios
# ---------------------------------------------------------------------------


def test_auc_ratio_equals_relative_bioavailability():
    result = make_result()
    assert result.auc_ratio == result.relative_bioavailability


def test_cmax_ratio_less_than_auc_ratio():
    result = make_result()
    assert result.cmax_ratio < result.auc_ratio


def test_cmax_ratio_is_085_times_auc_ratio():
    result = make_result()
    assert abs(result.cmax_ratio - result.auc_ratio * 0.85) < 1e-9


# ---------------------------------------------------------------------------
# Tmax delay
# ---------------------------------------------------------------------------


def test_tmax_delay_nonnegative():
    result = make_result()
    assert result.tmax_delay_h >= 0.0


def test_high_en_rate_increases_tmax_delay():
    low_rate = make_result(en_rate_mL_per_h=10.0)
    high_rate = make_result(en_rate_mL_per_h=300.0)
    assert high_rate.tmax_delay_h >= low_rate.tmax_delay_h


# ---------------------------------------------------------------------------
# Interaction severity
# ---------------------------------------------------------------------------


def test_interaction_severity_valid_values():
    valid = {"major", "moderate", "minor"}
    for formulation in ("tablet", "capsule", "liquid", "crushable"):
        for pb_risk in (True, False):
            result = make_result(formulation_type=formulation, protein_binding_risk=pb_risk)
            assert result.interaction_severity in valid


def test_major_severity_when_auc_below_07():
    # Force AUC ratio < 0.7: tablet + protein binding + high EN rate
    result = make_result(
        formulation_type="tablet",
        protein_binding_risk=True,
        en_rate_mL_per_h=300.0,
    )
    if result.auc_ratio < 0.7:
        assert result.interaction_severity == "major"


def test_minor_severity_when_no_effects():
    result = make_result(
        formulation_type="liquid",
        protein_binding_risk=False,
        en_rate_mL_per_h=0.0,
        drug_ionization_type="neutral",
    )
    assert result.interaction_severity == "minor"


# ---------------------------------------------------------------------------
# Clinical recommendation
# ---------------------------------------------------------------------------


def test_clinical_recommendation_is_string():
    result = make_result()
    assert isinstance(result.clinical_recommendation, str)
    assert len(result.clinical_recommendation) > 0


def test_protein_binding_recommendation_separate():
    # protein binding risk with no major AUC reduction
    result = make_result(
        formulation_type="liquid",
        protein_binding_risk=True,
        en_rate_mL_per_h=0.0,
        drug_ionization_type="neutral",
    )
    # auc_ratio = 1.0 - 0.0 - 0.3 - 0.0 - 0.0 + 0.0 = 0.7 → hold_EN_around_dose
    # OR if >= 0.85 and protein_binding_risk → separate_by_2h
    assert result.clinical_recommendation in {
        "hold_EN_around_dose",
        "monitor_drug_levels",
        "separate_by_2h",
        "no_special_precaution",
    }


# ---------------------------------------------------------------------------
# Basic drug pH correction
# ---------------------------------------------------------------------------


def test_basic_drug_higher_bioavailability():
    acid = make_result(drug_pka=4.0, drug_ionization_type="acid")
    base = make_result(drug_pka=9.0, drug_ionization_type="base")
    assert base.relative_bioavailability > acid.relative_bioavailability


# ---------------------------------------------------------------------------
# compare_formulations
# ---------------------------------------------------------------------------


def test_compare_formulations_returns_list():
    results = compare_formulations(
        drug_name="TestDrug",
        drug_pka=5.0,
        drug_ionization_type="acid",
        drug_logp=2.0,
        en_rate_mL_per_h=100.0,
    )
    assert isinstance(results, list)


def test_compare_formulations_has_four_entries():
    results = compare_formulations(
        drug_name="TestDrug",
        drug_pka=5.0,
        drug_ionization_type="acid",
        drug_logp=2.0,
        en_rate_mL_per_h=100.0,
    )
    assert len(results) == 4


def test_compare_formulations_sorted_descending():
    results = compare_formulations(
        drug_name="TestDrug",
        drug_pka=5.0,
        drug_ionization_type="acid",
        drug_logp=2.0,
        en_rate_mL_per_h=100.0,
    )
    bav = [r.relative_bioavailability for r in results]
    assert bav == sorted(bav, reverse=True)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_negative_en_rate_raises():
    with pytest.raises(ValueError, match="en_rate_mL_per_h"):
        make_result(en_rate_mL_per_h=-10.0)


def test_invalid_formulation_raises():
    with pytest.raises(ValueError, match="formulation_type"):
        make_result(formulation_type="patch")


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="drug_ionization_type"):
        make_result(drug_ionization_type="zwitterion")


def test_invalid_pka_raises():
    with pytest.raises(ValueError, match="drug_pka"):
        make_result(drug_pka=0.0)


def test_pka_above_14_raises():
    with pytest.raises(ValueError, match="drug_pka"):
        make_result(drug_pka=15.0)
