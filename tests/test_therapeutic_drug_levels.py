"""Tests for Phase 601 — therapeutic_drug_levels module."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.therapeutic_drug_levels import (
    DRUG_LEVEL_DATABASE,
    DrugLevelReference,
    TherapeuticLevelResult,
    evaluate_drug_level,
    list_available_drugs,
    lookup_drug_level,
)

# ---------------------------------------------------------------------------
# lookup_drug_level
# ---------------------------------------------------------------------------


def test_lookup_vancomycin():
    ref = lookup_drug_level("vancomycin")
    assert isinstance(ref, DrugLevelReference)
    assert ref.drug_name == "vancomycin"


def test_lookup_case_insensitive():
    ref_lower = lookup_drug_level("vancomycin")
    ref_upper = lookup_drug_level("Vancomycin")
    assert ref_lower.drug_name == ref_upper.drug_name


def test_lookup_unknown_raises():
    with pytest.raises(ValueError, match="not found"):
        lookup_drug_level("super_fake_drug_xyz")


# ---------------------------------------------------------------------------
# evaluate_drug_level — status
# ---------------------------------------------------------------------------


def test_evaluate_therapeutic_status():
    result = evaluate_drug_level("vancomycin", measured_conc_mg_L=15.0)
    assert result.status == "therapeutic"


def test_evaluate_subtherapeutic():
    result = evaluate_drug_level("vancomycin", measured_conc_mg_L=5.0)
    assert result.status == "subtherapeutic"


def test_evaluate_supratherapeutic():
    result = evaluate_drug_level("vancomycin", measured_conc_mg_L=25.0)
    assert result.status == "supratherapeutic"


def test_evaluate_toxic():
    result = evaluate_drug_level("vancomycin", measured_conc_mg_L=50.0)
    assert result.status == "toxic"


# ---------------------------------------------------------------------------
# safety_margin
# ---------------------------------------------------------------------------


def test_safety_margin_positive_when_below_toxic():
    result = evaluate_drug_level("vancomycin", measured_conc_mg_L=15.0)
    # toxic=40, measured=15 → margin = (40-15)/15 = 1.667
    assert result.safety_margin > 0.0


def test_safety_margin_negative_when_above_toxic():
    result = evaluate_drug_level("vancomycin", measured_conc_mg_L=50.0)
    # toxic=40, measured=50 → margin = (40-50)/50 = -0.2
    assert result.safety_margin < 0.0


# ---------------------------------------------------------------------------
# therapeutic_index
# ---------------------------------------------------------------------------


def test_therapeutic_index_positive():
    result = evaluate_drug_level("vancomycin", measured_conc_mg_L=15.0)
    assert result.therapeutic_index > 0.0


def test_nti_drug_low_ti():
    # digoxin: toxic=0.003, high=0.002 → TI = 1.5 < 5
    result = evaluate_drug_level("digoxin", measured_conc_mg_L=0.001)
    assert result.therapeutic_index < 5.0


# ---------------------------------------------------------------------------
# list_available_drugs
# ---------------------------------------------------------------------------


def test_list_drugs_returns_list():
    drugs = list_available_drugs()
    assert isinstance(drugs, list)


def test_list_drugs_contains_vancomycin():
    assert "vancomycin" in list_available_drugs()


def test_list_drugs_sorted():
    drugs = list_available_drugs()
    assert drugs == sorted(drugs)


def test_list_drugs_has_15_plus():
    assert len(list_available_drugs()) >= 15


# ---------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------


def test_override_parameters():
    result = evaluate_drug_level(
        "vancomycin",
        measured_conc_mg_L=8.0,
        therapeutic_low=5.0,
        therapeutic_high=15.0,
        toxic_threshold=25.0,
    )
    # 8 is within [5, 15]
    assert result.status == "therapeutic"
    assert result.therapeutic_low_mg_L == 5.0
    assert result.therapeutic_high_mg_L == 15.0
    assert result.toxic_threshold_mg_L == 25.0


# ---------------------------------------------------------------------------
# individual drug lookups
# ---------------------------------------------------------------------------


def test_phenytoin_lookup():
    ref = lookup_drug_level("phenytoin")
    assert ref.drug_class == "anticonvulsant"
    assert ref.therapeutic_low_mg_L == 10.0
    assert ref.therapeutic_high_mg_L == 20.0


def test_cyclosporine_lookup():
    ref = lookup_drug_level("cyclosporine")
    assert ref.drug_class == "immunosuppressant"


# ---------------------------------------------------------------------------
# result type and fields
# ---------------------------------------------------------------------------


def test_evaluate_returns_result():
    result = evaluate_drug_level("vancomycin", measured_conc_mg_L=15.0)
    assert isinstance(result, TherapeuticLevelResult)


def test_drug_class_stored():
    result = evaluate_drug_level("vancomycin", measured_conc_mg_L=15.0)
    ref = lookup_drug_level("vancomycin")
    assert result.drug_class == ref.drug_class


def test_recommendation_nonempty():
    result = evaluate_drug_level("vancomycin", measured_conc_mg_L=15.0)
    assert len(result.recommendation) > 0


def test_notes_nonempty():
    result = evaluate_drug_level("vancomycin", measured_conc_mg_L=15.0)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# status validity
# ---------------------------------------------------------------------------

VALID_STATUSES = {"subtherapeutic", "therapeutic", "supratherapeutic", "toxic"}

CONCENTRATIONS = {
    "vancomycin": [5.0, 15.0, 25.0, 50.0],
    "digoxin": [0.0004, 0.001, 0.0025, 0.004],
    "theophylline": [5.0, 15.0, 22.0, 30.0],
}


def test_all_status_values_valid():
    for drug, concs in CONCENTRATIONS.items():
        for conc in concs:
            result = evaluate_drug_level(drug, measured_conc_mg_L=conc)
            assert result.status in VALID_STATUSES, (
                f"Unexpected status '{result.status}' for {drug} at {conc} mg/L"
            )


# ---------------------------------------------------------------------------
# database class coverage
# ---------------------------------------------------------------------------


def test_database_has_antibiotic_class():
    classes = {ref.drug_class for ref in DRUG_LEVEL_DATABASE.values()}
    assert "antibiotic" in classes


def test_database_has_immunosuppressant_class():
    classes = {ref.drug_class for ref in DRUG_LEVEL_DATABASE.values()}
    assert "immunosuppressant" in classes
