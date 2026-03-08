"""Tests for Phase 442 — dose form bioavailability prediction."""

import pytest

from omega_pbpk.prediction.dose_form_bioavailability import (
    DOSE_FORMS,
    DoseFormBioavailabilityResult,
    compare_dose_forms,
    predict_dose_form_bioavailability,
)


def make_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_form="IR_tablet",
        bcs_class=1,
        intrinsic_f_oral=0.8,
    )
    defaults.update(kwargs)
    return predict_dose_form_bioavailability(**defaults)


# --- Return type ---
def test_return_type():
    result = make_result()
    assert isinstance(result, DoseFormBioavailabilityResult)


# --- Field preservation ---
def test_drug_name_preserved():
    result = make_result(drug_name="Metformin")
    assert result.drug_name == "Metformin"


def test_dose_form_preserved():
    for df in DOSE_FORMS:
        result = make_result(dose_form=df)
        assert result.dose_form == df


def test_bcs_class_preserved():
    for bcs in [1, 2, 3, 4]:
        result = make_result(bcs_class=bcs)
        assert result.bcs_class == bcs


# --- F oral range ---
def test_f_oral_estimated_between_0_and_1():
    for df in DOSE_FORMS:
        for bcs in [1, 2, 3, 4]:
            result = make_result(dose_form=df, bcs_class=bcs)
            assert 0.0 <= result.f_oral_estimated <= 1.0, (
                f"F out of range for {df}, BCS {bcs}: {result.f_oral_estimated}"
            )


# --- Solution has highest F ---
def test_solution_has_highest_f_for_same_drug():
    solution = make_result(dose_form="solution", bcs_class=1)
    for df in DOSE_FORMS:
        if df == "solution":
            continue
        other = make_result(dose_form=df, bcs_class=1)
        assert solution.f_oral_estimated >= other.f_oral_estimated, (
            f"Solution F should be >= {df} F, got {solution.f_oral_estimated} vs {other.f_oral_estimated}"
        )


# --- Tmax ordering ---
def test_er_tablet_has_higher_tmax_than_ir_tablet():
    er = make_result(dose_form="ER_tablet")
    ir = make_result(dose_form="IR_tablet")
    assert er.tmax_h > ir.tmax_h


# --- BCS class effect ---
def test_bcs4_lower_f_than_bcs1_same_form():
    bcs1 = make_result(bcs_class=1)
    bcs4 = make_result(bcs_class=4)
    assert bcs4.f_oral_estimated < bcs1.f_oral_estimated


def test_bcs3_lower_f_than_bcs1():
    bcs1 = make_result(bcs_class=1)
    bcs3 = make_result(bcs_class=3)
    assert bcs3.f_oral_estimated < bcs1.f_oral_estimated


# --- f_relative_to_solution ---
def test_f_relative_to_solution_le_1_for_all_forms():
    for df in DOSE_FORMS:
        result = make_result(dose_form=df, bcs_class=1)
        assert result.f_relative_to_solution <= 1.0 + 1e-9, (
            f"f_relative > 1.0 for {df}: {result.f_relative_to_solution}"
        )


def test_solution_f_relative_equals_1():
    result = make_result(dose_form="solution")
    assert abs(result.f_relative_to_solution - 1.0) < 1e-9


# --- Pediatric suitability ---
def test_solution_suitable_for_ped():
    result = make_result(dose_form="solution")
    assert result.suitable_for_ped is True


def test_ir_tablet_not_suitable_for_ped():
    result = make_result(dose_form="IR_tablet")
    assert result.suitable_for_ped is False


def test_odt_suitable_for_ped():
    result = make_result(dose_form="ODT")
    assert result.suitable_for_ped is True


# --- Elderly suitability ---
def test_all_forms_suitable_for_elderly():
    for df in DOSE_FORMS:
        result = make_result(dose_form=df)
        assert result.suitable_for_elderly is True


# --- compare_dose_forms ---
def test_compare_dose_forms_returns_9_results():
    results = compare_dose_forms("TestDrug", bcs_class=1)
    assert len(results) == 9


def test_compare_dose_forms_sorted_descending():
    results = compare_dose_forms("TestDrug", bcs_class=1)
    for i in range(len(results) - 1):
        assert results[i].f_oral_estimated >= results[i + 1].f_oral_estimated


def test_compare_dose_forms_all_forms_present():
    results = compare_dose_forms("TestDrug", bcs_class=2)
    forms = {r.dose_form for r in results}
    assert forms == set(DOSE_FORMS.keys())


# --- BCS II slow dissolution penalty ---
def test_bcs2_er_tablet_lower_f_than_bcs1_er():
    bcs1 = make_result(dose_form="ER_tablet", bcs_class=1)
    bcs2 = make_result(dose_form="ER_tablet", bcs_class=2)
    assert bcs2.f_oral_estimated < bcs1.f_oral_estimated


# --- Validation errors ---
def test_invalid_dose_form_raises():
    with pytest.raises(ValueError, match="Unknown dose_form"):
        predict_dose_form_bioavailability("D", "injection", 1)


def test_invalid_bcs_class_raises():
    with pytest.raises(ValueError):
        predict_dose_form_bioavailability("D", "IR_tablet", 5)


def test_invalid_bcs_class_zero_raises():
    with pytest.raises(ValueError):
        predict_dose_form_bioavailability("D", "IR_tablet", 0)


def test_intrinsic_f_oral_zero_raises():
    with pytest.raises(ValueError):
        predict_dose_form_bioavailability("D", "IR_tablet", 1, intrinsic_f_oral=0.0)


def test_intrinsic_f_oral_greater_than_1_raises():
    with pytest.raises(ValueError):
        predict_dose_form_bioavailability("D", "IR_tablet", 1, intrinsic_f_oral=1.5)
