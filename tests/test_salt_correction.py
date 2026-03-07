"""Tests for Phase 786 — salt_correction.py"""

import pytest

from omega_pbpk.prediction.salt_correction import (
    _COUNTERION_MW,
    SaltCorrectionResult,
    calculate_salt_factor,
    correct_dose_by_name,
    correct_dose_for_salt,
    screen_salt_forms,
)

# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_correct_dose_for_salt_returns_dataclass():
    result = correct_dose_for_salt("DrugA", 100.0, mw_base=300.0, mw_counterion=36.46)
    assert isinstance(result, SaltCorrectionResult)


def test_result_fields_types():
    result = correct_dose_for_salt("DrugA", 100.0, mw_base=300.0, mw_counterion=36.46)
    assert isinstance(result.drug_name, str)
    assert isinstance(result.salt_form, str)
    assert isinstance(result.salt_dose_mg, float)
    assert isinstance(result.salt_factor, float)
    assert isinstance(result.free_base_dose_mg, float)
    assert isinstance(result.mw_salt, float)
    assert isinstance(result.mw_base, float)
    assert isinstance(result.ionization_state, str)
    assert isinstance(result.bioequivalent_dose_mg, float)
    assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# Salt factor in (0, 1) for all common salts
# ---------------------------------------------------------------------------


def test_salt_factor_in_range_for_all_common_salts():
    mw_base = 350.0
    for salt_name, mw_counterion in _COUNTERION_MW.items():
        sf = calculate_salt_factor(mw_base, mw_counterion)
        assert 0.0 < sf < 1.0, f"Salt factor out of range for {salt_name}: {sf}"


def test_salt_factor_formula():
    mw_base = 300.0
    mw_counterion = 36.46  # HCl
    expected = mw_base / (mw_base + mw_counterion)
    sf = calculate_salt_factor(mw_base, mw_counterion)
    assert abs(sf - expected) < 1e-10


def test_salt_factor_with_two_counterions():
    mw_base = 300.0
    mw_counterion = 36.46
    n = 2
    expected = mw_base / (mw_base + n * mw_counterion)
    sf = calculate_salt_factor(mw_base, mw_counterion, n_counterions=n)
    assert abs(sf - expected) < 1e-10


# ---------------------------------------------------------------------------
# free_base_dose = salt_dose * salt_factor
# ---------------------------------------------------------------------------


def test_free_base_dose_equals_salt_dose_times_factor():
    salt_dose = 150.0
    mw_base = 300.0
    mw_counterion = 36.46
    result = correct_dose_for_salt("DrugB", salt_dose, mw_base=mw_base, mw_counterion=mw_counterion)
    expected_free_base = salt_dose * result.salt_factor
    assert abs(result.free_base_dose_mg - expected_free_base) < 1e-9


def test_bioequivalent_dose_equals_free_base_dose():
    result = correct_dose_for_salt("DrugB", 100.0, mw_base=300.0, mw_counterion=36.46)
    assert result.bioequivalent_dose_mg == result.free_base_dose_mg


# ---------------------------------------------------------------------------
# HCl salt gives higher salt_dose than free base dose
# ---------------------------------------------------------------------------


def test_hcl_salt_dose_greater_than_free_base_dose():
    result = correct_dose_by_name("DrugC", 100.0, mw_base=300.0, salt_form="HCl")
    assert result.salt_dose_mg > result.free_base_dose_mg


def test_free_base_dose_less_than_salt_dose_for_any_acid_salt():
    """For any salt, free-base dose is always less than salt dose."""
    for form in ["HCl", "HBr", "Mesylate", "Maleate"]:
        result = correct_dose_by_name("DrugD", 200.0, mw_base=400.0, salt_form=form)
        assert result.free_base_dose_mg < result.salt_dose_mg, f"Failed for salt form {form}"


# ---------------------------------------------------------------------------
# correct_dose_by_name
# ---------------------------------------------------------------------------


def test_correct_dose_by_name_hcl():
    mw_base = 300.0
    salt_dose = 100.0
    result = correct_dose_by_name("DrugE", salt_dose, mw_base=mw_base, salt_form="HCl")
    expected_sf = mw_base / (mw_base + _COUNTERION_MW["HCl"])
    assert abs(result.salt_factor - expected_sf) < 1e-10
    assert result.salt_form == "HCl"
    assert result.ionization_state == "HCl salt"


def test_correct_dose_by_name_na():
    mw_base = 250.0
    salt_dose = 100.0
    result = correct_dose_by_name("DrugF", salt_dose, mw_base=mw_base, salt_form="Na")
    expected_sf = mw_base / (mw_base + _COUNTERION_MW["Na"])
    assert abs(result.salt_factor - expected_sf) < 1e-10
    assert result.ionization_state == "Na salt"


def test_correct_dose_by_name_mesylate():
    result = correct_dose_by_name("DrugG", 100.0, mw_base=300.0, salt_form="Mesylate")
    assert result.salt_form == "Mesylate"
    assert 0.0 < result.salt_factor < 1.0


# ---------------------------------------------------------------------------
# screen_salt_forms
# ---------------------------------------------------------------------------


def test_screen_salt_forms_returns_sorted_list():
    results = screen_salt_forms("DrugH", target_free_base_dose_mg=100.0, mw_base=300.0)
    salt_doses = [r.salt_dose_mg for r in results]
    assert salt_doses == sorted(salt_doses)


def test_screen_salt_forms_correct_count_for_subset():
    forms = ["HCl", "Na", "Mesylate"]
    results = screen_salt_forms(
        "DrugH", target_free_base_dose_mg=100.0, mw_base=300.0, salt_forms=forms
    )
    assert len(results) == len(forms)


def test_screen_salt_forms_all_achieve_target_dose():
    target = 50.0
    mw_base = 300.0
    results = screen_salt_forms(
        "DrugI", target_free_base_dose_mg=target, mw_base=mw_base, salt_forms=["HCl", "HBr", "Na"]
    )
    for r in results:
        assert abs(r.free_base_dose_mg - target) < 1e-6, (
            f"Free base dose mismatch for {r.salt_form}: {r.free_base_dose_mg}"
        )


def test_screen_salt_forms_free_base_sorted_ascending():
    """Lighter counterion → smaller salt mass → first in sorted list."""
    results = screen_salt_forms(
        "DrugJ", target_free_base_dose_mg=100.0, mw_base=300.0, salt_forms=["HCl", "HBr"]
    )
    # HCl (MW=36.46) < HBr (MW=80.91) → HCl salt requires less mass
    assert results[0].salt_form == "HCl"
    assert results[1].salt_form == "HBr"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_calculate_salt_factor_zero_mw_base_raises():
    with pytest.raises(ValueError, match="mw_base"):
        calculate_salt_factor(0.0, 36.46)


def test_calculate_salt_factor_negative_mw_base_raises():
    with pytest.raises(ValueError, match="mw_base"):
        calculate_salt_factor(-100.0, 36.46)


def test_correct_dose_by_name_unknown_salt_raises():
    with pytest.raises(ValueError, match="salt_form"):
        correct_dose_by_name("DrugX", 100.0, mw_base=300.0, salt_form="XYZ_salt")


def test_correct_dose_for_salt_zero_mw_base_raises():
    with pytest.raises(ValueError, match="mw_base"):
        correct_dose_for_salt("DrugX", 100.0, mw_base=0.0, mw_counterion=36.46)


def test_screen_salt_forms_unknown_form_raises():
    with pytest.raises(ValueError, match="salt_form"):
        screen_salt_forms(
            "DrugX", target_free_base_dose_mg=100.0, mw_base=300.0, salt_forms=["HCl", "INVALID"]
        )
