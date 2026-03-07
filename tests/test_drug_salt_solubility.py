"""Tests for Phase 990 — drug_salt_solubility module."""

import pytest
from omega_pbpk.prediction.drug_salt_solubility import (
    DrugSaltSolubilityResult,
    compare_salt_forms,
    predict_salt_solubility,
)


def default_acid():
    return predict_salt_solubility("AcidDrug", pka=4.5, ionization_type="acid")


def default_base():
    return predict_salt_solubility("BaseDrug", pka=8.5, ionization_type="base", salt_form="HCl")


def default_neutral():
    return predict_salt_solubility("NeutralDrug", pka=7.0, ionization_type="neutral", salt_form="free_base")


# --- Return type ---

def test_returns_result_type():
    r = default_acid()
    assert isinstance(r, DrugSaltSolubilityResult)


# --- pH grid length ---

def test_ph_values_length():
    r = default_acid()
    assert len(r.ph_values) == 17


def test_solubility_grid_length():
    r = default_acid()
    assert len(r.solubility_mg_mL) == 17


# --- All solubility values positive ---

def test_all_solubility_positive_acid():
    r = default_acid()
    assert all(s > 0 for s in r.solubility_mg_mL)


def test_all_solubility_positive_base():
    r = default_base()
    assert all(s > 0 for s in r.solubility_mg_mL)


# --- Ionization behavior ---

def test_acid_higher_solubility_at_high_ph():
    r = default_acid()
    ph_vals = list(r.ph_values)
    sol_vals = list(r.solubility_mg_mL)
    idx_ph8 = ph_vals.index(8.0)
    idx_ph2 = ph_vals.index(2.0)
    assert sol_vals[idx_ph8] > sol_vals[idx_ph2]


def test_base_higher_solubility_at_low_ph():
    r = default_base()
    ph_vals = list(r.ph_values)
    sol_vals = list(r.solubility_mg_mL)
    idx_ph2 = ph_vals.index(2.0)
    idx_ph8 = ph_vals.index(8.0)
    assert sol_vals[idx_ph2] > sol_vals[idx_ph8]


def test_neutral_constant_solubility():
    r = default_neutral()
    sol_vals = list(r.solubility_mg_mL)
    assert all(abs(s - sol_vals[0]) < 1e-9 for s in sol_vals)


# --- Max/min ---

def test_max_greater_than_min():
    r = default_acid()
    assert r.max_solubility_mg_mL > r.min_solubility_mg_mL


# --- Gastric / intestinal ---

def test_gastric_solubility_positive():
    r = default_acid()
    assert r.gastric_solubility_mg_mL > 0.0


def test_intestinal_solubility_positive():
    r = default_acid()
    assert r.intestinal_solubility_mg_mL > 0.0


# --- logP effect ---

def test_higher_logp_lower_intrinsic_solubility():
    r_low = predict_salt_solubility("Drug", pka=5.0, ionization_type="acid", logp=1.0)
    r_high = predict_salt_solubility("Drug", pka=5.0, ionization_type="acid", logp=4.0)
    assert r_high.intrinsic_solubility_mg_mL < r_low.intrinsic_solubility_mg_mL


# --- Temperature effect ---

def test_higher_temperature_higher_solubility():
    r_cold = predict_salt_solubility("Drug", pka=5.0, ionization_type="acid", t_celsius=25.0)
    r_warm = predict_salt_solubility("Drug", pka=5.0, ionization_type="acid", t_celsius=37.0)
    assert r_warm.intrinsic_solubility_mg_mL > r_cold.intrinsic_solubility_mg_mL


# --- compare_salt_forms ---

def test_compare_salt_forms_sorted():
    salt_forms = ["HCl", "Na", "free_acid"]
    results = compare_salt_forms("Drug", pka=4.5, ionization_type="acid", logp=2.0, salt_forms=salt_forms)
    for i in range(len(results) - 1):
        assert results[i].intestinal_solubility_mg_mL >= results[i + 1].intestinal_solubility_mg_mL


# --- Validation errors ---

def test_invalid_pka_raises():
    with pytest.raises(ValueError):
        predict_salt_solubility("Drug", pka=-1.0, ionization_type="acid")


def test_invalid_pka_high_raises():
    with pytest.raises(ValueError):
        predict_salt_solubility("Drug", pka=15.0, ionization_type="acid")


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError):
        predict_salt_solubility("Drug", pka=5.0, ionization_type="zwitterion")


def test_invalid_salt_form_raises():
    with pytest.raises(ValueError):
        predict_salt_solubility("Drug", pka=5.0, ionization_type="acid", salt_form="invalid_form")
