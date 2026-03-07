"""Tests for Phase 954 — Salt Form PK Prediction."""

import pytest

from omega_pbpk.prediction.salt_form_pk import (
    SALT_FORMS,
    SaltFormPKResult,
    compare_salt_forms,
    predict_salt_form_pk,
)

DRUG = "TestDrug"
LOGP = 2.0
DOSE = 100.0


def default_result(**kwargs):
    params = dict(
        drug_name=DRUG,
        logp=LOGP,
        dose_mg=DOSE,
        salt_form="HCl_salt",
        cl_L_per_h=10.0,
        q_liver_L_per_h=90.0,
    )
    params.update(kwargs)
    return predict_salt_form_pk(**params)


class TestReturnType:
    def test_returns_salt_form_pk_result(self):
        r = default_result()
        assert isinstance(r, SaltFormPKResult)

    def test_is_frozen(self):
        r = default_result()
        with pytest.raises((AttributeError, TypeError)):
            r.auc_predicted_mg_h_per_L = 999.0  # type: ignore[misc]


class TestBasicFields:
    def test_s0_free_form_positive(self):
        r = default_result()
        assert r.s0_free_form_mg_mL > 0

    def test_s_salt_gte_s0(self):
        r = default_result()
        assert r.s_salt_mg_mL >= r.s0_free_form_mg_mL

    def test_solubility_factor_gte_one(self):
        for sf in SALT_FORMS:
            r = predict_salt_form_pk(DRUG, LOGP, DOSE, sf)
            assert r.solubility_factor >= 1.0, f"Salt {sf} has factor < 1"

    def test_fa_predicted_in_range(self):
        r = default_result()
        assert 0.0 <= r.fa_predicted <= 1.0

    def test_fa_base_in_range(self):
        r = default_result()
        assert 0.0 <= r.fa_base <= 1.0

    def test_auc_positive(self):
        r = default_result()
        assert r.auc_predicted_mg_h_per_L > 0

    def test_bioavailability_pct_in_range(self):
        r = default_result()
        assert 0.0 <= r.bioavailability_pct <= 100.0

    def test_dissolution_advantage_valid(self):
        r = default_result()
        assert r.dissolution_advantage in {"high", "moderate", "low"}

    def test_notes_non_empty(self):
        r = default_result()
        assert isinstance(r.notes, str) and len(r.notes) > 0

    def test_drug_name_preserved(self):
        r = predict_salt_form_pk("Ibuprofen", LOGP, DOSE, "Na_salt")
        assert r.drug_name == "Ibuprofen"


class TestComparisons:
    def test_na_salt_higher_solubility_than_free_acid(self):
        na = predict_salt_form_pk(DRUG, LOGP, DOSE, "Na_salt")
        fa = predict_salt_form_pk(DRUG, LOGP, DOSE, "free_acid")
        assert na.s_salt_mg_mL > fa.s_salt_mg_mL

    def test_hcl_higher_auc_than_free_base(self):
        hcl = predict_salt_form_pk(DRUG, LOGP, DOSE, "HCl_salt")
        fb = predict_salt_form_pk(DRUG, LOGP, DOSE, "free_base")
        assert hcl.auc_predicted_mg_h_per_L > fb.auc_predicted_mg_h_per_L

    def test_na_salt_solubility_factor_50(self):
        r = predict_salt_form_pk(DRUG, LOGP, DOSE, "Na_salt")
        assert r.solubility_factor == 50.0

    def test_hcl_salt_solubility_factor_20(self):
        r = predict_salt_form_pk(DRUG, LOGP, DOSE, "HCl_salt")
        assert r.solubility_factor == 20.0

    def test_free_base_factor_one(self):
        r = predict_salt_form_pk(DRUG, LOGP, DOSE, "free_base")
        assert r.solubility_factor == 1.0


class TestCompareSaltForms:
    def test_compare_returns_10_results(self):
        results = compare_salt_forms(DRUG, LOGP, DOSE)
        assert len(results) == 10

    def test_compare_sorted_by_auc_descending(self):
        results = compare_salt_forms(DRUG, LOGP, DOSE)
        for i in range(len(results) - 1):
            assert results[i].auc_predicted_mg_h_per_L >= results[i + 1].auc_predicted_mg_h_per_L

    def test_na_salt_or_hcl_near_top(self):
        results = compare_salt_forms(DRUG, LOGP, DOSE)
        top_3_forms = {r.salt_form for r in results[:3]}
        assert "Na_salt" in top_3_forms or "HCl_salt" in top_3_forms or "K_salt" in top_3_forms

    def test_free_forms_near_bottom(self):
        results = compare_salt_forms(DRUG, LOGP, DOSE)
        bottom_3_forms = {r.salt_form for r in results[-3:]}
        # free_base and free_acid should be among the lowest AUC
        assert "free_base" in bottom_3_forms or "free_acid" in bottom_3_forms

    def test_all_results_are_salt_form_pk_result(self):
        results = compare_salt_forms(DRUG, LOGP, DOSE)
        for r in results:
            assert isinstance(r, SaltFormPKResult)

    def test_all_salt_forms_covered(self):
        results = compare_salt_forms(DRUG, LOGP, DOSE)
        found_forms = {r.salt_form for r in results}
        assert found_forms == set(SALT_FORMS.keys())


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            predict_salt_form_pk(DRUG, LOGP, 0.0, "HCl_salt")

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            predict_salt_form_pk(DRUG, LOGP, -100.0, "HCl_salt")

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            predict_salt_form_pk(DRUG, LOGP, DOSE, "HCl_salt", cl_L_per_h=0.0)

    def test_cl_negative_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            predict_salt_form_pk(DRUG, LOGP, DOSE, "HCl_salt", cl_L_per_h=-5.0)

    def test_q_liver_zero_raises(self):
        with pytest.raises(ValueError, match="q_liver_L_per_h"):
            predict_salt_form_pk(DRUG, LOGP, DOSE, "HCl_salt", q_liver_L_per_h=0.0)

    def test_q_liver_negative_raises(self):
        with pytest.raises(ValueError, match="q_liver_L_per_h"):
            predict_salt_form_pk(DRUG, LOGP, DOSE, "HCl_salt", q_liver_L_per_h=-90.0)

    def test_invalid_salt_form_raises(self):
        with pytest.raises(ValueError, match="salt_form"):
            predict_salt_form_pk(DRUG, LOGP, DOSE, "invalid_salt_xyz")
