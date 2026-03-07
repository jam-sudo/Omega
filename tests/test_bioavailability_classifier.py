"""Tests for prediction/bioavailability_classifier.py (Phase 607)."""

from __future__ import annotations

from omega_pbpk.prediction.bioavailability_classifier import (
    BioavailabilityResult,
    classify_f,
    lipinski_violations,
    predict_bioavailability,
)

VALID_KWARGS = dict(
    drug_name="TestDrug",
    logP=2.0,
    mw=300.0,
    psa=60.0,
    solubility_mg_mL=1.0,
    dose_mg=100.0,
    n_hbd=2,
    n_hba=4,
    cyp3a4_er=0.3,
    fu_plasma=0.5,
)


def test_returns_result():
    result = predict_bioavailability(**VALID_KWARGS)
    assert isinstance(result, BioavailabilityResult)


def test_f_pct_in_0_100():
    result = predict_bioavailability(**VALID_KWARGS)
    assert 0.0 <= result.predicted_f_pct <= 100.0


def test_f_class_valid():
    result = predict_bioavailability(**VALID_KWARGS)
    assert result.f_class in {"very_low", "low", "moderate", "high", "very_high"}


def test_fa_in_0_1():
    result = predict_bioavailability(**VALID_KWARGS)
    assert 0.0 <= result.fa_fraction <= 1.0


def test_fg_in_0_1():
    result = predict_bioavailability(**VALID_KWARGS)
    assert 0.0 <= result.fg_fraction <= 1.0


def test_fh_in_0_1():
    result = predict_bioavailability(**VALID_KWARGS)
    assert 0.0 <= result.fh_fraction <= 1.0


def test_high_solubility_higher_f():
    r_high = predict_bioavailability(**{**VALID_KWARGS, "solubility_mg_mL": 10.0})
    r_low = predict_bioavailability(**{**VALID_KWARGS, "solubility_mg_mL": 0.001})
    assert r_high.predicted_f_pct > r_low.predicted_f_pct


def test_high_psa_lower_f():
    r_low_psa = predict_bioavailability(**{**VALID_KWARGS, "psa": 30.0})
    r_high_psa = predict_bioavailability(**{**VALID_KWARGS, "psa": 150.0})
    assert r_low_psa.predicted_f_pct > r_high_psa.predicted_f_pct


def test_high_cyp3a4_er_lower_f():
    r_low_er = predict_bioavailability(**{**VALID_KWARGS, "cyp3a4_er": 0.1})
    r_high_er = predict_bioavailability(**{**VALID_KWARGS, "cyp3a4_er": 0.8})
    assert r_low_er.predicted_f_pct > r_high_er.predicted_f_pct


def test_lipinski_pass_true_for_druglike():
    result = predict_bioavailability(
        **{**VALID_KWARGS, "mw": 300.0, "logP": 2.0, "n_hbd": 2, "n_hba": 4}
    )
    assert result.lipinski_pass is True


def test_lipinski_pass_false_for_violator():
    result = predict_bioavailability(
        **{**VALID_KWARGS, "mw": 600.0, "logP": 6.0, "n_hbd": 6, "n_hba": 11}
    )
    assert result.lipinski_pass is False


def test_bcs_class_valid():
    result = predict_bioavailability(**VALID_KWARGS)
    assert result.bcs_class in {"I", "II", "III", "IV"}


def test_bcs_i_for_high_sol_high_perm():
    # High sol (Do<=1), high perm (low PSA, good logP, low MW) -> I
    result = predict_bioavailability(
        drug_name="BCS_I",
        logP=2.0,
        mw=300.0,
        psa=40.0,
        solubility_mg_mL=10.0,
        dose_mg=100.0,
    )
    assert result.bcs_class in {"I", "II"}


def test_bcs_iv_for_low_sol_low_perm():
    # Low sol (Do>1), low perm (high PSA) -> IV
    result = predict_bioavailability(
        drug_name="BCS_IV",
        logP=-1.0,
        mw=550.0,
        psa=120.0,
        solubility_mg_mL=0.001,
        dose_mg=100.0,
    )
    assert result.bcs_class == "IV"


def test_dose_number_positive():
    result = predict_bioavailability(**VALID_KWARGS)
    assert result.predicted_dose_number > 0.0


def test_classify_f_very_low():
    assert classify_f(5.0) == "very_low"


def test_classify_f_very_high():
    assert classify_f(90.0) == "very_high"


def test_lipinski_violations_zero():
    # MW=300, logP=2, HBD=2, HBA=4 -> 0 violations
    assert lipinski_violations(300.0, 2.0, 2, 4) == 0


def test_lipinski_violations_counts():
    # MW=600>500, logP=6>5, HBD=6>5, HBA=11>10 -> 4 violations
    assert lipinski_violations(600.0, 6.0, 6, 11) == 4


def test_confidence_high_for_good_drug():
    result = predict_bioavailability(
        drug_name="GoodDrug",
        logP=2.0,
        mw=300.0,
        psa=60.0,
        solubility_mg_mL=1.0,
        dose_mg=100.0,
        n_hbd=2,
        n_hba=4,
    )
    assert result.confidence == "high"


def test_notes_nonempty():
    result = predict_bioavailability(**VALID_KWARGS)
    assert len(result.notes) > 0


def test_f_equals_fa_fg_fh_product():
    result = predict_bioavailability(**VALID_KWARGS)
    expected = result.fa_fraction * result.fg_fraction * result.fh_fraction * 100.0
    assert abs(result.predicted_f_pct - expected) < 1.0
