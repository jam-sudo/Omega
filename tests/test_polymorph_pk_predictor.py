"""Tests for Phase 857: Drug Crystal Polymorph PK Predictor."""

import math

import pytest

from omega_pbpk.prediction.polymorph_pk_predictor import (
    PolymorphPKResult,
    compare_polymorphs,
    predict_polymorph_pk,
)

# ---------------------------------------------------------------------------
# Basic return type and field preservation
# ---------------------------------------------------------------------------


def test_returns_polymorph_pk_result_type():
    result = predict_polymorph_pk("TestDrug", 100.0, "Form I", 1.0)
    assert isinstance(result, PolymorphPKResult)


def test_field_preservation_drug_name():
    result = predict_polymorph_pk("Aspirin", 100.0, "Form I", 1.0)
    assert result.drug_name == "Aspirin"


def test_field_preservation_polymorph_name():
    result = predict_polymorph_pk("Aspirin", 100.0, "Form II", 2.0)
    assert result.polymorph_name == "Form II"


def test_field_preservation_rsf():
    result = predict_polymorph_pk("TestDrug", 100.0, "Form I", 1.0)
    assert result.rsf == 1.0


def test_field_preservation_dose_mg():
    result = predict_polymorph_pk("TestDrug", 250.0, "Form I", 1.0)
    assert result.dose_mg == 250.0


def test_notes_is_string():
    result = predict_polymorph_pk("TestDrug", 100.0, "Form I", 1.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Dissolution fraction tests
# ---------------------------------------------------------------------------


def test_f_dissolved_2h_form1_range():
    """Form I (rsf=1.0): f_diss at 2h = 1 - exp(-0.2) ≈ 0.181"""
    result = predict_polymorph_pk("TestDrug", 100.0, "Form I", 1.0)
    expected = 1.0 - math.exp(-0.1 * 1.0 * 2.0)
    assert abs(result.f_dissolved_2h - expected) < 1e-9


def test_f_dissolved_4h_form1_range():
    """Form I (rsf=1.0): f_diss at 4h = 1 - exp(-0.4) ≈ 0.330"""
    result = predict_polymorph_pk("TestDrug", 100.0, "Form I", 1.0)
    expected = 1.0 - math.exp(-0.1 * 1.0 * 4.0)
    assert abs(result.f_dissolved_4h - expected) < 1e-9


def test_higher_rsf_higher_f_dissolved_2h():
    """Higher rsf → faster dissolution → higher f_dissolved_2h."""
    result_form1 = predict_polymorph_pk("TestDrug", 100.0, "Form I", 1.0)
    result_form2 = predict_polymorph_pk("TestDrug", 100.0, "Form II", 2.0)
    assert result_form2.f_dissolved_2h > result_form1.f_dissolved_2h


def test_higher_rsf_higher_f_dissolved_4h():
    """Higher rsf → higher f_dissolved_4h."""
    result_form1 = predict_polymorph_pk("TestDrug", 100.0, "Form I", 1.0)
    result_amorph = predict_polymorph_pk("TestDrug", 100.0, "Amorphous", 100.0)
    assert result_amorph.f_dissolved_4h > result_form1.f_dissolved_4h


def test_amorphous_f_dissolved_4h_near_1():
    """Amorphous (rsf=100) dissolves almost completely at 4h."""
    result = predict_polymorph_pk("TestDrug", 100.0, "Amorphous", 100.0)
    assert result.f_dissolved_4h > 0.99


def test_f_dissolved_capped_at_1():
    """f_dissolved never exceeds 1.0."""
    result = predict_polymorph_pk("TestDrug", 100.0, "Amorphous", 1000.0)
    assert result.f_dissolved_2h <= 1.0
    assert result.f_dissolved_4h <= 1.0


# ---------------------------------------------------------------------------
# PK metric monotonicity with rsf
# ---------------------------------------------------------------------------


def test_higher_rsf_higher_cmax():
    """Higher rsf → more drug absorbed → higher Cmax."""
    result_form1 = predict_polymorph_pk("TestDrug", 100.0, "Form I", 1.0)
    result_form2 = predict_polymorph_pk("TestDrug", 100.0, "Form II", 2.0)
    assert result_form2.cmax_mg_L > result_form1.cmax_mg_L


def test_higher_rsf_higher_auc():
    """Higher rsf → higher AUC."""
    result_form1 = predict_polymorph_pk("TestDrug", 100.0, "Form I", 1.0)
    result_form2 = predict_polymorph_pk("TestDrug", 100.0, "Form II", 2.0)
    assert result_form2.auc_mg_h_per_L > result_form1.auc_mg_h_per_L


def test_amorphous_highest_auc():
    """Amorphous (rsf=100) has highest AUC vs Form I and Form II."""
    r_form1 = predict_polymorph_pk("TestDrug", 100.0, "Form I", 1.0)
    r_form2 = predict_polymorph_pk("TestDrug", 100.0, "Form II", 2.0)
    r_amorph = predict_polymorph_pk("TestDrug", 100.0, "Amorphous", 100.0)
    assert r_amorph.auc_mg_h_per_L > r_form2.auc_mg_h_per_L
    assert r_form2.auc_mg_h_per_L > r_form1.auc_mg_h_per_L


# ---------------------------------------------------------------------------
# Stability classification
# ---------------------------------------------------------------------------


def test_form1_stability_class_stable():
    result = predict_polymorph_pk("TestDrug", 100.0, "Form I", 1.0)
    assert result.stability_class == "stable"


def test_form2_stability_class_metastable():
    result = predict_polymorph_pk("TestDrug", 100.0, "Form II", 2.0)
    assert result.stability_class == "metastable"


def test_amorphous_stability_class():
    result = predict_polymorph_pk("TestDrug", 100.0, "Amorphous", 100.0)
    assert result.stability_class == "amorphous"


def test_boundary_rsf_10_amorphous():
    """rsf == 10 exactly -> amorphous."""
    result = predict_polymorph_pk("TestDrug", 100.0, "Form X", 10.0)
    assert result.stability_class == "amorphous"


def test_boundary_rsf_1_stable():
    """rsf == 1.0 exactly -> stable."""
    result = predict_polymorph_pk("TestDrug", 100.0, "Form I", 1.0)
    assert result.stability_class == "stable"


# ---------------------------------------------------------------------------
# Relative bioavailability
# ---------------------------------------------------------------------------


def test_form1_bioavailability_relative_is_1():
    """Form I (rsf=1.0) must have bioavailability_relative == 1.0."""
    result = predict_polymorph_pk("TestDrug", 100.0, "Form I", 1.0)
    assert abs(result.bioavailability_relative - 1.0) < 1e-9


def test_higher_rsf_higher_relative_ba():
    """Higher rsf → higher relative BA vs Form I."""
    r1 = predict_polymorph_pk("TestDrug", 100.0, "Form I", 1.0)
    r2 = predict_polymorph_pk("TestDrug", 100.0, "Form II", 2.0)
    assert r1.bioavailability_relative == 1.0
    assert r2.bioavailability_relative > 1.0


# ---------------------------------------------------------------------------
# compare_polymorphs
# ---------------------------------------------------------------------------


def test_compare_polymorphs_returns_list():
    polymorphs = [
        {"name": "Form I", "rsf": 1.0},
        {"name": "Form II", "rsf": 2.0},
        {"name": "Amorphous", "rsf": 100.0},
    ]
    results = compare_polymorphs("TestDrug", 100.0, polymorphs)
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_polymorphs_sorted_by_auc_desc():
    """compare_polymorphs returns results sorted by AUC descending."""
    polymorphs = [
        {"name": "Form I", "rsf": 1.0},
        {"name": "Form II", "rsf": 2.0},
        {"name": "Amorphous", "rsf": 100.0},
    ]
    results = compare_polymorphs("TestDrug", 100.0, polymorphs)
    aucs = [r.auc_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_polymorphs_first_is_amorphous():
    """Amorphous (highest rsf) should be first after sorting by AUC descending."""
    polymorphs = [
        {"name": "Form I", "rsf": 1.0},
        {"name": "Amorphous", "rsf": 100.0},
    ]
    results = compare_polymorphs("TestDrug", 100.0, polymorphs)
    assert results[0].polymorph_name == "Amorphous"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        predict_polymorph_pk("TestDrug", 0.0, "Form I", 1.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        predict_polymorph_pk("TestDrug", -10.0, "Form I", 1.0)


def test_validation_rsf_zero():
    with pytest.raises(ValueError, match="rsf must be > 0"):
        predict_polymorph_pk("TestDrug", 100.0, "Form I", 0.0)


def test_validation_rsf_negative():
    with pytest.raises(ValueError, match="rsf must be > 0"):
        predict_polymorph_pk("TestDrug", 100.0, "Form I", -1.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h must be > 0"):
        predict_polymorph_pk("TestDrug", 100.0, "Form I", 1.0, cl_L_per_h=0.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L must be > 0"):
        predict_polymorph_pk("TestDrug", 100.0, "Form I", 1.0, vd_L=0.0)
