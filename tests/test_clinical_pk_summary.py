"""Tests for Phase 752 — Clinical PK Summary Generator."""

import sys

import pytest

sys.path.insert(0, "src")

from omega_pbpk.analysis.clinical_pk_summary import (
    ClinicalPKSummary,
    generate_clinical_pk_summary,
    screen_pk_profiles,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs) -> ClinicalPKSummary:
    defaults = dict(
        drug_name="TestDrug",
        t_half_h=12.0,
        vd_L_per_kg=1.0,
        cl_mL_per_min_per_kg=5.0,
        f_oral=0.6,
        protein_binding_pct=70.0,
        mec_mg_L=None,
        mtc_mg_L=None,
        fe_renal=0.4,
    )
    defaults.update(kwargs)
    return generate_clinical_pk_summary(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_clinical_pk_summary():
    res = _default()
    assert isinstance(res, ClinicalPKSummary)


def test_drug_name_preserved():
    res = _default(drug_name="Warfarin")
    assert res.drug_name == "Warfarin"


# ---------------------------------------------------------------------------
# Dosing interval recommendation
# ---------------------------------------------------------------------------


def test_long_thalf_gives_qd():
    res = _default(t_half_h=24.0)
    assert res.dosing_interval_recommendation == "QD"


def test_thalf_borderline_qd():
    res = _default(t_half_h=18.5)
    assert res.dosing_interval_recommendation == "QD"


def test_thalf_8_to_18_gives_bid():
    res = _default(t_half_h=10.0)
    assert res.dosing_interval_recommendation == "BID"


def test_thalf_4_to_8_gives_tid():
    res = _default(t_half_h=6.0)
    assert res.dosing_interval_recommendation == "TID"


def test_short_thalf_gives_qid():
    res = _default(t_half_h=2.0)
    assert res.dosing_interval_recommendation == "QID"


# ---------------------------------------------------------------------------
# Vd classification
# ---------------------------------------------------------------------------


def test_high_vd_classification():
    res = _default(vd_L_per_kg=5.0)
    assert res.vd_classification == "high"


def test_low_vd_classification():
    res = _default(vd_L_per_kg=0.2)
    assert res.vd_classification == "low"


def test_moderate_vd_classification():
    res = _default(vd_L_per_kg=1.0)
    assert res.vd_classification == "moderate"


# ---------------------------------------------------------------------------
# CL classification
# ---------------------------------------------------------------------------


def test_high_cl_classification():
    res = _default(cl_mL_per_min_per_kg=15.0)
    assert res.cl_classification == "high"


def test_low_cl_classification():
    res = _default(cl_mL_per_min_per_kg=1.0)
    assert res.cl_classification == "low"


# ---------------------------------------------------------------------------
# Bioavailability category
# ---------------------------------------------------------------------------


def test_excellent_bioavailability():
    res = _default(f_oral=0.90)
    assert res.bioavailability_category == "excellent"


def test_good_bioavailability():
    res = _default(f_oral=0.65)
    assert res.bioavailability_category == "good"


def test_moderate_bioavailability():
    res = _default(f_oral=0.35)
    assert res.bioavailability_category == "moderate"


def test_poor_bioavailability():
    res = _default(f_oral=0.10)
    assert res.bioavailability_category == "poor"


# ---------------------------------------------------------------------------
# Protein binding
# ---------------------------------------------------------------------------


def test_high_protein_binding():
    res = _default(protein_binding_pct=90.0)
    assert res.protein_binding_category == "high"


def test_low_protein_binding():
    res = _default(protein_binding_pct=30.0)
    assert res.protein_binding_category == "low"


# ---------------------------------------------------------------------------
# Therapeutic window
# ---------------------------------------------------------------------------


def test_therapeutic_window_width_computed():
    res = _default(mec_mg_L=1.0, mtc_mg_L=5.0)
    # tw_width = (5-1)/1 = 4.0
    assert res.therapeutic_window_width == pytest.approx(4.0, rel=1e-6)


def test_no_mec_mtc_gives_minus_one():
    res = _default(mec_mg_L=None, mtc_mg_L=None)
    assert res.therapeutic_window_width == pytest.approx(-1.0)


def test_narrow_therapeutic_index_true():
    # tw = (2-1)/1 = 1.0 < 2 → NTI
    res = _default(mec_mg_L=1.0, mtc_mg_L=2.0)
    assert res.narrow_therapeutic_index is True


def test_narrow_therapeutic_index_false_wide():
    # tw = (10-1)/1 = 9 ≥ 2 → not NTI
    res = _default(mec_mg_L=1.0, mtc_mg_L=10.0)
    assert res.narrow_therapeutic_index is False


def test_narrow_therapeutic_index_false_no_mec_mtc():
    res = _default()
    assert res.narrow_therapeutic_index is False


# ---------------------------------------------------------------------------
# Elimination route
# ---------------------------------------------------------------------------


def test_renal_dominant():
    res = _default(fe_renal=0.8)
    assert res.elimination_route_primary == "renal-dominant"


def test_hepatic_dominant():
    res = _default(fe_renal=0.1)
    assert res.elimination_route_primary == "hepatic-dominant"


def test_mixed_elimination():
    res = _default(fe_renal=0.4)
    assert res.elimination_route_primary == "mixed"


# ---------------------------------------------------------------------------
# Accumulation risk
# ---------------------------------------------------------------------------

_VALID_ACCUM = {"low", "moderate", "high"}


def test_accumulation_risk_valid_value():
    res = _default(t_half_h=12.0)
    assert res.accumulation_risk in _VALID_ACCUM


def test_accumulation_low_for_short_thalf_frequent_dosing():
    # t½=2h, QID (τ=6h) → ke=0.347/h, ratio = 1/(1-exp(-0.347*6)) ≈ 1/(1-0.125) ≈ 1.14 → low
    res = _default(t_half_h=2.0)
    assert res.accumulation_risk == "low"


def test_accumulation_high_for_long_thalf():
    # t½=100h, QD (τ=24h) → ke=0.00693/h, ratio = 1/(1-exp(-0.1663)) ≈ 6.6 → high
    res = _default(t_half_h=100.0)
    assert res.accumulation_risk == "high"


# ---------------------------------------------------------------------------
# screen_pk_profiles
# ---------------------------------------------------------------------------

_DRUG_A = dict(
    drug_name="DrugA",
    t_half_h=24.0,
    vd_L_per_kg=1.0,
    cl_mL_per_min_per_kg=5.0,
    f_oral=0.8,
    protein_binding_pct=60.0,
)
_DRUG_B = dict(
    drug_name="DrugB",
    t_half_h=4.0,
    vd_L_per_kg=2.0,
    cl_mL_per_min_per_kg=8.0,
    f_oral=0.5,
    protein_binding_pct=40.0,
)
_DRUG_C = dict(
    drug_name="DrugC",
    t_half_h=12.0,
    vd_L_per_kg=3.0,
    cl_mL_per_min_per_kg=3.0,
    f_oral=0.3,
    protein_binding_pct=80.0,
)


def test_screen_returns_list():
    results = screen_pk_profiles([_DRUG_A, _DRUG_B, _DRUG_C])
    assert isinstance(results, list)


def test_screen_returns_correct_length():
    results = screen_pk_profiles([_DRUG_A, _DRUG_B])
    assert len(results) == 2


def test_screen_sorted_by_thalf_descending():
    results = screen_pk_profiles([_DRUG_B, _DRUG_C, _DRUG_A])
    t_halfs = [r.t_half_h for r in results]
    assert t_halfs == sorted(t_halfs, reverse=True)


def test_screen_all_clinical_pk_summary_instances():
    results = screen_pk_profiles([_DRUG_A, _DRUG_B])
    for r in results:
        assert isinstance(r, ClinicalPKSummary)


# ---------------------------------------------------------------------------
# Clinical notes
# ---------------------------------------------------------------------------


def test_clinical_notes_nonempty():
    res = _default()
    assert isinstance(res.clinical_notes, str)
    assert len(res.clinical_notes) > 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_thalf_zero_raises():
    with pytest.raises(ValueError):
        _default(t_half_h=0.0)


def test_validation_thalf_negative_raises():
    with pytest.raises(ValueError):
        _default(t_half_h=-5.0)


def test_validation_vd_zero_raises():
    with pytest.raises(ValueError):
        _default(vd_L_per_kg=0.0)


def test_validation_cl_zero_raises():
    with pytest.raises(ValueError):
        _default(cl_mL_per_min_per_kg=0.0)
