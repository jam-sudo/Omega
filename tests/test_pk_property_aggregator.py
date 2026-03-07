"""Tests for Phase 673 — PK Property Aggregator."""

import math

import pytest

from omega_pbpk.prediction.pk_property_aggregator import (
    DrugProfile,
    build_drug_profile,
    compare_drug_profiles,
    profile_similarity,
)


def _default_profile(**kwargs) -> DrugProfile:
    defaults = dict(
        compound_name="TestDrug",
        logP=2.0,
        mw=350.0,
        psa=60.0,
        n_hbd=2,
        n_hba=4,
        clint_uL_per_min_per_mg=10.0,
        melting_point_c=150.0,
    )
    defaults.update(kwargs)
    return build_drug_profile(**defaults)


# ── Basic construction ───────────────────────────────────────────────────────


def test_build_drug_profile_returns_drug_profile():
    p = _default_profile()
    assert isinstance(p, DrugProfile)


def test_predicted_t_half_h_positive():
    p = _default_profile()
    assert p.predicted_t_half_h > 0.0


def test_predicted_vd_L_per_kg_positive():
    p = _default_profile()
    assert p.predicted_vd_L_per_kg > 0.0


def test_predicted_cl_L_per_h_per_kg_positive():
    p = _default_profile()
    assert p.predicted_cl_L_per_h_per_kg > 0.0


def test_predicted_f_oral_in_range():
    p = _default_profile()
    assert 0.0 < p.predicted_f_oral <= 1.0


def test_predicted_fup_in_range():
    p = _default_profile()
    assert 0.0 < p.predicted_fup <= 1.0


def test_predicted_logD_74_is_finite():
    p = _default_profile()
    assert math.isfinite(p.predicted_logD_74)


def test_predicted_solubility_positive():
    p = _default_profile()
    assert p.predicted_solubility_mg_mL > 0.0


def test_lipinski_score_range():
    p = _default_profile()
    assert 0 <= p.lipinski_score <= 4


def test_admet_profile_grade_valid():
    p = _default_profile()
    assert p.admet_profile_grade in {"A", "B", "C", "D"}


def test_notes_nonempty():
    p = _default_profile()
    assert len(p.notes) > 0


# ── Override params ──────────────────────────────────────────────────────────


def test_measured_fup_overrides_predicted():
    p = build_drug_profile(
        compound_name="X",
        logP=2.0,
        mw=300.0,
        psa=50.0,
        measured_fup=0.99,
    )
    assert p.predicted_fup == pytest.approx(0.99)


def test_measured_t_half_overrides_predicted():
    p = build_drug_profile(
        compound_name="X",
        logP=2.0,
        mw=300.0,
        psa=50.0,
        measured_t_half_h=12.5,
    )
    assert p.predicted_t_half_h == pytest.approx(12.5)


# ── Specific flag tests ──────────────────────────────────────────────────────


def test_bbb_penetration_likely_true_for_cns_drug():
    p = build_drug_profile(
        compound_name="CNSDrug",
        logP=2.0,
        mw=350.0,
        psa=60.0,
    )
    assert p.bbb_penetration_likely is True


def test_herg_risk_flag_true_for_cationic_lipophilic():
    p = build_drug_profile(
        compound_name="hERG",
        logP=5.0,
        mw=400.0,
        psa=50.0,
        pka_base=8.0,
    )
    assert p.herg_risk_flag is True


# ── Comparison / similarity ──────────────────────────────────────────────────


def test_compare_drug_profiles_returns_sorted_list():
    p1 = build_drug_profile("A", logP=-0.5, mw=200.0, psa=150.0)  # low f_oral
    p2 = build_drug_profile("B", logP=2.0, mw=350.0, psa=60.0)  # higher f_oral
    result = compare_drug_profiles([p1, p2], metric="predicted_f_oral")
    assert result[0].predicted_f_oral >= result[1].predicted_f_oral


def test_profile_similarity_returns_float_in_range():
    p1 = _default_profile(compound_name="A")
    p2 = _default_profile(compound_name="B", logP=3.0, mw=400.0, psa=80.0)
    sim = profile_similarity(p1, p2)
    assert isinstance(sim, float)
    assert 0.0 <= sim <= 1.0


def test_profile_similarity_identical_profiles():
    p1 = _default_profile(compound_name="A")
    p2 = _default_profile(compound_name="A")
    sim = profile_similarity(p1, p2)
    assert sim == pytest.approx(1.0)


# ── Validation errors ────────────────────────────────────────────────────────


def test_validation_mw_zero_raises():
    with pytest.raises(ValueError):
        build_drug_profile(compound_name="X", logP=2.0, mw=0.0, psa=60.0)


def test_validation_n_hbd_negative_raises():
    with pytest.raises(ValueError):
        build_drug_profile(compound_name="X", logP=2.0, mw=300.0, psa=60.0, n_hbd=-1)


# ── Property-driven tests ────────────────────────────────────────────────────


def test_high_logP_lower_f_oral():
    p_high = build_drug_profile("High", logP=6.0, mw=400.0, psa=50.0)
    p_low = build_drug_profile("Low", logP=2.0, mw=400.0, psa=50.0)
    assert p_high.predicted_f_oral < p_low.predicted_f_oral


def test_high_psa_low_permeability():
    p = build_drug_profile("HighPSA", logP=2.0, mw=400.0, psa=130.0)
    assert p.predicted_psa_permeability == "low"


def test_low_clint_stable_metabolic_stability():
    p = build_drug_profile("Stable", logP=2.0, mw=350.0, psa=60.0, clint_uL_per_min_per_mg=1.0)
    assert p.metabolic_stability_class == "stable"
