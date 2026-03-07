"""Tests for prediction/plasma_protein_affinity.py — Phase 658."""

import pytest

from omega_pbpk.prediction.plasma_protein_affinity import (
    PlasmaProteinAffinityResult,
    predict_plasma_protein_affinity,
    screen_protein_affinity,
)

# --- Basic return type and structure ---


def test_returns_result_type():
    result = predict_plasma_protein_affinity("Warfarin", logP=2.7, mw=308.3, psa=63.6)
    assert isinstance(result, PlasmaProteinAffinityResult)


def test_predicted_kd_positive():
    result = predict_plasma_protein_affinity("Warfarin", logP=2.7, mw=308.3, psa=63.6)
    assert result.predicted_kd_mg_L > 0.0


def test_predicted_bmax_positive():
    result = predict_plasma_protein_affinity("Warfarin", logP=2.7, mw=308.3, psa=63.6)
    assert result.predicted_bmax_umol_L > 0.0


def test_hill_coefficient_is_one():
    result = predict_plasma_protein_affinity("Warfarin", logP=2.7, mw=308.3, psa=63.6)
    assert result.hill_coefficient == 1.0


def test_fu_therapeutic_in_range():
    result = predict_plasma_protein_affinity("Warfarin", logP=2.7, mw=308.3, psa=63.6)
    assert 0.0 < result.fu_at_therapeutic_conc <= 1.0


def test_fu_high_in_range():
    result = predict_plasma_protein_affinity("Warfarin", logP=2.7, mw=308.3, psa=63.6)
    assert 0.0 < result.fu_at_high_conc <= 1.0


def test_binding_class_valid_values():
    result = predict_plasma_protein_affinity("Warfarin", logP=2.7, mw=308.3, psa=63.6)
    assert result.binding_class in {"strong", "moderate", "weak", "negligible"}


def test_dominant_binding_site_valid_values():
    result = predict_plasma_protein_affinity("Warfarin", logP=2.7, mw=308.3, psa=63.6)
    assert result.dominant_binding_site in {"site_i", "site_ii", "non_specific"}


def test_notes_nonempty():
    result = predict_plasma_protein_affinity("Warfarin", logP=2.7, mw=308.3, psa=63.6)
    assert len(result.notes) > 0


# --- Pharmacological sensitivity ---


def test_high_logp_lower_kd_albumin():
    """Higher logP → stronger albumin binding (lower Kd)."""
    r_low = predict_plasma_protein_affinity("LipophilicLow", logP=1.0, mw=300.0, psa=60.0)
    r_high = predict_plasma_protein_affinity("LipophilicHigh", logP=5.0, mw=300.0, psa=60.0)
    assert r_high.predicted_kd_mg_L < r_low.predicted_kd_mg_L


def test_acidic_drug_lower_kd_albumin():
    """Acidic drug (pKa<5) → albumin site I bonus → lower Kd."""
    r_neutral = predict_plasma_protein_affinity("Neutral", logP=2.0, mw=300.0, psa=60.0)
    r_acid = predict_plasma_protein_affinity("Acid", logP=2.0, mw=300.0, psa=60.0, pka_acid=3.5)
    assert r_acid.predicted_kd_mg_L < r_neutral.predicted_kd_mg_L


def test_basic_drug_lower_kd_agp():
    """Basic drug (pKa>7) → AGP binds more tightly (lower Kd)."""
    r_neutral = predict_plasma_protein_affinity(
        "Neutral", logP=2.0, mw=300.0, psa=60.0, protein="alpha1_agp"
    )
    r_basic = predict_plasma_protein_affinity(
        "Basic", logP=2.0, mw=300.0, psa=60.0, pka_base=9.0, protein="alpha1_agp"
    )
    assert r_basic.predicted_kd_mg_L < r_neutral.predicted_kd_mg_L


def test_agp_basic_drug_lower_kd_than_neutral():
    r_neutral = predict_plasma_protein_affinity(
        "Neutral", logP=2.5, mw=350.0, psa=50.0, protein="alpha1_agp"
    )
    r_basic = predict_plasma_protein_affinity(
        "Basic", logP=2.5, mw=350.0, psa=50.0, pka_base=8.5, protein="alpha1_agp"
    )
    assert r_basic.predicted_kd_mg_L < r_neutral.predicted_kd_mg_L


def test_fu_high_ge_fu_therapeutic():
    """Saturation at higher concentration should increase free fraction."""
    result = predict_plasma_protein_affinity("Drug", logP=3.0, mw=300.0, psa=50.0)
    assert result.fu_at_high_conc >= result.fu_at_therapeutic_conc


def test_saturable_binding_flag():
    """Saturable flag set correctly based on fu_high vs fu_therapeutic."""
    result = predict_plasma_protein_affinity("Drug", logP=3.0, mw=300.0, psa=50.0)
    expected = result.fu_at_high_conc > 1.5 * result.fu_at_therapeutic_conc
    assert result.saturable_binding == expected


# --- Screen function ---


def test_screen_returns_list_of_three():
    results = screen_protein_affinity("Ibuprofen", logP=3.5, mw=206.3, psa=37.3)
    assert len(results) == 3


def test_screen_sorted_by_kd_ascending():
    results = screen_protein_affinity("Ibuprofen", logP=3.5, mw=206.3, psa=37.3)
    kds = [r.predicted_kd_mg_L for r in results]
    assert kds == sorted(kds)


def test_screen_all_three_proteins_covered():
    results = screen_protein_affinity("Ibuprofen", logP=3.5, mw=206.3, psa=37.3)
    proteins = {r.protein for r in results}
    assert proteins == {"albumin", "alpha1_agp", "lipoprotein"}


# --- Validation ---


def test_validation_mw_zero_raises():
    with pytest.raises(ValueError):
        predict_plasma_protein_affinity("Drug", logP=2.0, mw=0.0, psa=50.0)


def test_validation_unknown_protein_raises():
    with pytest.raises(ValueError):
        predict_plasma_protein_affinity("Drug", logP=2.0, mw=300.0, psa=50.0, protein="unknown")


def test_validation_negative_n_hbd_raises():
    with pytest.raises(ValueError):
        predict_plasma_protein_affinity("Drug", logP=2.0, mw=300.0, psa=50.0, n_hbd=-1)


def test_validation_negative_psa_raises():
    with pytest.raises(ValueError):
        predict_plasma_protein_affinity("Drug", logP=2.0, mw=300.0, psa=-1.0)
