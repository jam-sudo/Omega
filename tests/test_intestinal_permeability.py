"""Tests for Phase 931 — intestinal permeability prediction."""

import pytest

from omega_pbpk.prediction.intestinal_permeability import (
    IntestinalPermeabilityResult,
    predict_intestinal_permeability,
    screen_permeability,
)

# --- Basic return type and field checks ---


def test_return_type():
    result = predict_intestinal_permeability("DrugA", logp=2.0, mw=300.0, pka=7.0)
    assert isinstance(result, IntestinalPermeabilityResult)


def test_frozen_dataclass():
    result = predict_intestinal_permeability("DrugA", logp=2.0, mw=300.0, pka=7.0)
    with pytest.raises((AttributeError, TypeError)):
        result.fa = 0.5  # type: ignore


def test_caco2_papp_positive():
    result = predict_intestinal_permeability("DrugA", logp=2.0, mw=300.0, pka=7.0)
    assert result.caco2_papp_cm_s > 0


def test_peff_positive():
    result = predict_intestinal_permeability("DrugA", logp=2.0, mw=300.0, pka=7.0)
    assert result.peff_cm_s > 0


def test_fa_in_range():
    result = predict_intestinal_permeability("DrugA", logp=2.0, mw=300.0, pka=7.0)
    assert 0.0 <= result.fa <= 1.0


def test_dose_number_positive():
    result = predict_intestinal_permeability(
        "DrugA", logp=2.0, mw=300.0, pka=7.0, dose_mg=100.0, cs_mg_mL=1.0
    )
    assert result.dose_number > 0


def test_bcs_class_valid():
    result = predict_intestinal_permeability("DrugA", logp=2.0, mw=300.0, pka=7.0)
    assert result.bcs_class in {"I", "II", "III", "IV"}


def test_high_permeability_is_bool():
    result = predict_intestinal_permeability("DrugA", logp=2.0, mw=300.0, pka=7.0)
    assert isinstance(result.high_permeability, bool)


def test_absorption_category_valid():
    result = predict_intestinal_permeability("DrugA", logp=2.0, mw=300.0, pka=7.0)
    assert result.absorption_category in {"excellent", "good", "moderate", "poor"}


# --- Physicochemical effects ---


def test_high_logp_higher_peff_than_low_logp():
    high = predict_intestinal_permeability("HighLogP", logp=4.0, mw=300.0, pka=7.0)
    low = predict_intestinal_permeability("LowLogP", logp=0.0, mw=300.0, pka=7.0)
    assert high.peff_cm_s > low.peff_cm_s


def test_high_mw_lower_peff_than_low_mw():
    high_mw = predict_intestinal_permeability("HeavyDrug", logp=2.0, mw=700.0, pka=7.0)
    low_mw = predict_intestinal_permeability("LightDrug", logp=2.0, mw=200.0, pka=7.0)
    assert high_mw.peff_cm_s < low_mw.peff_cm_s


def test_low_solubility_bcs_ii_or_iv():
    result = predict_intestinal_permeability(
        "PoorSolDrug", logp=4.0, mw=300.0, pka=7.0, dose_mg=500.0, cs_mg_mL=0.01
    )
    assert result.bcs_class in {"II", "IV"}
    assert not result.high_solubility


# --- Absorption category consistency ---


def test_high_fa_excellent_category():
    # Very high logP, low MW → high fa
    result = predict_intestinal_permeability("HighPermDrug", logp=5.0, mw=150.0, pka=7.0)
    if result.fa > 0.9:
        assert result.absorption_category == "excellent"


def test_low_fa_poor_category():
    # Very low logP, high MW → low fa
    result = predict_intestinal_permeability("PoorPermDrug", logp=-3.0, mw=800.0, pka=7.0)
    if result.fa <= 0.4:
        assert result.absorption_category == "poor"


def test_absorption_categories_consistent():
    result = predict_intestinal_permeability("DrugX", logp=2.0, mw=400.0, pka=7.0)
    fa = result.fa
    cat = result.absorption_category
    if fa > 0.9:
        assert cat == "excellent"
    elif fa > 0.7:
        assert cat == "good"
    elif fa > 0.4:
        assert cat == "moderate"
    else:
        assert cat == "poor"


# --- Drug name preserved ---


def test_drug_name_preserved():
    result = predict_intestinal_permeability(
        "Metformin", logp=-1.43, mw=129.0, pka=12.4, ionization_type="base"
    )
    assert result.drug_name == "Metformin"


# --- Notes ---


def test_notes_non_empty():
    result = predict_intestinal_permeability("DrugA", logp=2.0, mw=300.0, pka=7.0)
    assert isinstance(result.notes, str) and len(result.notes) > 0


# --- Validation errors ---


def test_mw_zero_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_intestinal_permeability("DrugA", logp=2.0, mw=0.0, pka=7.0)


def test_mw_negative_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_intestinal_permeability("DrugA", logp=2.0, mw=-100.0, pka=7.0)


def test_dose_mg_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_intestinal_permeability("DrugA", logp=2.0, mw=300.0, pka=7.0, dose_mg=0.0)


def test_dose_mg_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_intestinal_permeability("DrugA", logp=2.0, mw=300.0, pka=7.0, dose_mg=-10.0)


def test_cs_mg_mL_zero_raises():
    with pytest.raises(ValueError, match="cs_mg_mL"):
        predict_intestinal_permeability("DrugA", logp=2.0, mw=300.0, pka=7.0, cs_mg_mL=0.0)


def test_cs_mg_mL_negative_raises():
    with pytest.raises(ValueError, match="cs_mg_mL"):
        predict_intestinal_permeability("DrugA", logp=2.0, mw=300.0, pka=7.0, cs_mg_mL=-1.0)


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_intestinal_permeability(
            "DrugA", logp=2.0, mw=300.0, pka=7.0, ionization_type="zwitterion"
        )


# --- screen_permeability ---


def test_screen_returns_correct_length():
    compounds = [
        {"drug_name": "A", "logp": 1.0, "mw": 300.0, "pka": 7.0},
        {"drug_name": "B", "logp": 3.0, "mw": 250.0, "pka": 7.0},
        {"drug_name": "C", "logp": -1.0, "mw": 500.0, "pka": 7.0},
    ]
    results = screen_permeability(compounds)
    assert len(results) == 3


def test_screen_sorted_by_fa_descending():
    compounds = [
        {"drug_name": "A", "logp": -2.0, "mw": 600.0, "pka": 7.0},
        {"drug_name": "B", "logp": 4.0, "mw": 200.0, "pka": 7.0},
        {"drug_name": "C", "logp": 1.0, "mw": 350.0, "pka": 7.0},
    ]
    results = screen_permeability(compounds)
    fas = [r.fa for r in results]
    assert fas == sorted(fas, reverse=True)


def test_fa_monotone_with_logp():
    """Higher logP → higher fa (with same MW)."""
    high = predict_intestinal_permeability("High", logp=4.0, mw=300.0, pka=7.0)
    low = predict_intestinal_permeability("Low", logp=-1.0, mw=300.0, pka=7.0)
    assert high.fa >= low.fa


def test_neutral_no_ionization_correction():
    """Neutral drug: ionization correction = 0, pKa doesn't matter."""
    r1 = predict_intestinal_permeability(
        "N1", logp=2.0, mw=300.0, pka=4.0, ionization_type="neutral"
    )
    r2 = predict_intestinal_permeability(
        "N2", logp=2.0, mw=300.0, pka=10.0, ionization_type="neutral"
    )
    assert abs(r1.fa - r2.fa) < 1e-10


def test_acid_low_pka_ionization_correction():
    """Acid with pKa < 7.4 has no extra correction; pKa > 7.4 has correction → lower fa."""
    r_high_pka = predict_intestinal_permeability(
        "AcidHigh", logp=2.0, mw=300.0, pka=9.0, ionization_type="acid"
    )
    r_neutral = predict_intestinal_permeability(
        "Neutral", logp=2.0, mw=300.0, pka=9.0, ionization_type="neutral"
    )
    # Acid with pKa=9.0 > 7.4: correction = 0.5*(9-7.4)=0.8 → lower fa
    assert r_high_pka.fa < r_neutral.fa
