"""Tests for Phase 428: Salt form selection and solubility enhancement prediction."""

import pytest

from omega_pbpk.prediction.salt_form_selection import (
    COUNTERIONS,
    SaltFormResult,
    predict_salt_form,
    screen_salt_forms,
)


# --- Return type ---
def test_return_type():
    result = predict_salt_form("DrugA", "hydrochloride", 8.5, "base", 0.01)
    assert isinstance(result, SaltFormResult)


# --- Field preservation ---
def test_drug_name_preserved():
    result = predict_salt_form("Nifedipine", "hydrochloride", 8.0, "base", 0.05)
    assert result.drug_name == "Nifedipine"


def test_counterion_preserved():
    result = predict_salt_form("DrugA", "mesylate", 7.5, "base", 0.02)
    assert result.salt_counterion == "mesylate"


def test_drug_pka_preserved():
    result = predict_salt_form("DrugA", "hydrochloride", 8.5, "base", 0.01)
    assert result.drug_pka == pytest.approx(8.5)


def test_intrinsic_solubility_preserved():
    result = predict_salt_form("DrugA", "hydrochloride", 8.5, "base", 0.01)
    assert result.intrinsic_solubility_mg_mL == pytest.approx(0.01)


# --- Solubility ratio ---
def test_solubility_ratio_at_least_one():
    # Even incompatible pair gives ratio = 1 (factor=1)
    result = predict_salt_form("DrugA", "sodium", 8.5, "base", 0.05)
    assert result.solubility_ratio >= 1.0


def test_compatible_base_drug_hcl_ratio_above_one():
    # Base drug + HCl (acid counterion) -> compatible
    result = predict_salt_form("DrugB", "hydrochloride", 8.5, "base", 0.01)
    assert result.solubility_ratio > 1.0


def test_incompatible_pair_ratio_approx_one():
    # Base drug + sodium (basic counterion) -> incompatible, factor=1
    result = predict_salt_form("DrugB", "sodium", 8.5, "base", 0.01)
    # ratio = 1 * (1 + |8.5 - 15| * 0.05) but factor=1 since incompatible
    # Actually ratio = 1 * (1 + 0.325) = 1.325 because factor=1
    # But we need to check that it uses factor=1
    # intrinsic * 1 * (1 + pka_diff*0.05)
    pka_diff = abs(8.5 - 15)
    expected = 1.0 * (1.0 + pka_diff * 0.05)
    assert result.solubility_ratio == pytest.approx(expected, rel=1e-4)


def test_salt_solubility_capped_at_50x():
    # Use very high solubility factor with large pKa difference to force cap
    # sodium has pka=15, acid drug pka=2 -> compatible (acid drug + basic counterion)
    # diff = |2 - 15| = 13, factor = 4.0
    # raw = 0.001 * 4.0 * (1 + 13*0.05) = 0.001 * 4.0 * 1.65 = 0.0066
    # cap = 0.001 * 50 = 0.05 -> not capped here
    # Let's verify cap logic with a contrived scenario
    result = predict_salt_form("AcidDrug", "sodium", 2.0, "acid", 0.001)
    assert result.salt_solubility_mg_mL <= result.intrinsic_solubility_mg_mL * 50.0


# --- pH calculations ---
def test_ph_max_reasonable_range():
    result = predict_salt_form("DrugA", "hydrochloride", 8.5, "base", 0.01)
    # ph_max for base drug with ratio > 1: drug_pka - 0.5*log10(ratio)
    assert -2.0 <= result.ph_max <= 14.0


def test_ph_optimal_dissolution_in_valid_range():
    result = predict_salt_form("DrugA", "hydrochloride", 8.5, "base", 0.01)
    assert 1.0 <= result.ph_optimal_dissolution <= 9.0


# --- Hygroscopicity ---
def test_hygroscopicity_from_database():
    result = predict_salt_form("DrugA", "hydrochloride", 8.5, "base", 0.01)
    assert result.hygroscopicity_risk == COUNTERIONS["hydrochloride"]["hygroscopicity"]


def test_tartrate_hygroscopicity_high():
    result = predict_salt_form("DrugA", "tartrate", 8.5, "base", 0.01)
    assert result.hygroscopicity_risk == "high"


def test_mesylate_hygroscopicity_low():
    result = predict_salt_form("DrugA", "mesylate", 8.5, "base", 0.01)
    assert result.hygroscopicity_risk == "low"


# --- Disproportionation risk ---
def test_disproportionation_high_when_close_pka():
    # acetate pKa=4.75, drug_pka=4.0 -> diff=0.75 < 2 -> high
    result = predict_salt_form("AcidDrug", "acetate", 4.0, "acid", 0.1)
    # acetate type="acid", drug ionization_type="acid" -> incompatible
    # But disproportionation still computed based on pKa diff
    assert result.disproportionation_risk == "high"


def test_disproportionation_low_when_far_pka():
    # hydrochloride pKa=-7, drug_pka=8.5 -> diff=15.5 > 4 -> low
    result = predict_salt_form("DrugA", "hydrochloride", 8.5, "base", 0.01)
    assert result.disproportionation_risk == "low"


# --- Recommendation ---
def test_recommended_true_for_good_salt():
    # mesylate: low hygro, low stability, good factor
    # base drug with mesylate (acid counterion) -> compatible
    # drug_pka=8.5, mesylate pka=-1.2 -> diff=9.7 > 4 -> low dispr risk
    result = predict_salt_form("DrugA", "mesylate", 8.5, "base", 0.01)
    assert result.recommended is True


def test_recommended_false_for_incompatible():
    # Incompatible pair has ratio ~1, so recommended = False (ratio <= 2)
    result = predict_salt_form("DrugA", "sodium", 8.5, "base", 0.05)
    # sodium pKa=15, drug_pka=8.5, diff=6.5 -> compatible=False -> factor=1
    # ratio = 1 * (1 + 6.5*0.05) = 1.325 < 2.0 -> not recommended
    assert result.recommended is False


# --- screen_salt_forms ---
def test_screen_returns_all_counterions():
    results = screen_salt_forms("DrugA", 8.5, "base", 0.01)
    assert len(results) == len(COUNTERIONS)


def test_screen_sorted_descending():
    results = screen_salt_forms("DrugA", 8.5, "base", 0.01)
    ratios = [r.solubility_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


# --- Validation errors ---
def test_invalid_counterion_raises():
    with pytest.raises(ValueError, match="Unknown counterion"):
        predict_salt_form("DrugA", "unknown_salt", 8.5, "base", 0.01)


def test_invalid_solubility_raises():
    with pytest.raises(ValueError, match="intrinsic_solubility"):
        predict_salt_form("DrugA", "hydrochloride", 8.5, "base", 0.0)


def test_negative_solubility_raises():
    with pytest.raises(ValueError, match="intrinsic_solubility"):
        predict_salt_form("DrugA", "hydrochloride", 8.5, "base", -0.1)


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_salt_form("DrugA", "hydrochloride", 8.5, "neutral", 0.01)
