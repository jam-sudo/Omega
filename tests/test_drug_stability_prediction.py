"""Tests for Phase 972 — Drug chemical stability prediction."""

import pytest

from omega_pbpk.prediction.drug_stability_prediction import (
    DrugStabilityResult,
    predict_drug_stability,
    screen_stability,
)


def test_return_type_is_drug_stability_result():
    result = predict_drug_stability("DrugA", logp=1.0, mw=300.0)
    assert isinstance(result, DrugStabilityResult)


def test_frozen_dataclass():
    result = predict_drug_stability("DrugA", logp=1.0, mw=300.0)
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "other"  # type: ignore


def test_k_hyd_25_positive():
    result = predict_drug_stability("DrugA", logp=1.0, mw=300.0)
    assert result.k_hyd_25_per_h > 0


def test_k_hyd_40_greater_than_25():
    result = predict_drug_stability("DrugA", logp=1.0, mw=300.0)
    assert result.k_hyd_40_per_h > result.k_hyd_25_per_h


def test_shelf_life_25c_positive():
    result = predict_drug_stability("DrugA", logp=1.0, mw=300.0)
    assert result.shelf_life_months_25c > 0


def test_shelf_life_40c_shorter_than_25c():
    result = predict_drug_stability("DrugA", logp=1.0, mw=300.0)
    assert result.shelf_life_months_40c < result.shelf_life_months_25c


def test_oxidation_index_in_range():
    result = predict_drug_stability("DrugA", logp=1.0, mw=300.0)
    assert 0 <= result.oxidation_index <= 100


def test_light_sensitivity_valid_value():
    result = predict_drug_stability("DrugA", logp=1.0, mw=300.0)
    assert result.light_sensitivity in {"low", "moderate", "high"}


def test_stability_category_valid_value():
    result = predict_drug_stability("DrugA", logp=1.0, mw=300.0)
    assert result.stability_category in {"stable", "moderate", "unstable"}


def test_notes_non_empty():
    result = predict_drug_stability("DrugA", logp=1.0, mw=300.0)
    assert result.notes != ""


def test_drug_name_preserved():
    result = predict_drug_stability("Aspirin", logp=1.2, mw=180.0)
    assert result.drug_name == "Aspirin"


def test_has_ester_shorter_shelf_life():
    ester = predict_drug_stability("EsterDrug", logp=1.0, mw=300.0, has_ester=True)
    stable = predict_drug_stability("StableDrug", logp=1.0, mw=300.0)
    assert ester.shelf_life_months_25c < stable.shelf_life_months_25c


def test_catechol_high_oxidation_index():
    result = predict_drug_stability("Catechol", logp=1.0, mw=300.0, has_catechol=True)
    assert result.oxidation_index >= 40


def test_no_labile_groups_neutral_stable():
    result = predict_drug_stability(
        "StableMol",
        logp=1.0,
        mw=300.0,
        has_ester=False,
        has_lactam=False,
        has_amide_primary=False,
        has_catechol=False,
        has_thiol=False,
        has_aldehyde=False,
        ionization_type="neutral",
    )
    assert result.stability_category == "stable"


def test_ester_catechol_high_oxidation_and_short_shelf_life():
    result = predict_drug_stability(
        "Drug",
        logp=1.0,
        mw=300.0,
        has_ester=True,
        has_catechol=True,
    )
    assert result.oxidation_index >= 40
    assert result.shelf_life_months_25c < 24.0


def test_screen_stability_correct_length():
    compounds = [
        {"drug_name": "A", "logp": 1.0, "mw": 300.0},
        {"drug_name": "B", "logp": 2.0, "mw": 400.0, "has_ester": True},
        {"drug_name": "C", "logp": 0.5, "mw": 250.0, "has_lactam": True},
    ]
    results = screen_stability(compounds)
    assert len(results) == 3


def test_screen_stability_sorted_descending():
    compounds = [
        {"drug_name": "A", "logp": 1.0, "mw": 300.0, "has_ester": True},
        {"drug_name": "B", "logp": 1.0, "mw": 300.0},
        {"drug_name": "C", "logp": 1.0, "mw": 300.0, "has_lactam": True},
    ]
    results = screen_stability(compounds)
    shelf_lives = [r.shelf_life_months_25c for r in results]
    assert shelf_lives == sorted(shelf_lives, reverse=True)


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        predict_drug_stability("DrugA", logp=1.0, mw=0.0)


def test_validation_mw_negative():
    with pytest.raises(ValueError, match="mw"):
        predict_drug_stability("DrugA", logp=1.0, mw=-100.0)


def test_validation_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_drug_stability("DrugA", logp=1.0, mw=300.0, ionization_type="zwitterion")


def test_has_thiol_high_oxidation_index():
    result = predict_drug_stability("Thiol", logp=1.0, mw=300.0, has_thiol=True)
    assert result.oxidation_index >= 30


def test_stable_drug_shelf_life_over_24_months():
    result = predict_drug_stability(
        "StableDrug",
        logp=0.5,
        mw=200.0,
        has_ester=False,
        has_lactam=False,
        has_amide_primary=False,
        has_catechol=False,
        has_thiol=False,
        has_aldehyde=False,
        ionization_type="neutral",
    )
    assert result.shelf_life_months_25c > 24.0


def test_has_aldehyde_increases_oxidation_index():
    base = predict_drug_stability("Base", logp=1.0, mw=300.0)
    ald = predict_drug_stability("Ald", logp=1.0, mw=300.0, has_aldehyde=True)
    assert ald.oxidation_index > base.oxidation_index


def test_base_high_logp_increases_oxidation_index():
    neutral = predict_drug_stability("N", logp=3.0, mw=300.0, ionization_type="neutral")
    base = predict_drug_stability("B", logp=3.0, mw=300.0, ionization_type="base")
    assert base.oxidation_index > neutral.oxidation_index


def test_light_sensitivity_high_for_catechol():
    result = predict_drug_stability("C", logp=1.0, mw=300.0, has_catechol=True)
    assert result.light_sensitivity == "high"


def test_light_sensitivity_moderate_for_aromatic():
    result = predict_drug_stability("Ar", logp=2.0, mw=250.0)
    assert result.light_sensitivity == "moderate"


def test_light_sensitivity_low_for_small_hydrophilic():
    result = predict_drug_stability("Small", logp=0.5, mw=150.0)
    assert result.light_sensitivity == "low"
