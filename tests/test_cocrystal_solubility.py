"""Tests for cocrystal solubility prediction (Phase 1012)."""

import pytest

from omega_pbpk.prediction.cocrystal_solubility import (
    CocrystalSolubilityResult,
    predict_cocrystal_solubility,
    screen_coformers,
)


def test_returns_result_type():
    result = predict_cocrystal_solubility("DrugA", "saccharin")
    assert isinstance(result, CocrystalSolubilityResult)


def test_drug_solubility_positive():
    result = predict_cocrystal_solubility("DrugA", "saccharin")
    assert result.drug_solubility_mg_mL > 0


def test_cocrystal_solubility_greater_than_drug():
    result = predict_cocrystal_solubility("DrugA", "nicotinamide")
    assert result.cocrystal_solubility_mg_mL > result.drug_solubility_mg_mL


def test_solubility_enhancement_ratio_greater_than_one():
    result = predict_cocrystal_solubility("DrugA", "citric_acid")
    assert result.solubility_enhancement_ratio > 1.0


def test_eutectic_point_fraction_in_range():
    result = predict_cocrystal_solubility("DrugA", "caffeine")
    assert 0 < result.eutectic_point_fraction < 1


def test_stability_constant_positive():
    result = predict_cocrystal_solubility("DrugA", "glutaric_acid")
    assert result.stability_constant > 0


def test_phase_solubility_type_valid():
    result = predict_cocrystal_solubility("DrugA", "saccharin")
    assert result.phase_solubility_type in {"type_A", "type_B"}


def test_supersaturation_risk_is_bool():
    result = predict_cocrystal_solubility("DrugA", "saccharin")
    assert isinstance(result.supersaturation_risk, bool)


def test_high_logp_lower_drug_solubility():
    low_logp = predict_cocrystal_solubility("DrugA", "saccharin", drug_logp=1.0)
    high_logp = predict_cocrystal_solubility("DrugA", "saccharin", drug_logp=5.0)
    assert high_logp.drug_solubility_mg_mL < low_logp.drug_solubility_mg_mL


def test_citric_acid_higher_enhancement_than_oxalic_acid():
    citric = predict_cocrystal_solubility("DrugA", "citric_acid")
    oxalic = predict_cocrystal_solubility("DrugA", "oxalic_acid")
    assert citric.solubility_enhancement_ratio > oxalic.solubility_enhancement_ratio


def test_screen_coformers_returns_all_when_no_list():
    results = screen_coformers("DrugA", drug_logp=2.0, drug_pka=7.0)
    assert len(results) == 8


def test_screen_coformers_sorted_descending():
    results = screen_coformers("DrugA", drug_logp=2.0, drug_pka=7.0)
    ratios = [r.solubility_enhancement_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_invalid_coformer_raises_value_error():
    with pytest.raises(ValueError, match="coformer"):
        predict_cocrystal_solubility("DrugA", "unknown_coformer")


def test_invalid_stoichiometry_raises_value_error():
    with pytest.raises(ValueError, match="stoichiometry"):
        predict_cocrystal_solubility("DrugA", "saccharin", stoichiometry="3:1")


def test_stoichiometry_1_2_lower_than_1_1():
    result_1_1 = predict_cocrystal_solubility("DrugA", "citric_acid", stoichiometry="1:1")
    result_1_2 = predict_cocrystal_solubility("DrugA", "citric_acid", stoichiometry="1:2")
    assert result_1_2.solubility_enhancement_ratio < result_1_1.solubility_enhancement_ratio


def test_stoichiometry_2_1_higher_than_1_1():
    result_1_1 = predict_cocrystal_solubility("DrugA", "citric_acid", stoichiometry="1:1")
    result_2_1 = predict_cocrystal_solubility("DrugA", "citric_acid", stoichiometry="2:1")
    assert result_2_1.solubility_enhancement_ratio > result_1_1.solubility_enhancement_ratio


def test_screen_coformers_subset():
    results = screen_coformers("DrugA", drug_logp=2.0, drug_pka=7.0, coformers=["saccharin", "citric_acid"])
    assert len(results) == 2


def test_drug_name_preserved():
    result = predict_cocrystal_solubility("MyDrug", "saccharin")
    assert result.drug_name == "MyDrug"


def test_coformer_name_preserved():
    result = predict_cocrystal_solubility("DrugA", "nicotinamide")
    assert result.coformer_name == "nicotinamide"


def test_stoichiometry_preserved():
    result = predict_cocrystal_solubility("DrugA", "saccharin", stoichiometry="1:2")
    assert result.stoichiometry == "1:2"


def test_citric_acid_type_a():
    # citric_acid: enhancement=8, Ksp=0.7 → enhancement>5 and ksp>0.4 → type_A
    result = predict_cocrystal_solubility("DrugA", "citric_acid", stoichiometry="1:1")
    assert result.phase_solubility_type == "type_A"


def test_oxalic_acid_type_b():
    # oxalic_acid: enhancement=2, Ksp=0.2 → not > 5 → type_B
    result = predict_cocrystal_solubility("DrugA", "oxalic_acid", stoichiometry="1:1")
    assert result.phase_solubility_type == "type_B"
