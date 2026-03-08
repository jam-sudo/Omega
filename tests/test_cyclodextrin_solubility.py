"""Tests for Phase 583 - Drug Solubility Enhancement by Cyclodextrins."""

import pytest

from omega_pbpk.prediction.cyclodextrin_solubility import (
    CD_MW,
    CyclodextrinSolubilityResult,
    compare_cyclodextrins,
    predict_cd_solubility,
)


@pytest.fixture
def base_result():
    return predict_cd_solubility(
        drug_name="TestDrug",
        drug_logP=3.0,
        intrinsic_solubility_mg_mL=0.01,
        mw_drug_Da=400.0,
        cyclodextrin_type="HP_beta_CD",
        cyclodextrin_conc_mM=10.0,
    )


def test_return_type(base_result):
    assert isinstance(base_result, CyclodextrinSolubilityResult)


def test_all_fields_present(base_result):
    fields = [
        "drug_name",
        "cyclodextrin_type",
        "drug_logP",
        "intrinsic_solubility_mg_mL",
        "cyclodextrin_conc_mM",
        "apparent_solubility_mg_mL",
        "solubility_enhancement_ratio",
        "complexation_efficiency_pct",
        "ka_complex_per_mM",
        "drug_loading_pct",
        "formulation_feasibility",
        "notes",
    ]
    for f in fields:
        assert hasattr(base_result, f)


def test_apparent_greater_than_intrinsic(base_result):
    assert base_result.apparent_solubility_mg_mL > base_result.intrinsic_solubility_mg_mL


def test_se_ratio_calculation(base_result):
    expected = base_result.apparent_solubility_mg_mL / base_result.intrinsic_solubility_mg_mL
    assert abs(base_result.solubility_enhancement_ratio - expected) < 1e-9


def test_ka_hp_beta_cd():
    r = predict_cd_solubility("D", 3.0, 0.01, 400.0, "HP_beta_CD", 10.0)
    expected_ka = 500.0 * (1.0 + 0.5 * max(0, 3.0 - 1.0))
    assert abs(r.ka_complex_per_mM - expected_ka) < 1e-9


def test_ka_sbe_beta_cd():
    r = predict_cd_solubility("D", 3.0, 0.01, 400.0, "SBE_beta_CD", 10.0)
    expected_ka = 400.0 * (1.0 + 0.4 * max(0, 3.0 - 1.0))
    assert abs(r.ka_complex_per_mM - expected_ka) < 1e-9


def test_ka_alpha_cd():
    r = predict_cd_solubility("D", 3.0, 0.01, 400.0, "alpha_CD", 10.0)
    expected_ka = 200.0 * (1.0 + 0.2 * max(0, 3.0 - 1.0))
    assert abs(r.ka_complex_per_mM - expected_ka) < 1e-9


def test_higher_logp_higher_se():
    r_low = predict_cd_solubility("D", 1.0, 0.01, 400.0, "HP_beta_CD", 10.0)
    r_high = predict_cd_solubility("D", 4.0, 0.01, 400.0, "HP_beta_CD", 10.0)
    assert r_high.solubility_enhancement_ratio > r_low.solubility_enhancement_ratio


def test_hp_beta_better_than_alpha_for_lipophilic():
    r_hp = predict_cd_solubility("D", 4.0, 0.01, 400.0, "HP_beta_CD", 10.0)
    r_alpha = predict_cd_solubility("D", 4.0, 0.01, 400.0, "alpha_CD", 10.0)
    assert r_hp.solubility_enhancement_ratio > r_alpha.solubility_enhancement_ratio


def test_drug_loading_in_range(base_result):
    assert 0 <= base_result.drug_loading_pct <= 50


def test_drug_loading_calculation():
    r = predict_cd_solubility("D", 3.0, 0.01, 400.0, "HP_beta_CD", 10.0)
    delta_s = r.apparent_solubility_mg_mL - r.intrinsic_solubility_mg_mL
    cd_mass = r.cyclodextrin_conc_mM * CD_MW["HP_beta_CD"] / 1000.0
    expected = min(50.0, delta_s / (delta_s + cd_mass) * 100.0)
    assert abs(r.drug_loading_pct - expected) < 1e-9


def test_feasibility_excellent():
    # logP=4, HP_beta_CD → Ka=1250, SE=1+1250*10=12501 → excellent
    r = predict_cd_solubility("D", 4.0, 0.01, 400.0, "HP_beta_CD", 10.0)
    assert r.formulation_feasibility == "excellent"


def test_feasibility_poor():
    # Very low conc CD + low logP → small SE
    r = predict_cd_solubility("D", 0.5, 1.0, 400.0, "alpha_CD", 0.001)
    assert r.formulation_feasibility == "poor"


def test_feasibility_moderate():
    # Need SE in [2, 5): Ka*[CD] in [1, 4)
    # alpha_CD logP=0.5: Ka=200, need [CD] s.t. Ka*[CD] in [1, 4)
    # [CD] = 0.01 → Ka*[CD] = 2.0 → SE = 3.0
    r = predict_cd_solubility("D", 0.5, 0.01, 400.0, "alpha_CD", 0.01)
    assert r.formulation_feasibility == "moderate"


def test_feasibility_good():
    # Need SE in [5, 10]: Ka*[CD] in [4, 9)
    # alpha_CD logP=0.5: Ka=200, [CD]=0.03 → SE=1+6=7
    r = predict_cd_solubility("D", 0.5, 0.01, 400.0, "alpha_CD", 0.03)
    assert r.formulation_feasibility == "good"


def test_compare_returns_three():
    results = compare_cyclodextrins("D", 3.0, 0.01, 400.0)
    assert len(results) == 3


def test_compare_sorted_descending():
    results = compare_cyclodextrins("D", 3.0, 0.01, 400.0)
    se_ratios = [r.solubility_enhancement_ratio for r in results]
    assert se_ratios == sorted(se_ratios, reverse=True)


def test_compare_hp_beta_first_for_lipophilic():
    results = compare_cyclodextrins("D", 4.0, 0.01, 400.0)
    assert results[0].cyclodextrin_type == "HP_beta_CD"


def test_validation_solubility_zero():
    with pytest.raises(ValueError):
        predict_cd_solubility("D", 2.0, 0.0, 400.0, "HP_beta_CD")


def test_validation_solubility_negative():
    with pytest.raises(ValueError):
        predict_cd_solubility("D", 2.0, -1.0, 400.0, "HP_beta_CD")


def test_validation_conc_negative():
    with pytest.raises(ValueError):
        predict_cd_solubility("D", 2.0, 0.01, 400.0, "HP_beta_CD", -5.0)


def test_validation_mw_zero():
    with pytest.raises(ValueError):
        predict_cd_solubility("D", 2.0, 0.01, 0.0, "HP_beta_CD")


def test_validation_invalid_cd_type():
    with pytest.raises(ValueError):
        predict_cd_solubility("D", 2.0, 0.01, 400.0, "gamma_CD")


def test_logp_below_one():
    """LogP < 1 should use max(0, logP-1)=0, so Ka = base only."""
    r = predict_cd_solubility("D", 0.0, 0.01, 400.0, "HP_beta_CD", 10.0)
    assert abs(r.ka_complex_per_mM - 500.0) < 1e-9


def test_frozen_result(base_result):
    with pytest.raises(AttributeError):
        base_result.drug_name = "other"
