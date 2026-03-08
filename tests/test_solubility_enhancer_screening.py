"""Tests for Phase 1140: Solubility enhancer screening."""

import pytest

from omega_pbpk.prediction.solubility_enhancer_screening import (
    SolubilityEnhancerResult,
    find_optimal_enhancer,
    screen_solubility_enhancers,
)

# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    results = screen_solubility_enhancers("Drug", 0.01, 3.0, "neutral")
    assert isinstance(results, list)


def test_screen_returns_eight_results():
    results = screen_solubility_enhancers("Drug", 0.01, 3.0, "neutral")
    assert len(results) == 8


def test_all_results_are_correct_type():
    results = screen_solubility_enhancers("Drug", 0.01, 3.0, "neutral")
    for r in results:
        assert isinstance(r, SolubilityEnhancerResult)


def test_drug_name_preserved_in_results():
    results = screen_solubility_enhancers("Ibuprofen", 0.01, 3.0, "acid")
    for r in results:
        assert r.drug_name == "Ibuprofen"


def test_intrinsic_solubility_preserved():
    results = screen_solubility_enhancers("Drug", 0.05, 3.0, "neutral")
    for r in results:
        assert r.intrinsic_solubility_mg_mL == 0.05


def test_logP_preserved():
    results = screen_solubility_enhancers("Drug", 0.01, 2.5, "neutral")
    for r in results:
        assert r.logP == 2.5


def test_ionization_type_preserved():
    results = screen_solubility_enhancers("Drug", 0.01, 3.0, "base")
    for r in results:
        assert r.ionization_type == "base"


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


def test_results_sorted_descending_by_enhanced_solubility():
    results = screen_solubility_enhancers("Drug", 0.01, 3.0, "acid")
    for i in range(len(results) - 1):
        assert results[i].enhanced_solubility_mg_mL >= results[i + 1].enhanced_solubility_mg_mL


def test_results_sorted_descending_neutral():
    results = screen_solubility_enhancers("Drug", 0.01, 4.0, "neutral")
    for i in range(len(results) - 1):
        assert results[i].enhanced_solubility_mg_mL >= results[i + 1].enhanced_solubility_mg_mL


def test_results_sorted_descending_base():
    results = screen_solubility_enhancers("Drug", 0.02, 2.0, "base")
    for i in range(len(results) - 1):
        assert results[i].enhanced_solubility_mg_mL >= results[i + 1].enhanced_solubility_mg_mL


# ---------------------------------------------------------------------------
# Ionization-specific enhancers
# ---------------------------------------------------------------------------


def test_acid_drug_NaOH_pH_has_high_fold():
    results = screen_solubility_enhancers("AcidDrug", 0.01, 3.0, "acid")
    naoh = next(r for r in results if r.enhancer_name == "NaOH_pH")
    assert naoh.fold_enhancement == 100.0


def test_acid_drug_meglumine_salt_has_high_fold():
    results = screen_solubility_enhancers("AcidDrug", 0.01, 3.0, "acid")
    meg = next(r for r in results if r.enhancer_name == "meglumine_salt")
    assert meg.fold_enhancement == 200.0


def test_base_drug_HCl_pH_has_high_fold():
    results = screen_solubility_enhancers("BaseDrug", 0.01, 3.0, "base")
    hcl = next(r for r in results if r.enhancer_name == "HCl_pH")
    assert hcl.fold_enhancement == 100.0


def test_base_drug_HCl_salt_has_high_fold():
    results = screen_solubility_enhancers("BaseDrug", 0.01, 3.0, "base")
    hcl_salt = next(r for r in results if r.enhancer_name == "HCl_salt")
    assert hcl_salt.fold_enhancement == 150.0


def test_neutral_drug_NaOH_pH_gives_fold_one():
    results = screen_solubility_enhancers("NeutralDrug", 0.01, 3.0, "neutral")
    naoh = next(r for r in results if r.enhancer_name == "NaOH_pH")
    assert naoh.fold_enhancement == 1.0


def test_neutral_drug_HCl_pH_gives_fold_one():
    results = screen_solubility_enhancers("NeutralDrug", 0.01, 3.0, "neutral")
    hcl = next(r for r in results if r.enhancer_name == "HCl_pH")
    assert hcl.fold_enhancement == 1.0


def test_acid_drug_HCl_pH_gives_fold_one():
    """NaOH is for acids; HCl_pH is for bases — acid drug should get fold=1."""
    results = screen_solubility_enhancers("AcidDrug", 0.01, 3.0, "acid")
    hcl = next(r for r in results if r.enhancer_name == "HCl_pH")
    assert hcl.fold_enhancement == 1.0


# ---------------------------------------------------------------------------
# logP effects
# ---------------------------------------------------------------------------


def test_higher_logP_higher_fold_for_cosolvent():
    r_low = screen_solubility_enhancers("Drug", 0.01, 1.0, "neutral")
    r_high = screen_solubility_enhancers("Drug", 0.01, 5.0, "neutral")
    peg_low = next(r for r in r_low if r.enhancer_name == "PEG400")
    peg_high = next(r for r in r_high if r.enhancer_name == "PEG400")
    assert peg_high.fold_enhancement > peg_low.fold_enhancement


def test_higher_logP_higher_fold_for_cyclodextrin():
    r_low = screen_solubility_enhancers("Drug", 0.01, 1.0, "neutral")
    r_high = screen_solubility_enhancers("Drug", 0.01, 5.0, "neutral")
    hpbcd_low = next(r for r in r_low if r.enhancer_name == "HPBCD")
    hpbcd_high = next(r for r in r_high if r.enhancer_name == "HPBCD")
    assert hpbcd_high.fold_enhancement > hpbcd_low.fold_enhancement


def test_fold_capped_at_max_fold():
    """Very high logP should not exceed max_fold for cosolvents."""
    results = screen_solubility_enhancers("Drug", 0.01, 10.0, "neutral")
    peg = next(r for r in results if r.enhancer_name == "PEG400")
    assert peg.fold_enhancement <= 10.0


# ---------------------------------------------------------------------------
# Enhanced solubility calculation
# ---------------------------------------------------------------------------


def test_enhanced_solubility_equals_intrinsic_times_fold():
    results = screen_solubility_enhancers("Drug", 0.05, 3.0, "neutral")
    for r in results:
        expected = r.intrinsic_solubility_mg_mL * r.fold_enhancement
        assert abs(r.enhanced_solubility_mg_mL - expected) < 1e-10


def test_enhanced_solubility_always_ge_intrinsic():
    """Fold should always be >= 1, so enhanced >= intrinsic."""
    results = screen_solubility_enhancers("Drug", 0.01, -2.0, "neutral")
    for r in results:
        assert r.enhanced_solubility_mg_mL >= r.intrinsic_solubility_mg_mL


# ---------------------------------------------------------------------------
# Regulatory status
# ---------------------------------------------------------------------------


def test_cyclodextrin_regulatory_status():
    results = screen_solubility_enhancers("Drug", 0.01, 3.0, "neutral")
    hpbcd = next(r for r in results if r.enhancer_name == "HPBCD")
    assert hpbcd.regulatory_status == "established"


def test_cosolvent_regulatory_status():
    results = screen_solubility_enhancers("Drug", 0.01, 3.0, "neutral")
    peg = next(r for r in results if r.enhancer_name == "PEG400")
    assert peg.regulatory_status == "established"


def test_surfactant_regulatory_status():
    results = screen_solubility_enhancers("Drug", 0.01, 3.0, "neutral")
    ps80 = next(r for r in results if r.enhancer_name == "polysorbate80")
    assert ps80.regulatory_status == "established"


def test_salt_regulatory_status():
    results = screen_solubility_enhancers("Drug", 0.01, 3.0, "acid")
    meg = next(r for r in results if r.enhancer_name == "meglumine_salt")
    assert meg.regulatory_status == "generally_recognized"


def test_pH_adjustment_regulatory_status():
    results = screen_solubility_enhancers("Drug", 0.01, 3.0, "acid")
    naoh = next(r for r in results if r.enhancer_name == "NaOH_pH")
    assert naoh.regulatory_status == "generally_recognized"


# ---------------------------------------------------------------------------
# find_optimal_enhancer
# ---------------------------------------------------------------------------


def test_find_optimal_returns_single_result():
    result = find_optimal_enhancer("Drug", 0.01, 3.0, "acid", target_solubility_mg_mL=1.0)
    assert isinstance(result, SolubilityEnhancerResult)


def test_find_optimal_meets_target_when_possible():
    # With acid drug and very low target, meglumine_salt should easily meet it
    result = find_optimal_enhancer("Drug", 0.01, 3.0, "acid", target_solubility_mg_mL=0.5)
    assert result.target_met is True


def test_find_optimal_returns_best_when_target_not_met():
    # Very high target that no enhancer can meet
    result = find_optimal_enhancer("Drug", 0.001, 1.0, "neutral", target_solubility_mg_mL=1000.0)
    assert result.target_met is False
    # Should be the highest enhanced solubility available
    all_results = screen_solubility_enhancers("Drug", 0.001, 1.0, "neutral")
    assert result.enhanced_solubility_mg_mL == all_results[0].enhanced_solubility_mg_mL


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_zero_solubility_raises():
    with pytest.raises(ValueError, match="intrinsic_solubility"):
        screen_solubility_enhancers("Drug", 0.0, 2.0, "neutral")


def test_negative_solubility_raises():
    with pytest.raises(ValueError, match="intrinsic_solubility"):
        screen_solubility_enhancers("Drug", -0.1, 2.0, "neutral")


def test_logP_too_low_raises():
    with pytest.raises(ValueError, match="logP"):
        screen_solubility_enhancers("Drug", 0.01, -6.0, "neutral")


def test_logP_too_high_raises():
    with pytest.raises(ValueError, match="logP"):
        screen_solubility_enhancers("Drug", 0.01, 11.0, "neutral")


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        screen_solubility_enhancers("Drug", 0.01, 2.0, "zwitterion")


def test_find_optimal_invalid_ionization_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        find_optimal_enhancer("Drug", 0.01, 2.0, "cation", target_solubility_mg_mL=1.0)
