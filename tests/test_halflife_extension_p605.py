"""
Tests for Phase 605 — Drug Half-Life Extension Strategies
"""

import pytest

from omega_pbpk.clinical.halflife_extension_p605 import (
    VALID_STRATEGIES,
    HalflifeExtensionResult,
    compare_extension_strategies,
    predict_halflife_extension,
)

# ---------------------------------------------------------------------------
# Helper constants
# ---------------------------------------------------------------------------
DRUG = "TestDrug"
T_HALF = 2.0  # hours
MW = 10.0  # kDa


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = predict_halflife_extension(DRUG, T_HALF, MW, "none")
    assert isinstance(result, HalflifeExtensionResult)


def test_result_is_frozen():
    result = predict_halflife_extension(DRUG, T_HALF, MW, "none")
    with pytest.raises((AttributeError, TypeError)):
        result.t_half_extended_h = 999.0  # type: ignore[misc]


def test_fields_populated():
    result = predict_halflife_extension(DRUG, T_HALF, MW, "none")
    assert result.drug_name == DRUG
    assert result.strategy == "none"
    assert result.mw_original_kDa == MW
    assert result.t_half_original_h == T_HALF


# ---------------------------------------------------------------------------
# Strategy: none
# ---------------------------------------------------------------------------


def test_none_strategy_no_change():
    result = predict_halflife_extension(DRUG, T_HALF, MW, "none")
    assert result.t_half_extended_h == pytest.approx(T_HALF)
    assert result.extension_fold == pytest.approx(1.0)
    assert result.cl_reduction_factor == pytest.approx(1.0)
    assert result.vd_change_factor == pytest.approx(1.0)
    assert result.dosing_frequency_reduction == "same"
    assert result.manufacturing_complexity == "low"
    assert result.immunogenicity_risk_change == "none"


# ---------------------------------------------------------------------------
# Strategy: pegylation
# ---------------------------------------------------------------------------


def test_pegylation_cl_reduction():
    peg_kDa = 20.0
    expected_cl = 1.0 / (1.0 + peg_kDa / MW * 2.0)
    result = predict_halflife_extension(DRUG, T_HALF, MW, "pegylation", pegylation_kDa=peg_kDa)
    assert result.cl_reduction_factor == pytest.approx(expected_cl)


def test_pegylation_vd_change():
    peg_kDa = 20.0
    expected_vd = 1.0 + peg_kDa / MW * 0.5
    result = predict_halflife_extension(DRUG, T_HALF, MW, "pegylation", pegylation_kDa=peg_kDa)
    assert result.vd_change_factor == pytest.approx(expected_vd)


def test_pegylation_t_half_increased():
    result = predict_halflife_extension(DRUG, T_HALF, MW, "pegylation")
    assert result.t_half_extended_h > T_HALF


def test_pegylation_immunogenicity_reduced():
    result = predict_halflife_extension(DRUG, T_HALF, MW, "pegylation")
    assert result.immunogenicity_risk_change == "reduced"


def test_pegylation_manufacturing_medium():
    result = predict_halflife_extension(DRUG, T_HALF, MW, "pegylation")
    assert result.manufacturing_complexity == "medium"


# ---------------------------------------------------------------------------
# Strategy: albumin_fusion
# ---------------------------------------------------------------------------


def test_albumin_fusion_cl_factor():
    result = predict_halflife_extension(DRUG, T_HALF, MW, "albumin_fusion")
    assert result.cl_reduction_factor == pytest.approx(0.05)


def test_albumin_fusion_vd_factor():
    result = predict_halflife_extension(DRUG, T_HALF, MW, "albumin_fusion")
    assert result.vd_change_factor == pytest.approx(0.1)


def test_albumin_fusion_cap_720h():
    # A very long t_half should be capped at 720 h
    result = predict_halflife_extension(DRUG, 1000.0, MW, "albumin_fusion")
    assert result.t_half_extended_h <= 720.0


def test_albumin_fusion_short_drug_below_cap():
    # t_half * 0.1 / 0.05 = t_half * 2.0; for T_HALF=2, gives 4h — below cap
    result = predict_halflife_extension(DRUG, T_HALF, MW, "albumin_fusion")
    assert result.t_half_extended_h == pytest.approx(T_HALF * 0.1 / 0.05)


def test_albumin_fusion_immunogenicity_variable():
    result = predict_halflife_extension(DRUG, T_HALF, MW, "albumin_fusion")
    assert result.immunogenicity_risk_change == "variable"


def test_albumin_fusion_manufacturing_high():
    result = predict_halflife_extension(DRUG, T_HALF, MW, "albumin_fusion")
    assert result.manufacturing_complexity == "high"


# ---------------------------------------------------------------------------
# Strategy: nano_encapsulation
# ---------------------------------------------------------------------------


def test_nano_encapsulation_cl_factor():
    result = predict_halflife_extension(DRUG, T_HALF, MW, "nano_encapsulation")
    assert result.cl_reduction_factor == pytest.approx(0.3)


def test_nano_encapsulation_vd_factor():
    result = predict_halflife_extension(DRUG, T_HALF, MW, "nano_encapsulation")
    assert result.vd_change_factor == pytest.approx(2.0)


def test_nano_encapsulation_t_half_extended():
    result = predict_halflife_extension(DRUG, T_HALF, MW, "nano_encapsulation")
    expected = T_HALF * 2.0 / 0.3
    assert result.t_half_extended_h == pytest.approx(expected)


def test_nano_encapsulation_immunogenicity_increased():
    result = predict_halflife_extension(DRUG, T_HALF, MW, "nano_encapsulation")
    assert result.immunogenicity_risk_change == "increased"


# ---------------------------------------------------------------------------
# Strategy: sustained_release
# ---------------------------------------------------------------------------


def test_sustained_release_t_half_5x():
    result = predict_halflife_extension(DRUG, T_HALF, MW, "sustained_release")
    assert result.t_half_extended_h == pytest.approx(T_HALF * 5.0)


def test_sustained_release_cl_unchanged():
    result = predict_halflife_extension(DRUG, T_HALF, MW, "sustained_release")
    assert result.cl_reduction_factor == pytest.approx(1.0)


def test_sustained_release_immunogenicity_none():
    result = predict_halflife_extension(DRUG, T_HALF, MW, "sustained_release")
    assert result.immunogenicity_risk_change == "none"


def test_sustained_release_manufacturing_medium():
    result = predict_halflife_extension(DRUG, T_HALF, MW, "sustained_release")
    assert result.manufacturing_complexity == "medium"


# ---------------------------------------------------------------------------
# Dosing frequency reduction
# ---------------------------------------------------------------------------


def test_dosing_freq_same_when_fold_lt2():
    # none strategy => fold=1.0 => "same"
    result = predict_halflife_extension(DRUG, T_HALF, MW, "none")
    assert result.dosing_frequency_reduction == "same"


def test_dosing_freq_weekly():
    # Use a large t_half to push fold into weekly range (12–84)
    # nano_encapsulation: fold = 2/0.3 ≈ 6.67 -> "4x_less_frequent"
    # sustained_release: fold = 5.0 -> "4x_less_frequent"
    # Let's pick a t_half that gives fold in [12,84) with nano: fold=6.67x always
    # For weekly we need fold in [12, 84):
    # albumin_fusion for short drug: fold = 0.1/0.05 = 2.0 -> "2x_less_frequent"
    # We need fold >= 12 for weekly. sustained_release gives 5x -> "4x_less_frequent"
    # Use large drug with sustained_release won't help.
    # Let's use albumin_fusion with a drug that gives fold = 2*t_half/t_half after cap check
    # albumin: t_new = t * 0.1/0.05 = 2*t; fold=2.0 -> "2x_less_frequent"
    # We need fold >=12: pegylation with very heavy PEG vs small drug
    # e.g. mw=1, peg=20: cl=1/(1+20*2)=1/41≈0.0244, vd=1+20*0.5=11, t_new=T*11/0.0244=T*451 -> monthly
    result = predict_halflife_extension(DRUG, 1.0, 1.0, "pegylation", pegylation_kDa=5.0)
    # mw=1, peg=5: cl=1/(1+10)=1/11≈0.0909, vd=1+2.5=3.5, fold=3.5/0.0909≈38.5 -> weekly
    assert result.dosing_frequency_reduction == "weekly"


# ---------------------------------------------------------------------------
# Comparison function
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    results = compare_extension_strategies(DRUG, T_HALF, MW)
    assert isinstance(results, list)


def test_compare_returns_all_strategies():
    results = compare_extension_strategies(DRUG, T_HALF, MW)
    assert len(results) == len(VALID_STRATEGIES)


def test_compare_sorted_descending():
    results = compare_extension_strategies(DRUG, T_HALF, MW)
    t_halves = [r.t_half_extended_h for r in results]
    assert t_halves == sorted(t_halves, reverse=True)


def test_compare_all_have_same_drug_name():
    results = compare_extension_strategies(DRUG, T_HALF, MW)
    assert all(r.drug_name == DRUG for r in results)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_strategy_raises():
    with pytest.raises(ValueError, match="strategy"):
        predict_halflife_extension(DRUG, T_HALF, MW, "invalid_strategy")


def test_negative_t_half_raises():
    with pytest.raises(ValueError, match="t_half_original_h"):
        predict_halflife_extension(DRUG, -1.0, MW, "none")


def test_zero_mw_raises():
    with pytest.raises(ValueError, match="mw_kDa"):
        predict_halflife_extension(DRUG, T_HALF, 0.0, "none")


def test_zero_pegylation_kDa_raises():
    with pytest.raises(ValueError, match="pegylation_kDa"):
        predict_halflife_extension(DRUG, T_HALF, MW, "pegylation", pegylation_kDa=0.0)
