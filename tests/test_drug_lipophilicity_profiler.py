"""Tests for drug lipophilicity profiler module (Phase 702)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.drug_lipophilicity_profiler import (
    LipophilicityProfile,
    compare_lipophilicity_optimization,
    profile_lipophilicity,
    screen_lipophilicity,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_lipophilicity_profile():
    result = profile_lipophilicity("TestDrug", logP=2.0)
    assert isinstance(result, LipophilicityProfile)


def test_result_fields_present():
    result = profile_lipophilicity("TestDrug", logP=2.0, mw=350.0, psa=70.0)
    assert result.compound_name == "TestDrug"
    assert result.logP == 2.0
    assert result.mw == 350.0
    assert result.psa == 70.0


def test_notes_nonempty():
    result = profile_lipophilicity("TestDrug", logP=2.0)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# logD calculations
# ---------------------------------------------------------------------------


def test_logD_74_equals_logP_for_neutral():
    result = profile_lipophilicity("Neutral", logP=3.0)
    assert abs(result.logD_74 - 3.0) < 1e-9


def test_logD_68_equals_logP_for_neutral():
    result = profile_lipophilicity("Neutral", logP=3.0)
    assert abs(result.logD_68 - 3.0) < 1e-9


def test_logD_50_equals_logP_for_neutral():
    result = profile_lipophilicity("Neutral", logP=3.0)
    assert abs(result.logD_50 - 3.0) < 1e-9


def test_logD_74_less_than_logP_for_basic_drug():
    # Basic drug ionizes at pH 7.4 → logD < logP
    result = profile_lipophilicity("Base", logP=3.0, pka_base=9.0)
    assert result.logD_74 < result.logP


def test_logD_50_less_than_logD_74_for_basic_drug():
    # At lower pH, base is more ionized → logD_50 < logD_74
    result = profile_lipophilicity("Base", logP=3.0, pka_base=9.0)
    assert result.logD_50 < result.logD_74


def test_logD_74_less_than_logP_for_acidic_drug():
    # Acid deprotonated at pH 7.4 → logD < logP
    result = profile_lipophilicity("Acid", logP=2.0, pka_acid=4.0)
    assert result.logD_74 < result.logP


# ---------------------------------------------------------------------------
# logP classification
# ---------------------------------------------------------------------------


def test_logP_class_hydrophilic_when_negative():
    result = profile_lipophilicity("Drug", logP=-1.0)
    assert result.logP_class == "hydrophilic"


def test_logP_class_moderate_for_0_to_3():
    result = profile_lipophilicity("Drug", logP=2.0)
    assert result.logP_class == "moderate"


def test_logP_class_lipophilic_for_3_to_5():
    result = profile_lipophilicity("Drug", logP=4.0)
    assert result.logP_class == "lipophilic"


def test_logP_class_highly_lipophilic_above_5():
    result = profile_lipophilicity("Drug", logP=6.0)
    assert result.logP_class == "highly_lipophilic"


# ---------------------------------------------------------------------------
# Lipinski and oral optimal
# ---------------------------------------------------------------------------


def test_lipinski_pass_true_when_logP_le_5():
    result = profile_lipophilicity("Drug", logP=4.9)
    assert result.lipinski_pass is True


def test_lipinski_pass_false_when_logP_above_5():
    result = profile_lipophilicity("Drug", logP=5.5)
    assert result.lipinski_pass is False


def test_oral_optimal_true_when_logP_2_mw_300():
    result = profile_lipophilicity("Drug", logP=2.0, mw=300.0, psa=80.0)
    assert result.oral_optimal is True


def test_oral_optimal_false_when_mw_too_high():
    result = profile_lipophilicity("Drug", logP=2.0, mw=600.0, psa=80.0)
    assert result.oral_optimal is False


def test_oral_optimal_false_when_logP_too_high():
    result = profile_lipophilicity("Drug", logP=6.0, mw=300.0, psa=80.0)
    assert result.oral_optimal is False


# ---------------------------------------------------------------------------
# CNS penetration score
# ---------------------------------------------------------------------------


def test_cns_score_1_when_logP_2_psa_40():
    result = profile_lipophilicity("CNS", logP=2.0, psa=40.0)
    assert result.cns_penetration_score == 1.0


def test_cns_score_0_when_psa_150():
    result = profile_lipophilicity("NoPermeation", logP=2.0, psa=150.0)
    assert result.cns_penetration_score == 0.0


def test_cns_score_05_intermediate():
    # logP=2 in 0-5, psa=100 in 90-120 range
    result = profile_lipophilicity("Partial", logP=2.0, psa=100.0)
    assert result.cns_penetration_score == 0.5


# ---------------------------------------------------------------------------
# Adipose Kp and Papp
# ---------------------------------------------------------------------------


def test_adipose_kp_above_1_for_logP_3():
    result = profile_lipophilicity("Drug", logP=3.0)
    assert result.adipose_kp_estimate > 1.0


def test_adipose_kp_positive():
    result = profile_lipophilicity("Drug", logP=-2.0)
    assert result.adipose_kp_estimate > 0


def test_papp_caco2_positive():
    result = profile_lipophilicity("Drug", logP=2.0)
    assert result.papp_caco2_estimate > 0


# ---------------------------------------------------------------------------
# screen_lipophilicity
# ---------------------------------------------------------------------------


def test_screen_lipophilicity_sorted_descending_logD74():
    compounds = [
        {"compound_name": "A", "logP": 1.0},
        {"compound_name": "B", "logP": 4.0},
        {"compound_name": "C", "logP": 2.5},
    ]
    results = screen_lipophilicity(compounds)
    logD_vals = [r.logD_74 for r in results]
    assert logD_vals == sorted(logD_vals, reverse=True)


def test_screen_lipophilicity_returns_list():
    compounds = [
        {"compound_name": "A", "logP": 1.0},
        {"compound_name": "B", "logP": 3.0},
    ]
    results = screen_lipophilicity(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# compare_lipophilicity_optimization
# ---------------------------------------------------------------------------


def test_compare_returns_correct_length():
    variants = [1.0, 2.0, 3.0, 4.5]
    results = compare_lipophilicity_optimization("Lead", variants)
    assert len(results) == len(variants)


def test_compare_sorted_by_closeness_to_2_5():
    variants = [0.0, 2.5, 5.0]
    results = compare_lipophilicity_optimization("Lead", variants)
    # 2.5 should be first (distance 0)
    assert abs(results[0].logP - 2.5) < 1e-9


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_mw_zero_raises():
    with pytest.raises(ValueError):
        profile_lipophilicity("Drug", logP=2.0, mw=0.0)


def test_validation_mw_negative_raises():
    with pytest.raises(ValueError):
        profile_lipophilicity("Drug", logP=2.0, mw=-100.0)


def test_validation_psa_negative_raises():
    with pytest.raises(ValueError):
        profile_lipophilicity("Drug", logP=2.0, psa=-10.0)
