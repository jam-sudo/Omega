"""Tests for prediction/solubility_enhancement.py (Phase 593)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.solubility_enhancement import (
    SolubilityEnhancementResult,
    predict_enhancement,
    rank_strategies,
)

STRATEGIES = [
    "amorphous_dispersion",
    "cyclodextrin",
    "salt_form",
    "nanosizing",
    "lipid_formulation",
    "cosolvent",
]


def _default(**kwargs):
    params = dict(
        drug_name="TestDrug",
        baseline_solubility_mg_mL=0.01,
        dose_mg=100.0,
        logP=3.0,
        pka=None,
        strategy="amorphous_dispersion",
    )
    params.update(kwargs)
    return predict_enhancement(**params)


def test_returns_result():
    result = _default()
    assert isinstance(result, SolubilityEnhancementResult)


def test_enhancement_ratio_above_1():
    for strategy in STRATEGIES:
        result = _default(strategy=strategy)
        assert result.enhancement_ratio > 1.0, f"{strategy} ratio not > 1"


def test_enhanced_solubility_above_baseline():
    for strategy in STRATEGIES:
        result = _default(strategy=strategy)
        assert result.enhanced_solubility_mg_mL > result.baseline_solubility_mg_mL


def test_feasibility_in_0_100():
    for strategy in STRATEGIES:
        result = _default(strategy=strategy)
        assert 0.0 <= result.feasibility_score <= 100.0


def test_dose_number_positive():
    for strategy in STRATEGIES:
        result = _default(strategy=strategy)
        assert result.dose_number_enhanced > 0


def test_amorphous_high_logP_high_ratio():
    result_high = _default(strategy="amorphous_dispersion", logP=5.0)
    result_low = _default(strategy="amorphous_dispersion", logP=1.0)
    assert result_high.enhancement_ratio > result_low.enhancement_ratio


def test_salt_form_with_ionizable_drug():
    result_with_pka = _default(strategy="salt_form", pka=3.0)
    result_no_pka = _default(strategy="salt_form", pka=None)
    assert result_with_pka.enhancement_ratio > result_no_pka.enhancement_ratio


def test_bcs_class_valid_values():
    valid = {"I", "II", "III", "IV"}
    for strategy in STRATEGIES:
        result = _default(strategy=strategy)
        assert result.bcs_class_enhanced in valid


def test_rank_strategies_returns_list():
    result = rank_strategies("Drug", 0.01, 100.0, 3.0)
    assert isinstance(result, list)


def test_rank_strategies_sorted_descending():
    results = rank_strategies("Drug", 0.01, 100.0, 3.0)
    scores = [r.feasibility_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rank_strategies_all_results_type():
    results = rank_strategies("Drug", 0.01, 100.0, 3.0)
    for r in results:
        assert isinstance(r, SolubilityEnhancementResult)


def test_rank_strategies_count():
    results = rank_strategies("Drug", 0.01, 100.0, 3.0)
    assert len(results) == 6


def test_invalid_solubility_raises():
    with pytest.raises(ValueError):
        predict_enhancement("Drug", 0.0, 100.0, 3.0)


def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        predict_enhancement("Drug", 0.01, 0.0, 3.0)


def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        predict_enhancement("Drug", 0.01, 100.0, 3.0, strategy="magic_powder")


def test_notes_nonempty():
    for strategy in STRATEGIES:
        result = _default(strategy=strategy)
        assert len(result.notes) > 0


def test_enhancement_factor_override():
    result = predict_enhancement(
        "Drug", 0.01, 100.0, 3.0, strategy="amorphous_dispersion", enhancement_factor=100.0
    )
    assert result.enhancement_ratio == 100.0


def test_nanosizing_universal():
    for logP in [-1.0, 0.0, 2.0, 5.0, 8.0]:
        result = _default(strategy="nanosizing", logP=logP)
        assert result.feasibility_score > 0


def test_lipid_formulation_high_logP():
    result_high = _default(strategy="lipid_formulation", logP=5.0)
    result_low = _default(strategy="lipid_formulation", logP=0.5)
    assert result_high.feasibility_score > result_low.feasibility_score


def test_bcs_upgrade():
    # baseline BCS II: dose_number > 1 with high Peff (logP in 0-5)
    # solubility = 0.01 mg/mL, dose = 100 mg -> Do_baseline = 100/(250*0.01) = 40 (BCS II)
    baseline = predict_enhancement(
        "Drug",
        0.01,
        100.0,
        logP=3.0,
        strategy="amorphous_dispersion",
        enhancement_factor=1.0,
    )
    assert baseline.bcs_class_enhanced == "II"

    # With 400x enhancement: solubility = 4 -> Do = 100/(250*4) = 0.1 (BCS I)
    enhanced = predict_enhancement(
        "Drug",
        0.01,
        100.0,
        logP=3.0,
        strategy="amorphous_dispersion",
        enhancement_factor=400.0,
    )
    assert enhanced.bcs_class_enhanced == "I"


def test_cyclodextrin_moderate_logP():
    result = _default(strategy="cyclodextrin", logP=3.0)
    assert result.feasibility_score > 50.0


def test_cosolvent_low_logP():
    result = _default(strategy="cosolvent", logP=0.5)
    assert result.feasibility_score > 50.0
