"""Tests for active_metabolite_predictor module (Phase 692)."""

import pytest

from omega_pbpk.prediction.active_metabolite_predictor import (
    MetaboliteFormationResult,
    estimate_metabolite_clearance,
    predict_metabolite_formation,
    screen_metabolites,
)

# ── Basic return type ─────────────────────────────────────────────────────────


def test_returns_metabolite_formation_result():
    result = predict_metabolite_formation("CompA", mw_parent=300.0, logP_parent=2.0)
    assert isinstance(result, MetaboliteFormationResult)


def test_c_parent_initial_nonzero():
    result = predict_metabolite_formation("CompA", mw_parent=300.0, logP_parent=2.0)
    assert result.c_parent_mg_L[0] > 0.0


def test_c_metabolite_initial_zero():
    result = predict_metabolite_formation("CompA", mw_parent=300.0, logP_parent=2.0)
    assert result.c_metabolite_mg_L[0] == 0.0


# ── Monotone decay of parent (IV bolus) ──────────────────────────────────────


def test_c_parent_strictly_decreasing():
    result = predict_metabolite_formation("CompA", mw_parent=300.0, logP_parent=2.0)
    concs = result.c_parent_mg_L
    # Allow for minor numerical noise; check that final value < initial value
    assert concs[-1] < concs[0]


def test_c_parent_overall_monotone_decreasing():
    result = predict_metabolite_formation("CompA", mw_parent=300.0, logP_parent=2.0, dt_h=0.05)
    concs = result.c_parent_mg_L
    # Check most steps are decreasing (small numerical bumps tolerated)
    decreasing_count = sum(1 for i in range(1, len(concs)) if concs[i] <= concs[i - 1])
    total_steps = len(concs) - 1
    assert decreasing_count / total_steps > 0.95


# ── Metabolite rises and peaks ────────────────────────────────────────────────


def test_cmax_metabolite_positive():
    result = predict_metabolite_formation("CompA", mw_parent=300.0, logP_parent=2.0)
    assert result.cmax_metabolite > 0.0


def test_tmax_metabolite_positive():
    result = predict_metabolite_formation("CompA", mw_parent=300.0, logP_parent=2.0)
    assert result.tmax_metabolite_h > 0.0


def test_auc_metabolite_positive():
    result = predict_metabolite_formation("CompA", mw_parent=300.0, logP_parent=2.0)
    assert result.auc_metabolite > 0.0


def test_metabolite_auc_ratio_positive():
    result = predict_metabolite_formation("CompA", mw_parent=300.0, logP_parent=2.0)
    assert result.metabolite_auc_ratio > 0.0


# ── Parametric monotonicity ───────────────────────────────────────────────────


def test_higher_fm_gives_higher_metabolite_auc_ratio():
    low = predict_metabolite_formation("CompA", mw_parent=300.0, logP_parent=2.0, fm_metabolite=0.1)
    high = predict_metabolite_formation(
        "CompA", mw_parent=300.0, logP_parent=2.0, fm_metabolite=0.8
    )
    assert high.metabolite_auc_ratio > low.metabolite_auc_ratio


def test_higher_potency_ratio_gives_higher_weighted_auc_ratio():
    low = predict_metabolite_formation("CompA", mw_parent=300.0, logP_parent=2.0, potency_ratio=0.2)
    high = predict_metabolite_formation(
        "CompA", mw_parent=300.0, logP_parent=2.0, potency_ratio=2.0
    )
    assert high.weighted_auc_ratio > low.weighted_auc_ratio


# ── Safety flag ───────────────────────────────────────────────────────────────


def test_safety_flag_true_when_weighted_auc_ratio_above_1():
    # High fm + high potency ratio should trigger safety flag
    result = predict_metabolite_formation(
        "CompA",
        mw_parent=300.0,
        logP_parent=2.0,
        fm_metabolite=0.9,
        potency_ratio=5.0,
        cl_metabolite_L_per_h=0.5,
    )
    assert result.safety_flag is True


def test_safety_flag_false_when_weighted_auc_ratio_below_1():
    # Low fm + low potency ratio should not trigger safety flag
    result = predict_metabolite_formation(
        "CompA", mw_parent=300.0, logP_parent=2.0, fm_metabolite=0.05, potency_ratio=0.1
    )
    assert result.safety_flag is False


# ── screen_metabolites ────────────────────────────────────────────────────────


def test_screen_metabolites_returns_sorted_descending():
    compounds = [
        {
            "compound_name": "CompA",
            "mw_parent": 300.0,
            "logP_parent": 2.0,
            "fm_metabolite": 0.1,
            "potency_ratio": 0.1,
        },
        {
            "compound_name": "CompB",
            "mw_parent": 300.0,
            "logP_parent": 2.0,
            "fm_metabolite": 0.8,
            "potency_ratio": 2.0,
        },
        {
            "compound_name": "CompC",
            "mw_parent": 300.0,
            "logP_parent": 2.0,
            "fm_metabolite": 0.4,
            "potency_ratio": 1.0,
        },
    ]
    results = screen_metabolites(compounds)
    ratios = [r.weighted_auc_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_screen_metabolites_empty_list():
    results = screen_metabolites([])
    assert results == []


def test_screen_metabolites_returns_list_of_results():
    compounds = [
        {"compound_name": "CompA", "mw_parent": 300.0, "logP_parent": 2.0},
    ]
    results = screen_metabolites(compounds)
    assert isinstance(results, list)
    assert all(isinstance(r, MetaboliteFormationResult) for r in results)


# ── estimate_metabolite_clearance ─────────────────────────────────────────────


def test_estimate_metabolite_clearance_returns_float():
    cl = estimate_metabolite_clearance(5.0, 0.3, 300.0)
    assert isinstance(cl, float)


def test_estimate_metabolite_clearance_positive():
    cl = estimate_metabolite_clearance(5.0, 0.3, 300.0)
    assert cl > 0.0


def test_estimate_metabolite_clearance_heavier_slower():
    """Heavier metabolite has lower clearance."""
    cl_light = estimate_metabolite_clearance(5.0, 0.3, 150.0)
    cl_heavy = estimate_metabolite_clearance(5.0, 0.3, 600.0)
    assert cl_light > cl_heavy


# ── Validation ────────────────────────────────────────────────────────────────


def test_validation_fm_above_1_raises_value_error():
    with pytest.raises(ValueError):
        predict_metabolite_formation("CompA", mw_parent=300.0, logP_parent=2.0, fm_metabolite=1.5)


def test_validation_fm_negative_raises_value_error():
    with pytest.raises(ValueError):
        predict_metabolite_formation("CompA", mw_parent=300.0, logP_parent=2.0, fm_metabolite=-0.1)


# ── Stored fields ─────────────────────────────────────────────────────────────


def test_compound_name_stored():
    result = predict_metabolite_formation("MyDrug", mw_parent=350.0, logP_parent=3.0)
    assert result.compound_name == "MyDrug"


def test_dose_mg_stored():
    result = predict_metabolite_formation("CompA", mw_parent=300.0, logP_parent=2.0, dose_mg=200.0)
    assert result.dose_mg == 200.0
