"""Tests for Phase 263 -- Salt Bridge Stability Predictor."""

import pytest

from omega_pbpk.prediction.salt_bridge_stability_predictor import (
    SaltBridgeStabilityResult,
    predict_salt_bridge_stability,
    screen_salt_bridges,
)

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_predict_returns_correct_type():
    result = predict_salt_bridge_stability("DrugA", "Lys_Asp")
    assert isinstance(result, SaltBridgeStabilityResult)


# ---------------------------------------------------------------------------
# stability_score range
# ---------------------------------------------------------------------------


def test_stability_score_in_range_lys_asp():
    r = predict_salt_bridge_stability("DrugA", "Lys_Asp")
    assert 0 <= r.stability_score <= 100


def test_stability_score_in_range_his_asp():
    r = predict_salt_bridge_stability("DrugB", "His_Asp")
    assert 0 <= r.stability_score <= 100


def test_stability_score_in_range_arg_glu():
    r = predict_salt_bridge_stability("DrugC", "Arg_Glu")
    assert 0 <= r.stability_score <= 100


# ---------------------------------------------------------------------------
# probability_intact range
# ---------------------------------------------------------------------------


def test_probability_intact_in_range():
    for sb_type in ["Lys_Asp", "Arg_Glu", "Lys_Glu", "Arg_Asp", "His_Asp"]:
        r = predict_salt_bridge_stability("DrugX", sb_type)
        assert 0.0 <= r.probability_intact <= 1.0, f"Failed for {sb_type}"


# ---------------------------------------------------------------------------
# Higher ionic strength -> lower stability
# ---------------------------------------------------------------------------


def test_higher_ionic_strength_lowers_stability():
    r_low = predict_salt_bridge_stability("DrugA", "Lys_Asp", ionic_strength_mM=100.0)
    r_high = predict_salt_bridge_stability("DrugA", "Lys_Asp", ionic_strength_mM=400.0)
    assert r_high.stability_score < r_low.stability_score


def test_higher_ionic_strength_lowers_probability():
    r_low = predict_salt_bridge_stability("DrugA", "Arg_Glu", ionic_strength_mM=50.0)
    r_high = predict_salt_bridge_stability("DrugA", "Arg_Glu", ionic_strength_mM=450.0)
    assert r_high.probability_intact <= r_low.probability_intact


# ---------------------------------------------------------------------------
# Arg_Glu has higher base strength than His_Asp
# ---------------------------------------------------------------------------


def test_arg_glu_stronger_than_his_asp_at_physiological():
    r_arg = predict_salt_bridge_stability("DrugA", "Arg_Glu")
    r_his = predict_salt_bridge_stability("DrugA", "His_Asp")
    assert r_arg.interaction_strength_kcal_mol > r_his.interaction_strength_kcal_mol


def test_arg_glu_higher_stability_score_than_his_asp():
    r_arg = predict_salt_bridge_stability("DrugA", "Arg_Glu")
    r_his = predict_salt_bridge_stability("DrugA", "His_Asp")
    assert r_arg.stability_score > r_his.stability_score


# ---------------------------------------------------------------------------
# disruption_risk valid values
# ---------------------------------------------------------------------------


def test_disruption_risk_valid_values():
    valid = {"low", "moderate", "high"}
    for sb_type in ["Lys_Asp", "Arg_Glu", "Lys_Glu", "Arg_Asp", "His_Asp"]:
        r = predict_salt_bridge_stability("DrugX", sb_type)
        assert r.disruption_risk in valid


def test_disruption_risk_low_when_high_prob():
    # At physiological pH and low ionic strength, Arg_Glu should be stable
    r = predict_salt_bridge_stability("DrugA", "Arg_Glu", ionic_strength_mM=10.0)
    assert r.disruption_risk == "low"


def test_disruption_risk_high_when_low_prob():
    # His_Asp at extreme pH and high ionic strength -> very unstable
    r = predict_salt_bridge_stability(
        "DrugA", "His_Asp", physiological_ph=4.0, ionic_strength_mM=490.0
    )
    assert r.disruption_risk == "high"


# ---------------------------------------------------------------------------
# half_life_in_plasma_h > 0
# ---------------------------------------------------------------------------


def test_half_life_positive():
    r = predict_salt_bridge_stability("DrugA", "Lys_Asp")
    assert r.half_life_in_plasma_h > 0


def test_half_life_proportional_to_probability():
    r = predict_salt_bridge_stability("DrugA", "Arg_Glu")
    assert abs(r.half_life_in_plasma_h - r.probability_intact * 24.0) < 1e-9


# ---------------------------------------------------------------------------
# optimal_ph_for_stability
# ---------------------------------------------------------------------------


def test_optimal_ph_is_physiological():
    r = predict_salt_bridge_stability("DrugA", "Lys_Asp")
    assert r.optimal_ph_for_stability == 7.4


# ---------------------------------------------------------------------------
# screen_salt_bridges
# ---------------------------------------------------------------------------


def test_screen_returns_five_results():
    results = screen_salt_bridges("DrugA")
    assert len(results) == 5


def test_screen_sorted_descending():
    results = screen_salt_bridges("DrugA")
    scores = [r.stability_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_screen_all_are_correct_type():
    results = screen_salt_bridges("DrugA")
    assert all(isinstance(r, SaltBridgeStabilityResult) for r in results)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_salt_bridge_type_raises():
    with pytest.raises(ValueError, match="Unknown salt_bridge_type"):
        predict_salt_bridge_stability("DrugA", "InvalidType")


def test_ph_too_low_raises():
    with pytest.raises(ValueError, match="physiological_ph must be in"):
        predict_salt_bridge_stability("DrugA", "Lys_Asp", physiological_ph=3.9)


def test_ph_too_high_raises():
    with pytest.raises(ValueError, match="physiological_ph must be in"):
        predict_salt_bridge_stability("DrugA", "Lys_Asp", physiological_ph=10.1)


def test_negative_ionic_strength_raises():
    with pytest.raises(ValueError, match="ionic_strength_mM must be >= 0"):
        predict_salt_bridge_stability("DrugA", "Lys_Asp", ionic_strength_mM=-1.0)
