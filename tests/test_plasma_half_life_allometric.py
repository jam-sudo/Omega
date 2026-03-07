"""Tests for prediction/plasma_half_life_allometric.py — Phase 1123."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.plasma_half_life_allometric import (
    AllometricHalfLifeResult,
    predict_human_half_life,
    scale_across_species,
)


# ---------------------------------------------------------------------------
# predict_human_half_life — return type and fields
# ---------------------------------------------------------------------------


def test_returns_allometric_result():
    r = predict_human_half_life("DrugA", t_half_species_h=1.0, species="rat")
    assert isinstance(r, AllometricHalfLifeResult)


def test_drug_name_preserved():
    r = predict_human_half_life("Warfarin", t_half_species_h=2.0, species="rat")
    assert r.drug_name == "Warfarin"


def test_source_species_lowercase():
    r = predict_human_half_life("X", t_half_species_h=1.0, species="Rat")
    assert r.source_species == "rat"


def test_target_species_is_human():
    r = predict_human_half_life("X", t_half_species_h=1.0, species="dog")
    assert r.target_species == "human"


def test_target_bw_is_70():
    r = predict_human_half_life("X", t_half_species_h=1.0, species="rat")
    assert r.target_bw_kg == pytest.approx(70.0)


def test_scaling_exponent_is_025():
    r = predict_human_half_life("X", t_half_species_h=1.0, species="rat")
    assert r.scaling_exponent == pytest.approx(0.25)


def test_notes_is_nonempty_str():
    r = predict_human_half_life("X", t_half_species_h=1.0, species="rat")
    assert isinstance(r.notes, str)
    assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


def test_rat_to_human_1h():
    """Rat t½=1h → human t½ ≈ 1*(70/0.25)^0.25 ≈ 5.163h."""
    r = predict_human_half_life("Drug", t_half_species_h=1.0, species="rat")
    expected = 1.0 * (70.0 / 0.25) ** 0.25
    assert r.predicted_t_half_h == pytest.approx(expected, rel=1e-6)


def test_rat_to_human_5h():
    r = predict_human_half_life("Drug", t_half_species_h=5.0, species="rat")
    expected = 5.0 * (70.0 / 0.25) ** 0.25
    assert r.predicted_t_half_h == pytest.approx(expected, rel=1e-6)


def test_mouse_to_human_shorter_than_rat_to_human():
    """Mouse has lower BW than rat; prediction differs."""
    r_mouse = predict_human_half_life("Drug", t_half_species_h=1.0, species="mouse")
    r_rat = predict_human_half_life("Drug", t_half_species_h=1.0, species="rat")
    # Mouse BW=0.02 < rat BW=0.25 → (70/0.02)^0.25 > (70/0.25)^0.25
    assert r_mouse.predicted_t_half_h > r_rat.predicted_t_half_h


def test_dog_to_human_closer():
    """Dog BW=11 closer to 70 → smaller fold change."""
    r_dog = predict_human_half_life("Drug", t_half_species_h=1.0, species="dog")
    r_mouse = predict_human_half_life("Drug", t_half_species_h=1.0, species="mouse")
    assert r_dog.predicted_t_half_h < r_mouse.predicted_t_half_h


def test_fold_change_equals_ratio():
    r = predict_human_half_life("Drug", t_half_species_h=2.0, species="rat")
    assert r.fold_change == pytest.approx(r.predicted_t_half_h / r.source_t_half_h, rel=1e-9)


def test_human_to_human_fold_change_is_one():
    r = predict_human_half_life("Drug", t_half_species_h=4.0, species="human")
    assert r.fold_change == pytest.approx(1.0, rel=1e-6)


def test_custom_body_weight():
    """Override body_weight_kg changes the prediction."""
    r_default = predict_human_half_life("Drug", t_half_species_h=1.0, species="rat")
    r_custom = predict_human_half_life(
        "Drug", t_half_species_h=1.0, species="rat", body_weight_kg=1.0
    )
    expected_custom = 1.0 * (70.0 / 1.0) ** 0.25
    assert r_custom.predicted_t_half_h == pytest.approx(expected_custom, rel=1e-6)
    assert r_custom.predicted_t_half_h != pytest.approx(r_default.predicted_t_half_h, rel=1e-3)


# ---------------------------------------------------------------------------
# Confidence levels
# ---------------------------------------------------------------------------


def test_confidence_rat_is_high():
    r = predict_human_half_life("Drug", t_half_species_h=1.0, species="rat")
    assert r.confidence == "high"


def test_confidence_mouse_is_low():
    r = predict_human_half_life("Drug", t_half_species_h=1.0, species="mouse")
    assert r.confidence == "low"


def test_confidence_dog_is_moderate():
    r = predict_human_half_life("Drug", t_half_species_h=1.0, species="dog")
    assert r.confidence == "moderate"


def test_confidence_monkey_is_moderate():
    r = predict_human_half_life("Drug", t_half_species_h=1.0, species="monkey")
    assert r.confidence == "moderate"


# ---------------------------------------------------------------------------
# scale_across_species
# ---------------------------------------------------------------------------


def test_scale_returns_five_results():
    results = scale_across_species("Drug", t_half_rat_h=1.0)
    assert len(results) == 5


def test_scale_sorted_by_bw_ascending():
    results = scale_across_species("Drug", t_half_rat_h=1.0)
    bws = [r.target_bw_kg for r in results]
    assert bws == sorted(bws)


def test_scale_source_species_all_rat():
    results = scale_across_species("Drug", t_half_rat_h=1.0)
    assert all(r.source_species == "rat" for r in results)


def test_scale_larger_species_longer_half_life():
    results = scale_across_species("Drug", t_half_rat_h=1.0)
    bw_seq = [r.target_bw_kg for r in results]
    t_seq = [r.predicted_t_half_h for r in results]
    # Monotonically increasing since t½ ∝ BW^0.25
    for i in range(len(t_seq) - 1):
        if bw_seq[i] < bw_seq[i + 1]:
            assert t_seq[i] < t_seq[i + 1]


def test_scale_human_result_matches_predict():
    results = scale_across_species("Drug", t_half_rat_h=2.0)
    human_result = next(r for r in results if r.target_species == "human")
    r_direct = predict_human_half_life("Drug", t_half_species_h=2.0, species="rat")
    assert human_result.predicted_t_half_h == pytest.approx(r_direct.predicted_t_half_h, rel=1e-9)


def test_scale_rat_to_itself_fold_one():
    results = scale_across_species("Drug", t_half_rat_h=3.0)
    rat_result = next(r for r in results if r.target_species == "rat")
    assert rat_result.fold_change == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_t_half_zero():
    with pytest.raises(ValueError, match="t_half_species_h"):
        predict_human_half_life("Drug", t_half_species_h=0.0, species="rat")


def test_invalid_t_half_negative():
    with pytest.raises(ValueError, match="t_half_species_h"):
        predict_human_half_life("Drug", t_half_species_h=-1.0, species="rat")


def test_invalid_species():
    with pytest.raises(ValueError, match="species"):
        predict_human_half_life("Drug", t_half_species_h=1.0, species="elephant")


def test_scale_invalid_t_half():
    with pytest.raises(ValueError, match="t_half_rat_h"):
        scale_across_species("Drug", t_half_rat_h=0.0)
