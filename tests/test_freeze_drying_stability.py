"""Tests for Phase 1142 — prediction/freeze_drying_stability.py."""

import pytest

from omega_pbpk.prediction.freeze_drying_stability import (
    LyophilizationResult,
    assess_lyophilization_stability,
    screen_excipients,
)

# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


def test_returns_result_type():
    res = assess_lyophilization_stability("mAb", 5.0, "trehalose", 0.5, -20.0)
    assert isinstance(res, LyophilizationResult)


def test_drug_name_preserved():
    res = assess_lyophilization_stability("Infliximab", 10.0, "sucrose", 1.0, -25.0)
    assert res.drug_name == "Infliximab"


def test_excipient_type_preserved():
    res = assess_lyophilization_stability("X", 5.0, "mannitol", 1.5, -20.0)
    assert res.excipient_type == "mannitol"


def test_protein_conc_preserved():
    res = assess_lyophilization_stability("X", 7.5, "trehalose", 0.5, -20.0)
    assert res.protein_concentration_mg_mL == pytest.approx(7.5)


def test_residual_moisture_preserved():
    res = assess_lyophilization_stability("X", 5.0, "sucrose", 1.2, -20.0)
    assert res.residual_moisture_pct == pytest.approx(1.2)


def test_storage_temp_preserved():
    res = assess_lyophilization_stability("X", 5.0, "trehalose", 0.5, -20.0)
    assert res.storage_temp_C == pytest.approx(-20.0)


# ---------------------------------------------------------------------------
# Tg' and Tc values
# ---------------------------------------------------------------------------


def test_trehalose_tg_prime():
    res = assess_lyophilization_stability("X", 5.0, "trehalose", 0.5, -20.0)
    assert res.tg_prime_C == pytest.approx(-29.0)


def test_trehalose_tc():
    res = assess_lyophilization_stability("X", 5.0, "trehalose", 0.5, -20.0)
    assert res.tc_C == pytest.approx(-25.0)


def test_sucrose_tg_prime():
    res = assess_lyophilization_stability("X", 5.0, "sucrose", 0.5, -20.0)
    assert res.tg_prime_C == pytest.approx(-32.0)


def test_mannitol_tc():
    res = assess_lyophilization_stability("X", 5.0, "mannitol", 0.5, -20.0)
    assert res.tc_C == pytest.approx(-20.0)


# ---------------------------------------------------------------------------
# Collapse risk
# ---------------------------------------------------------------------------


def test_collapse_risk_true_at_or_above_tc():
    # mannitol tc = -20, storage at -20
    res = assess_lyophilization_stability("X", 5.0, "mannitol", 0.5, -20.0)
    assert res.collapse_risk is True


def test_collapse_risk_false_below_tc():
    # mannitol tc = -20, storage at -40
    res = assess_lyophilization_stability("X", 5.0, "mannitol", 0.5, -40.0)
    assert res.collapse_risk is False


def test_collapse_risk_true_above_tc():
    # trehalose tc = -25, storage at -10
    res = assess_lyophilization_stability("X", 5.0, "trehalose", 0.5, -10.0)
    assert res.collapse_risk is True


# ---------------------------------------------------------------------------
# Moisture score
# ---------------------------------------------------------------------------


def test_moisture_score_below_1pct():
    res = assess_lyophilization_stability("X", 5.0, "trehalose", 0.5, -40.0)
    assert res.moisture_score == pytest.approx(30.0)


def test_moisture_score_1_to_2pct():
    res = assess_lyophilization_stability("X", 5.0, "trehalose", 1.5, -40.0)
    assert res.moisture_score == pytest.approx(20.0)


def test_moisture_score_2_to_3pct():
    res = assess_lyophilization_stability("X", 5.0, "trehalose", 2.5, -40.0)
    assert res.moisture_score == pytest.approx(10.0)


def test_moisture_score_above_3pct():
    res = assess_lyophilization_stability("X", 5.0, "trehalose", 4.0, -40.0)
    assert res.moisture_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tg margin score
# ---------------------------------------------------------------------------


def test_tg_margin_score_large_margin():
    # trehalose tg' = -29, storage = -80: margin = -29 - (-80) = 51 > 20
    res = assess_lyophilization_stability("X", 5.0, "trehalose", 0.5, -80.0)
    assert res.tg_margin_score == pytest.approx(30.0)


def test_tg_margin_score_medium_margin():
    # trehalose tg' = -29, storage = -45: margin = 16
    res = assess_lyophilization_stability("X", 5.0, "trehalose", 0.5, -45.0)
    assert res.tg_margin_score == pytest.approx(20.0)


def test_tg_margin_score_small_margin():
    # trehalose tg' = -29, storage = -35: margin = 6
    res = assess_lyophilization_stability("X", 5.0, "trehalose", 0.5, -35.0)
    assert res.tg_margin_score == pytest.approx(10.0)


def test_tg_margin_score_no_margin():
    # trehalose tg' = -29, storage = -10: margin = -19
    res = assess_lyophilization_stability("X", 5.0, "trehalose", 0.5, -10.0)
    assert res.tg_margin_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Stability ordering
# ---------------------------------------------------------------------------


def test_trehalose_better_than_sucrose():
    """Trehalose has higher lyoprotection (0.95 vs 0.9), so should score >= sucrose."""
    treb = assess_lyophilization_stability("X", 5.0, "trehalose", 0.5, -40.0)
    suc = assess_lyophilization_stability("X", 5.0, "sucrose", 0.5, -40.0)
    # trehalose tg' = -29 vs sucrose -32; both have large margin at -40
    # lyoprotection 0.95 vs 0.90 → trehalose wins
    assert treb.stability_score >= suc.stability_score


def test_sucrose_better_than_mannitol():
    """Sucrose lyoprotection 0.9 > mannitol 0.7; at -55 both have tg margin >20 so lyoprotection decides."""
    # sucrose tg'=-32, margin at -55 = 23 (>20, score=30)
    # mannitol tg'=-25, margin at -55 = 30 (>20, score=30)
    # tie on tg margin → lyoprotection_score: sucrose 18 vs mannitol 14 → sucrose wins
    suc = assess_lyophilization_stability("X", 5.0, "sucrose", 0.5, -55.0)
    man = assess_lyophilization_stability("X", 5.0, "mannitol", 0.5, -55.0)
    assert suc.stability_score > man.stability_score


def test_mannitol_ge_glycine():
    """Mannitol lyoprotection 0.7 >= glycine 0.6 → mannitol >= glycine."""
    man = assess_lyophilization_stability("X", 5.0, "mannitol", 0.5, -40.0)
    gly = assess_lyophilization_stability("X", 5.0, "glycine", 0.5, -40.0)
    assert man.stability_score >= gly.stability_score


# ---------------------------------------------------------------------------
# Shelf life thresholds
# ---------------------------------------------------------------------------


def test_shelf_life_3_years_high_score():
    # Low moisture, large tg margin, trehalose, low conc
    res = assess_lyophilization_stability("X", 5.0, "trehalose", 0.5, -80.0)
    assert res.stability_score >= 75.0
    assert res.shelf_life_estimate_years == pytest.approx(3.0)


def test_shelf_life_05_years_low_score():
    # High moisture, no tg margin, poor lyoprotection
    res = assess_lyophilization_stability("X", 200.0, "glycine", 5.0, -10.0)
    assert res.stability_score < 40.0
    assert res.shelf_life_estimate_years == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Stability class
# ---------------------------------------------------------------------------


def test_stability_class_excellent():
    res = assess_lyophilization_stability("X", 5.0, "trehalose", 0.5, -80.0)
    assert res.stability_class == "excellent"


def test_stability_class_valid_values():
    res = assess_lyophilization_stability("X", 5.0, "mannitol", 2.0, -25.0)
    assert res.stability_class in {"excellent", "good", "moderate", "poor"}


# ---------------------------------------------------------------------------
# screen_excipients
# ---------------------------------------------------------------------------


def test_screen_excipients_returns_four():
    results = screen_excipients("mAb", 5.0, 0.5, -40.0)
    assert len(results) == 4


def test_screen_excipients_sorted_descending():
    results = screen_excipients("mAb", 5.0, 0.5, -40.0)
    scores = [r.stability_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_screen_excipients_first_is_trehalose_or_sucrose():
    results = screen_excipients("mAb", 5.0, 0.5, -40.0)
    assert results[0].excipient_type in {"trehalose", "sucrose"}


def test_screen_excipients_all_same_drug():
    results = screen_excipients("AbcDrug", 5.0, 0.5, -40.0)
    assert all(r.drug_name == "AbcDrug" for r in results)


def test_screen_excipients_all_excipient_types_present():
    results = screen_excipients("mAb", 5.0, 0.5, -40.0)
    types = {r.excipient_type for r in results}
    assert types == {"sucrose", "trehalose", "mannitol", "glycine"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_zero_protein_concentration_raises():
    with pytest.raises(ValueError, match="protein_concentration"):
        assess_lyophilization_stability("X", 0.0, "sucrose", 1.0, -20.0)


def test_negative_protein_concentration_raises():
    with pytest.raises(ValueError, match="protein_concentration"):
        assess_lyophilization_stability("X", -1.0, "sucrose", 1.0, -20.0)


def test_moisture_above_10_raises():
    with pytest.raises(ValueError, match="residual_moisture"):
        assess_lyophilization_stability("X", 5.0, "sucrose", 11.0, -20.0)


def test_moisture_negative_raises():
    with pytest.raises(ValueError, match="residual_moisture"):
        assess_lyophilization_stability("X", 5.0, "sucrose", -0.1, -20.0)


def test_storage_temp_too_high_raises():
    with pytest.raises(ValueError, match="storage_temp"):
        assess_lyophilization_stability("X", 5.0, "sucrose", 1.0, 40.0)


def test_storage_temp_too_low_raises():
    with pytest.raises(ValueError, match="storage_temp"):
        assess_lyophilization_stability("X", 5.0, "sucrose", 1.0, -100.0)


def test_invalid_excipient_raises():
    with pytest.raises(ValueError, match="excipient_type"):
        assess_lyophilization_stability("X", 5.0, "glucose", 1.0, -20.0)
