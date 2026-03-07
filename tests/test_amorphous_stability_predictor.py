"""Tests for Amorphous Stability Predictor — Phase 814."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.amorphous_stability_predictor import (
    AmorphousStabilityResult,
    optimize_drug_loading,
    predict_amorphous_stability,
)

# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_returns_amorphous_stability_result():
    result = predict_amorphous_stability("DrugA", logp=2.0, mw_da=350.0)
    assert isinstance(result, AmorphousStabilityResult)


def test_compound_name_preserved():
    result = predict_amorphous_stability("TestDrug", logp=1.5, mw_da=400.0)
    assert result.compound_name == "TestDrug"


def test_polymer_name_preserved():
    result = predict_amorphous_stability("DrugA", logp=2.0, mw_da=350.0, polymer_name="PVP")
    assert result.polymer_name == "PVP"


def test_drug_loading_preserved():
    result = predict_amorphous_stability("DrugA", logp=2.0, mw_da=350.0, drug_loading_pct=25.0)
    assert result.drug_loading_pct == 25.0


def test_notes_nonempty():
    result = predict_amorphous_stability("DrugA", logp=2.0, mw_da=350.0)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Tg blend
# ---------------------------------------------------------------------------


def test_tg_blend_below_polymer_tg():
    # Adding drug (Tg_drug < Tg_polymer) should lower blend Tg
    result = predict_amorphous_stability(
        "DrugA",
        logp=2.0,
        mw_da=350.0,
        tg_drug_celsius=80.0,
        polymer_name="HPMC-AS",
        drug_loading_pct=30.0,
    )
    # Blend Tg should be between drug and polymer Tg
    assert result.tg_blend_celsius < 120.0  # HPMC-AS Tg ~120


def test_tg_blend_above_drug_tg_at_low_loading():
    # Low drug loading → blend Tg close to polymer Tg
    result = predict_amorphous_stability(
        "DrugA",
        logp=2.0,
        mw_da=350.0,
        tg_drug_celsius=60.0,
        polymer_name="HPMC-AS",
        drug_loading_pct=5.0,
    )
    # With 5% drug, blend Tg should be close to polymer Tg (>80)
    assert result.tg_blend_celsius > 80.0


# ---------------------------------------------------------------------------
# t_above_tg
# ---------------------------------------------------------------------------


def test_t_above_tg_is_float():
    result = predict_amorphous_stability("DrugA", logp=2.0, mw_da=350.0)
    assert isinstance(result.t_above_tg_celsius, float)


def test_t_above_tg_negative_when_tg_above_storage():
    # High polymer Tg, low storage temp → blend Tg > storage → t_above_tg < 0
    result = predict_amorphous_stability(
        "DrugA",
        logp=2.0,
        mw_da=350.0,
        tg_drug_celsius=120.0,
        polymer_name="HPMC-AS",
        drug_loading_pct=10.0,
        storage_temp_celsius=25.0,
    )
    # Blend Tg should be >> 25°C → t_above_tg < 0
    assert result.t_above_tg_celsius < 0.0


# ---------------------------------------------------------------------------
# Crystallization tendency
# ---------------------------------------------------------------------------


def test_crystallization_tendency_is_string():
    result = predict_amorphous_stability("DrugA", logp=2.0, mw_da=350.0)
    assert isinstance(result.crystallization_tendency, str)
    assert len(result.crystallization_tendency) > 0


def test_crystallization_tendency_valid_values():
    result = predict_amorphous_stability("DrugA", logp=2.0, mw_da=350.0)
    valid = {"low", "moderate", "high", "very_high"}
    assert result.crystallization_tendency in valid


def test_high_logp_tends_to_crystallize():
    # Very high logP → poor miscibility → higher crystallization tendency
    r_low = predict_amorphous_stability("A", logp=0.5, mw_da=300.0)
    r_high = predict_amorphous_stability("B", logp=6.0, mw_da=300.0)
    # r_high should have worse or equal crystallization tendency
    order = ["low", "moderate", "high", "very_high"]
    assert order.index(r_high.crystallization_tendency) >= order.index(
        r_low.crystallization_tendency
    )


# ---------------------------------------------------------------------------
# Shelf life
# ---------------------------------------------------------------------------


def test_shelf_life_positive():
    result = predict_amorphous_stability("DrugA", logp=2.0, mw_da=350.0)
    assert result.predicted_shelf_life_months > 0.0


def test_high_tg_blend_gives_longer_shelf_life():
    # Higher Tg blend → slower crystallization → longer shelf life
    r_high_tg = predict_amorphous_stability(
        "DrugA",
        logp=2.0,
        mw_da=350.0,
        tg_drug_celsius=150.0,
        drug_loading_pct=10.0,
    )
    r_low_tg = predict_amorphous_stability(
        "DrugA",
        logp=2.0,
        mw_da=350.0,
        tg_drug_celsius=50.0,
        drug_loading_pct=60.0,
        storage_temp_celsius=40.0,
    )
    assert r_high_tg.predicted_shelf_life_months >= r_low_tg.predicted_shelf_life_months


# ---------------------------------------------------------------------------
# Amorphous fraction at 2 years
# ---------------------------------------------------------------------------


def test_amorphous_fraction_2yr_between_0_and_1():
    result = predict_amorphous_stability("DrugA", logp=2.0, mw_da=350.0)
    assert 0.0 <= result.amorphous_fraction_2yr <= 1.0


# ---------------------------------------------------------------------------
# Miscibility
# ---------------------------------------------------------------------------


def test_miscibility_is_string():
    result = predict_amorphous_stability("DrugA", logp=2.0, mw_da=350.0)
    assert isinstance(result.miscibility, str)


def test_miscibility_valid_values():
    result = predict_amorphous_stability("DrugA", logp=2.0, mw_da=350.0)
    valid = {"miscible", "partially_miscible", "immiscible"}
    assert result.miscibility in valid


# ---------------------------------------------------------------------------
# Recommended drug loading
# ---------------------------------------------------------------------------


def test_recommended_drug_loading_in_range():
    result = predict_amorphous_stability("DrugA", logp=2.0, mw_da=350.0)
    assert 0.0 < result.recommended_drug_loading_pct < 100.0


# ---------------------------------------------------------------------------
# Excipient recommendation
# ---------------------------------------------------------------------------


def test_excipient_recommendation_nonempty():
    result = predict_amorphous_stability("DrugA", logp=2.0, mw_da=350.0)
    assert len(result.excipient_recommendation) > 0


# ---------------------------------------------------------------------------
# Polymer variants
# ---------------------------------------------------------------------------


def test_pvp_polymer():
    result = predict_amorphous_stability("DrugA", logp=2.0, mw_da=350.0, polymer_name="PVP")
    assert isinstance(result, AmorphousStabilityResult)


def test_pvp_va_polymer():
    result = predict_amorphous_stability("DrugA", logp=2.0, mw_da=350.0, polymer_name="PVP-VA")
    assert isinstance(result, AmorphousStabilityResult)


def test_unknown_polymer_uses_tg_param():
    result = predict_amorphous_stability(
        "DrugA",
        logp=2.0,
        mw_da=350.0,
        polymer_name="CustomPolymer",
        tg_polymer_celsius=90.0,
    )
    assert isinstance(result, AmorphousStabilityResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_drug_loading_zero_raises():
    with pytest.raises(ValueError, match="drug_loading_pct"):
        predict_amorphous_stability("DrugA", logp=2.0, mw_da=350.0, drug_loading_pct=0.0)


def test_invalid_drug_loading_100_raises():
    with pytest.raises(ValueError, match="drug_loading_pct"):
        predict_amorphous_stability("DrugA", logp=2.0, mw_da=350.0, drug_loading_pct=100.0)


# ---------------------------------------------------------------------------
# optimize_drug_loading
# ---------------------------------------------------------------------------


def test_optimize_returns_list():
    results = optimize_drug_loading("DrugA", logp=2.0, mw_da=350.0)
    assert isinstance(results, list)


def test_optimize_default_5_loadings():
    results = optimize_drug_loading("DrugA", logp=2.0, mw_da=350.0)
    assert len(results) == 5


def test_optimize_sorted_by_shelf_life_descending():
    results = optimize_drug_loading("DrugA", logp=2.0, mw_da=350.0)
    for i in range(len(results) - 1):
        assert results[i].predicted_shelf_life_months >= results[i + 1].predicted_shelf_life_months


def test_optimize_custom_loadings():
    results = optimize_drug_loading("DrugA", logp=2.0, mw_da=350.0, loadings=[10.0, 20.0, 30.0])
    assert len(results) == 3


def test_optimize_returns_amorphous_stability_result():
    results = optimize_drug_loading("DrugA", logp=2.0, mw_da=350.0)
    for r in results:
        assert isinstance(r, AmorphousStabilityResult)
