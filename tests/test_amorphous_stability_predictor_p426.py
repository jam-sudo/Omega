"""Tests for Phase 426 — amorphous_stability_predictor_p426 module."""

import pytest

from omega_pbpk.prediction.amorphous_stability_predictor_p426 import (
    POLYMERS,
    AmorphousStabilityResult,
    predict_amorphous_stability,
    screen_polymers_stability,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        polymer_name="HPMC-AS",
        drug_tg_C=60.0,
        drug_loading_pct=20.0,
        storage_temp_C=25.0,
    )
    defaults.update(kwargs)
    return predict_amorphous_stability(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _base()
    assert isinstance(result, AmorphousStabilityResult)


# ---------------------------------------------------------------------------
# Preserved fields
# ---------------------------------------------------------------------------


def test_drug_name_preserved():
    result = _base(drug_name="Itraconazole")
    assert result.drug_name == "Itraconazole"


def test_polymer_name_preserved():
    for polymer in POLYMERS:
        result = _base(polymer_name=polymer)
        assert result.polymer_name == polymer


# ---------------------------------------------------------------------------
# Gordon-Taylor composite Tg
# ---------------------------------------------------------------------------


def test_composite_tg_between_drug_and_polymer_tg():
    drug_tg = 50.0
    polymer_tg = 120.0  # HPMC-AS
    result = _base(drug_tg_C=drug_tg, polymer_name="HPMC-AS", drug_loading_pct=30.0)
    # Composite should be between drug_tg and polymer_tg
    lo, hi = min(drug_tg, polymer_tg), max(drug_tg, polymer_tg)
    assert lo <= result.composite_tg_C <= hi


def test_higher_polymer_tg_higher_composite_tg():
    # PVP (Tg=168) vs PVP-VA (Tg=105)
    result_pvp = _base(polymer_name="PVP", drug_loading_pct=20.0)
    result_pvpva = _base(polymer_name="PVP-VA", drug_loading_pct=20.0)
    assert result_pvp.composite_tg_C > result_pvpva.composite_tg_C


# ---------------------------------------------------------------------------
# Stability / shelf life
# ---------------------------------------------------------------------------


def test_shelf_life_positive():
    result = _base()
    assert result.shelf_life_months > 0.0


def test_mobility_parameter_positive():
    result = _base()
    assert result.mobility_parameter > 0.0


def test_t_onset_greater_than_shelf_life():
    result = _base()
    assert result.t_onset_recrystallization_months > result.shelf_life_months


def test_shelf_life_equals_08_times_t_onset():
    result = _base()
    assert abs(result.shelf_life_months - result.t_onset_recrystallization_months * 0.8) < 1e-9


def test_higher_composite_tg_longer_shelf_life():
    # Higher composite Tg at same storage temp → less mobility → longer shelf life
    result_pvp = _base(polymer_name="PVP", drug_loading_pct=20.0, storage_temp_C=25.0)
    result_pvpva = _base(polymer_name="PVP-VA", drug_loading_pct=20.0, storage_temp_C=25.0)
    assert result_pvp.shelf_life_months >= result_pvpva.shelf_life_months


# ---------------------------------------------------------------------------
# Stability class assignment
# ---------------------------------------------------------------------------


def test_stability_class_excellent():
    # Store well below Tg → very long shelf life
    result = _base(drug_tg_C=80.0, polymer_name="PVP", drug_loading_pct=10.0, storage_temp_C=25.0)
    if result.shelf_life_months > 24:
        assert result.stability_class == "excellent"


def test_stability_class_poor():
    # Store at same temperature as Tg → high mobility → poor stability
    result = _base(
        drug_tg_C=20.0, polymer_name="PVP-VA", drug_loading_pct=50.0, storage_temp_C=25.0
    )
    if result.shelf_life_months < 6:
        assert result.stability_class == "poor"


# ---------------------------------------------------------------------------
# Recommended storage
# ---------------------------------------------------------------------------


def test_room_temperature_when_well_below_tg():
    # High Tg polymer, low loading, moderate storage temp → room_temperature
    result = _base(drug_tg_C=100.0, polymer_name="PVP", drug_loading_pct=10.0, storage_temp_C=25.0)
    # composite_tg should be well above 25 + 50 = 75
    if result.composite_tg_C - result.storage_temp_C > 50:
        assert result.recommended_storage == "room_temperature"


def test_storage_recommendation_is_valid():
    valid_options = {"room_temperature", "refrigerator", "freezer"}
    for polymer in POLYMERS:
        result = _base(polymer_name=polymer)
        assert result.recommended_storage in valid_options


# ---------------------------------------------------------------------------
# screen_polymers_stability
# ---------------------------------------------------------------------------


def test_screen_returns_five_results():
    results = screen_polymers_stability("Drug", 60.0, 20.0)
    assert len(results) == 5


def test_screen_sorted_descending():
    results = screen_polymers_stability("Drug", 60.0, 20.0)
    shelf_lives = [r.shelf_life_months for r in results]
    assert shelf_lives == sorted(shelf_lives, reverse=True)


def test_screen_covers_all_polymers():
    results = screen_polymers_stability("Drug", 60.0, 20.0)
    polymer_names = {r.polymer_name for r in results}
    assert polymer_names == set(POLYMERS.keys())


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_polymer_raises():
    with pytest.raises(ValueError, match="polymer_name"):
        predict_amorphous_stability(
            drug_name="X",
            polymer_name="CelluloseMagic",
            drug_tg_C=60.0,
            drug_loading_pct=20.0,
        )


def test_drug_loading_zero_raises():
    with pytest.raises(ValueError):
        _base(drug_loading_pct=0.0)


def test_drug_loading_100_raises():
    with pytest.raises(ValueError):
        _base(drug_loading_pct=100.0)


def test_storage_temp_too_high_raises():
    with pytest.raises(ValueError):
        _base(storage_temp_C=70.0)


def test_storage_temp_too_low_raises():
    with pytest.raises(ValueError):
        _base(storage_temp_C=-40.0)
