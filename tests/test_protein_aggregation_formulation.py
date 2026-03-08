"""
Tests for Phase 410 — Protein Aggregation Kinetics in Formulation
"""

import pytest

from omega_pbpk.prediction.protein_aggregation_formulation import (
    FormulationAggregationResult,
    screen_excipients_aggregation,
    simulate_formulation_aggregation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_result(**kwargs):
    defaults = dict(
        drug_name="mAb-X",
        formulation_ph=7.0,
        ionic_strength_mM=150.0,
        temperature_C=25.0,
        excipient="none",
    )
    defaults.update(kwargs)
    return simulate_formulation_aggregation(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_formulation_aggregation_result(self):
        result = _base_result()
        assert isinstance(result, FormulationAggregationResult)

    def test_has_times_list(self):
        result = _base_result()
        assert isinstance(result.times_h, list)
        assert len(result.times_h) > 0

    def test_has_monomer_pct_list(self):
        result = _base_result()
        assert isinstance(result.monomer_pct, list)
        assert len(result.monomer_pct) == len(result.times_h)

    def test_has_aggregate_pct_list(self):
        result = _base_result()
        assert isinstance(result.aggregate_pct, list)
        assert len(result.aggregate_pct) == len(result.times_h)

    def test_monomer_starts_at_100_pct(self):
        result = _base_result()
        assert abs(result.monomer_pct[0] - 100.0) < 1e-6

    def test_aggregate_starts_at_0_pct(self):
        result = _base_result()
        assert abs(result.aggregate_pct[0]) < 1e-6


# ---------------------------------------------------------------------------
# Sucrose is most stable (lowest aggregation rate)
# ---------------------------------------------------------------------------


class TestExcipientStability:
    def test_sucrose_lower_rate_than_none(self):
        r_none = _base_result(excipient="none")
        r_sucrose = _base_result(excipient="sucrose")
        assert r_sucrose.aggregation_rate_per_h < r_none.aggregation_rate_per_h

    def test_sucrose_longer_shelf_life_than_none(self):
        r_none = _base_result(excipient="none")
        r_sucrose = _base_result(excipient="sucrose")
        assert r_sucrose.shelf_life_years > r_none.shelf_life_years

    def test_none_least_stable_aggregation_rate(self):
        """'none' excipient should have the highest aggregation rate."""
        excipients = ["none", "sucrose", "polysorbate", "arginine", "histidine"]
        rates = {exc: _base_result(excipient=exc).aggregation_rate_per_h for exc in excipients}
        assert rates["none"] == max(rates.values())

    def test_sucrose_most_stable_shelf_life(self):
        """Sucrose has factor 0.3 — should give longest shelf life."""
        excipients = ["none", "sucrose", "polysorbate", "arginine", "histidine"]
        shelf = {exc: _base_result(excipient=exc).shelf_life_years for exc in excipients}
        assert shelf["sucrose"] == max(shelf.values())


# ---------------------------------------------------------------------------
# pH effect
# ---------------------------------------------------------------------------


class TestPhEffect:
    def test_extreme_low_ph_faster_aggregation(self):
        r_neutral = _base_result(formulation_ph=7.0)
        r_acid = _base_result(formulation_ph=4.0)
        assert r_acid.aggregation_rate_per_h > r_neutral.aggregation_rate_per_h

    def test_extreme_high_ph_faster_aggregation(self):
        r_neutral = _base_result(formulation_ph=7.0)
        r_basic = _base_result(formulation_ph=10.0)
        assert r_basic.aggregation_rate_per_h > r_neutral.aggregation_rate_per_h

    def test_optimal_ph_is_7(self):
        result = _base_result()
        assert result.optimal_ph == 7.0

    def test_symmetric_ph_effect(self):
        """pH 6.5 and 7.5 should give same ph_factor due to symmetric U-shape."""
        r_low = _base_result(formulation_ph=6.5)
        r_high = _base_result(formulation_ph=7.5)
        assert abs(r_low.aggregation_rate_per_h - r_high.aggregation_rate_per_h) < 1e-12


# ---------------------------------------------------------------------------
# Ionic strength effect
# ---------------------------------------------------------------------------


class TestIonicStrengthEffect:
    def test_higher_is_faster_aggregation(self):
        r_low = _base_result(ionic_strength_mM=100.0)
        r_high = _base_result(ionic_strength_mM=500.0)
        assert r_high.aggregation_rate_per_h > r_low.aggregation_rate_per_h

    def test_is_below_150_no_extra_effect(self):
        """IS <= 150 mM has no additional IS factor (factor=1.0 baseline)."""
        r_100 = _base_result(ionic_strength_mM=100.0)
        r_150 = _base_result(ionic_strength_mM=150.0)
        assert abs(r_100.aggregation_rate_per_h - r_150.aggregation_rate_per_h) < 1e-12


# ---------------------------------------------------------------------------
# Temperature / Arrhenius
# ---------------------------------------------------------------------------


class TestTemperatureEffect:
    def test_higher_temperature_faster_aggregation(self):
        r_25 = _base_result(temperature_C=25.0)
        r_40 = _base_result(temperature_C=40.0)
        assert r_40.aggregation_rate_per_h > r_25.aggregation_rate_per_h

    def test_shelf_life_at_4C_longer_than_25C(self):
        """Shelf life is calculated at 4 °C, so it should be longer than t10 at 25 °C."""
        result = _base_result(temperature_C=25.0)
        t10_at_25C_years = result.t10_h / 8760.0
        assert result.shelf_life_years > t10_at_25C_years

    def test_lower_temperature_more_stable(self):
        r_4 = _base_result(temperature_C=4.0)
        r_25 = _base_result(temperature_C=25.0)
        assert r_4.aggregation_rate_per_h < r_25.aggregation_rate_per_h


# ---------------------------------------------------------------------------
# t50 and t10 consistency
# ---------------------------------------------------------------------------


class TestTimepointConsistency:
    def test_t50_greater_than_t10(self):
        result = _base_result()
        assert result.t50_h > result.t10_h

    def test_t10_positive(self):
        result = _base_result()
        assert result.t10_h > 0

    def test_t50_positive(self):
        result = _base_result()
        assert result.t50_h > 0

    def test_shelf_life_positive(self):
        result = _base_result()
        assert result.shelf_life_years > 0


# ---------------------------------------------------------------------------
# screen_excipients_aggregation
# ---------------------------------------------------------------------------


class TestScreenExcipients:
    def test_returns_5_results(self):
        results = screen_excipients_aggregation("mAb-X", 7.0)
        assert len(results) == 5

    def test_all_excipients_represented(self):
        results = screen_excipients_aggregation("mAb-X", 7.0)
        excipients = {r.excipient for r in results}
        assert excipients == {"none", "sucrose", "polysorbate", "arginine", "histidine"}

    def test_sorted_by_shelf_life_descending(self):
        results = screen_excipients_aggregation("mAb-X", 7.0)
        shelf_lives = [r.shelf_life_years for r in results]
        assert shelf_lives == sorted(shelf_lives, reverse=True)

    def test_sucrose_first_in_screen(self):
        """Sucrose (factor 0.3) should have the longest shelf life."""
        results = screen_excipients_aggregation("mAb-X", 7.0)
        assert results[0].excipient == "sucrose"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_ph_low_raises(self):
        with pytest.raises(ValueError, match="formulation_ph"):
            simulate_formulation_aggregation("X", formulation_ph=3.0, ionic_strength_mM=150.0)

    def test_invalid_ph_high_raises(self):
        with pytest.raises(ValueError, match="formulation_ph"):
            simulate_formulation_aggregation("X", formulation_ph=11.0, ionic_strength_mM=150.0)

    def test_negative_ionic_strength_raises(self):
        with pytest.raises(ValueError, match="ionic_strength_mM"):
            simulate_formulation_aggregation("X", formulation_ph=7.0, ionic_strength_mM=-10.0)

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature_C"):
            simulate_formulation_aggregation(
                "X", formulation_ph=7.0, ionic_strength_mM=150.0, temperature_C=90.0
            )

    def test_invalid_excipient_raises(self):
        with pytest.raises(ValueError, match="excipient"):
            simulate_formulation_aggregation(
                "X", formulation_ph=7.0, ionic_strength_mM=150.0, excipient="invalid_excipient"
            )
