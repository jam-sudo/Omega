"""Tests for Phase 213 — amide bond stability prediction."""

import math

import pytest

from omega_pbpk.prediction.amide_bond_stability import (
    AmideStabilityResult,
    compare_ph_stability,
    predict_amide_stability,
)

DRUG = "TestDrug"

# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_amide_stability_result(self):
        r = predict_amide_stability(DRUG, ph=7.4, temperature_C=37.0)
        assert isinstance(r, AmideStabilityResult)

    def test_drug_name_preserved(self):
        r = predict_amide_stability("MyDrug", ph=7.4, temperature_C=37.0)
        assert r.drug_name == "MyDrug"

    def test_ph_preserved(self):
        r = predict_amide_stability(DRUG, ph=6.8, temperature_C=37.0)
        assert r.ph == pytest.approx(6.8)

    def test_temperature_preserved(self):
        r = predict_amide_stability(DRUG, ph=7.4, temperature_C=25.0)
        assert r.temperature_C == pytest.approx(25.0)

    def test_flanking_groups_preserved(self):
        r = predict_amide_stability(
            DRUG, ph=7.4, temperature_C=37.0, flanking_groups="electron_withdrawing"
        )
        assert r.flanking_groups == "electron_withdrawing"

    def test_notes_preserved(self):
        r = predict_amide_stability(DRUG, ph=7.4, temperature_C=37.0, notes="batch 42")
        assert r.notes == "batch 42"


# ---------------------------------------------------------------------------
# Positivity checks
# ---------------------------------------------------------------------------


class TestPositivity:
    def test_k_hydrolysis_positive(self):
        r = predict_amide_stability(DRUG, ph=7.4, temperature_C=37.0)
        assert r.k_hydrolysis_per_h > 0

    def test_t_half_positive(self):
        r = predict_amide_stability(DRUG, ph=7.4, temperature_C=37.0)
        assert r.t_half_h > 0

    def test_k_t_half_consistent(self):
        r = predict_amide_stability(DRUG, ph=7.4, temperature_C=37.0)
        expected = math.log(2) / r.k_hydrolysis_per_h
        assert r.t_half_h == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# pH effect
# ---------------------------------------------------------------------------


class TestPhEffect:
    def test_ph74_lower_k_than_ph2(self):
        r74 = predict_amide_stability(DRUG, ph=7.4, temperature_C=37.0)
        r2 = predict_amide_stability(DRUG, ph=2.0, temperature_C=37.0)
        assert r74.k_hydrolysis_per_h < r2.k_hydrolysis_per_h

    def test_ph74_lower_k_than_ph12(self):
        r74 = predict_amide_stability(DRUG, ph=7.4, temperature_C=37.0)
        r12 = predict_amide_stability(DRUG, ph=12.0, temperature_C=37.0)
        assert r74.k_hydrolysis_per_h < r12.k_hydrolysis_per_h

    def test_extreme_acid_base_symmetric_around_74(self):
        # |pH - 7.4| identical for pH=5.4 and pH=9.4 → same k
        r5 = predict_amide_stability(DRUG, ph=5.4, temperature_C=37.0)
        r9 = predict_amide_stability(DRUG, ph=9.4, temperature_C=37.0)
        assert r5.k_hydrolysis_per_h == pytest.approx(r9.k_hydrolysis_per_h, rel=1e-6)


# ---------------------------------------------------------------------------
# Temperature effect
# ---------------------------------------------------------------------------


class TestTemperatureEffect:
    def test_higher_temp_higher_k(self):
        r37 = predict_amide_stability(DRUG, ph=7.4, temperature_C=37.0)
        r60 = predict_amide_stability(DRUG, ph=7.4, temperature_C=60.0)
        assert r60.k_hydrolysis_per_h > r37.k_hydrolysis_per_h

    def test_lower_temp_lower_k(self):
        r37 = predict_amide_stability(DRUG, ph=7.4, temperature_C=37.0)
        r4 = predict_amide_stability(DRUG, ph=7.4, temperature_C=4.0)
        assert r4.k_hydrolysis_per_h < r37.k_hydrolysis_per_h


# ---------------------------------------------------------------------------
# Flanking group effect
# ---------------------------------------------------------------------------


class TestFlankingGroup:
    def test_electron_withdrawing_faster_than_none(self):
        rew = predict_amide_stability(
            DRUG, ph=7.4, temperature_C=37.0, flanking_groups="electron_withdrawing"
        )
        rnone = predict_amide_stability(DRUG, ph=7.4, temperature_C=37.0, flanking_groups="none")
        assert rew.k_hydrolysis_per_h > rnone.k_hydrolysis_per_h

    def test_electron_donating_slower_than_none(self):
        red = predict_amide_stability(
            DRUG, ph=7.4, temperature_C=37.0, flanking_groups="electron_donating"
        )
        rnone = predict_amide_stability(DRUG, ph=7.4, temperature_C=37.0, flanking_groups="none")
        assert red.k_hydrolysis_per_h < rnone.k_hydrolysis_per_h

    def test_electron_withdrawing_5x_faster_than_none(self):
        rew = predict_amide_stability(
            DRUG, ph=7.4, temperature_C=37.0, flanking_groups="electron_withdrawing"
        )
        rnone = predict_amide_stability(DRUG, ph=7.4, temperature_C=37.0, flanking_groups="none")
        assert rew.k_hydrolysis_per_h == pytest.approx(5.0 * rnone.k_hydrolysis_per_h, rel=1e-6)

    def test_electron_donating_03x_none(self):
        red = predict_amide_stability(
            DRUG, ph=7.4, temperature_C=37.0, flanking_groups="electron_donating"
        )
        rnone = predict_amide_stability(DRUG, ph=7.4, temperature_C=37.0, flanking_groups="none")
        assert red.k_hydrolysis_per_h == pytest.approx(0.3 * rnone.k_hydrolysis_per_h, rel=1e-6)


# ---------------------------------------------------------------------------
# Percentage remaining
# ---------------------------------------------------------------------------


class TestPctRemaining:
    def test_pct_remaining_24h_in_range(self):
        r = predict_amide_stability(DRUG, ph=7.4, temperature_C=37.0)
        assert 0.0 <= r.pct_remaining_24h <= 100.0

    def test_pct_remaining_7d_in_range(self):
        r = predict_amide_stability(DRUG, ph=7.4, temperature_C=37.0)
        assert 0.0 <= r.pct_remaining_7d <= 100.0

    def test_pct_7d_less_than_pct_24h(self):
        r = predict_amide_stability(DRUG, ph=7.4, temperature_C=37.0)
        assert r.pct_remaining_7d < r.pct_remaining_24h

    def test_pct_remaining_24h_near_100_at_ref_conditions(self):
        # At pH 7.4, 37 °C, k~1e-6 /h → negligible loss in 24 h
        r = predict_amide_stability(DRUG, ph=7.4, temperature_C=37.0)
        assert r.pct_remaining_24h > 99.99


# ---------------------------------------------------------------------------
# Stability class
# ---------------------------------------------------------------------------

VALID_CLASSES = {"highly_stable", "stable", "labile", "very_labile"}


class TestStabilityClass:
    def test_stability_class_valid(self):
        r = predict_amide_stability(DRUG, ph=7.4, temperature_C=37.0)
        assert r.stability_class in VALID_CLASSES

    def test_ref_conditions_highly_stable(self):
        r = predict_amide_stability(DRUG, ph=7.4, temperature_C=37.0)
        assert r.stability_class == "highly_stable"

    def test_very_labile_at_extreme_conditions(self):
        # Very high temperature and extreme pH → very_labile
        r = predict_amide_stability(
            DRUG, ph=1.0, temperature_C=80.0, flanking_groups="electron_withdrawing"
        )
        assert r.stability_class == "very_labile"


# ---------------------------------------------------------------------------
# compare_ph_stability
# ---------------------------------------------------------------------------


class TestComparePhStability:
    def test_returns_list(self):
        results = compare_ph_stability(DRUG, [7.4, 2.0, 12.0], temperature_C=37.0)
        assert isinstance(results, list)

    def test_length_matches_input(self):
        ph_values = [5.0, 7.4, 9.0]
        results = compare_ph_stability(DRUG, ph_values, temperature_C=37.0)
        assert len(results) == 3

    def test_sorted_by_t_half_descending(self):
        results = compare_ph_stability(DRUG, [2.0, 7.4, 12.0], temperature_C=37.0)
        t_halves = [r.t_half_h for r in results]
        assert t_halves == sorted(t_halves, reverse=True)

    def test_ph74_first_in_sorted(self):
        # pH 7.4 is most stable → appears first
        results = compare_ph_stability(DRUG, [2.0, 7.4, 12.0], temperature_C=37.0)
        assert results[0].ph == pytest.approx(7.4)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_temperature_below_absolute_zero(self):
        with pytest.raises(ValueError, match="[Tt]emperature"):
            predict_amide_stability(DRUG, ph=7.4, temperature_C=-300.0)

    def test_invalid_flanking_group(self):
        with pytest.raises(ValueError, match="flanking_groups"):
            predict_amide_stability(DRUG, ph=7.4, temperature_C=37.0, flanking_groups="bad_value")
