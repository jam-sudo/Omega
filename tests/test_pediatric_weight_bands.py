"""Tests for clinical/pediatric_weight_bands.py — Phase 527."""

import pytest

from omega_pbpk.clinical.pediatric_weight_bands import (
    PediatricRegimenResult,
    generate_weight_band_table,
    pediatric_dosing_regimen,
    round_to_nearest,
    weight_band_dose,
)

# ---------------------------------------------------------------------------
# round_to_nearest
# ---------------------------------------------------------------------------


class TestRoundToNearest:
    def test_round_to_50_from_options(self):
        assert round_to_nearest(47.0, [25.0, 50.0, 100.0]) == 50.0

    def test_round_to_25(self):
        assert round_to_nearest(30.0, [25.0, 50.0, 100.0]) == 25.0

    def test_round_to_100(self):
        assert round_to_nearest(80.0, [25.0, 50.0, 100.0]) == 100.0

    def test_exact_match(self):
        assert round_to_nearest(50.0, [25.0, 50.0, 100.0]) == 50.0

    def test_single_option(self):
        assert round_to_nearest(999.0, [250.0]) == 250.0

    def test_tie_rounds_up(self):
        # Midpoint between 50 and 100 → should pick 100 (round half up)
        assert round_to_nearest(75.0, [50.0, 100.0]) == 100.0

    def test_empty_options_raises(self):
        with pytest.raises(ValueError):
            round_to_nearest(50.0, [])


# ---------------------------------------------------------------------------
# weight_band_dose
# ---------------------------------------------------------------------------


class TestWeightBandDose:
    def test_basic_calculation(self):
        result = weight_band_dose(10.0, 5.0)
        assert result["target_dose_mg"] == pytest.approx(50.0)
        assert result["rounded_dose_mg"] == pytest.approx(50.0)
        assert result["weight_kg"] == 10.0

    def test_rounding_to_nearest_option(self):
        # 15 kg × 5 mg/kg = 75 → nearest of [50, 100, 150] = 100
        result = weight_band_dose(15.0, 5.0, dose_options_mg=[50.0, 100.0, 150.0])
        assert result["target_dose_mg"] == pytest.approx(75.0)
        assert result["rounded_dose_mg"] == pytest.approx(100.0)

    def test_max_adult_dose_cap(self):
        # 70 kg × 10 mg/kg = 700 → capped at 500
        result = weight_band_dose(70.0, 10.0, max_adult_dose_mg=500.0)
        assert result["target_dose_mg"] == pytest.approx(500.0)
        assert result["rounded_dose_mg"] == pytest.approx(500.0)

    def test_max_adult_dose_not_triggered(self):
        # 10 kg × 5 mg/kg = 50 → well below cap of 500
        result = weight_band_dose(10.0, 5.0, max_adult_dose_mg=500.0)
        assert result["target_dose_mg"] == pytest.approx(50.0)

    def test_actual_mg_per_kg(self):
        result = weight_band_dose(20.0, 5.0, dose_options_mg=[50.0, 100.0, 150.0])
        assert result["actual_mg_per_kg"] == pytest.approx(result["rounded_dose_mg"] / 20.0)

    def test_within_10pct_true(self):
        # 10 kg × 10 mg/kg = 100 → rounded 100, exact match → True
        result = weight_band_dose(10.0, 10.0, dose_options_mg=[100.0])
        assert result["within_10pct"] is True

    def test_within_10pct_false(self):
        # target 85, rounded 100 → (100-85)/85 ≈ 17.6% → False
        result = weight_band_dose(17.0, 5.0, dose_options_mg=[100.0])
        # target = 17 × 5 = 85, rounded = 100, deviation ≈ 17.6%
        assert result["within_10pct"] is False

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            weight_band_dose(-5.0, 5.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            weight_band_dose(10.0, 0.0)

    def test_rounding_and_cap_combined(self):
        # 60 kg × 10 = 600 → cap 500, then round to [250, 500] → 500
        result = weight_band_dose(
            60.0, 10.0, dose_options_mg=[250.0, 500.0], max_adult_dose_mg=500.0
        )
        assert result["target_dose_mg"] == pytest.approx(500.0)
        assert result["rounded_dose_mg"] == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# generate_weight_band_table
# ---------------------------------------------------------------------------


class TestGenerateWeightBandTable:
    def test_default_who_bands_count(self):
        table = generate_weight_band_table(5.0)
        assert len(table) == 10  # 10 WHO weight bands

    def test_custom_bands(self):
        bands = [(5, 9.9), (10, 19.9), (20, 29.9)]
        table = generate_weight_band_table(5.0, weight_bands_kg=bands)
        assert len(table) == 3

    def test_doses_increase_with_weight(self):
        table = generate_weight_band_table(5.0)
        # Target doses should increase as weight bands increase
        targets = [row["target_dose_mg"] for row in table]
        for i in range(len(targets) - 1):
            assert targets[i] < targets[i + 1], (
                f"Doses should increase: band {i} ({targets[i]}) >= band {i + 1} ({targets[i + 1]})"
            )

    def test_daily_dose_is_freq_times_rounded(self):
        table = generate_weight_band_table(5.0, freq_per_day=3)
        for row in table:
            assert row["daily_dose_mg"] == pytest.approx(row["rounded_dose_mg"] * 3)

    def test_weight_band_str_format(self):
        table = generate_weight_band_table(5.0)
        assert "kg" in table[0]["weight_band_str"]

    def test_representative_weight_is_midpoint(self):
        bands = [(10, 20)]
        table = generate_weight_band_table(5.0, weight_bands_kg=bands)
        assert table[0]["representative_weight_kg"] == pytest.approx(15.0)

    def test_adult_dose_cap_applied(self):
        # With cap of 50 mg and high dose per kg, large bands should be capped
        table = generate_weight_band_table(10.0, max_adult_dose_mg=50.0)
        last = table[-1]
        assert last["target_dose_mg"] == pytest.approx(50.0)

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            generate_weight_band_table(-1.0)


# ---------------------------------------------------------------------------
# pediatric_dosing_regimen
# ---------------------------------------------------------------------------


class TestPediatricDosingRegimen:
    def test_returns_result_type(self):
        result = pediatric_dosing_regimen("Amoxicillin", 25.0, [125.0, 250.0, 500.0], 500.0)
        assert isinstance(result, PediatricRegimenResult)

    def test_drug_name_stored(self):
        result = pediatric_dosing_regimen("Paracetamol", 15.0, [120.0, 250.0, 500.0], 1000.0)
        assert result.drug_name == "Paracetamol"

    def test_weight_bands_populated(self):
        result = pediatric_dosing_regimen("DrugX", 10.0, [50.0, 100.0, 200.0], 200.0)
        assert len(result.weight_bands) == 10

    def test_freq_and_route_stored(self):
        result = pediatric_dosing_regimen(
            "DrugY", 5.0, [50.0, 100.0], 200.0, freq_per_day=3, route="iv"
        )
        assert result.freq_per_day == 3
        assert result.route == "iv"

    def test_dose_options_stored(self):
        opts = [25.0, 50.0, 100.0]
        result = pediatric_dosing_regimen("DrugZ", 5.0, opts, 100.0)
        assert result.dose_options_mg == opts

    def test_notes_generated(self):
        result = pediatric_dosing_regimen("DrugA", 5.0, [25.0, 50.0, 100.0], 500.0)
        assert isinstance(result.notes, list)

    def test_pipeline_complete(self):
        """Full end-to-end check for a realistic drug."""
        result = pediatric_dosing_regimen(
            "Azithromycin",
            10.0,
            [100.0, 200.0, 500.0],
            500.0,
            freq_per_day=1,
            route="oral",
        )
        assert result.drug_name == "Azithromycin"
        assert result.dose_mg_per_kg == 10.0
        assert all("rounded_dose_mg" in row for row in result.weight_bands)
        # All rounded doses should be in the options list
        for row in result.weight_bands:
            assert row["rounded_dose_mg"] in [100.0, 200.0, 500.0]
