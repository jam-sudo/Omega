"""Tests for pediatric weight-band dosing table — Phase 270."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.pediatric_weight_dosing import (
    DosingTable,
    generate_dosing_table,
    round_to_practical_dose,
)

# ---------------------------------------------------------------------------
# round_to_practical_dose
# ---------------------------------------------------------------------------


def test_round_small_value_to_half():
    assert round_to_practical_dose(0.3) == 0.5


def test_round_1_2_to_1():
    assert round_to_practical_dose(1.2) == 1.0


def test_round_2_8_to_3():
    # 2.8 is closer to 3.0 (diff 0.2) than to 2.5 (diff 0.3)
    assert round_to_practical_dose(2.8) == 3.0


def test_round_8_to_7_5():
    assert round_to_practical_dose(8.0) == 7.5


def test_round_11_to_10():
    assert round_to_practical_dose(11.0) == 10.0


def test_round_zero_returns_zero():
    assert round_to_practical_dose(0.0) == 0.0


def test_round_negative_returns_zero():
    assert round_to_practical_dose(-5.0) == 0.0


# ---------------------------------------------------------------------------
# Input validation for generate_dosing_table
# ---------------------------------------------------------------------------


def test_mg_per_kg_zero_raises():
    with pytest.raises(ValueError, match="mg_per_kg must be > 0"):
        generate_dosing_table("Drug", mg_per_kg=0.0)


def test_mg_per_kg_negative_raises():
    with pytest.raises(ValueError, match="mg_per_kg must be > 0"):
        generate_dosing_table("Drug", mg_per_kg=-1.0)


def test_frequency_zero_raises():
    with pytest.raises(ValueError, match="frequency_per_day must be >= 1"):
        generate_dosing_table("Drug", mg_per_kg=5.0, frequency_per_day=0)


def test_min_dose_zero_raises():
    with pytest.raises(ValueError, match="min_dose_mg must be > 0"):
        generate_dosing_table("Drug", mg_per_kg=5.0, min_dose_mg=0.0)


# ---------------------------------------------------------------------------
# Default weight bands
# ---------------------------------------------------------------------------


def test_default_nine_weight_bands():
    table = generate_dosing_table("Amoxicillin", mg_per_kg=25.0)
    assert len(table.weight_bands) == 9


def test_each_band_rounded_dose_gte_min_dose():
    table = generate_dosing_table("Drug", mg_per_kg=5.0, min_dose_mg=2.5)
    for band in table.weight_bands:
        assert band.rounded_dose_mg >= 2.5


def test_total_daily_dose_equals_rounded_times_frequency():
    table = generate_dosing_table("Drug", mg_per_kg=10.0, frequency_per_day=3)
    for band in table.weight_bands:
        expected = band.rounded_dose_mg * band.frequency_per_day
        assert abs(band.total_daily_dose_mg - expected) < 1e-9


def test_mg_per_kg_actual_positive():
    table = generate_dosing_table("Drug", mg_per_kg=5.0)
    for band in table.weight_bands:
        assert band.mg_per_kg_actual > 0.0


def test_max_dose_cap_applied():
    table = generate_dosing_table("Drug", mg_per_kg=50.0, max_dose_mg=100.0)
    for band in table.weight_bands:
        assert band.rounded_dose_mg <= 100.0


def test_representative_weight_is_midpoint():
    table = generate_dosing_table("Drug", mg_per_kg=5.0)
    for band in table.weight_bands:
        expected_mid = (band.weight_min_kg + band.weight_max_kg) / 2.0
        assert abs(band.representative_weight_kg - expected_mid) < 1e-9


def test_drug_name_stored():
    table = generate_dosing_table("Ibuprofen", mg_per_kg=10.0)
    assert table.drug_name == "Ibuprofen"


# ---------------------------------------------------------------------------
# Custom weight bands
# ---------------------------------------------------------------------------


def test_custom_weight_bands_accepted():
    custom_bands = [(5.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    table = generate_dosing_table("Drug", mg_per_kg=5.0, weight_bands_kg=custom_bands)
    assert len(table.weight_bands) == 3


# ---------------------------------------------------------------------------
# Higher mg/kg → higher rounded doses
# ---------------------------------------------------------------------------


def test_higher_mg_per_kg_gives_higher_doses():
    low = generate_dosing_table("Drug", mg_per_kg=2.0)
    high = generate_dosing_table("Drug", mg_per_kg=20.0)
    low_doses = [b.rounded_dose_mg for b in low.weight_bands]
    high_doses = [b.rounded_dose_mg for b in high.weight_bands]
    # At least most bands should have higher doses at 20 mg/kg
    higher_count = sum(h > lo for h, lo in zip(high_doses, low_doses))
    assert higher_count >= len(low_doses) - 1


def test_result_is_dosing_table():
    result = generate_dosing_table("Drug", mg_per_kg=5.0)
    assert isinstance(result, DosingTable)


def test_frequency_stored_in_table():
    table = generate_dosing_table("Drug", mg_per_kg=5.0, frequency_per_day=3)
    assert table.frequency_per_day == 3
    for band in table.weight_bands:
        assert band.frequency_per_day == 3
