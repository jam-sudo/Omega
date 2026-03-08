"""Tests for Phase 343 — Inhalation Toxicology Dosimetry."""

import pytest

from omega_pbpk.risk.inhalation_toxicology_dosimetry import (
    InhalationDosimetryResult,
    calculate_inhalation_dosimetry,
    compare_exposure_durations,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_result():
    return calculate_inhalation_dosimetry(
        chemical_name="TestChem",
        airborne_conc_mg_per_m3=1.0,
        exposure_duration_h=8.0,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type(base_result):
    assert isinstance(base_result, InhalationDosimetryResult)


def test_result_is_frozen(base_result):
    with pytest.raises((AttributeError, TypeError)):
        base_result.chemical_name = "hacked"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Basic physics
# ---------------------------------------------------------------------------


def test_total_inhaled_positive(base_result):
    assert base_result.total_inhaled_mg_per_h > 0


def test_alveolar_le_total_inhaled(base_result):
    assert base_result.alveolar_deposited_mg_per_h <= base_result.total_inhaled_mg_per_h


def test_absorbed_le_alveolar(base_result):
    assert base_result.absorbed_systemic_mg_per_h <= base_result.alveolar_deposited_mg_per_h


def test_hec_positive(base_result):
    assert base_result.human_equivalent_conc_mg_per_kg > 0


def test_moe_equals_noael_over_hec():
    noael = 50.0
    result = calculate_inhalation_dosimetry(
        chemical_name="A",
        airborne_conc_mg_per_m3=2.0,
        exposure_duration_h=4.0,
        noael_mg_per_kg=noael,
    )
    assert abs(result.moe - noael / result.human_equivalent_conc_mg_per_kg) < 1e-9


def test_risk_class_valid(base_result):
    assert base_result.risk_class in {"low", "moderate", "high"}


# ---------------------------------------------------------------------------
# Risk classification boundaries
# ---------------------------------------------------------------------------


def test_low_risk_high_noael():
    # Very high NOAEL → MOE > 1000 → low risk
    result = calculate_inhalation_dosimetry(
        chemical_name="Safe",
        airborne_conc_mg_per_m3=0.001,
        exposure_duration_h=1.0,
        noael_mg_per_kg=10_000.0,
    )
    assert result.risk_class == "low"
    assert result.moe > 1000


def test_high_risk_low_noael():
    # Very low NOAEL → MOE < 100 → high risk
    result = calculate_inhalation_dosimetry(
        chemical_name="Toxic",
        airborne_conc_mg_per_m3=100.0,
        exposure_duration_h=24.0,
        noael_mg_per_kg=0.1,
    )
    assert result.risk_class == "high"
    assert result.moe < 100


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


def test_higher_conc_higher_absorbed():
    low = calculate_inhalation_dosimetry("C", 1.0, 8.0)
    high = calculate_inhalation_dosimetry("C", 10.0, 8.0)
    assert high.total_absorbed_mg > low.total_absorbed_mg


def test_higher_duration_higher_total_absorbed():
    short = calculate_inhalation_dosimetry("C", 1.0, 4.0)
    long_ = calculate_inhalation_dosimetry("C", 1.0, 8.0)
    assert long_.total_absorbed_mg > short.total_absorbed_mg


# ---------------------------------------------------------------------------
# Minute ventilation formula
# ---------------------------------------------------------------------------


def test_minute_ventilation_formula():
    result = calculate_inhalation_dosimetry(
        chemical_name="X",
        airborne_conc_mg_per_m3=1.0,
        exposure_duration_h=1.0,
        tidal_volume_mL=600.0,
        respiratory_rate_per_min=20.0,
    )
    expected_mv = 600.0 * 20.0 / 1000.0
    assert abs(result.minute_ventilation_L_per_min - expected_mv) < 1e-9


# ---------------------------------------------------------------------------
# compare_exposure_durations
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    results = compare_exposure_durations("X", 1.0, [1.0, 4.0, 8.0])
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_sorted_ascending():
    results = compare_exposure_durations("X", 1.0, [8.0, 1.0, 4.0])
    absorbed = [r.total_absorbed_mg for r in results]
    assert absorbed == sorted(absorbed)


def test_compare_all_same_chemical():
    results = compare_exposure_durations("ChemA", 2.0, [2.0, 6.0])
    for r in results:
        assert r.chemical_name == "ChemA"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_airborne_conc_zero():
    with pytest.raises(ValueError, match="airborne_conc_mg_per_m3"):
        calculate_inhalation_dosimetry("X", 0.0, 8.0)


def test_validation_airborne_conc_negative():
    with pytest.raises(ValueError, match="airborne_conc_mg_per_m3"):
        calculate_inhalation_dosimetry("X", -1.0, 8.0)


def test_validation_duration_zero():
    with pytest.raises(ValueError, match="exposure_duration_h"):
        calculate_inhalation_dosimetry("X", 1.0, 0.0)


def test_validation_duration_negative():
    with pytest.raises(ValueError, match="exposure_duration_h"):
        calculate_inhalation_dosimetry("X", 1.0, -5.0)


def test_validation_deposition_fractions_wrong_sum():
    with pytest.raises(ValueError, match="sum to 1.0"):
        calculate_inhalation_dosimetry(
            "X",
            1.0,
            8.0,
            regional_deposition_fraction={"et": 0.5, "tb": 0.5, "al": 0.5},
        )


def test_validation_body_weight_zero():
    with pytest.raises(ValueError, match="body_weight_kg"):
        calculate_inhalation_dosimetry("X", 1.0, 8.0, body_weight_kg=0.0)


def test_validation_body_weight_negative():
    with pytest.raises(ValueError, match="body_weight_kg"):
        calculate_inhalation_dosimetry("X", 1.0, 8.0, body_weight_kg=-70.0)
