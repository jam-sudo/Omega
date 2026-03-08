"""Tests for Phase 1032 — Drug Photodegradation Kinetics."""

import pytest

from omega_pbpk.prediction.photodegradation_kinetics import (
    PhotodegradationResult,
    predict_photodegradation,
    screen_packaging_conditions,
)

_VALID_CLASSES = {"stable", "slightly_unstable", "unstable", "very_unstable"}


# ── Return type ────────────────────────────────────────────────────────────


def test_returns_correct_type():
    result = predict_photodegradation("DrugA", mw_Da=300.0)
    assert isinstance(result, PhotodegradationResult)


def test_result_is_frozen():
    result = predict_photodegradation("DrugA", mw_Da=300.0)
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Modified"  # type: ignore[misc]


# ── Fraction remaining ─────────────────────────────────────────────────────


def test_fraction_remaining_in_range():
    result = predict_photodegradation("DrugA", mw_Da=300.0)
    assert 0.0 <= result.fraction_remaining <= 1.0


def test_fraction_remaining_short_exposure():
    result = predict_photodegradation(
        "DrugA", mw_Da=300.0, chromophore="none", light_source="incandescent", exposure_h=1.0
    )
    assert result.fraction_remaining > 0.99


# ── Half-life and t90 ─────────────────────────────────────────────────────


def test_t50_positive():
    result = predict_photodegradation("DrugA", mw_Da=300.0)
    assert result.t50_h > 0.0


def test_t90_greater_than_t50():
    result = predict_photodegradation("DrugA", mw_Da=300.0)
    # t90 loss is ln(10)/k, t50 is ln(2)/k → ratio ~3.32
    assert result.t90_percent_loss_h > result.t50_h


def test_t90_approx_332_times_t50():
    result = predict_photodegradation("DrugA", mw_Da=300.0)
    ratio = result.t90_percent_loss_h / result.t50_h
    assert abs(ratio - 3.321928) < 0.01


# ── Photostability class ───────────────────────────────────────────────────


def test_photostability_class_valid():
    result = predict_photodegradation("DrugA", mw_Da=300.0)
    assert result.photostability_class in _VALID_CLASSES


def test_protection_needed_consistent_with_class():
    result = predict_photodegradation("DrugA", mw_Da=300.0)
    expected = result.photostability_class != "stable"
    assert result.protection_needed == expected


def test_no_chromophore_incandescent_is_stable():
    result = predict_photodegradation(
        "Stable", mw_Da=200.0, chromophore="none", light_source="incandescent", exposure_h=24.0
    )
    assert result.photostability_class == "stable"


# ── Quantum yield ─────────────────────────────────────────────────────────


def test_quantum_yield_in_range():
    result = predict_photodegradation("DrugA", mw_Da=300.0)
    assert 0.0 <= result.quantum_yield_estimate <= 1.0


# ── Chromophore effect ────────────────────────────────────────────────────


def test_nitro_lower_fraction_than_none():
    r_none = predict_photodegradation(
        "D", mw_Da=300.0, chromophore="none", light_source="fluorescent"
    )
    r_nitro = predict_photodegradation(
        "D", mw_Da=300.0, chromophore="nitro", light_source="fluorescent"
    )
    assert r_nitro.fraction_remaining < r_none.fraction_remaining


# ── Light source effect ────────────────────────────────────────────────────


def test_uv_lower_fraction_than_fluorescent():
    r_fl = predict_photodegradation(
        "D", mw_Da=300.0, chromophore="aromatic", light_source="fluorescent"
    )
    r_uv = predict_photodegradation("D", mw_Da=300.0, chromophore="aromatic", light_source="UV")
    assert r_uv.fraction_remaining < r_fl.fraction_remaining


# ── Oxygen effect ─────────────────────────────────────────────────────────


def test_oxygen_reduces_fraction_remaining():
    r_no_o2 = predict_photodegradation(
        "D", mw_Da=300.0, chromophore="aromatic", oxygen_present=False
    )
    r_o2 = predict_photodegradation("D", mw_Da=300.0, chromophore="aromatic", oxygen_present=True)
    assert r_o2.fraction_remaining < r_no_o2.fraction_remaining


# ── Antioxidant effect ────────────────────────────────────────────────────


def test_antioxidant_increases_fraction_remaining():
    r_no_aox = predict_photodegradation(
        "D", mw_Da=300.0, chromophore="phenol", antioxidant_conc_mM=0.0
    )
    r_aox = predict_photodegradation(
        "D", mw_Da=300.0, chromophore="phenol", antioxidant_conc_mM=5.0
    )
    assert r_aox.fraction_remaining > r_no_aox.fraction_remaining


# ── Notes ─────────────────────────────────────────────────────────────────


def test_notes_nonempty():
    result = predict_photodegradation("DrugA", mw_Da=300.0)
    assert len(result.notes) > 0


# ── screen_packaging_conditions ───────────────────────────────────────────


def test_screen_returns_four_results():
    results = screen_packaging_conditions("DrugB", mw_Da=400.0)
    assert len(results) == 4


def test_screen_sorted_descending():
    results = screen_packaging_conditions("DrugB", mw_Da=400.0)
    fractions = [r.fraction_remaining for r in results]
    assert fractions == sorted(fractions, reverse=True)


def test_screen_all_photodegradation_results():
    results = screen_packaging_conditions("DrugB", mw_Da=400.0)
    for r in results:
        assert isinstance(r, PhotodegradationResult)


# ── Validation errors ──────────────────────────────────────────────────────


def test_conc_zero_raises():
    with pytest.raises(ValueError, match="initial_conc_mg_mL"):
        predict_photodegradation("Drug", mw_Da=300.0, initial_conc_mg_mL=0.0)


def test_exposure_zero_raises():
    with pytest.raises(ValueError, match="exposure_h"):
        predict_photodegradation("Drug", mw_Da=300.0, exposure_h=0.0)


def test_invalid_chromophore_raises():
    with pytest.raises(ValueError, match="chromophore"):
        predict_photodegradation("Drug", mw_Da=300.0, chromophore="invalid")


def test_invalid_light_source_raises():
    with pytest.raises(ValueError, match="light_source"):
        predict_photodegradation("Drug", mw_Da=300.0, light_source="laser")


def test_negative_antioxidant_raises():
    with pytest.raises(ValueError, match="antioxidant_conc_mM"):
        predict_photodegradation("Drug", mw_Da=300.0, antioxidant_conc_mM=-1.0)
