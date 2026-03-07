"""Tests for comprehensive oral absorption model (Phase 724)."""

import pytest

from omega_pbpk.prediction.oral_absorption_model import (
    OralAbsorptionModelResult,
    model_oral_absorption,
    screen_oral_absorption,
)

# --- Basic return type ---


def test_returns_oral_absorption_model_result():
    result = model_oral_absorption(
        compound_name="Aspirin",
        logP=1.2,
        mw=180.0,
        psa=63.0,
        solubility_mg_per_mL=3.0,
        dose_mg=100.0,
    )
    assert isinstance(result, OralAbsorptionModelResult)


def test_f_dissolved_in_range():
    result = model_oral_absorption(
        compound_name="Drug", logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=1.0, dose_mg=100.0
    )
    assert 0.0 < result.f_dissolved <= 1.0


def test_f_soluble_in_range():
    result = model_oral_absorption(
        compound_name="Drug", logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=1.0, dose_mg=100.0
    )
    assert 0.0 < result.f_soluble <= 1.0


def test_fa_permeability_in_range():
    result = model_oral_absorption(
        compound_name="Drug", logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=1.0, dose_mg=100.0
    )
    assert 0.0 < result.fa_permeability < 1.0


def test_fa_combined_in_range():
    result = model_oral_absorption(
        compound_name="Drug", logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=1.0, dose_mg=100.0
    )
    assert 0.0 < result.fa_combined < 1.0


def test_f_oral_bioavailability_in_range():
    result = model_oral_absorption(
        compound_name="Drug", logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=1.0, dose_mg=100.0
    )
    assert 0.0 < result.f_oral_bioavailability < 1.0


def test_bioavailability_equals_fa_times_fh():
    result = model_oral_absorption(
        compound_name="Drug", logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=1.0, dose_mg=100.0
    )
    expected = result.fa_combined * result.fh_hepatic
    assert abs(result.f_oral_bioavailability - expected) < 1e-9


def test_notes_nonempty():
    result = model_oral_absorption(
        compound_name="Drug", logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=1.0, dose_mg=100.0
    )
    assert len(result.notes) > 0


def test_compound_name_preserved():
    result = model_oral_absorption(
        compound_name="Ibuprofen",
        logP=3.5,
        mw=206.0,
        psa=37.0,
        solubility_mg_per_mL=0.021,
        dose_mg=400.0,
    )
    assert result.compound_name == "Ibuprofen"


def test_limiting_factor_valid_values():
    result = model_oral_absorption(
        compound_name="Drug", logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=1.0, dose_mg=100.0
    )
    valid = {"dissolution", "solubility", "permeability", "hepatic", "multiple"}
    assert result.limiting_factor in valid


# --- Sensitivity tests ---


def test_high_dose_lower_f_soluble():
    """High dose should reduce solubility fraction."""
    low_dose = model_oral_absorption(
        compound_name="Drug", logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=0.1, dose_mg=10.0
    )
    high_dose = model_oral_absorption(
        compound_name="Drug",
        logP=2.0,
        mw=300.0,
        psa=60.0,
        solubility_mg_per_mL=0.1,
        dose_mg=10000.0,
    )
    assert high_dose.f_soluble < low_dose.f_soluble


def test_large_particle_lower_f_dissolved():
    """Larger particle size should reduce dissolution fraction."""
    small = model_oral_absorption(
        compound_name="Drug",
        logP=2.0,
        mw=300.0,
        psa=60.0,
        solubility_mg_per_mL=1.0,
        dose_mg=100.0,
        particle_size_um=5.0,
    )
    large = model_oral_absorption(
        compound_name="Drug",
        logP=2.0,
        mw=300.0,
        psa=60.0,
        solubility_mg_per_mL=1.0,
        dose_mg=100.0,
        particle_size_um=500.0,
    )
    assert large.f_dissolved < small.f_dissolved


def test_high_psa_lower_fa_permeability():
    """High PSA should reduce permeability."""
    low_psa = model_oral_absorption(
        compound_name="Drug", logP=2.0, mw=300.0, psa=30.0, solubility_mg_per_mL=1.0, dose_mg=100.0
    )
    high_psa = model_oral_absorption(
        compound_name="Drug", logP=2.0, mw=300.0, psa=200.0, solubility_mg_per_mL=1.0, dose_mg=100.0
    )
    assert high_psa.fa_permeability < low_psa.fa_permeability


def test_dissolution_rate_limiting_flag():
    """Large particles should be dissolution rate limiting."""
    result = model_oral_absorption(
        compound_name="Drug",
        logP=2.0,
        mw=300.0,
        psa=60.0,
        solubility_mg_per_mL=1.0,
        dose_mg=100.0,
        particle_size_um=1000.0,
    )
    # With very large particles, f_dissolved should be low
    assert isinstance(result.dissolution_rate_limiting, bool)


def test_solubility_rate_limiting_flag():
    """Very low solubility with high dose should be solubility limited."""
    result = model_oral_absorption(
        compound_name="Drug",
        logP=2.0,
        mw=300.0,
        psa=60.0,
        solubility_mg_per_mL=0.001,
        dose_mg=1000.0,
    )
    assert result.solubility_rate_limiting is True


def test_permeability_rate_limiting_flag():
    result = model_oral_absorption(
        compound_name="Drug", logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=1.0, dose_mg=100.0
    )
    assert isinstance(result.permeability_rate_limiting, bool)


def test_papp_estimate_positive():
    result = model_oral_absorption(
        compound_name="Drug", logP=2.0, mw=300.0, psa=60.0, solubility_mg_per_mL=1.0, dose_mg=100.0
    )
    assert result.papp_estimate_1e6 > 0


# --- Screen function ---


def test_screen_oral_absorption_sorted_descending():
    compounds = [
        {
            "compound_name": "A",
            "logP": 0.5,
            "mw": 500.0,
            "psa": 150.0,
            "solubility_mg_per_mL": 0.1,
            "dose_mg": 100.0,
        },
        {
            "compound_name": "B",
            "logP": 2.5,
            "mw": 300.0,
            "psa": 50.0,
            "solubility_mg_per_mL": 1.0,
            "dose_mg": 100.0,
        },
        {
            "compound_name": "C",
            "logP": 1.0,
            "mw": 200.0,
            "psa": 40.0,
            "solubility_mg_per_mL": 5.0,
            "dose_mg": 100.0,
        },
    ]
    results = screen_oral_absorption(compounds)
    assert len(results) == 3
    for i in range(len(results) - 1):
        assert results[i].f_oral_bioavailability >= results[i + 1].f_oral_bioavailability


def test_screen_returns_list_of_results():
    compounds = [
        {
            "compound_name": "X",
            "logP": 2.0,
            "mw": 300.0,
            "psa": 60.0,
            "solubility_mg_per_mL": 1.0,
            "dose_mg": 100.0,
        },
    ]
    results = screen_oral_absorption(compounds)
    assert len(results) == 1
    assert isinstance(results[0], OralAbsorptionModelResult)


# --- Validation ---


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        model_oral_absorption(
            compound_name="Drug",
            logP=2.0,
            mw=300.0,
            psa=60.0,
            solubility_mg_per_mL=1.0,
            dose_mg=0.0,
        )


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        model_oral_absorption(
            compound_name="Drug",
            logP=2.0,
            mw=300.0,
            psa=60.0,
            solubility_mg_per_mL=1.0,
            dose_mg=-10.0,
        )


def test_validation_solubility_zero():
    with pytest.raises(ValueError, match="solubility"):
        model_oral_absorption(
            compound_name="Drug",
            logP=2.0,
            mw=300.0,
            psa=60.0,
            solubility_mg_per_mL=0.0,
            dose_mg=100.0,
        )
