"""Tests for Phase 763 — Drug Photodegradation Risk Prediction."""

import math

from omega_pbpk.prediction.photodegradation_risk import (
    PhotodegradationResult,
    predict_photodegradation,
    screen_photodegradation,
)

# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_returns_photodegradation_result():
    result = predict_photodegradation("TestDrug")
    assert isinstance(result, PhotodegradationResult)


def test_compound_name_preserved():
    result = predict_photodegradation("Aspirin")
    assert result.compound_name == "Aspirin"


# ---------------------------------------------------------------------------
# No chromophores → stable
# ---------------------------------------------------------------------------


def test_no_chromophores_high_fraction_remaining():
    result = predict_photodegradation(
        "StableDrug",
        has_conjugated_system=False,
        has_nitro_group=False,
        has_carbonyl_extended=False,
        has_aromatic_amine=False,
        logp=1.0,
    )
    assert result.fraction_remaining_24h_50klux > 0.9
    assert result.photostability_category == "stable"


def test_no_chromophores_ich_q1b_compliant():
    result = predict_photodegradation("StableDrug2")
    assert result.ich_q1b_compliant is True


# ---------------------------------------------------------------------------
# Multiple chromophores → less stable
# ---------------------------------------------------------------------------


def test_multiple_chromophores_lower_fraction_remaining():
    stable = predict_photodegradation("Stable", logp=1.0)
    labile = predict_photodegradation(
        "Labile",
        has_conjugated_system=True,
        has_nitro_group=True,
        has_carbonyl_extended=True,
        has_aromatic_amine=True,
        logp=4.0,
    )
    assert labile.fraction_remaining_24h_50klux < stable.fraction_remaining_24h_50klux


def test_multiple_chromophores_labile_category():
    result = predict_photodegradation(
        "LabildDrug",
        has_conjugated_system=True,
        has_nitro_group=True,
        has_carbonyl_extended=True,
        has_aromatic_amine=True,
        logp=4.0,
    )
    assert result.photostability_category == "labile"


# ---------------------------------------------------------------------------
# Value ranges
# ---------------------------------------------------------------------------


def test_fraction_remaining_between_0_and_1():
    result = predict_photodegradation("RangeDrug", has_conjugated_system=True)
    assert 0.0 < result.fraction_remaining_24h_50klux <= 1.0


def test_fraction_remaining_1h_5klux_between_0_and_1():
    result = predict_photodegradation("RangeDrug2", has_nitro_group=True)
    assert 0.0 < result.fraction_remaining_1h_5klux <= 1.0


def test_t_half_50klux_h_positive():
    result = predict_photodegradation("ThalfDrug")
    assert result.t_half_50klux_h > 0.0


def test_k_photo_positive():
    result = predict_photodegradation("KDrug")
    assert result.k_photo_per_h_per_klux > 0.0


# ---------------------------------------------------------------------------
# t_half computation check
# ---------------------------------------------------------------------------


def test_t_half_consistent_with_k_photo():
    result = predict_photodegradation("ThalfCheck", has_conjugated_system=True)
    expected_t_half = math.log(2.0) / (result.k_photo_per_h_per_klux * 50.0)
    assert abs(result.t_half_50klux_h - expected_t_half) < 1e-9


# ---------------------------------------------------------------------------
# photostability_category valid values
# ---------------------------------------------------------------------------


def test_photostability_category_valid():
    for has_conj in [False, True]:
        result = predict_photodegradation("CatDrug", has_conjugated_system=has_conj)
        assert result.photostability_category in ("stable", "moderate", "labile")


# ---------------------------------------------------------------------------
# ich_q1b_compliant
# ---------------------------------------------------------------------------


def test_ich_q1b_compliant_when_fraction_above_0_9():
    result = predict_photodegradation("CompDrug")
    expected = result.fraction_remaining_24h_50klux > 0.9
    assert result.ich_q1b_compliant == expected


def test_ich_q1b_not_compliant_for_labile_drug():
    result = predict_photodegradation(
        "LabDrug",
        has_conjugated_system=True,
        has_nitro_group=True,
        has_aromatic_amine=True,
        logp=4.0,
    )
    assert result.ich_q1b_compliant is False


# ---------------------------------------------------------------------------
# packaging_recommendation
# ---------------------------------------------------------------------------


def test_packaging_recommendation_is_string():
    result = predict_photodegradation("PackDrug")
    assert isinstance(result.packaging_recommendation, str)
    assert result.packaging_recommendation in (
        "amber bottle",
        "foil overwrap",
        "standard packaging",
    )


def test_stable_drug_standard_packaging():
    result = predict_photodegradation("StablePackDrug")
    assert result.packaging_recommendation == "standard packaging"


def test_labile_drug_foil_overwrap():
    result = predict_photodegradation(
        "FoilDrug",
        has_conjugated_system=True,
        has_nitro_group=True,
        has_aromatic_amine=True,
        logp=4.0,
    )
    assert result.packaging_recommendation == "foil overwrap"


# ---------------------------------------------------------------------------
# screen_photodegradation sorted by fraction_remaining desc
# ---------------------------------------------------------------------------


def test_screen_photodegradation_returns_list():
    compounds = [
        {"compound_name": "A"},
        {"compound_name": "B", "has_conjugated_system": True},
    ]
    results = screen_photodegradation(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_photodegradation_sorted_desc():
    compounds = [
        {"compound_name": "Labile", "has_conjugated_system": True, "has_nitro_group": True},
        {"compound_name": "Stable"},
        {"compound_name": "Mid", "has_carbonyl_extended": True},
    ]
    results = screen_photodegradation(compounds)
    fracs = [r.fraction_remaining_24h_50klux for r in results]
    assert fracs == sorted(fracs, reverse=True)


# ---------------------------------------------------------------------------
# Aromatic amine → lower stability
# ---------------------------------------------------------------------------


def test_aromatic_amine_lower_stability():
    no_amine = predict_photodegradation("NoAmine", has_aromatic_amine=False)
    with_amine = predict_photodegradation("WithAmine", has_aromatic_amine=True)
    assert with_amine.fraction_remaining_24h_50klux < no_amine.fraction_remaining_24h_50klux


# ---------------------------------------------------------------------------
# notes non-empty
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = predict_photodegradation("NotesDrug")
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


def test_notes_nonempty_for_labile_drug():
    result = predict_photodegradation(
        "LabNotesDrug",
        has_conjugated_system=True,
        has_aromatic_amine=True,
    )
    assert len(result.notes) > 0
