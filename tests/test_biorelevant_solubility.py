"""Tests for Phase 1090 — Drug Solubility in Biorelevant Media."""

import pytest

from omega_pbpk.prediction.biorelevant_solubility import (
    VALID_MEDIA,
    BiorelevantSolubilityResult,
    predict_biorelevant_solubility,
    screen_media,
)

VALID_SOLUBILITY_CLASSES = {"high", "moderate", "low"}


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = predict_biorelevant_solubility("drug_a", pka=4.0, logP=2.0, ionization_type="acid")
    assert isinstance(result, BiorelevantSolubilityResult)


# ---------------------------------------------------------------------------
# FeSSIF > FaSSIF for lipophilic drug
# ---------------------------------------------------------------------------


def test_fessif_greater_than_fassif_for_lipophilic():
    """FeSSIF has more bile salts and lipids → higher solubility for lipophilic drug."""
    fassif = predict_biorelevant_solubility(
        "lipophilic",
        pka=7.0,
        logP=3.0,
        ionization_type="neutral",
        intrinsic_solubility_mg_mL=0.05,
        medium="FaSSIF",
    )
    fessif = predict_biorelevant_solubility(
        "lipophilic",
        pka=7.0,
        logP=3.0,
        ionization_type="neutral",
        intrinsic_solubility_mg_mL=0.05,
        medium="FeSSIF",
    )
    assert fessif.s_biorelevant_mg_mL > fassif.s_biorelevant_mg_mL


# ---------------------------------------------------------------------------
# s_biorelevant >= s_intrinsic * ionization_factor (always enhanced by micelles)
# ---------------------------------------------------------------------------


def test_biorelevant_at_least_ionized_solubility():
    """Micellar enhancement only adds to solubility, never subtracts."""
    result = predict_biorelevant_solubility(
        "acid_drug",
        pka=6.0,
        logP=2.0,
        ionization_type="acid",
        intrinsic_solubility_mg_mL=0.1,
        medium="FaSSIF",
    )
    expected_min = result.s_intrinsic_mg_mL * result.ionization_factor
    assert result.s_biorelevant_mg_mL >= expected_min - 1e-10


def test_biorelevant_water_no_micellar():
    """Water has no bile/lecithin/lipid — micellar_enhancement should be 1.0."""
    result = predict_biorelevant_solubility(
        "neutral_drug",
        pka=7.0,
        logP=1.0,
        ionization_type="neutral",
        intrinsic_solubility_mg_mL=0.2,
        medium="water",
    )
    assert abs(result.micellar_enhancement - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Ionization direction checks
# ---------------------------------------------------------------------------


def test_acid_higher_ph_increases_ionization_factor():
    """For acid drug: higher pH → more ionization → higher ionization factor."""
    fassif = predict_biorelevant_solubility(
        "acid",
        pka=5.0,
        logP=1.0,
        ionization_type="acid",
        intrinsic_solubility_mg_mL=0.05,
        medium="FaSSIF",
    )  # pH=6.5
    water = predict_biorelevant_solubility(
        "acid",
        pka=5.0,
        logP=1.0,
        ionization_type="acid",
        intrinsic_solubility_mg_mL=0.05,
        medium="water",
    )  # pH=7.0
    assert water.ionization_factor > fassif.ionization_factor


def test_base_lower_ph_increases_ionization_factor():
    """For base drug: lower pH → more ionization → higher ionization factor."""
    fassif = predict_biorelevant_solubility(
        "base",
        pka=8.0,
        logP=1.0,
        ionization_type="base",
        intrinsic_solubility_mg_mL=0.05,
        medium="FaSSIF",
    )  # pH=6.5
    fassgf = predict_biorelevant_solubility(
        "base",
        pka=8.0,
        logP=1.0,
        ionization_type="base",
        intrinsic_solubility_mg_mL=0.05,
        medium="FaSSGF",
    )  # pH=1.6
    assert fassgf.ionization_factor > fassif.ionization_factor


def test_neutral_ionization_factor_is_one():
    """Neutral drug: ionization factor == 1.0 in any medium."""
    for medium in ["FaSSIF", "FeSSIF", "FaSSGF", "water", "PBS_pH6_8"]:
        result = predict_biorelevant_solubility(
            "neutral", pka=7.0, logP=1.5, ionization_type="neutral", medium=medium
        )
        assert abs(result.ionization_factor - 1.0) < 1e-10, f"Failed for {medium}"


# ---------------------------------------------------------------------------
# screen_media
# ---------------------------------------------------------------------------


def test_screen_media_returns_five_results():
    results = screen_media("drug_x", pka=4.5, logP=2.0, ionization_type="acid")
    assert len(results) == 5


def test_screen_media_sorted_descending():
    results = screen_media("drug_x", pka=4.5, logP=2.0, ionization_type="acid")
    sols = [r.s_biorelevant_mg_mL for r in results]
    assert sols == sorted(sols, reverse=True)


def test_screen_media_all_five_media_present():
    results = screen_media("drug_x", pka=7.0, logP=1.0, ionization_type="neutral")
    media_found = {r.medium for r in results}
    assert media_found == {"FaSSIF", "FeSSIF", "FaSSGF", "water", "PBS_pH6_8"}


def test_screen_media_food_effect_ratio_computed():
    """food_effect_ratio should be FeSSIF/FaSSIF, > 0 for lipophilic drug."""
    results = screen_media(
        "drug_x", pka=7.0, logP=3.0, ionization_type="neutral", intrinsic_solubility_mg_mL=0.05
    )
    # All results share the same food_effect_ratio
    ratios = {r.food_effect_ratio for r in results}
    assert len(ratios) == 1
    food_ratio = list(ratios)[0]
    assert food_ratio > 0.0


def test_screen_media_fessif_over_fassif_ratio():
    """food_effect_ratio should equal FeSSIF/FaSSIF."""
    results = screen_media(
        "drug_x", pka=7.0, logP=3.0, ionization_type="neutral", intrinsic_solubility_mg_mL=0.05
    )
    result_map = {r.medium: r for r in results}
    expected_ratio = (
        result_map["FeSSIF"].s_biorelevant_mg_mL / result_map["FaSSIF"].s_biorelevant_mg_mL
    )
    for r in results:
        assert abs(r.food_effect_ratio - expected_ratio) < 1e-9


# ---------------------------------------------------------------------------
# Solubility class
# ---------------------------------------------------------------------------


def test_solubility_class_valid():
    for medium in VALID_MEDIA:
        result = predict_biorelevant_solubility(
            "drug", pka=7.0, logP=1.0, ionization_type="neutral", medium=medium
        )
        assert result.solubility_class in VALID_SOLUBILITY_CLASSES


def test_solubility_class_high():
    result = predict_biorelevant_solubility(
        "soluble_drug",
        pka=7.0,
        logP=-2.0,
        ionization_type="neutral",
        intrinsic_solubility_mg_mL=5.0,
    )
    assert result.solubility_class == "high"


def test_solubility_class_low():
    result = predict_biorelevant_solubility(
        "insoluble_drug",
        pka=7.0,
        logP=5.0,
        ionization_type="neutral",
        intrinsic_solubility_mg_mL=0.001,
        medium="water",
    )
    # water: no micellar enhancement; neutral: f_ion=1; very low intrinsic
    assert result.solubility_class == "low"


# ---------------------------------------------------------------------------
# s_biorelevant > 0
# ---------------------------------------------------------------------------


def test_s_biorelevant_always_positive():
    for medium in VALID_MEDIA:
        result = predict_biorelevant_solubility(
            "drug",
            pka=4.0,
            logP=2.0,
            ionization_type="acid",
            intrinsic_solubility_mg_mL=0.01,
            medium=medium,
        )
        assert result.s_biorelevant_mg_mL > 0.0


# ---------------------------------------------------------------------------
# Base drug higher solubility in acidic FaSSGF
# ---------------------------------------------------------------------------


def test_base_drug_higher_solubility_in_fassgf_vs_water():
    """Base drug (pKa=9) ionizes more at pH 1.6 than pH 7.0."""
    fassgf = predict_biorelevant_solubility(
        "base_drug",
        pka=9.0,
        logP=1.0,
        ionization_type="base",
        intrinsic_solubility_mg_mL=0.05,
        medium="FaSSGF",
    )
    water = predict_biorelevant_solubility(
        "base_drug",
        pka=9.0,
        logP=1.0,
        ionization_type="base",
        intrinsic_solubility_mg_mL=0.05,
        medium="water",
    )
    # FaSSGF pH=1.6 vs water pH=7.0 → base more ionized at lower pH
    assert fassgf.ionization_factor > water.ionization_factor


# ---------------------------------------------------------------------------
# Validation / error cases
# ---------------------------------------------------------------------------


def test_invalid_medium_raises_value_error():
    with pytest.raises(ValueError, match="medium"):
        predict_biorelevant_solubility(
            "drug", pka=7.0, logP=2.0, ionization_type="neutral", medium="unknown_medium"
        )


def test_invalid_ionization_type_raises_value_error():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_biorelevant_solubility("drug", pka=7.0, logP=2.0, ionization_type="zwitterion")


def test_pka_out_of_range_raises():
    with pytest.raises(ValueError, match="pka"):
        predict_biorelevant_solubility("drug", pka=15.0, logP=2.0, ionization_type="neutral")


def test_logp_out_of_range_raises():
    with pytest.raises(ValueError, match="logP"):
        predict_biorelevant_solubility("drug", pka=7.0, logP=12.0, ionization_type="neutral")


def test_negative_solubility_raises():
    with pytest.raises(ValueError, match="intrinsic_solubility"):
        predict_biorelevant_solubility(
            "drug", pka=7.0, logP=2.0, ionization_type="neutral", intrinsic_solubility_mg_mL=-0.1
        )


# ---------------------------------------------------------------------------
# Additional
# ---------------------------------------------------------------------------


def test_single_medium_food_effect_ratio_zero():
    """Single medium call: food_effect_ratio = 0.0 (not computable)."""
    result = predict_biorelevant_solubility("drug", pka=7.0, logP=2.0, ionization_type="neutral")
    assert result.food_effect_ratio == 0.0


def test_pbs_ph6_8_no_micellar():
    """PBS_pH6_8 has no bile/lecithin/lipid → micellar_enhancement == 1.0."""
    result = predict_biorelevant_solubility(
        "drug", pka=7.0, logP=3.0, ionization_type="neutral", medium="PBS_pH6_8"
    )
    assert abs(result.micellar_enhancement - 1.0) < 1e-9


def test_intrinsic_solubility_stored_correctly():
    result = predict_biorelevant_solubility(
        "drug", pka=7.0, logP=2.0, ionization_type="neutral", intrinsic_solubility_mg_mL=0.123
    )
    assert abs(result.s_intrinsic_mg_mL - 0.123) < 1e-10


def test_drug_name_stored():
    result = predict_biorelevant_solubility("ibuprofen", pka=4.4, logP=3.5, ionization_type="acid")
    assert result.drug_name == "ibuprofen"


def test_acid_drug_lower_ionization_in_fassgf():
    """Acid drug at very low pH: minimal ionization → lower ionization factor."""
    fassgf = predict_biorelevant_solubility(
        "acid_drug", pka=6.0, logP=2.0, ionization_type="acid", medium="FaSSGF"
    )  # pH=1.6
    fassif = predict_biorelevant_solubility(
        "acid_drug", pka=6.0, logP=2.0, ionization_type="acid", medium="FaSSIF"
    )  # pH=6.5
    # acid: f = 1 + 10^(pH - pKa); higher pH → more ionization
    assert fassif.ionization_factor > fassgf.ionization_factor


def test_notes_nonempty():
    result = predict_biorelevant_solubility("drug", pka=7.0, logP=2.0, ionization_type="neutral")
    assert result.notes != ""
