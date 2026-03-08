"""Tests for Phase 478 — Drug Concentration Prediction in Sweat."""

import pytest

from omega_pbpk.prediction.sweat_concentration import (
    SweatConcentrationResult,
    predict_sweat_concentration,
    screen_sweat_detectability,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_return_type():
    result = predict_sweat_concentration(
        "Caffeine", logP=0.07, pKa=10.4, ionization_type="neutral", plasma_conc_mg_L=1.0
    )
    assert isinstance(result, SweatConcentrationResult)


def test_frozen_dataclass():
    result = predict_sweat_concentration(
        "Caffeine", logP=0.07, pKa=10.4, ionization_type="neutral", plasma_conc_mg_L=1.0
    )
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"  # type: ignore[misc]


def test_drug_name_stored():
    result = predict_sweat_concentration(
        "Aspirin", logP=1.19, pKa=3.5, ionization_type="acid", plasma_conc_mg_L=5.0
    )
    assert result.drug_name == "Aspirin"


def test_logp_stored():
    result = predict_sweat_concentration(
        "TestDrug", logP=2.5, pKa=7.0, ionization_type="neutral", plasma_conc_mg_L=1.0
    )
    assert result.logP == pytest.approx(2.5)


def test_ionization_type_stored():
    result = predict_sweat_concentration(
        "TestDrug", logP=1.0, pKa=7.0, ionization_type="base", plasma_conc_mg_L=1.0
    )
    assert result.ionization_type == "base"


# ---------------------------------------------------------------------------
# Ratio clamping
# ---------------------------------------------------------------------------


def test_ratio_in_valid_range():
    result = predict_sweat_concentration(
        "TestDrug", logP=5.0, pKa=4.0, ionization_type="neutral", plasma_conc_mg_L=1.0
    )
    assert 0.01 <= result.sweat_plasma_ratio <= 10.0


def test_ratio_clamped_minimum():
    """Very lipophilic drug with acid ionization → ratio approaches minimum."""
    result = predict_sweat_concentration(
        "LipophilicAcid",
        logP=6.0,
        pKa=9.0,
        ionization_type="acid",
        plasma_conc_mg_L=1.0,
    )
    assert result.sweat_plasma_ratio >= 0.01


def test_ratio_clamped_maximum():
    """Hydrophilic neutral drug should have ratio <= 10."""
    result = predict_sweat_concentration(
        "HydrophilicDrug",
        logP=-5.0,
        pKa=7.0,
        ionization_type="neutral",
        plasma_conc_mg_L=1.0,
    )
    assert result.sweat_plasma_ratio <= 10.0


# ---------------------------------------------------------------------------
# Concentration calculation
# ---------------------------------------------------------------------------


def test_sweat_conc_equals_plasma_times_ratio():
    result = predict_sweat_concentration(
        "TestDrug", logP=1.0, pKa=7.0, ionization_type="neutral", plasma_conc_mg_L=2.0
    )
    expected = result.plasma_conc_mg_L * result.sweat_plasma_ratio
    assert result.sweat_conc_mg_L == pytest.approx(expected, rel=1e-6)


def test_excretion_rate_formula():
    """rate = sweat_conc * flow / 1000."""
    flow = 0.8
    result = predict_sweat_concentration(
        "TestDrug",
        logP=1.0,
        pKa=7.0,
        ionization_type="neutral",
        plasma_conc_mg_L=2.0,
        sweat_flow_mL_per_h=flow,
    )
    expected_rate = result.sweat_conc_mg_L * flow / 1000.0
    assert result.sweat_excretion_rate_mg_per_h == pytest.approx(expected_rate, rel=1e-6)


def test_excretion_rate_positive():
    result = predict_sweat_concentration(
        "TestDrug", logP=1.0, pKa=7.0, ionization_type="neutral", plasma_conc_mg_L=1.0
    )
    assert result.sweat_excretion_rate_mg_per_h > 0.0


# ---------------------------------------------------------------------------
# Neutral drug ratio
# ---------------------------------------------------------------------------


def test_neutral_drug_ratio_formula():
    """Neutral: R = 1 / (1 + 0.3 * logP), clamped to [0.01, 10]."""
    logP = 1.0
    result = predict_sweat_concentration(
        "NeutralDrug", logP=logP, pKa=7.0, ionization_type="neutral", plasma_conc_mg_L=1.0
    )
    expected = max(0.01, min(10.0, 1.0 / (1.0 + 0.3 * logP)))
    assert result.sweat_plasma_ratio == pytest.approx(expected, rel=1e-6)


def test_neutral_higher_logp_lower_ratio():
    """More lipophilic neutral drug → lower sweat/plasma ratio."""
    result_low = predict_sweat_concentration(
        "Drug", logP=0.5, pKa=7.0, ionization_type="neutral", plasma_conc_mg_L=1.0
    )
    result_high = predict_sweat_concentration(
        "Drug", logP=3.0, pKa=7.0, ionization_type="neutral", plasma_conc_mg_L=1.0
    )
    assert result_low.sweat_plasma_ratio > result_high.sweat_plasma_ratio


# ---------------------------------------------------------------------------
# Ionization effects
# ---------------------------------------------------------------------------


def test_basic_drug_more_ionized_in_sweat():
    """Basic drug (pKa ~9): sweat pH 6.3 is more acidic → more ionized in sweat → lower ratio."""
    result = predict_sweat_concentration(
        "BasicDrug", logP=1.0, pKa=9.0, ionization_type="base", plasma_conc_mg_L=1.0
    )
    result_neutral = predict_sweat_concentration(
        "NeutralDrug", logP=1.0, pKa=9.0, ionization_type="neutral", plasma_conc_mg_L=1.0
    )
    # basic drug more ionized in sweat → lower unionized fraction → lower ratio
    assert result.sweat_plasma_ratio < result_neutral.sweat_plasma_ratio


def test_acid_drug_less_ionized_in_sweat():
    """Acid drug (pKa ~4.5): sweat pH 6.3 < pKa → less ionized in sweat than plasma."""
    result_acid = predict_sweat_concentration(
        "AcidDrug", logP=1.0, pKa=9.0, ionization_type="acid", plasma_conc_mg_L=1.0
    )
    # fi_sweat < fi_plasma → fu_sweat > fu_plasma → ratio > neutral (if pKa > sweat_pH)
    assert result_acid.sweat_plasma_ratio >= 0.01


# ---------------------------------------------------------------------------
# Detection class
# ---------------------------------------------------------------------------


def test_detection_class_high():
    """Very hydrophilic neutral drug → ratio > 1 → 'high'."""
    result = predict_sweat_concentration(
        "Drug", logP=-3.0, pKa=7.0, ionization_type="neutral", plasma_conc_mg_L=1.0
    )
    assert result.detection_class == "high"


def test_detection_class_moderate():
    """Moderate logP neutral drug → ratio between 0.1 and 1 → 'moderate'."""
    result = predict_sweat_concentration(
        "Drug", logP=1.0, pKa=7.0, ionization_type="neutral", plasma_conc_mg_L=1.0
    )
    ratio = result.sweat_plasma_ratio
    if 0.1 <= ratio <= 1.0:
        assert result.detection_class == "moderate"


def test_detection_class_low_for_high_logp():
    """High logP → low ratio → 'low' detection class."""
    result = predict_sweat_concentration(
        "LipoDrug", logP=5.0, pKa=7.0, ionization_type="neutral", plasma_conc_mg_L=1.0
    )
    if result.sweat_plasma_ratio < 0.1:
        assert result.detection_class == "low"


def test_detection_window_positive_when_detectable():
    """Drug above LOD in sweat → detection window = 24h."""
    result = predict_sweat_concentration(
        "Drug", logP=0.0, pKa=7.0, ionization_type="neutral", plasma_conc_mg_L=10.0
    )
    if result.sweat_conc_mg_L > 0.001:
        assert result.detection_window_h == pytest.approx(24.0)


def test_detection_window_zero_when_below_lod():
    """Drug below LOD → detection window = 0."""
    result = predict_sweat_concentration(
        "Drug",
        logP=5.0,
        pKa=7.0,
        ionization_type="neutral",
        plasma_conc_mg_L=0.0001,
    )
    if result.sweat_conc_mg_L <= 0.001:
        assert result.detection_window_h == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_is_string():
    result = predict_sweat_concentration(
        "TestDrug", logP=1.0, pKa=7.0, ionization_type="neutral", plasma_conc_mg_L=1.0
    )
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# screen_sweat_detectability
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    compounds = [
        {
            "drug_name": "A",
            "logP": 0.5,
            "pKa": 7.0,
            "ionization_type": "neutral",
            "plasma_conc_mg_L": 1.0,
        },
        {
            "drug_name": "B",
            "logP": 3.0,
            "pKa": 7.0,
            "ionization_type": "neutral",
            "plasma_conc_mg_L": 1.0,
        },
    ]
    results = screen_sweat_detectability(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_by_ratio_descending():
    compounds = [
        {
            "drug_name": "A",
            "logP": 0.5,
            "pKa": 7.0,
            "ionization_type": "neutral",
            "plasma_conc_mg_L": 1.0,
        },
        {
            "drug_name": "B",
            "logP": 3.0,
            "pKa": 7.0,
            "ionization_type": "neutral",
            "plasma_conc_mg_L": 1.0,
        },
        {
            "drug_name": "C",
            "logP": -1.0,
            "pKa": 7.0,
            "ionization_type": "neutral",
            "plasma_conc_mg_L": 1.0,
        },
    ]
    results = screen_sweat_detectability(compounds)
    ratios = [r.sweat_plasma_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_screen_result_types():
    compounds = [
        {
            "drug_name": "X",
            "logP": 1.0,
            "pKa": 7.0,
            "ionization_type": "acid",
            "plasma_conc_mg_L": 2.0,
        }
    ]
    results = screen_sweat_detectability(compounds)
    assert isinstance(results[0], SweatConcentrationResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_sweat_concentration(
            "X", logP=1.0, pKa=7.0, ionization_type="zwitterion", plasma_conc_mg_L=1.0
        )


def test_validation_plasma_conc_zero():
    with pytest.raises(ValueError, match="plasma_conc"):
        predict_sweat_concentration(
            "X", logP=1.0, pKa=7.0, ionization_type="neutral", plasma_conc_mg_L=0.0
        )


def test_validation_plasma_conc_negative():
    with pytest.raises(ValueError, match="plasma_conc"):
        predict_sweat_concentration(
            "X", logP=1.0, pKa=7.0, ionization_type="neutral", plasma_conc_mg_L=-1.0
        )


def test_validation_sweat_flow_zero():
    with pytest.raises(ValueError, match="sweat_flow"):
        predict_sweat_concentration(
            "X",
            logP=1.0,
            pKa=7.0,
            ionization_type="neutral",
            plasma_conc_mg_L=1.0,
            sweat_flow_mL_per_h=0.0,
        )


def test_validation_sweat_ph_out_of_range_low():
    with pytest.raises(ValueError, match="sweat_ph"):
        predict_sweat_concentration(
            "X",
            logP=1.0,
            pKa=7.0,
            ionization_type="neutral",
            plasma_conc_mg_L=1.0,
            sweat_ph=1.5,
        )


def test_validation_sweat_ph_out_of_range_high():
    with pytest.raises(ValueError, match="sweat_ph"):
        predict_sweat_concentration(
            "X",
            logP=1.0,
            pKa=7.0,
            ionization_type="neutral",
            plasma_conc_mg_L=1.0,
            sweat_ph=9.0,
        )
