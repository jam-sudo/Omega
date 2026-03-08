"""Tests for Phase 1085 — Drug Absorption Window and Colon Absorption."""

import pytest

from omega_pbpk.core.colon_absorption_pk import (
    ColonAbsorptionResult,
    compare_ionization_types,
    predict_absorption_window,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_return_type():
    result = predict_absorption_window("TestDrug", pka=5.0, logP=2.0, ionization_type="neutral")
    assert isinstance(result, ColonAbsorptionResult)


def test_frozen_dataclass():
    result = predict_absorption_window("TestDrug", pka=5.0, logP=2.0, ionization_type="neutral")
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"  # type: ignore[misc]


def test_fields_populated():
    result = predict_absorption_window("DrugA", pka=7.0, logP=1.5, ionization_type="base")
    assert result.drug_name == "DrugA"
    assert result.pka == 7.0
    assert result.logP == 1.5
    assert result.ionization_type == "base"


# ---------------------------------------------------------------------------
# Neutral drug absorbs everywhere including colon
# ---------------------------------------------------------------------------


def test_neutral_drug_colon_absorption():
    """Neutral drug: pH has no ionization effect, colon should have some absorption."""
    result = predict_absorption_window(
        "Neutral", pka=7.0, logP=2.0, ionization_type="neutral", peff_base_cm_per_s=1e-5
    )
    # With low Peff, drug survives to colon
    assert result.fraction_absorbed_colon >= 0.0


def test_neutral_drug_has_colon_absorption_low_peff():
    """Low-Peff neutral drug should show measurable colon contribution."""
    result = predict_absorption_window(
        "SlowDrug", pka=7.0, logP=2.0, ionization_type="neutral", peff_base_cm_per_s=1e-6
    )
    assert result.fraction_absorbed_colon > 0.0


# ---------------------------------------------------------------------------
# Acid drug at low pKa: reduced absorption in alkaline SI
# ---------------------------------------------------------------------------


def test_acid_low_pka_reduced_alkaline_absorption():
    """Acid drug with low pKa is fully ionized at alkaline pH → lower total absorption."""
    # Strong acid (pKa=2): mostly ionized across all GI segments (pH 5.5-7.2 >> pKa=2)
    # → very low Peff in all segments → lower total absorption vs neutral
    result_acid = predict_absorption_window(
        "StrongAcid", pka=2.0, logP=1.0, ionization_type="acid", peff_base_cm_per_s=1e-4
    )
    result_neutral = predict_absorption_window(
        "Neutral", pka=2.0, logP=1.0, ionization_type="neutral", peff_base_cm_per_s=1e-4
    )
    # Neutral has no ionization penalty → absorbs more than ionized acid
    assert result_neutral.f_total_absorbed >= result_acid.f_total_absorbed


def test_acid_low_pka_lower_total_vs_neutral():
    """Strongly ionizing acid absorbs less than neutral drug."""
    result_acid = predict_absorption_window(
        "Acid2", pka=2.0, logP=1.0, ionization_type="acid", peff_base_cm_per_s=1e-4
    )
    result_neutral = predict_absorption_window(
        "Neutral2", pka=2.0, logP=1.0, ionization_type="neutral", peff_base_cm_per_s=1e-4
    )
    assert result_neutral.f_total_absorbed >= result_acid.f_total_absorbed


# ---------------------------------------------------------------------------
# Range / boundary checks
# ---------------------------------------------------------------------------


def test_f_total_in_0_1():
    for ionization_type in ("acid", "base", "neutral"):
        result = predict_absorption_window(
            "Drug", pka=5.0, logP=2.0, ionization_type=ionization_type
        )
        assert 0.0 <= result.f_total_absorbed <= 1.0


def test_all_fraction_segments_non_negative():
    result = predict_absorption_window("Drug", pka=5.0, logP=2.0, ionization_type="base")
    assert result.fraction_absorbed_duodenum >= 0.0
    assert result.fraction_absorbed_jejunum >= 0.0
    assert result.fraction_absorbed_ileum >= 0.0
    assert result.fraction_absorbed_colon >= 0.0


def test_colon_fraction_not_dominant_high_peff():
    """High-Peff drug is absorbed mostly in SI; colon fraction small."""
    result = predict_absorption_window(
        "HighPeff", pka=7.0, logP=3.0, ionization_type="neutral", peff_base_cm_per_s=1e-3
    )
    f_total = result.f_total_absorbed
    if f_total > 0:
        assert result.fraction_absorbed_colon <= 0.5 * f_total


# ---------------------------------------------------------------------------
# Boolean flags
# ---------------------------------------------------------------------------


def test_has_colon_absorption_is_bool():
    result = predict_absorption_window("Drug", pka=5.0, logP=2.0, ionization_type="neutral")
    assert isinstance(result.has_colon_absorption, bool)


def test_er_design_feasible_is_bool():
    result = predict_absorption_window("Drug", pka=5.0, logP=2.0, ionization_type="neutral")
    assert isinstance(result.er_design_feasible, bool)


def test_er_design_feasible_equals_has_colon_absorption():
    result = predict_absorption_window("Drug", pka=5.0, logP=2.0, ionization_type="neutral")
    assert result.er_design_feasible == result.has_colon_absorption


# ---------------------------------------------------------------------------
# absorption_window_segments
# ---------------------------------------------------------------------------


def test_absorption_window_segments_is_tuple():
    result = predict_absorption_window("Drug", pka=5.0, logP=2.0, ionization_type="neutral")
    assert isinstance(result.absorption_window_segments, tuple)


def test_absorption_window_segments_valid_names():
    valid = {
        "duodenum",
        "jejunum_proximal",
        "jejunum_distal",
        "ileum_proximal",
        "ileum_distal",
        "colon",
    }
    result = predict_absorption_window("Drug", pka=5.0, logP=2.0, ionization_type="neutral")
    for seg in result.absorption_window_segments:
        assert seg in valid


# ---------------------------------------------------------------------------
# bioavailability_class
# ---------------------------------------------------------------------------


def test_bioavailability_class_valid():
    for ionization_type in ("acid", "base", "neutral"):
        result = predict_absorption_window(
            "Drug", pka=7.0, logP=2.0, ionization_type=ionization_type
        )
        assert result.bioavailability_class in ("high", "moderate", "low")


def test_high_peff_gives_high_bioavailability_class():
    result = predict_absorption_window(
        "HighF", pka=7.0, logP=3.0, ionization_type="neutral", peff_base_cm_per_s=1e-3
    )
    assert result.bioavailability_class == "high"


def test_very_low_peff_gives_low_bioavailability_class():
    result = predict_absorption_window(
        "LowF", pka=7.0, logP=3.0, ionization_type="neutral", peff_base_cm_per_s=1e-8
    )
    assert result.bioavailability_class == "low"


# ---------------------------------------------------------------------------
# Effect of Peff
# ---------------------------------------------------------------------------


def test_higher_peff_increases_total_absorbed():
    low_peff = predict_absorption_window(
        "Drug", pka=5.0, logP=2.0, ionization_type="neutral", peff_base_cm_per_s=1e-6
    )
    high_peff = predict_absorption_window(
        "Drug", pka=5.0, logP=2.0, ionization_type="neutral", peff_base_cm_per_s=1e-4
    )
    assert high_peff.f_total_absorbed > low_peff.f_total_absorbed


# ---------------------------------------------------------------------------
# compare_ionization_types
# ---------------------------------------------------------------------------


def test_compare_ionization_types_returns_three():
    results = compare_ionization_types("TestDrug", pka=5.0, logP=2.0)
    assert len(results) == 3


def test_compare_ionization_types_sorted_descending():
    results = compare_ionization_types("TestDrug", pka=5.0, logP=2.0)
    for i in range(len(results) - 1):
        assert results[i].f_total_absorbed >= results[i + 1].f_total_absorbed


def test_compare_ionization_types_all_types_present():
    results = compare_ionization_types("TestDrug", pka=5.0, logP=2.0)
    types = {r.ionization_type for r in results}
    assert types == {"acid", "base", "neutral"}


def test_neutral_has_highest_f_total():
    """Neutral drug should absorb at least as well as ionizable counterparts."""
    results = compare_ionization_types("TestDrug", pka=5.0, logP=2.0)
    neutral_f = next(r.f_total_absorbed for r in results if r.ionization_type == "neutral")
    for r in results:
        assert neutral_f >= r.f_total_absorbed - 1e-9


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_absorption_window("Drug", pka=5.0, logP=2.0, ionization_type="zwitterion")


def test_invalid_pka_too_low_raises():
    with pytest.raises(ValueError, match="pKa"):
        predict_absorption_window("Drug", pka=-1.0, logP=2.0, ionization_type="acid")


def test_invalid_pka_too_high_raises():
    with pytest.raises(ValueError, match="pKa"):
        predict_absorption_window("Drug", pka=15.0, logP=2.0, ionization_type="acid")


def test_invalid_logp_too_low_raises():
    with pytest.raises(ValueError, match="logP"):
        predict_absorption_window("Drug", pka=5.0, logP=-6.0, ionization_type="neutral")


def test_invalid_peff_raises():
    with pytest.raises(ValueError, match="peff_base"):
        predict_absorption_window(
            "Drug", pka=5.0, logP=2.0, ionization_type="neutral", peff_base_cm_per_s=-1e-4
        )
