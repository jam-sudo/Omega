"""Tests for Phase 988 — drug ionization profile."""

import math

import pytest

from omega_pbpk.prediction.drug_ionization_profile import (
    IonizationProfileResult,
    compare_ionization_profiles,
    compute_ionization_profile,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_result_type():
    result = compute_ionization_profile("Aspirin", pka=3.5, ionization_type="acid")
    assert isinstance(result, IonizationProfileResult)


def test_ph_values_length():
    result = compute_ionization_profile("Test", pka=7.0)
    assert len(result.ph_values) == 28


def test_fraction_ionized_length():
    result = compute_ionization_profile("Test", pka=7.0)
    assert len(result.fraction_ionized) == 28


def test_fraction_neutral_length():
    result = compute_ionization_profile("Test", pka=7.0)
    assert len(result.fraction_neutral) == 28


def test_ph_values_start_at_0_5():
    result = compute_ionization_profile("Test", pka=5.0)
    assert math.isclose(result.ph_values[0], 0.5)


def test_ph_values_end_at_14():
    result = compute_ionization_profile("Test", pka=5.0)
    assert math.isclose(result.ph_values[-1], 14.0)


def test_all_fractions_in_0_1():
    result = compute_ionization_profile("Test", pka=6.0)
    for fi in result.fraction_ionized:
        assert 0.0 <= fi <= 1.0
    for fn in result.fraction_neutral:
        assert 0.0 <= fn <= 1.0


def test_ionized_plus_neutral_equals_one():
    result = compute_ionization_profile("Test", pka=6.0)
    for fi, fn in zip(result.fraction_ionized, result.fraction_neutral):
        assert math.isclose(fi + fn, 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Henderson-Hasselbalch correctness
# ---------------------------------------------------------------------------


def test_acid_high_ph_mostly_ionized():
    """Acid at pH >> pKa should be >90% ionized."""
    result = compute_ionization_profile("WeakAcid", pka=3.0, ionization_type="acid")
    # At pH 14.0 (last point), pKa=3 → exponent=3-14=-11 → ~100% ionized
    assert result.fraction_ionized[-1] > 0.9


def test_acid_low_ph_mostly_neutral():
    """Acid at pH << pKa should be <10% ionized."""
    result = compute_ionization_profile("WeakAcid", pka=10.0, ionization_type="acid")
    # At pH 0.5 (first point), pKa=10 → exponent=10-0.5=9.5 → ~0% ionized
    assert result.fraction_ionized[0] < 0.1


def test_base_low_ph_mostly_ionized():
    """Base at pH << pKa should be >90% ionized."""
    result = compute_ionization_profile("WeakBase", pka=10.0, ionization_type="base")
    # At pH 0.5, pKa=10 → exponent=0.5-10=-9.5 → ~100% ionized
    assert result.fraction_ionized[0] > 0.9


def test_base_high_ph_mostly_neutral():
    """Base at pH >> pKa should be <10% ionized."""
    result = compute_ionization_profile("WeakBase", pka=3.0, ionization_type="base")
    # At pH 14.0, pKa=3 → exponent=14-3=11 → ~0% ionized
    assert result.fraction_ionized[-1] < 0.1


def test_neutral_all_zero_ionized():
    """Neutral compound: all fraction_ionized should be 0.0."""
    result = compute_ionization_profile("Neutral", pka=7.0, ionization_type="neutral")
    for fi in result.fraction_ionized:
        assert fi == 0.0


def test_ph_at_50pct_ionized_equals_pka():
    pka = 6.5
    result = compute_ionization_profile("Drug", pka=pka)
    assert math.isclose(result.ph_at_50pct_ionized, pka)


# ---------------------------------------------------------------------------
# Physiologically relevant pH points
# ---------------------------------------------------------------------------


def test_weak_acid_gastric_fasted_mostly_neutral():
    """Weak acid pKa=5 at gastric pH 2.0 → mostly neutral (fraction_ionized < 0.1)."""
    result = compute_ionization_profile("WeakAcid", pka=5.0, ionization_type="acid")
    assert result.gi_ionization_fasted < 0.1


def test_weak_base_gastric_fasted_mostly_ionized():
    """Weak base pKa=8 at gastric pH 2.0 → mostly ionized (fraction_ionized > 0.9)."""
    result = compute_ionization_profile("WeakBase", pka=8.0, ionization_type="base")
    assert result.gi_ionization_fasted > 0.9


def test_plasma_ionization_in_0_1():
    result = compute_ionization_profile("Drug", pka=7.4)
    assert 0.0 <= result.plasma_ionization <= 1.0


def test_acid_at_pka_is_50pct_ionized_plasma():
    """Acid with pKa=7.4: exactly 50% ionized at plasma pH."""
    result = compute_ionization_profile("Drug", pka=7.4, ionization_type="acid")
    assert math.isclose(result.plasma_ionization, 0.5, abs_tol=1e-6)


def test_base_at_pka_is_50pct_ionized_intestinal():
    """Base with pKa=6.8: exactly 50% ionized at intestinal pH."""
    result = compute_ionization_profile("Drug", pka=6.8, ionization_type="base")
    assert math.isclose(result.intestinal_ionization, 0.5, abs_tol=1e-6)


def test_notes_is_string():
    result = compute_ionization_profile("Drug", pka=5.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# compare_ionization_profiles
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    drugs = [
        {"drug_name": "AcidDrug", "pka": 4.0, "ionization_type": "acid"},
        {"drug_name": "BaseDrug", "pka": 9.0, "ionization_type": "base"},
    ]
    results = compare_ionization_profiles(drugs)
    assert isinstance(results, list)


def test_compare_sorted_by_plasma_ionization():
    drugs = [
        {"drug_name": "HighIonized", "pka": 3.0, "ionization_type": "acid"},
        {"drug_name": "LowIonized", "pka": 12.0, "ionization_type": "acid"},
    ]
    results = compare_ionization_profiles(drugs)
    # Sorted ascending by plasma_ionization
    assert results[0].plasma_ionization <= results[1].plasma_ionization


def test_compare_returns_correct_count():
    drugs = [
        {"drug_name": f"Drug{i}", "pka": float(i + 1), "ionization_type": "acid"} for i in range(5)
    ]
    results = compare_ionization_profiles(drugs)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_pka_low_raises():
    with pytest.raises(ValueError):
        compute_ionization_profile("Drug", pka=-1.0)


def test_invalid_pka_high_raises():
    with pytest.raises(ValueError):
        compute_ionization_profile("Drug", pka=15.0)


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError):
        compute_ionization_profile("Drug", pka=5.0, ionization_type="zwitterion")
