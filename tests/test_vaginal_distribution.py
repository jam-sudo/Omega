"""Tests for Phase 911 — Drug Vaginal Distribution."""

import math
import pytest

from omega_pbpk.clinical.vaginal_distribution import (
    VaginalDistributionResult,
    predict_vaginal_distribution,
    screen_vaginal_drugs,
)


# ---------------------------------------------------------------------------
# Basic instantiation / return type
# ---------------------------------------------------------------------------

def test_returns_dataclass():
    result = predict_vaginal_distribution("clotrimazole", dose_mg=100.0, pka=5.0, logp=3.0)
    assert isinstance(result, VaginalDistributionResult)


def test_drug_name_preserved():
    result = predict_vaginal_distribution("miconazole", dose_mg=200.0, pka=6.0, logp=4.0)
    assert result.drug_name == "miconazole"


def test_dose_preserved():
    result = predict_vaginal_distribution("fluconazole", dose_mg=150.0, pka=2.0, logp=0.5)
    assert result.dose_mg == 150.0


def test_pka_preserved():
    result = predict_vaginal_distribution("drugA", dose_mg=50.0, pka=4.5, logp=1.0)
    assert result.pka == 4.5


def test_logp_preserved():
    result = predict_vaginal_distribution("drugB", dose_mg=50.0, pka=4.5, logp=2.5)
    assert result.logp == 2.5


def test_ionization_type_default_acid():
    result = predict_vaginal_distribution("drugC", dose_mg=50.0, pka=4.5, logp=1.0)
    assert result.ionization_type == "acid"


def test_vaginal_ph_default():
    result = predict_vaginal_distribution("drugD", dose_mg=50.0, pka=4.5, logp=1.0)
    assert result.vaginal_ph == 4.2


def test_vaginal_ph_custom():
    result = predict_vaginal_distribution(
        "drugE", dose_mg=50.0, pka=4.5, logp=1.0, vaginal_ph=3.8
    )
    assert result.vaginal_ph == 3.8


# ---------------------------------------------------------------------------
# Henderson-Hasselbalch ratio logic
# ---------------------------------------------------------------------------

def test_acid_ratio_higher_than_neutral():
    """Acid with pKa above vaginal pH has less ionization locally → smaller local retention."""
    acid = predict_vaginal_distribution(
        "acid_drug", dose_mg=100.0, pka=6.0, logp=1.0, ionization_type="acid"
    )
    neutral = predict_vaginal_distribution(
        "neutral_drug", dose_mg=100.0, pka=6.0, logp=1.0, ionization_type="neutral"
    )
    # Acid with pKa > vaginal_ph but < plasma_pH: more ionized at plasma → trapped in vagina
    # vaginal_plasma_ratio for acid = (1+10^(4.2-6))/(1+10^(7.4-6))
    assert acid.vaginal_plasma_ratio != neutral.vaginal_plasma_ratio


def test_base_trapped_at_acid_ph():
    """Base with pKa > vaginal pH is highly ionized at vaginal pH → high ratio."""
    base = predict_vaginal_distribution(
        "base_drug", dose_mg=100.0, pka=9.0, logp=1.0, ionization_type="base"
    )
    # ratio = (1+10^(9-4.2))/(1+10^(9-7.4)) → very large
    assert base.vaginal_plasma_ratio > 10.0


def test_neutral_ratio_equals_one():
    result = predict_vaginal_distribution(
        "neutral_drug", dose_mg=100.0, pka=7.0, logp=1.0, ionization_type="neutral"
    )
    assert result.vaginal_plasma_ratio == 1.0


def test_ratio_clamped_min():
    """Very unfavorable case should still be at least 0.5."""
    result = predict_vaginal_distribution(
        "drugF", dose_mg=10.0, pka=1.0, logp=0.0, ionization_type="acid"
    )
    assert result.vaginal_plasma_ratio >= 0.5


def test_ratio_clamped_max():
    """Ratio should not exceed 200."""
    result = predict_vaginal_distribution(
        "drugG", dose_mg=10.0, pka=12.0, logp=0.0, ionization_type="base"
    )
    assert result.vaginal_plasma_ratio <= 200.0


# ---------------------------------------------------------------------------
# Systemic absorption fraction
# ---------------------------------------------------------------------------

def test_acid_lower_systemic_absorption_than_base():
    acid = predict_vaginal_distribution(
        "acid", dose_mg=100.0, pka=5.0, logp=1.0, ionization_type="acid"
    )
    base = predict_vaginal_distribution(
        "base", dose_mg=100.0, pka=5.0, logp=1.0, ionization_type="base"
    )
    assert acid.systemic_absorption_fraction < base.systemic_absorption_fraction


def test_high_logp_increases_absorption():
    low = predict_vaginal_distribution(
        "low_logp", dose_mg=100.0, pka=5.0, logp=0.0, ionization_type="neutral"
    )
    high = predict_vaginal_distribution(
        "high_logp", dose_mg=100.0, pka=5.0, logp=5.0, ionization_type="neutral"
    )
    assert high.systemic_absorption_fraction > low.systemic_absorption_fraction


def test_absorption_fraction_clamped():
    result = predict_vaginal_distribution(
        "high_logp_base", dose_mg=100.0, pka=5.0, logp=20.0, ionization_type="base"
    )
    assert result.systemic_absorption_fraction <= 0.8
    assert result.systemic_absorption_fraction >= 0.01


# ---------------------------------------------------------------------------
# Concentration and selectivity
# ---------------------------------------------------------------------------

def test_vaginal_conc_positive():
    result = predict_vaginal_distribution("clotrimazole", dose_mg=100.0, pka=5.0, logp=3.0)
    assert result.predicted_vaginal_conc_ug_mL > 0.0


def test_antifungal_adequacy_flag():
    """High dose should flag as adequate for antifungal."""
    result = predict_vaginal_distribution("clotrimazole", dose_mg=100.0, pka=5.0, logp=3.0)
    assert result.antifungal_local_conc_adequate is True


def test_low_dose_may_be_inadequate():
    """Very low dose may produce inadequate concentrations."""
    result = predict_vaginal_distribution(
        "low_dose_drug", dose_mg=0.001, pka=7.0, logp=1.0, ionization_type="neutral"
    )
    assert result.antifungal_local_conc_adequate is False


def test_local_selectivity_positive():
    result = predict_vaginal_distribution("clotrimazole", dose_mg=100.0, pka=5.0, logp=2.0)
    assert result.local_selectivity_index > 0.0


def test_notes_not_empty():
    result = predict_vaginal_distribution("clotrimazole", dose_mg=100.0, pka=5.0, logp=2.0)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_vaginal_distribution("drug", dose_mg=-1.0, pka=5.0, logp=1.0)


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_vaginal_distribution("drug", dose_mg=0.0, pka=5.0, logp=1.0)


def test_invalid_pka_low_raises():
    with pytest.raises(ValueError, match="pka"):
        predict_vaginal_distribution("drug", dose_mg=100.0, pka=-1.0, logp=1.0)


def test_invalid_pka_high_raises():
    with pytest.raises(ValueError, match="pka"):
        predict_vaginal_distribution("drug", dose_mg=100.0, pka=15.0, logp=1.0)


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_vaginal_distribution(
            "drug", dose_mg=100.0, pka=5.0, logp=1.0, ionization_type="zwitterion"
        )


# ---------------------------------------------------------------------------
# Screen function
# ---------------------------------------------------------------------------

def test_screen_returns_list():
    candidates = [
        {"name": "drug1", "pka": 5.0, "logp": 2.0, "dose_mg": 100.0, "ionization_type": "acid"},
        {"name": "drug2", "pka": 9.0, "logp": 1.0, "dose_mg": 50.0, "ionization_type": "base"},
        {"name": "drug3", "pka": 7.0, "logp": 1.0, "dose_mg": 200.0, "ionization_type": "neutral"},
    ]
    results = screen_vaginal_drugs(candidates)
    assert len(results) == 3
    assert all(isinstance(r, VaginalDistributionResult) for r in results)


def test_screen_sorted_by_selectivity():
    candidates = [
        {"name": "drug1", "pka": 5.0, "logp": 2.0, "dose_mg": 100.0, "ionization_type": "acid"},
        {"name": "drug2", "pka": 9.0, "logp": 1.0, "dose_mg": 50.0, "ionization_type": "base"},
        {"name": "drug3", "pka": 7.0, "logp": 1.0, "dose_mg": 200.0, "ionization_type": "neutral"},
    ]
    results = screen_vaginal_drugs(candidates)
    indices = [r.local_selectivity_index for r in results]
    assert indices == sorted(indices, reverse=True)


def test_screen_default_ionization_type():
    """Missing ionization_type should default to 'acid'."""
    candidates = [
        {"name": "default_acid", "pka": 5.0, "logp": 1.0, "dose_mg": 100.0},
    ]
    results = screen_vaginal_drugs(candidates)
    assert results[0].ionization_type == "acid"


def test_screen_empty_list():
    results = screen_vaginal_drugs([])
    assert results == []


def test_plasma_conc_scales_with_dose():
    low = predict_vaginal_distribution("drug", dose_mg=50.0, pka=5.0, logp=1.0)
    high = predict_vaginal_distribution("drug", dose_mg=200.0, pka=5.0, logp=1.0)
    assert high.plasma_conc_mg_L > low.plasma_conc_mg_L


def test_high_selectivity_note_included():
    """Base drug with high pKa at acid vaginal pH should have high selectivity note."""
    result = predict_vaginal_distribution(
        "base_high_pka", dose_mg=10.0, pka=10.0, logp=0.0, ionization_type="base"
    )
    if result.local_selectivity_index > 10.0:
        assert "High local selectivity" in result.notes


def test_result_is_frozen():
    result = predict_vaginal_distribution("clotrimazole", dose_mg=100.0, pka=5.0, logp=3.0)
    with pytest.raises(Exception):
        result.drug_name = "other"  # type: ignore[misc]
