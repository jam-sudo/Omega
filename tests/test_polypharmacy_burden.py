"""Tests for Phase 263: polypharmacy burden scoring."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.polypharmacy_burden import (
    DrugEntry,
    PolypharmacyResult,
    assess_polypharmacy_burden,
)

VALID_CATS = {"low", "moderate", "high", "very_high"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_safe_drug() -> list[DrugEntry]:
    return [DrugEntry(name="aspirin_low", drug_class="other")]


def _high_risk_regimen() -> list[DrugEntry]:
    """10 high-risk drugs with major DDIs each."""
    classes = [
        "anticoagulant",
        "nsaid",
        "benzodiazepine",
        "opioid",
        "antipsychotic",
        "digoxin",
        "insulin",
        "anticholinergic",
        "diuretic",
        "antiepileptic",
    ]
    drugs = []
    for i, cls in enumerate(classes):
        # Each drug has one major DDI with the next drug (bidirectional)
        other = classes[(i + 1) % len(classes)]
        drugs.append(
            DrugEntry(
                name=f"drug_{cls}",
                drug_class=cls,
                ddi_pairs=[f"drug_{other}"],
                ddi_severity=["major"],
            )
        )
    return drugs


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_empty_drugs_raises():
    with pytest.raises(ValueError, match="empty"):
        assess_polypharmacy_burden([])


def test_invalid_age_zero_raises():
    with pytest.raises(ValueError, match="patient_age"):
        assess_polypharmacy_burden(_single_safe_drug(), patient_age=0)


def test_invalid_age_negative_raises():
    with pytest.raises(ValueError, match="patient_age"):
        assess_polypharmacy_burden(_single_safe_drug(), patient_age=-5)


# ---------------------------------------------------------------------------
# Basic structural tests
# ---------------------------------------------------------------------------


def test_returns_polypharmacy_result():
    result = assess_polypharmacy_burden(_single_safe_drug())
    assert isinstance(result, PolypharmacyResult)


def test_single_safe_drug_low_risk():
    result = assess_polypharmacy_burden(_single_safe_drug())
    assert result.risk_category == "low"


def test_n_drugs_equals_len_drugs():
    drugs = _single_safe_drug() + [DrugEntry(name="metformin", drug_class="other")]
    result = assess_polypharmacy_burden(drugs)
    assert result.n_drugs == len(drugs)


def test_drug_names_match():
    drugs = [DrugEntry(name="warfarin", drug_class="anticoagulant")]
    result = assess_polypharmacy_burden(drugs)
    assert "warfarin" in result.drug_names


def test_n_high_risk_drugs_le_n_drugs():
    drugs = _high_risk_regimen()
    result = assess_polypharmacy_burden(drugs)
    assert result.n_high_risk_drugs <= result.n_drugs


def test_total_burden_score_nonnegative():
    result = assess_polypharmacy_burden(_single_safe_drug())
    assert result.total_burden_score >= 0.0


def test_risk_category_valid():
    drugs = _high_risk_regimen()
    result = assess_polypharmacy_burden(drugs)
    assert result.risk_category in VALID_CATS


def test_recommendation_nonempty():
    result = assess_polypharmacy_burden(_single_safe_drug())
    assert len(result.recommendation) > 0


# ---------------------------------------------------------------------------
# High-risk regimen
# ---------------------------------------------------------------------------


def test_high_risk_regimen_high_or_very_high():
    drugs = _high_risk_regimen()
    result = assess_polypharmacy_burden(drugs)
    assert result.risk_category in {"high", "very_high"}


def test_high_risk_regimen_positive_ddi_score():
    drugs = _high_risk_regimen()
    result = assess_polypharmacy_burden(drugs)
    assert result.ddi_burden_score > 0.0


def test_high_risk_drug_names_contains_anticoagulant():
    drugs = [DrugEntry(name="warfarin", drug_class="anticoagulant")]
    result = assess_polypharmacy_burden(drugs)
    assert "warfarin" in result.high_risk_drug_names


def test_other_drug_class_not_high_risk():
    drugs = [DrugEntry(name="metformin", drug_class="other")]
    result = assess_polypharmacy_burden(drugs)
    assert result.n_high_risk_drugs == 0


# ---------------------------------------------------------------------------
# Age effect
# ---------------------------------------------------------------------------


def test_elderly_higher_score_than_young():
    drugs = [
        DrugEntry(name="lorazepam", drug_class="benzodiazepine"),
        DrugEntry(name="oxycodone", drug_class="opioid"),
    ]
    result_old = assess_polypharmacy_burden(drugs, patient_age=80)
    result_young = assess_polypharmacy_burden(drugs, patient_age=30)
    assert result_old.total_burden_score > result_young.total_burden_score


def test_age_boundary_65():
    drugs = [DrugEntry(name="digoxin_drug", drug_class="digoxin")]
    result_64 = assess_polypharmacy_burden(drugs, patient_age=64)
    result_65 = assess_polypharmacy_burden(drugs, patient_age=65)
    assert result_65.total_burden_score > result_64.total_burden_score


# ---------------------------------------------------------------------------
# DDI scoring
# ---------------------------------------------------------------------------


def test_ddi_burden_increases_with_more_pairs():
    single_ddi = [
        DrugEntry(name="drugA", drug_class="other", ddi_pairs=["drugB"], ddi_severity=["major"]),
        DrugEntry(name="drugB", drug_class="other", ddi_pairs=["drugA"], ddi_severity=["major"]),
    ]
    multi_ddi = [
        DrugEntry(
            name="drugA",
            drug_class="other",
            ddi_pairs=["drugB", "drugC"],
            ddi_severity=["major", "major"],
        ),
        DrugEntry(name="drugB", drug_class="other", ddi_pairs=["drugA"], ddi_severity=["major"]),
        DrugEntry(name="drugC", drug_class="other", ddi_pairs=["drugA"], ddi_severity=["major"]),
    ]
    r1 = assess_polypharmacy_burden(single_ddi)
    r2 = assess_polypharmacy_burden(multi_ddi)
    assert r2.ddi_burden_score > r1.ddi_burden_score


def test_no_ddi_zero_ddi_score():
    drugs = [DrugEntry(name="safe_drug", drug_class="other")]
    result = assess_polypharmacy_burden(drugs)
    assert result.ddi_burden_score == 0.0


def test_minor_ddi_lower_than_major():
    minor = [
        DrugEntry(name="A", drug_class="other", ddi_pairs=["B"], ddi_severity=["minor"]),
        DrugEntry(name="B", drug_class="other", ddi_pairs=["A"], ddi_severity=["minor"]),
    ]
    major = [
        DrugEntry(name="A", drug_class="other", ddi_pairs=["B"], ddi_severity=["major"]),
        DrugEntry(name="B", drug_class="other", ddi_pairs=["A"], ddi_severity=["major"]),
    ]
    r_minor = assess_polypharmacy_burden(minor)
    r_major = assess_polypharmacy_burden(major)
    assert r_major.ddi_burden_score > r_minor.ddi_burden_score


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_contains_total():
    drugs = _single_safe_drug()
    result = assess_polypharmacy_burden(drugs)
    assert "Total=" in result.notes
