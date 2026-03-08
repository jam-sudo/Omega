"""Tests for phase 671 — Drug Microbiome Interaction."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.microbiome_interaction_p671 import (
    MicrobiomeInteractionResult,
    predict_microbiome_interaction,
)

# ── helpers ──────────────────────────────────────────────────────────────


def _default(**kw):
    defaults = dict(
        drug_name="TestDrug",
        drug_class="other",
        logP=2.0,
        mw_Da=300.0,
    )
    defaults.update(kw)
    return predict_microbiome_interaction(**defaults)


# ── type & immutability ─────────────────────────────────────────────────


class TestResultType:
    def test_returns_correct_type(self):
        r = _default()
        assert isinstance(r, MicrobiomeInteractionResult)

    def test_frozen_raises_on_mutation(self):
        r = _default()
        with pytest.raises(AttributeError):
            r.drug_name = "X"  # type: ignore[misc]

    def test_frozen_raises_on_new_field(self):
        r = _default()
        with pytest.raises(AttributeError):
            r.new_field = 42  # type: ignore[attr-defined]


# ── antibiotic behaviour ────────────────────────────────────────────────


class TestAntibiotics:
    def test_antibiotic_high_disruption(self):
        r = _default(drug_class="antibiotic", is_antibiotic=True, antibiotic_spectrum="broad")
        assert r.microbiome_disruption_score > 5

    def test_broad_greater_than_narrow(self):
        broad = _default(drug_class="antibiotic", is_antibiotic=True, antibiotic_spectrum="broad")
        narrow = _default(drug_class="antibiotic", is_antibiotic=True, antibiotic_spectrum="narrow")
        assert broad.microbiome_disruption_score > narrow.microbiome_disruption_score

    def test_antibiotic_low_bacterial_met(self):
        r = _default(drug_class="antibiotic", is_antibiotic=True, antibiotic_spectrum="broad")
        assert r.bacterial_metabolism_fraction == pytest.approx(0.02)

    def test_antibiotic_class_set(self):
        r = _default(drug_class="antibiotic", is_antibiotic=True, antibiotic_spectrum="broad")
        assert r.antibiotic_class == "broad"

    def test_non_antibiotic_class_na(self):
        r = _default()
        assert r.antibiotic_class == "N/A"


# ── logP effects ────────────────────────────────────────────────────────


class TestLogPEffects:
    def test_high_logP_low_bacterial_met(self):
        r = _default(logP=5.0)
        assert r.bacterial_metabolism_fraction <= 0.02

    def test_negative_logP_higher_bacterial_met(self):
        r = _default(logP=-1.0)
        assert r.bacterial_metabolism_fraction >= 0.15

    def test_moderate_logP(self):
        r = _default(logP=1.0)
        assert r.bacterial_metabolism_fraction == pytest.approx(0.15)

    def test_logP_between_2_and_4(self):
        r = _default(logP=3.0)
        assert r.bacterial_metabolism_fraction == pytest.approx(0.05)


# ── bioavailability ─────────────────────────────────────────────────────


class TestBioavailability:
    def test_with_microbiome_le_without(self):
        r = _default()
        assert r.drug_bioavailability_with_microbiome <= r.drug_bioavailability_without_microbiome

    def test_bioavailability_change_negative_or_zero(self):
        r = _default()
        assert r.bioavailability_change_pct <= 0

    def test_bioavailability_bounded(self):
        r = _default(logP=10.0)
        assert 0 <= r.drug_bioavailability_with_microbiome <= 1
        assert 0 <= r.drug_bioavailability_without_microbiome <= 1


# ── disruption score ────────────────────────────────────────────────────


class TestDisruptionScore:
    def test_score_in_range(self):
        for lp in [-2, 0, 2, 5, 8]:
            r = _default(logP=lp)
            assert 0 <= r.microbiome_disruption_score <= 10

    def test_ppi_disruption(self):
        r = _default(drug_class="PPI")
        assert r.microbiome_disruption_score == pytest.approx(4.0)

    def test_nsaid_disruption(self):
        r = _default(drug_class="NSAID")
        assert r.microbiome_disruption_score == pytest.approx(3.0)


# ── recovery time ───────────────────────────────────────────────────────


class TestRecoveryTime:
    def test_recovery_non_negative(self):
        r = _default()
        assert r.recovery_time_weeks >= 0

    def test_higher_disruption_longer_recovery(self):
        abx = _default(drug_class="antibiotic", is_antibiotic=True, antibiotic_spectrum="broad")
        other = _default(drug_class="other", logP=0.5)
        assert abx.recovery_time_weeks > other.recovery_time_weeks


# ── clinical impact ─────────────────────────────────────────────────────


class TestClinicalImpact:
    def test_significant(self):
        r = _default(drug_class="antibiotic", is_antibiotic=True, antibiotic_spectrum="broad")
        assert r.clinical_impact == "significant"

    def test_moderate(self):
        r = _default(drug_class="PPI")
        assert r.clinical_impact == "moderate"

    def test_minimal(self):
        r = _default(drug_class="other", logP=1.0)
        assert r.clinical_impact == "minimal"


# ── active metabolite ───────────────────────────────────────────────────


class TestActiveMetabolite:
    def test_non_negative(self):
        r = _default()
        assert r.active_metabolite_formed_pct >= 0

    def test_nsaid_with_high_met(self):
        r = _default(drug_class="NSAID", logP=-1.0)
        assert r.active_metabolite_formed_pct > 0


# ── validation ──────────────────────────────────────────────────────────


class TestValidation:
    def test_invalid_mw(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _default(mw_Da=0)

    def test_invalid_drug_class(self):
        with pytest.raises(ValueError, match="drug_class"):
            _default(drug_class="unknown")
