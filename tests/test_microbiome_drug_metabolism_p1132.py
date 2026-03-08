"""Tests for Phase 1132 — Gut microbiome drug metabolism prediction."""

import math

import pytest

from omega_pbpk.prediction.microbiome_drug_metabolism import (
    MicrobiomeMetabolismResult,
    predict_microbiome_metabolism,
    rank_microbiome_substrates,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def default_predict(**kwargs) -> MicrobiomeMetabolismResult:
    params = dict(
        drug_name="TestDrug",
        logP=2.0,
        ionization_type="neutral",
        has_azo_bond=False,
        has_nitro_group=False,
        has_glucuronide=False,
        residence_time_h=24.0,
        colonic_pH=6.5,
    )
    params.update(kwargs)
    return predict_microbiome_metabolism(**params)


# ---------------------------------------------------------------------------
# Pathway-specific tests
# ---------------------------------------------------------------------------


class TestPathways:
    def test_azo_bond_increases_rate(self):
        no_azo = default_predict(has_azo_bond=False)
        with_azo = default_predict(has_azo_bond=True)
        assert with_azo.total_metabolic_rate_per_h > no_azo.total_metabolic_rate_per_h

    def test_nitro_group_increases_rate(self):
        no_nitro = default_predict(has_nitro_group=False)
        with_nitro = default_predict(has_nitro_group=True)
        assert with_nitro.total_metabolic_rate_per_h > no_nitro.total_metabolic_rate_per_h

    def test_glucuronide_increases_rate(self):
        no_gluc = default_predict(has_glucuronide=False)
        with_gluc = default_predict(has_glucuronide=True)
        assert with_gluc.total_metabolic_rate_per_h > no_gluc.total_metabolic_rate_per_h

    def test_glucuronide_higher_than_passive_alone(self):
        passive = default_predict(logP=5.0)
        gluc = default_predict(logP=5.0, has_glucuronide=True)
        assert gluc.total_metabolic_rate_per_h > passive.total_metabolic_rate_per_h

    def test_no_functional_groups_passive_only(self):
        r = default_predict(
            has_azo_bond=False,
            has_nitro_group=False,
            has_glucuronide=False,
            logP=0.0,
        )
        # passive_rate = 0.01 * max(0, 3-0) = 0.03
        assert r.primary_pathway == "passive_fermentation"

    def test_glucuronide_primary_pathway(self):
        r = default_predict(has_glucuronide=True, logP=5.0)
        assert r.primary_pathway == "glucuronide_cleavage"

    def test_azo_primary_pathway_low_ph(self):
        # azoreduction rate at pH 5.0 = 0.15 * (8-5)/3 = 0.15
        r = default_predict(has_azo_bond=True, colonic_pH=5.0, logP=5.0)
        assert r.primary_pathway == "azoreduction"

    def test_azo_ph_factor_at_ph8_lower_than_ph5(self):
        r_ph8 = default_predict(has_azo_bond=True, colonic_pH=8.0, logP=5.0)
        r_ph5 = default_predict(has_azo_bond=True, colonic_pH=5.0, logP=5.0)
        assert r_ph5.total_metabolic_rate_per_h > r_ph8.total_metabolic_rate_per_h


# ---------------------------------------------------------------------------
# Hydrophilic passive fermentation
# ---------------------------------------------------------------------------


class TestPassiveFermentation:
    def test_hydrophilic_higher_passive_rate(self):
        r_hydrophilic = default_predict(logP=-2.0)
        r_lipophilic = default_predict(logP=5.0)
        assert r_hydrophilic.total_metabolic_rate_per_h > r_lipophilic.total_metabolic_rate_per_h

    def test_logp_above_3_zero_passive(self):
        r = default_predict(logP=4.0)
        # max(0, 3-4) = 0 → passive_rate = 0, no other pathways
        assert r.total_metabolic_rate_per_h == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# metabolic_fraction bounds
# ---------------------------------------------------------------------------


class TestMetabolicFraction:
    def test_fraction_between_0_and_1(self):
        for logP in [-5.0, 0.0, 2.0, 5.0, 10.0]:
            r = default_predict(logP=logP)
            assert 0.0 <= r.metabolic_fraction <= 1.0

    def test_fraction_with_all_pathways(self):
        r = default_predict(
            has_azo_bond=True,
            has_nitro_group=True,
            has_glucuronide=True,
            logP=0.0,
            colonic_pH=5.0,
        )
        assert 0.0 < r.metabolic_fraction <= 1.0

    def test_metabolized_pct_equals_fraction_times_100(self):
        r = default_predict(has_glucuronide=True)
        assert r.metabolized_pct == pytest.approx(r.metabolic_fraction * 100, rel=1e-6)

    def test_longer_residence_higher_fraction(self):
        r_short = default_predict(logP=0.0, residence_time_h=1.0)
        r_long = default_predict(logP=0.0, residence_time_h=24.0)
        assert r_long.metabolic_fraction > r_short.metabolic_fraction

    def test_fraction_formula(self):
        r = default_predict(has_glucuronide=True, logP=5.0, residence_time_h=10.0)
        # only glucuronide: rate=0.20, fraction = 1-exp(-0.20*10)
        expected = 1.0 - math.exp(-0.20 * 10.0)
        assert r.metabolic_fraction == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Reabsorption and bioavailability impact
# ---------------------------------------------------------------------------


class TestReabsorptionAndImpact:
    def test_base_higher_reabsorption(self):
        r_base = default_predict(ionization_type="base")
        r_acid = default_predict(ionization_type="acid")
        assert r_base.reabsorption_fraction > r_acid.reabsorption_fraction

    def test_neutral_reabsorption_is_01(self):
        r = default_predict(ionization_type="neutral")
        assert r.reabsorption_fraction == pytest.approx(0.1)

    def test_high_reduction_impact(self):
        # large glucuronide rate + long residence → >50%
        r = default_predict(has_glucuronide=True, residence_time_h=48.0)
        assert r.bioavailability_impact == "high_reduction"

    def test_low_impact_when_minimal_metabolism(self):
        # No pathways active at logP=5 → metabolized_pct=0 → low
        r = default_predict(logP=5.0)
        assert r.bioavailability_impact == "low"

    def test_bioavailability_impact_in_valid_set(self):
        for logP in [0.0, 2.0, 5.0]:
            r = default_predict(logP=logP)
            assert r.bioavailability_impact in {"high_reduction", "moderate", "low"}


# ---------------------------------------------------------------------------
# rank_microbiome_substrates
# ---------------------------------------------------------------------------


class TestRanking:
    def _make_drug(self, name, logP, gluc=False, azo=False, nitro=False):
        return dict(
            drug_name=name,
            logP=logP,
            ionization_type="neutral",
            has_azo_bond=azo,
            has_nitro_group=nitro,
            has_glucuronide=gluc,
            residence_time_h=24.0,
            colonic_pH=6.5,
        )

    def test_sorted_descending(self):
        drugs = [
            self._make_drug("NoPathway", logP=5.0),
            self._make_drug("Glucuronide", logP=5.0, gluc=True),
            self._make_drug("Hydrophilic", logP=0.0),
        ]
        ranked = rank_microbiome_substrates(drugs)
        fractions = [r.metabolic_fraction for r in ranked]
        assert fractions == sorted(fractions, reverse=True)

    def test_returns_list_of_results(self):
        drugs = [self._make_drug("A", logP=2.0)]
        ranked = rank_microbiome_substrates(drugs)
        assert isinstance(ranked, list)
        assert isinstance(ranked[0], MicrobiomeMetabolismResult)

    def test_single_drug(self):
        drugs = [self._make_drug("Only", logP=1.0)]
        ranked = rank_microbiome_substrates(drugs)
        assert len(ranked) == 1

    def test_empty_list(self):
        ranked = rank_microbiome_substrates([])
        assert ranked == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_logp_too_low(self):
        with pytest.raises(ValueError, match="logP"):
            default_predict(logP=-6.0)

    def test_logp_too_high(self):
        with pytest.raises(ValueError, match="logP"):
            default_predict(logP=11.0)

    def test_invalid_ionization_type(self):
        with pytest.raises(ValueError, match="ionization_type"):
            default_predict(ionization_type="zwitterion")

    def test_residence_time_zero(self):
        with pytest.raises(ValueError, match="residence_time_h"):
            default_predict(residence_time_h=0.0)

    def test_residence_time_negative(self):
        with pytest.raises(ValueError, match="residence_time_h"):
            default_predict(residence_time_h=-1.0)

    def test_residence_time_over_72(self):
        with pytest.raises(ValueError, match="residence_time_h"):
            default_predict(residence_time_h=73.0)

    def test_colonic_ph_too_low(self):
        with pytest.raises(ValueError, match="colonic_pH"):
            default_predict(colonic_pH=4.9)

    def test_colonic_ph_too_high(self):
        with pytest.raises(ValueError, match="colonic_pH"):
            default_predict(colonic_pH=8.1)


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


class TestResultStructure:
    def test_result_type(self):
        r = default_predict()
        assert isinstance(r, MicrobiomeMetabolismResult)

    def test_drug_name_preserved(self):
        r = default_predict(drug_name="MyDrug")
        assert r.drug_name == "MyDrug"

    def test_notes_non_empty(self):
        r = default_predict()
        assert len(r.notes) > 0

    def test_result_is_frozen(self):
        r = default_predict()
        with pytest.raises(Exception):
            r.drug_name = "changed"  # type: ignore[misc]
