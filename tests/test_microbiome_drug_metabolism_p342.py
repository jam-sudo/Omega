"""Tests for Phase 342 — Microbiome Drug Metabolism Model."""

import pytest

from omega_pbpk.prediction.microbiome_drug_metabolism_p342 import (
    MicrobiomeDrugMetabolismResult,
    simulate_microbiome_drug_metabolism,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _default(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        k_microbial_metabolism_per_h=0.1,
        f_abs_without_microbiome=0.8,
        microbiome_density=1e11,
        transit_time_h=6.0,
        drug_class="other",
        gut_dysbiosis=False,
    )
    defaults.update(kwargs)
    return simulate_microbiome_drug_metabolism(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        result = _default()
        assert isinstance(result, MicrobiomeDrugMetabolismResult)

    def test_is_frozen_dataclass(self):
        result = _default()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Core invariants
# ---------------------------------------------------------------------------


class TestCoreInvariants:
    def test_f_abs_with_leq_f_abs_without(self):
        result = _default()
        assert result.f_abs_with_microbiome <= result.f_abs_without_microbiome

    def test_fraction_metabolized_in_range(self):
        result = _default()
        assert 0.0 <= result.fraction_metabolized_by_microbiome <= 1.0

    def test_fraction_metabolized_max_0_9(self):
        # Even with very high microbial rate, fraction caps at 0.9
        result = _default(k_microbial_metabolism_per_h=5.0, transit_time_h=100.0)
        assert result.fraction_metabolized_by_microbiome <= 0.9

    def test_bioavailability_reduction_in_range(self):
        result = _default()
        assert 0.0 <= result.bioavailability_reduction_pct <= 100.0

    def test_metabolite_fraction_non_negative(self):
        result = _default()
        assert result.metabolite_fraction >= 0.0

    def test_metabolite_auc_fraction_non_negative(self):
        result = _default()
        assert result.metabolite_auc_fraction >= 0.0


# ---------------------------------------------------------------------------
# Dose and density effects
# ---------------------------------------------------------------------------


class TestRateEffects:
    def test_higher_k_microbial_lower_f_abs(self):
        r_low = _default(k_microbial_metabolism_per_h=0.05)
        r_high = _default(k_microbial_metabolism_per_h=1.0)
        assert r_high.f_abs_with_microbiome <= r_low.f_abs_with_microbiome

    def test_zero_microbial_rate_no_reduction(self):
        result = _default(k_microbial_metabolism_per_h=0.0)
        assert abs(result.f_abs_with_microbiome - result.f_abs_without_microbiome) < 1e-9

    def test_higher_density_more_metabolism(self):
        r_low = _default(microbiome_density=1e9)
        r_high = _default(microbiome_density=1e12)
        assert r_high.fraction_metabolized_by_microbiome >= r_low.fraction_metabolized_by_microbiome


# ---------------------------------------------------------------------------
# Dysbiosis effects
# ---------------------------------------------------------------------------


class TestDysbiosisEffects:
    def test_dysbiosis_reduces_metabolism(self):
        r_normal = _default(gut_dysbiosis=False)
        r_dysbiosis = _default(gut_dysbiosis=True)
        # Reduced microbiome → less metabolism → higher f_abs
        assert r_dysbiosis.f_abs_with_microbiome >= r_normal.f_abs_with_microbiome

    def test_dysbiosis_impact_label_normal(self):
        result = _default(gut_dysbiosis=False)
        assert result.dysbiosis_impact == "none"

    def test_dysbiosis_impact_label_dysbiosis(self):
        result = _default(gut_dysbiosis=True)
        assert result.dysbiosis_impact == "reduced_metabolism"


# ---------------------------------------------------------------------------
# Drug class effects
# ---------------------------------------------------------------------------


class TestDrugClassEffects:
    def test_antibiotic_reduces_effective_rate(self):
        r_other = _default(drug_class="other", k_microbial_metabolism_per_h=0.5)
        r_abx = _default(drug_class="antibiotic", k_microbial_metabolism_per_h=0.5)
        # Antibiotic has self-depletion → lower effective rate
        assert r_abx.effective_microbial_rate_per_h <= r_other.effective_microbial_rate_per_h

    def test_all_drug_classes_accepted(self):
        for drug_class in ("antibiotic", "NSAID", "cardiovascular", "other"):
            result = _default(drug_class=drug_class)
            assert result.drug_class == drug_class


# ---------------------------------------------------------------------------
# Clinical implication
# ---------------------------------------------------------------------------


class TestClinicalImplication:
    def test_clinical_implication_valid_values(self):
        result = _default()
        assert result.clinical_implication in {"negligible", "minor", "significant"}

    def test_negligible_when_no_metabolism(self):
        result = _default(k_microbial_metabolism_per_h=0.0)
        assert result.clinical_implication == "negligible"

    def test_significant_when_high_metabolism(self):
        result = _default(
            k_microbial_metabolism_per_h=3.0,
            transit_time_h=12.0,
            microbiome_density=1e12,
        )
        assert result.clinical_implication == "significant"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_k_microbial_negative(self):
        with pytest.raises(ValueError, match="k_microbial"):
            _default(k_microbial_metabolism_per_h=-0.1)

    def test_k_microbial_too_high(self):
        with pytest.raises(ValueError, match="k_microbial"):
            _default(k_microbial_metabolism_per_h=6.0)

    def test_f_abs_zero(self):
        with pytest.raises(ValueError, match="f_abs_without"):
            _default(f_abs_without_microbiome=0.0)

    def test_f_abs_too_high(self):
        with pytest.raises(ValueError, match="f_abs_without"):
            _default(f_abs_without_microbiome=1.5)

    def test_transit_time_zero(self):
        with pytest.raises(ValueError, match="transit_time"):
            _default(transit_time_h=0.0)

    def test_transit_time_negative(self):
        with pytest.raises(ValueError, match="transit_time"):
            _default(transit_time_h=-1.0)

    def test_invalid_drug_class(self):
        with pytest.raises(ValueError, match="drug_class"):
            _default(drug_class="unknown_class")
