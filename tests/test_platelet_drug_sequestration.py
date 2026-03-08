"""Tests for Phase 391 — Platelet-Mediated Drug Sequestration Model."""

import pytest

from omega_pbpk.core.platelet_drug_sequestration import (
    PlateletSequestrationResult,
    calculate_platelet_sequestration,
    compare_platelet_counts,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def basic_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        total_plasma_conc_mg_L=1.0,
        drug_type="basic_lipophilic",
        logp=3.0,
        pka_base=8.5,
        platelet_count_per_uL=250000.0,
        fup_baseline=0.1,
    )
    defaults.update(kwargs)
    return calculate_platelet_sequestration(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        r = basic_result()
        assert isinstance(r, PlateletSequestrationResult)

    def test_frozen(self):
        r = basic_result()
        with pytest.raises((AttributeError, TypeError)):
            r.drug_name = "other"  # type: ignore[misc]

    def test_field_names(self):
        r = basic_result()
        assert hasattr(r, "drug_name")
        assert hasattr(r, "platelet_count_per_uL")
        assert hasattr(r, "drug_type")
        assert hasattr(r, "total_plasma_conc_mg_L")
        assert hasattr(r, "free_plasma_conc_mg_L")
        assert hasattr(r, "platelet_bound_fraction")
        assert hasattr(r, "apparent_fup")
        assert hasattr(r, "kp_platelet")
        assert hasattr(r, "sequestration_effect_pct")
        assert hasattr(r, "clinical_impact")
        assert hasattr(r, "notes")


# ---------------------------------------------------------------------------
# Basic values
# ---------------------------------------------------------------------------


class TestBasicValues:
    def test_drug_name_preserved(self):
        r = basic_result(drug_name="Chlorpromazine")
        assert r.drug_name == "Chlorpromazine"

    def test_total_conc_preserved(self):
        r = basic_result(total_plasma_conc_mg_L=5.0)
        assert r.total_plasma_conc_mg_L == pytest.approx(5.0)

    def test_free_conc_less_than_total(self):
        r = basic_result()
        assert r.free_plasma_conc_mg_L < r.total_plasma_conc_mg_L

    def test_bound_fraction_range(self):
        r = basic_result()
        assert 0.0 <= r.platelet_bound_fraction <= 1.0

    def test_effect_pct_equals_bound_fraction_times_100(self):
        r = basic_result()
        assert r.sequestration_effect_pct == pytest.approx(
            r.platelet_bound_fraction * 100.0, rel=1e-6
        )

    def test_apparent_fup_le_baseline(self):
        r = basic_result(fup_baseline=0.1)
        assert r.apparent_fup <= 0.1 + 1e-9

    def test_apparent_fup_positive(self):
        r = basic_result()
        assert r.apparent_fup > 0


# ---------------------------------------------------------------------------
# Drug type comparison
# ---------------------------------------------------------------------------


class TestDrugTypeComparison:
    def test_basic_lipophilic_higher_sequestration_than_neutral(self):
        r_basic = basic_result(drug_type="basic_lipophilic", logp=3.0, pka_base=9.0)
        r_neutral = basic_result(drug_type="neutral", logp=3.0, pka_base=None)
        assert r_basic.platelet_bound_fraction > r_neutral.platelet_bound_fraction

    def test_acidic_low_sequestration(self):
        r_acidic = basic_result(drug_type="acidic", logp=2.0, pka_base=None)
        assert r_acidic.platelet_bound_fraction < 0.05

    def test_basic_lipophilic_kp_greater_than_acidic(self):
        r_basic = basic_result(drug_type="basic_lipophilic", logp=3.0, pka_base=8.0)
        r_acidic = basic_result(drug_type="acidic", logp=3.0, pka_base=None)
        assert r_basic.kp_platelet > r_acidic.kp_platelet

    def test_neutral_kp_positive_logp(self):
        r = basic_result(drug_type="neutral", logp=2.0, pka_base=None)
        assert r.kp_platelet == pytest.approx(1.0 + 2.0 * 0.3, rel=1e-6)

    def test_neutral_kp_negative_logp(self):
        r = basic_result(drug_type="neutral", logp=-1.0, pka_base=None)
        assert r.kp_platelet == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Platelet count effect
# ---------------------------------------------------------------------------


class TestPlateletCountEffect:
    def test_higher_count_more_sequestration(self):
        r_low = basic_result(platelet_count_per_uL=100000.0)
        r_high = basic_result(platelet_count_per_uL=500000.0)
        assert r_high.platelet_bound_fraction > r_low.platelet_bound_fraction

    def test_higher_count_lower_free_conc(self):
        r_low = basic_result(platelet_count_per_uL=100000.0)
        r_high = basic_result(platelet_count_per_uL=500000.0)
        assert r_high.free_plasma_conc_mg_L < r_low.free_plasma_conc_mg_L


# ---------------------------------------------------------------------------
# Clinical impact mapping
# ---------------------------------------------------------------------------


class TestClinicalImpact:
    def test_none_impact_acidic(self):
        r = basic_result(drug_type="acidic", logp=1.0, pka_base=None)
        assert r.clinical_impact == "none"

    def test_major_impact_high_basic(self):
        # Very high kp → major
        r = basic_result(
            drug_type="basic_lipophilic",
            logp=5.0,
            pka_base=10.0,
            platelet_count_per_uL=1000000.0,
        )
        assert r.clinical_impact in ("moderate", "major")

    def test_valid_impact_values(self):
        for drug_type in ("basic_lipophilic", "acidic", "neutral"):
            r = basic_result(
                drug_type=drug_type, pka_base=8.0 if drug_type == "basic_lipophilic" else None
            )
            assert r.clinical_impact in ("none", "minor", "moderate", "major")


# ---------------------------------------------------------------------------
# compare_platelet_counts
# ---------------------------------------------------------------------------


class TestComparePlateletCounts:
    def test_returns_list_of_5(self):
        results = compare_platelet_counts(
            drug_name="Drug",
            total_plasma_conc_mg_L=1.0,
            drug_type="basic_lipophilic",
            logp=3.0,
            pka_base=8.5,
        )
        assert len(results) == 5

    def test_all_results_are_correct_type(self):
        results = compare_platelet_counts(
            drug_name="Drug",
            total_plasma_conc_mg_L=1.0,
            drug_type="basic_lipophilic",
            logp=3.0,
        )
        for r in results:
            assert isinstance(r, PlateletSequestrationResult)

    def test_sorted_descending_free_conc(self):
        results = compare_platelet_counts(
            drug_name="Drug",
            total_plasma_conc_mg_L=1.0,
            drug_type="basic_lipophilic",
            logp=3.0,
            pka_base=8.5,
        )
        free_concs = [r.free_plasma_conc_mg_L for r in results]
        assert free_concs == sorted(free_concs, reverse=True)

    def test_custom_counts(self):
        results = compare_platelet_counts(
            drug_name="Drug",
            total_plasma_conc_mg_L=1.0,
            drug_type="neutral",
            logp=2.0,
            counts=[100000.0, 300000.0, 600000.0],
        )
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_conc_raises(self):
        with pytest.raises(ValueError, match="total_plasma_conc_mg_L"):
            calculate_platelet_sequestration(
                drug_name="D", total_plasma_conc_mg_L=-1.0, drug_type="neutral", logp=2.0
            )

    def test_zero_conc_raises(self):
        with pytest.raises(ValueError, match="total_plasma_conc_mg_L"):
            calculate_platelet_sequestration(
                drug_name="D", total_plasma_conc_mg_L=0.0, drug_type="neutral", logp=2.0
            )

    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError, match="drug_type"):
            calculate_platelet_sequestration(
                drug_name="D", total_plasma_conc_mg_L=1.0, drug_type="unknown", logp=2.0
            )

    def test_zero_platelet_count_raises(self):
        with pytest.raises(ValueError, match="platelet_count_per_uL"):
            calculate_platelet_sequestration(
                drug_name="D",
                total_plasma_conc_mg_L=1.0,
                drug_type="neutral",
                logp=2.0,
                platelet_count_per_uL=0.0,
            )

    def test_invalid_fup_raises(self):
        with pytest.raises(ValueError, match="fup_baseline"):
            calculate_platelet_sequestration(
                drug_name="D",
                total_plasma_conc_mg_L=1.0,
                drug_type="neutral",
                logp=2.0,
                fup_baseline=0.0,
            )

    def test_fup_greater_than_1_raises(self):
        with pytest.raises(ValueError, match="fup_baseline"):
            calculate_platelet_sequestration(
                drug_name="D",
                total_plasma_conc_mg_L=1.0,
                drug_type="neutral",
                logp=2.0,
                fup_baseline=1.5,
            )
