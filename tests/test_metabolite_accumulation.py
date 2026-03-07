"""Tests for Phase 340 — Drug Metabolite Accumulation."""

import pytest

from omega_pbpk.clinical.metabolite_accumulation import (
    MetaboliteAccumulationResult,
    compare_metabolite_half_lives,
    simulate_metabolite_accumulation,
)

# ---------------------------------------------------------------------------
# Helper: sensible defaults
# ---------------------------------------------------------------------------

_DEFAULTS = dict(
    drug_name="DrugA",
    metabolite_name="MetabA",
    dose_mg=100.0,
    cl_parent_L_per_h=5.0,
    vd_parent_L=50.0,
    cl_metabolite_L_per_h=10.0,
    vd_metabolite_L=60.0,
    fm=0.3,
)


def _sim(**overrides):
    kw = {**_DEFAULTS, **overrides}
    return simulate_metabolite_accumulation(**kw)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_cl_parent_zero_raises(self):
        with pytest.raises(ValueError, match="cl_parent"):
            _sim(cl_parent_L_per_h=0.0)

    def test_vd_parent_zero_raises(self):
        with pytest.raises(ValueError, match="vd_parent"):
            _sim(vd_parent_L=0.0)

    def test_cl_metabolite_zero_raises(self):
        with pytest.raises(ValueError, match="cl_metabolite"):
            _sim(cl_metabolite_L_per_h=0.0)

    def test_vd_metabolite_zero_raises(self):
        with pytest.raises(ValueError, match="vd_metabolite"):
            _sim(vd_metabolite_L=0.0)

    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _sim(dose_mg=0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _sim(dose_mg=-100.0)

    def test_fm_zero_raises(self):
        with pytest.raises(ValueError, match="fm"):
            _sim(fm=0.0)

    def test_fm_greater_than_one_raises(self):
        with pytest.raises(ValueError, match="fm"):
            _sim(fm=1.5)

    def test_f_oral_zero_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            _sim(f_oral=0.0)

    def test_f_oral_greater_than_one_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            _sim(f_oral=1.5)

    def test_n_doses_zero_raises(self):
        with pytest.raises(ValueError, match="n_doses"):
            _sim(n_doses=0)

    def test_interval_h_zero_raises(self):
        with pytest.raises(ValueError, match="interval_h"):
            _sim(interval_h=0.0)


# ---------------------------------------------------------------------------
# Concentration non-negativity
# ---------------------------------------------------------------------------


class TestNonNegativity:
    def test_parent_concentrations_non_negative(self):
        r = _sim()
        assert all(c >= 0.0 for c in r.c_parent_mg_L)

    def test_metabolite_concentrations_non_negative(self):
        r = _sim()
        assert all(c >= 0.0 for c in r.c_metabolite_mg_L)

    def test_times_monotone_increasing(self):
        r = _sim()
        for i in range(1, len(r.times_h)):
            assert r.times_h[i] >= r.times_h[i - 1]


# ---------------------------------------------------------------------------
# fm = 1 (maximum metabolite formation)
# ---------------------------------------------------------------------------


class TestMaxFM:
    def test_fm_one_gives_metabolite(self):
        r = _sim(fm=1.0)
        assert max(r.c_metabolite_mg_L) > 0.0

    def test_fm_one_metabolite_peaks_exist(self):
        r = _sim(fm=1.0)
        assert r.c_metabolite_peak_ss > 0.0


# ---------------------------------------------------------------------------
# Accumulation ratios
# ---------------------------------------------------------------------------


class TestAccumulationRatios:
    def test_parent_accumulation_ratio_gte_one(self):
        r = _sim(n_doses=10)
        assert r.parent_accumulation_ratio >= 1.0 - 1e-6

    def test_metabolite_accumulation_ratio_gte_one(self):
        r = _sim(n_doses=10)
        assert r.metabolite_accumulation_ratio >= 1.0 - 1e-6

    def test_long_metabolite_half_life_high_accumulation(self):
        # Very slow metabolite elimination → high accumulation
        r_long = _sim(cl_metabolite_L_per_h=0.5, n_doses=15)
        r_short = _sim(cl_metabolite_L_per_h=20.0, n_doses=15)
        assert r_long.metabolite_accumulation_ratio > r_short.metabolite_accumulation_ratio

    def test_short_metabolite_half_life_low_accumulation(self):
        # Fast metabolite clearance → accumulation close to 1
        r = _sim(cl_metabolite_L_per_h=100.0, n_doses=10)
        # Should still be >= 1
        assert r.metabolite_accumulation_ratio >= 1.0 - 1e-6


# ---------------------------------------------------------------------------
# AUC ratio
# ---------------------------------------------------------------------------


class TestAUCRatio:
    def test_auc_ratio_positive_when_fm_positive(self):
        r = _sim(fm=0.5)
        assert r.metabolite_parent_auc_ratio > 0.0

    def test_high_fm_increases_auc_ratio(self):
        r_low = _sim(fm=0.1)
        r_high = _sim(fm=0.9)
        assert r_high.metabolite_parent_auc_ratio > r_low.metabolite_parent_auc_ratio


# ---------------------------------------------------------------------------
# Metabolite concern classification
# ---------------------------------------------------------------------------


class TestConcernClassification:
    def test_concern_string_valid_values(self):
        r = _sim()
        assert r.metabolite_concern in {"none", "moderate", "high"}

    def test_high_concern_for_very_slow_metabolite(self):
        # Very slow elimination and high fm → concern should be "moderate" or "high"
        r = _sim(fm=1.0, cl_metabolite_L_per_h=0.1, n_doses=20)
        assert r.metabolite_concern in {"moderate", "high"}


# ---------------------------------------------------------------------------
# Steady-state summary
# ---------------------------------------------------------------------------


class TestSteadyStateSummary:
    def test_peak_gte_trough_parent(self):
        r = _sim()
        assert r.c_parent_peak_ss >= r.c_parent_trough_ss

    def test_peak_gte_trough_metabolite(self):
        r = _sim()
        assert r.c_metabolite_peak_ss >= r.c_metabolite_trough_ss

    def test_ss_peak_parent_positive(self):
        r = _sim()
        assert r.c_parent_peak_ss > 0.0


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


class TestNotes:
    def test_notes_contain_drug_name(self):
        r = simulate_metabolite_accumulation(
            drug_name="Morphine",
            metabolite_name="Morphine-6-glucuronide",
            dose_mg=10.0,
            cl_parent_L_per_h=5.0,
            vd_parent_L=50.0,
            cl_metabolite_L_per_h=3.0,
            vd_metabolite_L=60.0,
            fm=0.5,
        )
        assert "Morphine" in r.notes

    def test_notes_contain_metabolite_name(self):
        r = simulate_metabolite_accumulation(
            drug_name="Morphine",
            metabolite_name="Morphine-6-glucuronide",
            dose_mg=10.0,
            cl_parent_L_per_h=5.0,
            vd_parent_L=50.0,
            cl_metabolite_L_per_h=3.0,
            vd_metabolite_L=60.0,
            fm=0.5,
        )
        assert "Morphine-6-glucuronide" in r.notes


# ---------------------------------------------------------------------------
# Dataclass immutability
# ---------------------------------------------------------------------------


class TestDataclass:
    def test_frozen_dataclass(self):
        r = _sim()
        with pytest.raises((AttributeError, TypeError)):
            r.drug_name = "OtherDrug"  # type: ignore[misc]

    def test_result_fields_accessible(self):
        r = _sim()
        assert hasattr(r, "drug_name")
        assert hasattr(r, "metabolite_name")
        assert hasattr(r, "metabolite_concern")
        assert hasattr(r, "times_h")
        assert hasattr(r, "c_parent_mg_L")
        assert hasattr(r, "c_metabolite_mg_L")


# ---------------------------------------------------------------------------
# compare_metabolite_half_lives
# ---------------------------------------------------------------------------


class TestCompareHalfLives:
    def test_returns_correct_length(self):
        results = compare_metabolite_half_lives(
            drug_name="DrugA",
            metabolite_name="MetabA",
            dose_mg=100.0,
            cl_parent_L_per_h=5.0,
            vd_parent_L=50.0,
            fm=0.3,
        )
        assert len(results) == 4  # default: [2, 6, 12, 24]

    def test_sorted_by_accumulation_ratio_descending(self):
        results = compare_metabolite_half_lives(
            drug_name="DrugA",
            metabolite_name="MetabA",
            dose_mg=100.0,
            cl_parent_L_per_h=5.0,
            vd_parent_L=50.0,
            fm=0.3,
        )
        ratios = [r.metabolite_accumulation_ratio for r in results]
        assert ratios == sorted(ratios, reverse=True)

    def test_custom_half_lives_length(self):
        results = compare_metabolite_half_lives(
            drug_name="DrugB",
            metabolite_name="MetabB",
            dose_mg=200.0,
            cl_parent_L_per_h=8.0,
            vd_parent_L=80.0,
            fm=0.4,
            t_half_metabolite_values_h=[3.0, 8.0, 20.0],
        )
        assert len(results) == 3

    def test_longest_half_life_highest_accumulation(self):
        results = compare_metabolite_half_lives(
            drug_name="DrugC",
            metabolite_name="MetabC",
            dose_mg=50.0,
            cl_parent_L_per_h=4.0,
            vd_parent_L=40.0,
            fm=0.5,
            t_half_metabolite_values_h=[2.0, 24.0],
        )
        # Sorted descending, so first element = highest accumulation
        assert results[0].metabolite_accumulation_ratio >= results[1].metabolite_accumulation_ratio

    def test_all_results_are_metabolite_accumulation_result(self):
        results = compare_metabolite_half_lives(
            drug_name="DrugD",
            metabolite_name="MetabD",
            dose_mg=100.0,
            cl_parent_L_per_h=5.0,
            vd_parent_L=50.0,
            fm=0.3,
        )
        for r in results:
            assert isinstance(r, MetaboliteAccumulationResult)
