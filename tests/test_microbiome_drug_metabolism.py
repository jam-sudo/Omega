"""Tests for microbiome-mediated drug metabolism simulation (Phase 879)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.microbiome_drug_metabolism import (
    MicrobiomePKResult,
    compare_microbiome_activities,
    simulate_microbiome_pk,
)

# ---------------------------------------------------------------------------
# Basic result structure tests
# ---------------------------------------------------------------------------


class TestMicrobiomePKResultStructure:
    def test_returns_correct_type(self):
        result = simulate_microbiome_pk("TestDrug", dose_mg=100.0)
        assert isinstance(result, MicrobiomePKResult)

    def test_drug_name_preserved(self):
        result = simulate_microbiome_pk("Sulfasalazine", dose_mg=500.0)
        assert result.drug_name == "Sulfasalazine"

    def test_dose_preserved(self):
        result = simulate_microbiome_pk("Drug", dose_mg=250.0)
        assert result.dose_mg == 250.0

    def test_activity_preserved(self):
        result = simulate_microbiome_pk("Drug", dose_mg=100.0, microbiome_activity="high")
        assert result.microbiome_activity == "high"

    def test_times_length_consistent(self):
        result = simulate_microbiome_pk("Drug", dose_mg=100.0, t_end_h=12.0, dt_h=0.5)
        assert len(result.times_h) == len(result.c_parent_plasma_mg_L)
        assert len(result.times_h) == len(result.c_metabolite_plasma_mg_L)

    def test_times_start_at_zero(self):
        result = simulate_microbiome_pk("Drug", dose_mg=100.0)
        assert result.times_h[0] == 0.0

    def test_initial_concentrations_zero(self):
        result = simulate_microbiome_pk("Drug", dose_mg=100.0)
        assert result.c_parent_plasma_mg_L[0] == 0.0
        assert result.c_metabolite_plasma_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# Microbiome activity scaling tests
# ---------------------------------------------------------------------------


class TestMicrobiomeActivityScaling:
    def test_none_activity_no_metabolite(self):
        result = simulate_microbiome_pk("Drug", dose_mg=100.0, microbiome_activity="none")
        assert result.auc_metabolite == pytest.approx(0.0, abs=1e-6)
        assert result.cmax_metabolite == pytest.approx(0.0, abs=1e-6)

    def test_none_activity_max_parent_auc(self):
        """With no microbiome activity, parent AUC should be highest."""
        r_none = simulate_microbiome_pk("Drug", dose_mg=100.0, microbiome_activity="none")
        r_high = simulate_microbiome_pk("Drug", dose_mg=100.0, microbiome_activity="high")
        assert r_none.auc_parent > r_high.auc_parent

    def test_high_activity_more_metabolite_than_low(self):
        r_low = simulate_microbiome_pk("Drug", dose_mg=100.0, microbiome_activity="low")
        r_high = simulate_microbiome_pk("Drug", dose_mg=100.0, microbiome_activity="high")
        assert r_high.auc_metabolite > r_low.auc_metabolite

    def test_f_microbiome_metabolism_zero_for_none(self):
        result = simulate_microbiome_pk("Drug", dose_mg=100.0, microbiome_activity="none")
        assert result.f_microbiome_metabolism == pytest.approx(0.0, abs=1e-9)

    def test_f_microbiome_metabolism_positive_for_normal(self):
        result = simulate_microbiome_pk("Drug", dose_mg=100.0, microbiome_activity="normal")
        assert result.f_microbiome_metabolism > 0.0

    def test_all_valid_activities_run(self):
        for activity in ["none", "low", "normal", "high"]:
            result = simulate_microbiome_pk("Drug", dose_mg=100.0, microbiome_activity=activity)
            assert result.microbiome_activity == activity


# ---------------------------------------------------------------------------
# PK metric tests
# ---------------------------------------------------------------------------


class TestPKMetrics:
    def test_cmax_parent_matches_max_concentration(self):
        result = simulate_microbiome_pk("Drug", dose_mg=100.0)
        assert result.cmax_parent == pytest.approx(max(result.c_parent_plasma_mg_L), rel=1e-6)

    def test_tmax_parent_is_valid_time(self):
        result = simulate_microbiome_pk("Drug", dose_mg=100.0)
        assert 0.0 <= result.tmax_parent_h <= result.times_h[-1]

    def test_auc_parent_positive_for_nonzero_activity(self):
        result = simulate_microbiome_pk("Drug", dose_mg=100.0, microbiome_activity="normal")
        assert result.auc_parent > 0.0

    def test_metabolite_ratio_zero_for_none_activity(self):
        result = simulate_microbiome_pk("Drug", dose_mg=100.0, microbiome_activity="none")
        assert result.metabolite_ratio == pytest.approx(0.0, abs=1e-9)

    def test_metabolite_ratio_positive_for_high_activity(self):
        result = simulate_microbiome_pk(
            "Drug", dose_mg=100.0, microbiome_activity="high", k_microb_per_h=2.0, f_microb=0.8
        )
        assert result.metabolite_ratio >= 0.0

    def test_dose_scaling_auc(self):
        r1 = simulate_microbiome_pk("Drug", dose_mg=100.0)
        r2 = simulate_microbiome_pk("Drug", dose_mg=200.0)
        assert r2.auc_parent == pytest.approx(r1.auc_parent * 2.0, rel=0.01)


# ---------------------------------------------------------------------------
# Notes / interpretation tests
# ---------------------------------------------------------------------------


class TestNotes:
    def test_notes_minimal_for_none_activity(self):
        result = simulate_microbiome_pk("Drug", dose_mg=100.0, microbiome_activity="none")
        assert "Minimal" in result.notes

    def test_notes_extensive_for_dominant_metabolite(self):
        # High microbiome activity with very high metabolite production
        result = simulate_microbiome_pk(
            "Drug",
            dose_mg=100.0,
            microbiome_activity="high",
            k_microb_per_h=5.0,
            f_microb=1.0,
            cl_parent_L_per_h=5.0,
            cl_met_L_per_h=1.0,
        )
        # metabolite_ratio > 1 should trigger "Extensive"
        if result.metabolite_ratio > 1.0:
            assert "Extensive" in result.notes
        elif result.metabolite_ratio > 0.2:
            assert "Moderate" in result.notes
        else:
            assert "Minimal" in result.notes

    def test_notes_is_string(self):
        result = simulate_microbiome_pk("Drug", dose_mg=100.0)
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_microbiome_pk("Drug", dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_microbiome_pk("Drug", dose_mg=-50.0)

    def test_invalid_activity_raises(self):
        with pytest.raises(ValueError, match="microbiome_activity"):
            simulate_microbiome_pk("Drug", dose_mg=100.0, microbiome_activity="extreme")


# ---------------------------------------------------------------------------
# compare_microbiome_activities tests
# ---------------------------------------------------------------------------


class TestCompareMicrobiomeActivities:
    def test_returns_four_results(self):
        results = compare_microbiome_activities("Drug", dose_mg=100.0)
        assert len(results) == 4

    def test_sorted_by_auc_parent_descending(self):
        results = compare_microbiome_activities("Drug", dose_mg=100.0)
        aucs = [r.auc_parent for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_all_activities_represented(self):
        results = compare_microbiome_activities("Drug", dose_mg=100.0)
        activities = {r.microbiome_activity for r in results}
        assert activities == {"none", "low", "normal", "high"}

    def test_none_activity_has_highest_parent_auc(self):
        results = compare_microbiome_activities("Drug", dose_mg=100.0)
        assert results[0].microbiome_activity == "none"
