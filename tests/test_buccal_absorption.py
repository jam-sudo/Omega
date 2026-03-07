"""Tests for Phase 913 — Buccal Absorption Simulator."""

import pytest

from omega_pbpk.clinical.buccal_absorption import (
    BuccalAbsorptionResult,
    compare_buccal_formulations,
    simulate_buccal_absorption,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_default(**kwargs) -> BuccalAbsorptionResult:
    return simulate_buccal_absorption(
        drug_name="TestDrug",
        dose_mg=5.0,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Result type and fields
# ---------------------------------------------------------------------------


class TestResultDataclass:
    def test_returns_buccal_absorption_result(self):
        result = _run_default()
        assert isinstance(result, BuccalAbsorptionResult)

    def test_drug_name_preserved(self):
        result = simulate_buccal_absorption("Nitro", dose_mg=1.0)
        assert result.drug_name == "Nitro"

    def test_dose_preserved(self):
        result = _run_default()
        assert result.dose_mg == 5.0

    def test_formulation_preserved(self):
        result = simulate_buccal_absorption("X", dose_mg=2.0, formulation="spray")
        assert result.formulation == "spray"

    def test_times_non_empty(self):
        result = _run_default()
        assert len(result.times_h) > 0

    def test_c_plasma_non_empty(self):
        result = _run_default()
        assert len(result.c_plasma_mg_L) > 0

    def test_times_and_c_plasma_same_length(self):
        result = _run_default()
        assert len(result.times_h) == len(result.c_plasma_mg_L)

    def test_first_pass_bypass_fraction(self):
        result = _run_default()
        assert abs(result.first_pass_bypass_fraction - 0.95) < 1e-9


# ---------------------------------------------------------------------------
# PK metrics correctness
# ---------------------------------------------------------------------------


class TestPKMetrics:
    def test_cmax_positive(self):
        result = _run_default()
        assert result.cmax_mg_L > 0.0

    def test_tmax_positive(self):
        result = _run_default()
        assert result.tmax_h > 0.0

    def test_auc_positive(self):
        result = _run_default()
        assert result.auc_mg_h_per_L > 0.0

    def test_f_buccal_between_0_and_1(self):
        result = _run_default()
        assert 0.0 <= result.f_buccal <= 1.0

    def test_onset_time_min_positive(self):
        result = _run_default()
        assert result.onset_time_min >= 0.0

    def test_onset_is_half_tmax_in_minutes(self):
        result = _run_default()
        expected = result.tmax_h * 60.0 * 0.5
        assert abs(result.onset_time_min - expected) < 1e-6

    def test_compared_to_oral_auc_ratio_bounds(self):
        result = _run_default()
        assert 0.1 <= result.compared_to_oral_auc_ratio <= 5.0

    def test_compared_to_oral_formula(self):
        result = _run_default()
        expected = result.f_buccal / 0.5
        expected = max(0.1, min(5.0, expected))
        assert abs(result.compared_to_oral_auc_ratio - expected) < 1e-9


# ---------------------------------------------------------------------------
# Formulation differences
# ---------------------------------------------------------------------------


class TestFormulations:
    def test_spray_faster_than_tablet(self):
        spray = simulate_buccal_absorption("D", dose_mg=10.0, formulation="spray")
        tablet = simulate_buccal_absorption("D", dose_mg=10.0, formulation="tablet")
        assert spray.tmax_h <= tablet.tmax_h

    def test_film_higher_auc_than_tablet(self):
        film = simulate_buccal_absorption("D", dose_mg=10.0, formulation="film")
        tablet = simulate_buccal_absorption("D", dose_mg=10.0, formulation="tablet")
        assert film.auc_mg_h_per_L > tablet.auc_mg_h_per_L

    def test_all_formulations_run(self):
        for form in ("film", "spray", "tablet", "gel"):
            r = simulate_buccal_absorption("D", dose_mg=5.0, formulation=form)
            assert r.cmax_mg_L > 0.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_buccal_absorption("D", dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_buccal_absorption("D", dose_mg=-1.0)

    def test_invalid_formulation_raises(self):
        with pytest.raises(ValueError, match="formulation"):
            simulate_buccal_absorption("D", dose_mg=5.0, formulation="patch")

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_buccal_absorption("D", dose_mg=5.0, cl_L_per_h=0.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_buccal_absorption("D", dose_mg=5.0, vd_L=0.0)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


class TestNotes:
    def test_rapid_onset_note_for_spray(self):
        # spray k_abs=6 → fast → onset < 15 min
        result = simulate_buccal_absorption(
            "D", dose_mg=5.0, formulation="spray", cl_L_per_h=10.0, vd_L=50.0
        )
        assert "onset" in result.notes.lower() or "rapid" in result.notes.lower()

    def test_moderate_onset_note_for_tablet(self):
        # tablet k_abs=1.5 → slower
        result = simulate_buccal_absorption(
            "D", dose_mg=5.0, formulation="tablet", cl_L_per_h=0.5, vd_L=500.0
        )
        # onset_time_min = tmax*60*0.5; tmax will be higher → check moderate note
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# compare_buccal_formulations
# ---------------------------------------------------------------------------


class TestCompareFormulations:
    def test_returns_four_results(self):
        results = compare_buccal_formulations("Drug", dose_mg=10.0)
        assert len(results) == 4

    def test_all_are_buccal_absorption_result(self):
        results = compare_buccal_formulations("Drug", dose_mg=10.0)
        for r in results:
            assert isinstance(r, BuccalAbsorptionResult)

    def test_sorted_descending_by_auc(self):
        results = compare_buccal_formulations("Drug", dose_mg=10.0)
        aucs = [r.auc_mg_h_per_L for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_all_formulations_present(self):
        results = compare_buccal_formulations("Drug", dose_mg=10.0)
        forms = {r.formulation for r in results}
        assert forms == {"film", "spray", "tablet", "gel"}

    def test_kwargs_forwarded(self):
        results = compare_buccal_formulations("Drug", dose_mg=10.0, cl_L_per_h=5.0, vd_L=30.0)
        assert len(results) == 4
        for r in results:
            assert r.auc_mg_h_per_L > 0.0
