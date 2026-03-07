"""Tests for Phase 914 — Sublingual to Systemic PK Simulator."""

import pytest

from omega_pbpk.clinical.sublingual_systemic_pk import (
    SublingualSystemicPKResult,
    compare_drug_types,
    simulate_sublingual_systemic_pk,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_default(**kwargs) -> SublingualSystemicPKResult:
    return simulate_sublingual_systemic_pk(
        drug_name="TestDrug",
        dose_mg=10.0,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Result type and fields
# ---------------------------------------------------------------------------


class TestResultDataclass:
    def test_returns_sublingual_systemic_pk_result(self):
        result = _run_default()
        assert isinstance(result, SublingualSystemicPKResult)

    def test_drug_name_preserved(self):
        result = simulate_sublingual_systemic_pk("Buprenorphine", dose_mg=8.0)
        assert result.drug_name == "Buprenorphine"

    def test_dose_preserved(self):
        result = _run_default()
        assert result.dose_mg == 10.0

    def test_drug_type_preserved(self):
        result = simulate_sublingual_systemic_pk("X", dose_mg=5.0, drug_type="opioid")
        assert result.drug_type == "opioid"

    def test_times_non_empty(self):
        result = _run_default()
        assert len(result.times_h) > 0

    def test_c_plasma_non_empty(self):
        result = _run_default()
        assert len(result.c_plasma_mg_L) > 0

    def test_times_and_c_plasma_same_length(self):
        result = _run_default()
        assert len(result.times_h) == len(result.c_plasma_mg_L)


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

    def test_f_sublingual_between_0_and_1(self):
        result = _run_default()
        assert 0.0 <= result.f_sublingual <= 1.0

    def test_time_to_onset_non_negative(self):
        result = _run_default()
        assert result.time_to_onset_min >= 0.0

    def test_time_to_peak_equals_tmax_times_60(self):
        result = _run_default()
        assert abs(result.time_to_peak_min - result.tmax_h * 60.0) < 1e-6

    def test_duration_above_half_max_non_negative(self):
        result = _run_default()
        assert result.duration_above_half_max_h >= 0.0

    def test_duration_above_half_max_less_than_t_end(self):
        result = _run_default(t_end_h=4.0)
        assert result.duration_above_half_max_h <= 4.0

    def test_onset_before_peak(self):
        result = _run_default()
        assert result.time_to_onset_min <= result.time_to_peak_min


# ---------------------------------------------------------------------------
# Drug type differences
# ---------------------------------------------------------------------------


class TestDrugTypes:
    def test_small_molecule_higher_f_than_peptide(self):
        sm = simulate_sublingual_systemic_pk("D", dose_mg=10.0, drug_type="small_molecule")
        pep = simulate_sublingual_systemic_pk("D", dose_mg=10.0, drug_type="peptide")
        assert sm.f_sublingual > pep.f_sublingual

    def test_opioid_between_small_molecule_and_peptide_f(self):
        sm = simulate_sublingual_systemic_pk("D", dose_mg=10.0, drug_type="small_molecule")
        op = simulate_sublingual_systemic_pk("D", dose_mg=10.0, drug_type="opioid")
        pep = simulate_sublingual_systemic_pk("D", dose_mg=10.0, drug_type="peptide")
        assert sm.auc_mg_h_per_L > op.auc_mg_h_per_L > pep.auc_mg_h_per_L

    def test_all_drug_types_run(self):
        for dt in ("small_molecule", "opioid", "peptide"):
            r = simulate_sublingual_systemic_pk("D", dose_mg=10.0, drug_type=dt)
            assert r.cmax_mg_L > 0.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_sublingual_systemic_pk("D", dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_sublingual_systemic_pk("D", dose_mg=-5.0)

    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError, match="drug_type"):
            simulate_sublingual_systemic_pk("D", dose_mg=10.0, drug_type="liquid")

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_sublingual_systemic_pk("D", dose_mg=10.0, cl_L_per_h=0.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_sublingual_systemic_pk("D", dose_mg=10.0, vd_L=0.0)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


class TestNotes:
    def test_peptide_note(self):
        result = simulate_sublingual_systemic_pk("D", dose_mg=10.0, drug_type="peptide")
        assert "peptide" in result.notes.lower() or "poor" in result.notes.lower()

    def test_notes_non_empty(self):
        result = _run_default()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_small_molecule_rapid_note_or_standard(self):
        result = simulate_sublingual_systemic_pk("D", dose_mg=10.0, drug_type="small_molecule")
        assert result.notes in (
            "Very rapid systemic onset — suitable for acute/emergency dosing",
            "Standard sublingual kinetics",
        )


# ---------------------------------------------------------------------------
# compare_drug_types
# ---------------------------------------------------------------------------


class TestCompareDrugTypes:
    def test_returns_three_results(self):
        results = compare_drug_types("Drug", dose_mg=10.0)
        assert len(results) == 3

    def test_all_are_sublingual_systemic_pk_result(self):
        results = compare_drug_types("Drug", dose_mg=10.0)
        for r in results:
            assert isinstance(r, SublingualSystemicPKResult)

    def test_sorted_descending_by_auc(self):
        results = compare_drug_types("Drug", dose_mg=10.0)
        aucs = [r.auc_mg_h_per_L for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_all_drug_types_present(self):
        results = compare_drug_types("Drug", dose_mg=10.0)
        types = {r.drug_type for r in results}
        assert types == {"small_molecule", "opioid", "peptide"}

    def test_kwargs_forwarded(self):
        results = compare_drug_types("Drug", dose_mg=10.0, cl_L_per_h=10.0, vd_L=80.0)
        assert len(results) == 3
        for r in results:
            assert r.auc_mg_h_per_L > 0.0
