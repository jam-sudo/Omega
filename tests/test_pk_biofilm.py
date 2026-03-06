"""Tests for antibiotic PK/PD in biofilm infections module."""

import pytest

from omega_pbpk.clinical.pk_biofilm import (
    BiofilmPKResult,
    compare_dosing_regimens,
    simulate_biofilm_pk,
)


def _default():
    return simulate_biofilm_pk("Cipro", dose_mg=500, cl_L_per_h=30, vd_L=150)


class TestSimulateBiofilmPK:
    def test_returns_result(self):
        r = _default()
        assert isinstance(r, BiofilmPKResult)

    def test_dose_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_biofilm_pk("x", dose_mg=0, cl_L_per_h=30, vd_L=150)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_biofilm_pk("x", dose_mg=500, cl_L_per_h=0, vd_L=150)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_biofilm_pk("x", dose_mg=500, cl_L_per_h=30, vd_L=0)

    def test_penetration_negative_raises(self):
        with pytest.raises(ValueError):
            simulate_biofilm_pk("x", dose_mg=500, cl_L_per_h=30, vd_L=150, biofilm_penetration=-0.1)

    def test_penetration_above_one_raises(self):
        with pytest.raises(ValueError):
            simulate_biofilm_pk("x", dose_mg=500, cl_L_per_h=30, vd_L=150, biofilm_penetration=1.5)

    def test_cmax_plasma_positive(self):
        r = _default()
        assert r.cmax_plasma > 0

    def test_lower_penetration_lower_biofilm_cmax(self):
        kw = dict(drug_name="x", dose_mg=500, cl_L_per_h=30, vd_L=150)
        low_p = simulate_biofilm_pk(**kw, biofilm_penetration=0.01)
        high_p = simulate_biofilm_pk(**kw, biofilm_penetration=0.1)
        assert high_p.cmax_biofilm > low_p.cmax_biofilm

    def test_auc_plasma_positive(self):
        r = _default()
        assert r.auc_24h_plasma > 0

    def test_auc_biofilm_positive(self):
        r = _default()
        assert r.auc_24h_biofilm > 0

    def test_ft_above_mic_in_range(self):
        r = _default()
        assert 0 <= r.ft_above_mic_pct <= 100

    def test_planktonic_kill_is_bool(self):
        r = _default()
        assert isinstance(r.planktonic_kill_expected, bool)

    def test_biofilm_eradication_is_bool(self):
        r = _default()
        assert isinstance(r.biofilm_eradication_expected, bool)

    def test_high_dose_higher_auc_mic(self):
        low = simulate_biofilm_pk("x", dose_mg=100, cl_L_per_h=30, vd_L=150)
        high = simulate_biofilm_pk("x", dose_mg=1000, cl_L_per_h=30, vd_L=150)
        assert high.auc_mic_ratio > low.auc_mic_ratio

    def test_zero_penetration_zero_biofilm(self):
        r = simulate_biofilm_pk("x", dose_mg=500, cl_L_per_h=30, vd_L=150, biofilm_penetration=0.0)
        assert r.cmax_biofilm < 1e-10

    def test_very_low_dose_no_biofilm_eradication(self):
        r = simulate_biofilm_pk("x", dose_mg=0.001, cl_L_per_h=30, vd_L=150)
        assert r.biofilm_eradication_expected is False

    def test_multiple_doses_accumulation(self):
        single = simulate_biofilm_pk("x", dose_mg=500, cl_L_per_h=30, vd_L=150, n_doses=1)
        multi = simulate_biofilm_pk("x", dose_mg=500, cl_L_per_h=30, vd_L=150, n_doses=3)
        assert multi.cmax_plasma >= single.cmax_plasma

    def test_biofilm_eradication_harder_than_planktonic(self):
        """With default MBEC >> MIC, biofilm eradication is harder at low doses."""
        r = simulate_biofilm_pk("x", dose_mg=1, cl_L_per_h=30, vd_L=150)
        assert r.mbec_mg_L > r.mic_mg_L

    def test_notes_contains_name(self):
        r = simulate_biofilm_pk("MyDrug", dose_mg=500, cl_L_per_h=30, vd_L=150)
        assert "MyDrug" in r.notes

    def test_times_length(self):
        r = _default()
        assert len(r.times_h) > 1

    def test_cmax_mic_ratio_positive(self):
        r = _default()
        assert r.cmax_mic_ratio > 0

    def test_mbec_gt_mic(self):
        r = _default()
        assert r.mbec_mg_L > r.mic_mg_L


class TestCompareDosing:
    def test_returns_correct_length(self):
        regimens = [
            {"dose_mg": 250, "n_doses": 3, "dosing_interval_h": 8},
            {"dose_mg": 500, "n_doses": 2, "dosing_interval_h": 12},
        ]
        results = compare_dosing_regimens("x", cl_L_per_h=30, vd_L=150, regimens=regimens)
        assert len(results) == 2

    def test_higher_dose_higher_ratio(self):
        regimens = [
            {"dose_mg": 100},
            {"dose_mg": 1000},
        ]
        results = compare_dosing_regimens("x", cl_L_per_h=30, vd_L=150, regimens=regimens)
        assert results[1].auc_mic_ratio > results[0].auc_mic_ratio
