"""Tests for biliary excretion and enterohepatic circulation model."""

from __future__ import annotations

import pytest

from omega_pbpk.core.biliary_excretion import (
    EHCResult,
    biliary_clearance,
    enterohepatic_circulation_factor,
    secondary_peak_detection,
    simulate_enterohepatic_circulation,
)


class TestBiliaryClearance:
    def test_basic_calculation(self):
        result = biliary_clearance(cl_renal_L_per_h=5.0, cl_total_L_per_h=20.0)
        assert result["cl_biliary_L_per_h"] == pytest.approx(15.0)
        assert result["cl_total_L_per_h"] == pytest.approx(20.0)

    def test_fe_bile_correct(self):
        result = biliary_clearance(cl_renal_L_per_h=5.0, cl_total_L_per_h=20.0)
        assert result["fe_bile"] == pytest.approx(0.75)

    def test_fe_urine_correct(self):
        result = biliary_clearance(cl_renal_L_per_h=5.0, cl_total_L_per_h=20.0)
        assert result["fe_urine"] == pytest.approx(0.25)

    def test_fe_bile_plus_fe_urine_le_1(self):
        result = biliary_clearance(cl_renal_L_per_h=8.0, cl_total_L_per_h=20.0)
        assert result["fe_bile"] + result["fe_urine"] <= 1.0 + 1e-9

    def test_zero_renal_clearance(self):
        result = biliary_clearance(cl_renal_L_per_h=0.0, cl_total_L_per_h=10.0)
        assert result["cl_biliary_L_per_h"] == pytest.approx(10.0)
        assert result["fe_bile"] == pytest.approx(1.0)
        assert result["fe_urine"] == pytest.approx(0.0)

    def test_all_renal_clearance(self):
        result = biliary_clearance(cl_renal_L_per_h=10.0, cl_total_L_per_h=10.0)
        assert result["cl_biliary_L_per_h"] == pytest.approx(0.0)
        assert result["fe_bile"] == pytest.approx(0.0)

    def test_invalid_total_zero(self):
        with pytest.raises(ValueError):
            biliary_clearance(cl_renal_L_per_h=0.0, cl_total_L_per_h=0.0)

    def test_invalid_renal_exceeds_total(self):
        with pytest.raises(ValueError):
            biliary_clearance(cl_renal_L_per_h=15.0, cl_total_L_per_h=10.0)

    def test_negative_renal_clearance(self):
        with pytest.raises(ValueError):
            biliary_clearance(cl_renal_L_per_h=-1.0, cl_total_L_per_h=10.0)


class TestEHCFactor:
    def test_no_reabsorption_unchanged_half_life(self):
        t_half = 10.0
        result = enterohepatic_circulation_factor(
            f_reabsorbed=0.0, t_recirculation_h=6.0, t_half_h=t_half
        )
        assert result == pytest.approx(t_half)

    def test_high_reabsorption_extends_half_life(self):
        t_half_base = 10.0
        result = enterohepatic_circulation_factor(
            f_reabsorbed=0.9, t_recirculation_h=4.0, t_half_h=t_half_base
        )
        assert result > t_half_base

    def test_moderate_reabsorption(self):
        result = enterohepatic_circulation_factor(
            f_reabsorbed=0.5, t_recirculation_h=6.0, t_half_h=8.0
        )
        assert result > 8.0

    def test_invalid_f_reabsorbed_above_1(self):
        with pytest.raises(ValueError):
            enterohepatic_circulation_factor(f_reabsorbed=1.5, t_recirculation_h=6.0, t_half_h=10.0)

    def test_invalid_negative_f_reabsorbed(self):
        with pytest.raises(ValueError):
            enterohepatic_circulation_factor(
                f_reabsorbed=-0.1, t_recirculation_h=6.0, t_half_h=10.0
            )

    def test_invalid_zero_recirculation(self):
        with pytest.raises(ValueError):
            enterohepatic_circulation_factor(f_reabsorbed=0.5, t_recirculation_h=0.0, t_half_h=10.0)


class TestSecondaryPeakDetection:
    def test_no_peaks_monotone_decline(self):
        times = [0.0, 1.0, 2.0, 3.0, 4.0]
        conc = [10.0, 5.0, 2.5, 1.25, 0.625]
        peaks = secondary_peak_detection(times, conc)
        assert peaks == []

    def test_secondary_peak_detected(self):
        # Simulate a profile with one secondary peak
        times = list(range(20))
        conc = [
            10.0,
            6.0,
            4.0,
            3.0,
            2.5,
            2.0,
            1.8,
            2.5,
            3.0,
            2.8,
            2.0,
            1.5,
            1.0,
            0.8,
            0.6,
            0.4,
            0.3,
            0.2,
            0.1,
            0.05,
        ]
        peaks = secondary_peak_detection(times, conc)
        assert len(peaks) >= 1

    def test_peak_at_correct_time(self):
        times = [float(i) for i in range(15)]
        conc = [10.0, 8.0, 6.0, 4.0, 3.0, 2.5, 2.0, 3.0, 4.0, 3.5, 2.0, 1.0, 0.5, 0.3, 0.1]
        peaks = secondary_peak_detection(times, conc)
        # Peak should be around index 8 (time=8.0)
        assert any(7.0 <= t <= 9.0 for t in peaks)

    def test_below_prominence_threshold_ignored(self):
        times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        conc = [10.0, 5.0, 4.9, 4.95, 4.85, 0.1]  # tiny bump, not prominent
        peaks = secondary_peak_detection(times, conc, min_prominence=0.5)
        assert peaks == []

    def test_empty_input(self):
        peaks = secondary_peak_detection([], [])
        assert peaks == []

    def test_short_input(self):
        peaks = secondary_peak_detection([0.0, 1.0], [10.0, 5.0])
        assert peaks == []


class TestSimulateEHC:
    def test_returns_ehc_result(self):
        result = simulate_enterohepatic_circulation(
            drug_name="TestDrug",
            dose_mg=100.0,
            cl_hepatic_L_per_h=5.0,
            vd_L=50.0,
        )
        assert isinstance(result, EHCResult)

    def test_auc_greater_than_no_ehc(self):
        """EHC with reabsorption should give higher AUC than without."""
        result_ehc = simulate_enterohepatic_circulation(
            drug_name="Drug",
            dose_mg=100.0,
            cl_hepatic_L_per_h=5.0,
            vd_L=50.0,
            f_biliary=0.3,
            f_reabsorbed=0.7,
        )
        result_no_ehc = simulate_enterohepatic_circulation(
            drug_name="Drug",
            dose_mg=100.0,
            cl_hepatic_L_per_h=5.0,
            vd_L=50.0,
            f_biliary=0.3,
            f_reabsorbed=0.0,
        )
        assert result_ehc.auc > result_no_ehc.auc

    def test_no_ehc_single_peak_only(self):
        """Without EHC (f_reabsorbed=0), no secondary peaks."""
        result = simulate_enterohepatic_circulation(
            drug_name="Drug",
            dose_mg=100.0,
            cl_hepatic_L_per_h=5.0,
            vd_L=50.0,
            f_biliary=0.3,
            f_reabsorbed=0.0,
            t_end_h=48.0,
        )
        assert len(result.secondary_peaks) == 0

    def test_times_start_at_zero(self):
        result = simulate_enterohepatic_circulation(
            drug_name="Drug",
            dose_mg=100.0,
            cl_hepatic_L_per_h=5.0,
            vd_L=50.0,
        )
        assert result.times_h[0] == pytest.approx(0.0)

    def test_initial_plasma_conc_equals_dose_over_vd(self):
        dose_mg = 100.0
        vd_L = 50.0
        result = simulate_enterohepatic_circulation(
            drug_name="Drug",
            dose_mg=dose_mg,
            cl_hepatic_L_per_h=5.0,
            vd_L=vd_L,
        )
        assert result.c_plasma_mg_L[0] == pytest.approx(dose_mg / vd_L)

    def test_auc_positive(self):
        result = simulate_enterohepatic_circulation(
            drug_name="Drug",
            dose_mg=100.0,
            cl_hepatic_L_per_h=5.0,
            vd_L=50.0,
        )
        assert result.auc > 0.0

    def test_invalid_f_biliary(self):
        with pytest.raises(ValueError):
            simulate_enterohepatic_circulation(
                drug_name="Drug",
                dose_mg=100.0,
                cl_hepatic_L_per_h=5.0,
                vd_L=50.0,
                f_biliary=1.5,
            )

    def test_invalid_f_reabsorbed(self):
        with pytest.raises(ValueError):
            simulate_enterohepatic_circulation(
                drug_name="Drug",
                dose_mg=100.0,
                cl_hepatic_L_per_h=5.0,
                vd_L=50.0,
                f_reabsorbed=-0.1,
            )

    def test_bile_accumulates_initially(self):
        result = simulate_enterohepatic_circulation(
            drug_name="Drug",
            dose_mg=200.0,
            cl_hepatic_L_per_h=10.0,
            vd_L=50.0,
            f_biliary=0.5,
            f_reabsorbed=0.8,
            t_end_h=48.0,
        )
        # Bile should accumulate from zero initially
        assert result.a_bile_mg[0] == pytest.approx(0.0)
        # And reach some positive value
        assert max(result.a_bile_mg) > 0.0

    def test_drug_name_in_result(self):
        result = simulate_enterohepatic_circulation(
            drug_name="MyDrug",
            dose_mg=100.0,
            cl_hepatic_L_per_h=5.0,
            vd_L=50.0,
        )
        assert result.drug_name == "MyDrug"

    def test_notes_populated(self):
        result = simulate_enterohepatic_circulation(
            drug_name="Drug",
            dose_mg=100.0,
            cl_hepatic_L_per_h=5.0,
            vd_L=50.0,
        )
        assert len(result.notes) > 0
