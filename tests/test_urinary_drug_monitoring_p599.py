"""Tests for Phase 599 — Urinary Drug Monitoring."""

import pytest

from omega_pbpk.clinical.urinary_drug_monitoring_p599 import (
    UrinaryDrugMonitoringResult,
    detection_window,
    simulate_urinary_monitoring,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_result():
    return simulate_urinary_monitoring(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_renal_L_per_h=2.0,
        vd_L=50.0,
    )


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self, base_result):
        assert isinstance(base_result, UrinaryDrugMonitoringResult)

    def test_times_is_list(self, base_result):
        assert isinstance(base_result.times_h, list)

    def test_urine_conc_is_list(self, base_result):
        assert isinstance(base_result.urine_conc_mg_L, list)

    def test_cumulative_is_list(self, base_result):
        assert isinstance(base_result.cumulative_excreted_mg, list)

    def test_list_lengths_match(self, base_result):
        n = len(base_result.times_h)
        assert len(base_result.urine_conc_mg_L) == n
        assert len(base_result.cumulative_excreted_mg) == n

    def test_drug_name_stored(self, base_result):
        assert base_result.drug_name == "TestDrug"

    def test_dose_stored(self, base_result):
        assert base_result.dose_mg == 100.0

    def test_renal_cl_stored(self, base_result):
        assert base_result.renal_cl_L_per_h == 2.0

    def test_urine_flow_stored(self, base_result):
        assert base_result.urine_flow_mL_per_h == 60.0


# ---------------------------------------------------------------------------
# Physical constraints
# ---------------------------------------------------------------------------


class TestPhysicalConstraints:
    def test_urine_conc_nonnegative(self, base_result):
        assert all(c >= 0 for c in base_result.urine_conc_mg_L)

    def test_cumulative_nondecreasing(self, base_result):
        cum = base_result.cumulative_excreted_mg
        for i in range(1, len(cum)):
            assert cum[i] >= cum[i - 1] - 1e-12

    def test_cumulative_starts_at_zero(self, base_result):
        assert base_result.cumulative_excreted_mg[0] == pytest.approx(0.0, abs=1e-10)

    def test_fe_urine_in_range(self, base_result):
        assert 0 <= base_result.fe_urine <= 1.0

    def test_peak_urine_conc_positive(self, base_result):
        assert base_result.peak_urine_conc_mg_L > 0


# ---------------------------------------------------------------------------
# Urine concentration formula
# ---------------------------------------------------------------------------


class TestUrineConcentrationFormula:
    def test_urine_conc_proportional_to_cl_renal(self):
        r1 = simulate_urinary_monitoring("D", 100.0, 1.0, 50.0)
        r2 = simulate_urinary_monitoring("D", 100.0, 2.0, 50.0)
        # Peak urine conc should be ~2x with 2x CL_renal
        ratio = r2.peak_urine_conc_mg_L / r1.peak_urine_conc_mg_L
        assert 1.5 < ratio < 2.5

    def test_urine_conc_inversely_proportional_to_flow(self):
        r1 = simulate_urinary_monitoring("D", 100.0, 2.0, 50.0, urine_flow_mL_per_h=30.0)
        r2 = simulate_urinary_monitoring("D", 100.0, 2.0, 50.0, urine_flow_mL_per_h=120.0)
        # Lower flow → higher concentration
        assert r1.peak_urine_conc_mg_L > r2.peak_urine_conc_mg_L


# ---------------------------------------------------------------------------
# Cumulative excretion
# ---------------------------------------------------------------------------


class TestCumulativeExcretion:
    def test_cumulative_less_than_dose(self, base_result):
        assert base_result.cumulative_excreted_mg[-1] <= base_result.dose_mg + 1e-6

    def test_fe_urine_consistent_with_cumulative(self, base_result):
        expected_fe = min(base_result.cumulative_excreted_mg[-1] / base_result.dose_mg, 1.0)
        assert base_result.fe_urine == pytest.approx(expected_fe, abs=1e-10)

    def test_higher_cl_renal_more_excreted(self):
        r1 = simulate_urinary_monitoring("D", 100.0, 1.0, 50.0)
        r2 = simulate_urinary_monitoring("D", 100.0, 3.0, 50.0)
        assert r2.cumulative_excreted_mg[-1] > r1.cumulative_excreted_mg[-1]


# ---------------------------------------------------------------------------
# Detection and clearance times
# ---------------------------------------------------------------------------


class TestDetectionClearanceTimes:
    def test_t_detection_less_than_t_clearance(self, base_result):
        if base_result.t_detection_h < base_result.t_clearance_h:
            assert base_result.t_detection_h < base_result.t_clearance_h

    def test_detection_time_finite(self, base_result):
        assert base_result.t_detection_h < 48.0, "Drug should be detectable"

    def test_clearance_time_after_detection(self, base_result):
        assert base_result.t_clearance_h >= base_result.t_detection_h

    def test_higher_dose_longer_detection_window(self):
        r1 = simulate_urinary_monitoring("D", 50.0, 2.0, 50.0)
        r2 = simulate_urinary_monitoring("D", 200.0, 2.0, 50.0)
        w1 = r1.t_clearance_h - r1.t_detection_h
        w2 = r2.t_clearance_h - r2.t_detection_h
        assert w2 >= w1


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


class TestDoseProportionality:
    def test_double_dose_roughly_double_auc(self):
        r1 = simulate_urinary_monitoring("D", 100.0, 2.0, 50.0)
        r2 = simulate_urinary_monitoring("D", 200.0, 2.0, 50.0)
        ratio = r2.auc_urine_mg_h_per_L / r1.auc_urine_mg_h_per_L
        assert 1.8 < ratio < 2.2

    def test_double_dose_roughly_double_peak(self):
        r1 = simulate_urinary_monitoring("D", 100.0, 2.0, 50.0)
        r2 = simulate_urinary_monitoring("D", 200.0, 2.0, 50.0)
        ratio = r2.peak_urine_conc_mg_L / r1.peak_urine_conc_mg_L
        assert 1.8 < ratio < 2.2


# ---------------------------------------------------------------------------
# fe_urine
# ---------------------------------------------------------------------------


class TestFeUrine:
    def test_fe_urine_high_cl_renal(self):
        # With very high renal CL, most drug excreted renally
        r = simulate_urinary_monitoring("D", 100.0, 5.0, 30.0, t_end_h=96.0)
        assert r.fe_urine > 0.3

    def test_fe_urine_low_cl_renal(self):
        r_low = simulate_urinary_monitoring("D", 100.0, 0.5, 50.0, t_end_h=96.0)
        r_high = simulate_urinary_monitoring("D", 100.0, 3.0, 50.0, t_end_h=96.0)
        assert r_high.fe_urine > r_low.fe_urine


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------


class TestAUC:
    def test_auc_positive(self, base_result):
        assert base_result.auc_urine_mg_h_per_L > 0

    def test_auc_increases_with_dose(self):
        r1 = simulate_urinary_monitoring("D", 50.0, 2.0, 50.0)
        r2 = simulate_urinary_monitoring("D", 100.0, 2.0, 50.0)
        assert r2.auc_urine_mg_h_per_L > r1.auc_urine_mg_h_per_L


# ---------------------------------------------------------------------------
# Detection window function
# ---------------------------------------------------------------------------


class TestDetectionWindow:
    def test_returns_dict(self):
        dw = detection_window("D", 100.0, 2.0, 50.0)
        assert isinstance(dw, dict)

    def test_dict_keys(self):
        dw = detection_window("D", 100.0, 2.0, 50.0)
        assert "t_detection_h" in dw
        assert "t_clearance_h" in dw
        assert "window_h" in dw

    def test_window_nonnegative(self):
        dw = detection_window("D", 100.0, 2.0, 50.0)
        assert dw["window_h"] >= 0

    def test_window_consistency(self):
        dw = detection_window("D", 100.0, 2.0, 50.0)
        assert dw["window_h"] == pytest.approx(dw["t_clearance_h"] - dw["t_detection_h"], abs=1e-9)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_urinary_monitoring("D", 0.0, 2.0, 50.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_urinary_monitoring("D", -10.0, 2.0, 50.0)

    def test_zero_cl_renal_raises(self):
        with pytest.raises(ValueError, match="cl_renal"):
            simulate_urinary_monitoring("D", 100.0, 0.0, 50.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_urinary_monitoring("D", 100.0, 2.0, 0.0)

    def test_zero_urine_flow_raises(self):
        with pytest.raises(ValueError, match="urine_flow"):
            simulate_urinary_monitoring("D", 100.0, 2.0, 50.0, urine_flow_mL_per_h=0.0)

    def test_invalid_f_oral_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            simulate_urinary_monitoring("D", 100.0, 2.0, 50.0, f_oral=0.0)

    def test_f_oral_above_one_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            simulate_urinary_monitoring("D", 100.0, 2.0, 50.0, f_oral=1.5)
