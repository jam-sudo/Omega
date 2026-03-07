"""Tests for clinical/urine_monitoring.py — Phase 320."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.urine_monitoring import (
    UrineDrugResult,
    detection_window,
    simulate_urine_drug,
    therapeutic_drug_monitoring_urine,
    urine_concentration_window,
)


# ---------------------------------------------------------------------------
# simulate_urine_drug
# ---------------------------------------------------------------------------

class TestSimulateUrineDrug:
    def _iv_result(self) -> UrineDrugResult:
        return simulate_urine_drug(
            drug_name="TestDrug",
            dose_mg=100.0,
            cl_renal_L_per_h=2.0,
            cl_total_L_per_h=10.0,
            vd_L=50.0,
            route="iv",
            t_end_h=24.0,
        )

    def test_returns_dataclass(self):
        result = self._iv_result()
        assert isinstance(result, UrineDrugResult)

    def test_frozen(self):
        result = self._iv_result()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_fe_calculation(self):
        result = self._iv_result()
        assert result.fe == pytest.approx(0.20)

    def test_total_excreted_reasonable(self):
        result = self._iv_result()
        # fe=0.2 → total excreted ≈ 20 mg (not exact due to numerical integration)
        assert 0 < result.total_excreted_mg < 100.0

    def test_fraction_recovered_approx_fe(self):
        result = self._iv_result()
        # Over 24h (several t½), should approach fe=0.2
        assert result.fraction_recovered == pytest.approx(0.20, abs=0.02)

    def test_times_length_matches_arrays(self):
        result = self._iv_result()
        n = len(result.times_h)
        assert len(result.c_plasma) == n
        assert len(result.c_urine_mg_L) == n
        assert len(result.cumulative_excreted_mg) == n

    def test_cumulative_excreted_monotone(self):
        result = self._iv_result()
        cum = result.cumulative_excreted_mg
        for i in range(len(cum) - 1):
            assert cum[i + 1] >= cum[i] - 1e-9

    def test_plasma_conc_declines(self):
        result = self._iv_result()
        cp = result.c_plasma
        # Plasma should decline over 24h for IV
        assert cp[-1] < cp[0]

    def test_oral_route(self):
        result = simulate_urine_drug(
            drug_name="Oral",
            dose_mg=200.0,
            cl_renal_L_per_h=3.0,
            cl_total_L_per_h=15.0,
            vd_L=60.0,
            route="oral",
            t_end_h=24.0,
        )
        assert result.route == "oral"
        # Oral: plasma starts at 0
        assert result.c_plasma[0] == pytest.approx(0.0)
        assert result.total_excreted_mg > 0

    def test_peak_urine_conc_positive(self):
        result = self._iv_result()
        assert result.peak_urine_conc_mg_L > 0

    def test_zero_renal_cl_no_excretion(self):
        result = simulate_urine_drug(
            drug_name="NoRenal",
            dose_mg=100.0,
            cl_renal_L_per_h=0.0,
            cl_total_L_per_h=10.0,
            vd_L=50.0,
            route="iv",
        )
        assert result.total_excreted_mg == pytest.approx(0.0)
        assert result.fraction_recovered == pytest.approx(0.0)
        assert result.fe == pytest.approx(0.0)

    def test_invalid_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_urine_drug("X", -10.0, 1.0, 5.0, 20.0)

    def test_invalid_cl_renal_negative(self):
        with pytest.raises(ValueError, match="cl_renal_L_per_h"):
            simulate_urine_drug("X", 100.0, -1.0, 5.0, 20.0)

    def test_invalid_cl_total_zero(self):
        with pytest.raises(ValueError, match="cl_total_L_per_h"):
            simulate_urine_drug("X", 100.0, 1.0, 0.0, 20.0)

    def test_invalid_cl_renal_gt_total(self):
        with pytest.raises(ValueError, match="cl_renal_L_per_h"):
            simulate_urine_drug("X", 100.0, 10.0, 5.0, 20.0)

    def test_invalid_vd(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_urine_drug("X", 100.0, 1.0, 5.0, 0.0)

    def test_invalid_route(self):
        with pytest.raises(ValueError, match="route"):
            simulate_urine_drug("X", 100.0, 1.0, 5.0, 20.0, route="im")

    def test_notes_non_empty(self):
        result = self._iv_result()
        assert len(result.notes) > 5

    def test_drug_name_preserved(self):
        result = simulate_urine_drug("Amoxicillin", 500.0, 5.0, 8.0, 20.0, route="oral")
        assert result.drug_name == "Amoxicillin"


# ---------------------------------------------------------------------------
# urine_concentration_window
# ---------------------------------------------------------------------------

class TestUrineConcentrationWindow:
    def _setup(self):
        result = simulate_urine_drug(
            drug_name="W",
            dose_mg=100.0,
            cl_renal_L_per_h=2.0,
            cl_total_L_per_h=10.0,
            vd_L=50.0,
            route="iv",
            t_end_h=24.0,
        )
        return result.times_h, result.c_urine_mg_L

    def test_returns_dict(self):
        times, conc = self._setup()
        out = urine_concentration_window(times, conc, (0.0, 12.0))
        assert isinstance(out, dict)

    def test_mean_within_window(self):
        times, conc = self._setup()
        out = urine_concentration_window(times, conc, (0.0, 6.0))
        assert out["mean"] >= 0.0

    def test_min_le_mean_le_max(self):
        times, conc = self._setup()
        out = urine_concentration_window(times, conc, (2.0, 10.0))
        assert out["min"] <= out["mean"] <= out["max"] + 1e-9

    def test_empty_window_returns_zeros(self):
        times = [0.0, 1.0, 2.0]
        conc = [1.0, 0.5, 0.25]
        out = urine_concentration_window(times, conc, (5.0, 6.0))
        assert out["mean"] == 0.0
        assert out["n_points"] == 0

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError, match="collection_window_h"):
            urine_concentration_window([0.0, 1.0], [1.0, 0.5], (5.0, 3.0))

    def test_creatinine_normalized_positive(self):
        times, conc = self._setup()
        out = urine_concentration_window(times, conc, (0.0, 24.0))
        assert out["creatinine_normalized_mg_g"] >= 0.0

    def test_n_points_correct(self):
        times = [0.0, 1.0, 2.0, 3.0, 4.0]
        conc = [1.0, 0.8, 0.6, 0.4, 0.2]
        out = urine_concentration_window(times, conc, (1.0, 3.0))
        assert out["n_points"] == 3  # t=1, 2, 3


# ---------------------------------------------------------------------------
# therapeutic_drug_monitoring_urine
# ---------------------------------------------------------------------------

class TestTherapeuticDrugMonitoringUrine:
    def test_within_range(self):
        out = therapeutic_drug_monitoring_urine("Drug", 5.0, (2.0, 10.0))
        assert out["in_range"] is True
        assert out["interpretation"] == "within_range"

    def test_below_range(self):
        out = therapeutic_drug_monitoring_urine("Drug", 0.5, (2.0, 10.0))
        assert out["in_range"] is False
        assert out["interpretation"] == "below_range"

    def test_above_range(self):
        out = therapeutic_drug_monitoring_urine("Drug", 15.0, (2.0, 10.0))
        assert out["in_range"] is False
        assert out["interpretation"] == "above_range"

    def test_notes_non_empty(self):
        out = therapeutic_drug_monitoring_urine("Drug", 5.0, (2.0, 10.0))
        assert len(out["notes"]) > 5

    def test_invalid_reference_range(self):
        with pytest.raises(ValueError, match="reference_range"):
            therapeutic_drug_monitoring_urine("Drug", 5.0, (10.0, 2.0))

    def test_invalid_negative_conc(self):
        with pytest.raises(ValueError, match="c_urine_mg_L"):
            therapeutic_drug_monitoring_urine("Drug", -1.0, (2.0, 10.0))

    def test_returns_dict_with_expected_keys(self):
        out = therapeutic_drug_monitoring_urine("Drug", 5.0, (2.0, 10.0))
        for key in ("drug_name", "c_urine_mg_L", "reference_low", "reference_high",
                    "in_range", "interpretation", "notes"):
            assert key in out


# ---------------------------------------------------------------------------
# detection_window
# ---------------------------------------------------------------------------

class TestDetectionWindow:
    def test_returns_float(self):
        dw = detection_window("Drug", 100.0, 2.0, 10.0, 50.0)
        assert isinstance(dw, float)

    def test_detection_window_positive(self):
        dw = detection_window("Drug", 100.0, 2.0, 10.0, 50.0)
        assert dw > 0.0

    def test_higher_dose_longer_detection(self):
        dw_low = detection_window("Drug", 50.0, 2.0, 10.0, 50.0)
        dw_high = detection_window("Drug", 200.0, 2.0, 10.0, 50.0)
        assert dw_high > dw_low

    def test_lower_lloq_longer_detection(self):
        dw1 = detection_window("Drug", 100.0, 2.0, 10.0, 50.0, lloq_urine_mg_L=0.01)
        dw2 = detection_window("Drug", 100.0, 2.0, 10.0, 50.0, lloq_urine_mg_L=0.0001)
        assert dw2 > dw1

    def test_zero_renal_cl_gives_zero(self):
        dw = detection_window("Drug", 100.0, 0.0, 10.0, 50.0)
        assert dw == pytest.approx(0.0)

    def test_invalid_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            detection_window("Drug", 0.0, 2.0, 10.0, 50.0)

    def test_invalid_cl_renal_negative(self):
        with pytest.raises(ValueError, match="cl_renal_L_per_h"):
            detection_window("Drug", 100.0, -1.0, 10.0, 50.0)

    def test_invalid_cl_total_zero(self):
        with pytest.raises(ValueError, match="cl_total_L_per_h"):
            detection_window("Drug", 100.0, 2.0, 0.0, 50.0)

    def test_invalid_cl_renal_gt_total(self):
        with pytest.raises(ValueError, match="cl_renal_L_per_h"):
            detection_window("Drug", 100.0, 15.0, 10.0, 50.0)

    def test_invalid_vd(self):
        with pytest.raises(ValueError, match="vd_L"):
            detection_window("Drug", 100.0, 2.0, 10.0, 0.0)

    def test_invalid_lloq(self):
        with pytest.raises(ValueError, match="lloq_urine_mg_L"):
            detection_window("Drug", 100.0, 2.0, 10.0, 50.0, lloq_urine_mg_L=0.0)

    def test_analytical_formula_check(self):
        # c_urine_0 = cl_renal * (dose/vd) / urine_flow_L_per_h
        # ke = cl_total / vd
        # t = ln(c_urine_0 / lloq) / ke
        cl_renal = 2.0
        cl_total = 10.0
        vd = 50.0
        dose = 100.0
        lloq = 0.001
        urine_flow = 0.060  # L/h default
        ke = cl_total / vd
        c_urine_0 = cl_renal * (dose / vd) / urine_flow
        expected = math.log(c_urine_0 / lloq) / ke
        result = detection_window("Drug", dose, cl_renal, cl_total, vd, lloq)
        assert abs(result - expected) < 1e-6
