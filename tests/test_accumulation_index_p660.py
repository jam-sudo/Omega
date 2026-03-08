"""Tests for drug accumulation index (phase 660)."""

import pytest

from omega_pbpk.clinical.accumulation_index_p660 import (
    AccumulationIndexResult,
    calculate_accumulation_index,
    compare_regimens,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    dosing_interval_h=8.0,
    cl_L_per_h=5.0,
    vd_L=50.0,
)


def _run(**overrides):
    params = {**BASE, **overrides}
    return calculate_accumulation_index(**params)


# ---------------------------------------------------------------------------
# Return type & frozen
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_result_dataclass(self):
        r = _run()
        assert isinstance(r, AccumulationIndexResult)

    def test_frozen_raises_on_mutation(self):
        r = _run()
        with pytest.raises(AttributeError):
            r.dose_mg = 999.0  # type: ignore[misc]

    def test_drug_name_stored(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_dose_stored(self):
        r = _run()
        assert r.dose_mg == 100.0


# ---------------------------------------------------------------------------
# Accumulation index properties
# ---------------------------------------------------------------------------


class TestAccumulationIndex:
    def test_ai_greater_than_one_short_interval(self):
        # t_half = 0.693 / (5/50) = 6.93 h; interval 8 h < 5*t_half
        r = _run()
        assert r.accumulation_index > 1.0

    def test_ai_approaches_one_long_interval(self):
        # Very long interval relative to half-life
        r = _run(dosing_interval_h=200.0)
        assert abs(r.accumulation_index - 1.0) < 0.05

    def test_short_interval_more_accumulation(self):
        r_short = _run(dosing_interval_h=4.0)
        r_long = _run(dosing_interval_h=24.0)
        assert r_short.accumulation_index > r_long.accumulation_index


# ---------------------------------------------------------------------------
# Time to steady state
# ---------------------------------------------------------------------------


class TestSteadyState:
    def test_time_to_90pct_approx_3_32_t_half(self):
        r = _run()
        expected = 3.32 * r.t_half_h
        ratio = r.time_to_90pct_steady_state_h / expected
        assert 0.9 < ratio < 1.1  # within 10%

    def test_n_doses_positive(self):
        r = _run()
        assert r.n_doses_to_steady_state > 0

    def test_n_doses_is_int(self):
        r = _run()
        assert isinstance(r.n_doses_to_steady_state, int)


# ---------------------------------------------------------------------------
# Peak / trough
# ---------------------------------------------------------------------------


class TestPeakTrough:
    def test_cmax_ss_greater_than_cmin_ss(self):
        r = _run()
        assert r.peak_steady_state_mg_L > r.trough_steady_state_mg_L

    def test_cmax_ratio_positive(self):
        r = _run()
        assert r.cmax_ratio > 0

    def test_cmin_ratio_positive(self):
        r = _run()
        assert r.cmin_ratio > 0

    def test_trough_positive(self):
        r = _run()
        assert r.trough_steady_state_mg_L > 0

    def test_peak_positive(self):
        r = _run()
        assert r.peak_steady_state_mg_L > 0


# ---------------------------------------------------------------------------
# Fluctuation
# ---------------------------------------------------------------------------


class TestFluctuation:
    def test_fluctuation_positive(self):
        r = _run()
        assert r.fluctuation_index > 0

    def test_fluctuation_clamped(self):
        r = _run()
        assert r.fluctuation_index <= 100.0


# ---------------------------------------------------------------------------
# AUC ratio
# ---------------------------------------------------------------------------


class TestAUCRatio:
    def test_auc_ratio_equals_ai(self):
        r = _run()
        assert abs(r.auc_ratio_ss_to_single - r.accumulation_index) < 1e-10


# ---------------------------------------------------------------------------
# compare_regimens
# ---------------------------------------------------------------------------


class TestCompareRegimens:
    def test_returns_four_results(self):
        results = compare_regimens("TestDrug", 5.0, 50.0, 400.0)
        assert len(results) == 4

    def test_sorted_by_fluctuation_ascending(self):
        results = compare_regimens("TestDrug", 5.0, 50.0, 400.0)
        flucts = [r.fluctuation_index for r in results]
        assert flucts == sorted(flucts)

    def test_all_results_are_correct_type(self):
        results = compare_regimens("TestDrug", 5.0, 50.0, 400.0)
        assert all(isinstance(r, AccumulationIndexResult) for r in results)

    def test_custom_intervals(self):
        results = compare_regimens("TestDrug", 5.0, 50.0, 400.0, intervals=[6, 12])
        assert len(results) == 2


# ---------------------------------------------------------------------------
# ka ~ k_el edge case
# ---------------------------------------------------------------------------


class TestKaEqualsKel:
    def test_ka_equals_kel_no_error(self):
        k_el = 5.0 / 50.0  # 0.1
        r = _run(ka_per_h=k_el)
        assert r.accumulation_index > 0
        assert r.peak_steady_state_mg_L > 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-10.0)

    def test_zero_interval_raises(self):
        with pytest.raises(ValueError, match="dosing_interval"):
            _run(dosing_interval_h=0.0)

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            _run(cl_L_per_h=-1.0)

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            _run(vd_L=-10.0)

    def test_negative_ka_raises(self):
        with pytest.raises(ValueError, match="ka_per_h"):
            _run(ka_per_h=-0.5)
