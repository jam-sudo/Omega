"""Tests for clinical/accumulation_index.py (Phase 529)."""

import math

import pytest

from omega_pbpk.clinical.accumulation_index import (
    AccumulationResult,
    accumulation_index,
    optimal_dosing_interval,
    peak_trough_ratio,
    simulate_multiple_dose_accumulation,
    time_to_steady_state,
)

# ---------------------------------------------------------------------------
# accumulation_index
# ---------------------------------------------------------------------------


class TestAccumulationIndex:
    def test_long_half_life_short_interval_high_accumulation(self):
        # t½=24h, interval=6h → high accumulation
        rac = accumulation_index(t_half_h=24.0, interval_h=6.0)
        assert rac > 3.0

    def test_short_half_life_long_interval_near_one(self):
        # t½=1h, interval=24h → drug almost fully eliminated; Rac ≈ 1
        rac = accumulation_index(t_half_h=1.0, interval_h=24.0)
        assert rac < 1.1

    def test_equal_half_life_and_interval(self):
        # t½=tau → Rac = 1/(1-0.5) = 2.0
        rac = accumulation_index(t_half_h=12.0, interval_h=12.0)
        assert abs(rac - 2.0) < 0.01

    def test_formula_consistency(self):
        t_half = 8.0
        interval = 12.0
        ke = math.log(2) / t_half
        expected = 1.0 / (1.0 - math.exp(-ke * interval))
        result = accumulation_index(t_half, interval)
        assert abs(result - expected) < 1e-10

    def test_returns_float(self):
        assert isinstance(accumulation_index(10.0, 8.0), float)

    def test_invalid_t_half(self):
        with pytest.raises(ValueError):
            accumulation_index(t_half_h=0, interval_h=12.0)

    def test_invalid_interval(self):
        with pytest.raises(ValueError):
            accumulation_index(t_half_h=12.0, interval_h=0)

    def test_negative_t_half_raises(self):
        with pytest.raises(ValueError):
            accumulation_index(t_half_h=-5.0, interval_h=12.0)


# ---------------------------------------------------------------------------
# time_to_steady_state
# ---------------------------------------------------------------------------


class TestTimeToSteadyState:
    def test_90_percent_approx_3_32_half_lives(self):
        t_half = 10.0
        t_ss = time_to_steady_state(t_half_h=t_half, fraction=0.90)
        expected = 3.3219 * t_half  # -ln(0.1)/ke * ke/ln2 = -ln(0.1)/ln2 * t½
        assert abs(t_ss - expected) < 0.05

    def test_50_percent_equals_one_half_life(self):
        t_half = 6.0
        t_ss = time_to_steady_state(t_half_h=t_half, fraction=0.50)
        assert abs(t_ss - t_half) < 0.01

    def test_97_percent_approx_5_half_lives(self):
        t_half = 10.0
        t_ss = time_to_steady_state(t_half_h=t_half, fraction=0.97)
        # -ln(0.03)/ln2 ≈ 5.06 t½
        assert 4.5 * t_half < t_ss < 6.0 * t_half

    def test_invalid_fraction_zero(self):
        with pytest.raises(ValueError):
            time_to_steady_state(t_half_h=10.0, fraction=0.0)

    def test_invalid_fraction_one(self):
        with pytest.raises(ValueError):
            time_to_steady_state(t_half_h=10.0, fraction=1.0)

    def test_invalid_t_half(self):
        with pytest.raises(ValueError):
            time_to_steady_state(t_half_h=-1.0, fraction=0.90)


# ---------------------------------------------------------------------------
# simulate_multiple_dose_accumulation
# ---------------------------------------------------------------------------


class TestSimulateMultipleDoseAccumulation:
    def _basic_result(self):
        return simulate_multiple_dose_accumulation(
            t_half_h=4.0,
            dose_mg=100.0,
            vd_L=10.0,
            interval_h=4.0,
            n_doses=8,
            dt_h=0.05,
        )

    def test_returns_accumulation_result(self):
        result = self._basic_result()
        assert isinstance(result, AccumulationResult)

    def test_cmax_increases_across_doses(self):
        result = self._basic_result()
        # Cmax should increase from dose 1 to dose 2
        assert result.cmax_per_dose[1] > result.cmax_per_dose[0]

    def test_cmax_plateaus_at_steady_state(self):
        result = self._basic_result()
        # Later doses should be close to each other (within 5%)
        cmax_last = result.cmax_per_dose[-1]
        cmax_second_last = result.cmax_per_dose[-2]
        assert abs(cmax_last - cmax_second_last) / cmax_last < 0.05

    def test_accumulation_ratio_observed_matches_theoretical(self):
        # Use short t½ relative to interval so Rac is near 2 (converges faster)
        # t½=6h, interval=6h → Rac=2.0 exactly; should converge in ~10 doses
        result = simulate_multiple_dose_accumulation(
            t_half_h=6.0,
            dose_mg=100.0,
            vd_L=10.0,
            interval_h=6.0,
            n_doses=15,
            dt_h=0.05,
        )
        theoretical = accumulation_index(6.0, 6.0)
        # Observed ratio (Cmax_last / Cmax_dose1) should approach theoretical
        # within 25% after 15 doses
        assert abs(result.accumulation_ratio_observed - theoretical) / theoretical < 0.25

    def test_times_length_matches_concentrations(self):
        result = self._basic_result()
        assert len(result.times_h) == len(result.c_plasma_mg_L)

    def test_cmax_per_dose_length_equals_n_doses(self):
        result = self._basic_result()
        assert len(result.cmax_per_dose) == result.n_doses

    def test_ctrough_per_dose_length_equals_n_doses(self):
        result = self._basic_result()
        assert len(result.ctrough_per_dose) == result.n_doses

    def test_notes_is_string(self):
        result = self._basic_result()
        assert isinstance(result.notes, str)

    def test_first_cmax_near_dose_over_vd(self):
        # For IV bolus, first Cmax = dose/Vd
        result = simulate_multiple_dose_accumulation(
            t_half_h=0.5,  # very short t½ → no accumulation
            dose_mg=100.0,
            vd_L=10.0,
            interval_h=24.0,  # very long interval
            n_doses=2,
            dt_h=0.01,
        )
        expected_cmax1 = 100.0 / 10.0  # 10 mg/L
        assert abs(result.cmax_per_dose[0] - expected_cmax1) / expected_cmax1 < 0.05

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_multiple_dose_accumulation(t_half_h=4.0, dose_mg=0, vd_L=10.0, interval_h=4.0)

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError):
            simulate_multiple_dose_accumulation(
                t_half_h=4.0, dose_mg=100.0, vd_L=-1.0, interval_h=4.0
            )


# ---------------------------------------------------------------------------
# peak_trough_ratio
# ---------------------------------------------------------------------------


class TestPeakTroughRatio:
    def test_long_half_life_low_ptr(self):
        # t½=24h, interval=6h → most drug remains → PTR close to 1
        ptr = peak_trough_ratio(t_half_h=24.0, interval_h=6.0)
        assert 1.0 < ptr < 2.0

    def test_short_half_life_high_ptr(self):
        # t½=2h, interval=12h → most drug eliminated → high PTR
        ptr = peak_trough_ratio(t_half_h=2.0, interval_h=12.0)
        assert ptr > 10.0

    def test_ptr_formula(self):
        t_half = 6.0
        interval = 8.0
        ke = math.log(2) / t_half
        expected = math.exp(ke * interval)
        result = peak_trough_ratio(t_half, interval)
        assert abs(result - expected) < 1e-10

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            peak_trough_ratio(t_half_h=0, interval_h=12.0)
        with pytest.raises(ValueError):
            peak_trough_ratio(t_half_h=12.0, interval_h=0)


# ---------------------------------------------------------------------------
# optimal_dosing_interval
# ---------------------------------------------------------------------------


class TestOptimalDosingInterval:
    def test_rac_2_gives_correct_tau(self):
        # For Rac=2, tau = t½ (since 1/(1-e^(-ln2)) = 2)
        t_half = 8.0
        tau = optimal_dosing_interval(t_half_h=t_half, target_accumulation_ratio=2.0)
        # Round-trip check: accumulation_index(t_half, tau) ≈ 2.0
        rac = accumulation_index(t_half, tau)
        assert abs(rac - 2.0) < 0.001

    def test_higher_target_gives_shorter_interval(self):
        t_half = 12.0
        tau_3 = optimal_dosing_interval(t_half_h=t_half, target_accumulation_ratio=3.0)
        tau_2 = optimal_dosing_interval(t_half_h=t_half, target_accumulation_ratio=2.0)
        assert tau_3 < tau_2

    def test_roundtrip_various_targets(self):
        for target in [1.5, 2.0, 3.0, 5.0]:
            tau = optimal_dosing_interval(t_half_h=10.0, target_accumulation_ratio=target)
            rac = accumulation_index(10.0, tau)
            assert abs(rac - target) < 0.001, f"Failed for target={target}"

    def test_invalid_target_le_1(self):
        with pytest.raises(ValueError):
            optimal_dosing_interval(t_half_h=10.0, target_accumulation_ratio=1.0)

    def test_invalid_target_less_than_1(self):
        with pytest.raises(ValueError):
            optimal_dosing_interval(t_half_h=10.0, target_accumulation_ratio=0.5)

    def test_invalid_t_half(self):
        with pytest.raises(ValueError):
            optimal_dosing_interval(t_half_h=0, target_accumulation_ratio=2.0)
