"""Tests for signal_transduction_pd module (Phase 206)."""

import pytest

from omega_pbpk.clinical.signal_transduction_pd import (
    SignalTransductionResult,
    compare_signal_drugs,
    simulate_signal_cascade,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim(**overrides):
    kwargs = dict(
        drug_name="DrugA",
        drug_conc_mg_L=1.0,
        ec50_mg_L=1.0,
        t_end_h=24.0,
        dt_h=0.1,
    )
    kwargs.update(overrides)
    return simulate_signal_cascade(**kwargs)


# ---------------------------------------------------------------------------
# Zero-drug baseline
# ---------------------------------------------------------------------------


class TestZeroDrug:
    def test_zero_conc_zero_peak_response(self):
        r = _sim(drug_conc_mg_L=0.0)
        assert r.peak_response_pct == 0.0

    def test_zero_conc_zero_sustained_response(self):
        r = _sim(drug_conc_mg_L=0.0)
        assert r.sustained_response_pct == 0.0

    def test_zero_conc_zero_desensitization(self):
        r = _sim(drug_conc_mg_L=0.0)
        assert r.receptor_desensitization_pct == pytest.approx(0.0, abs=1e-6)

    def test_zero_conc_zero_second_messenger(self):
        r = _sim(drug_conc_mg_L=0.0)
        assert all(v == pytest.approx(0.0, abs=1e-9) for v in r.second_messenger_norm)


# ---------------------------------------------------------------------------
# Drug concentration effects
# ---------------------------------------------------------------------------


class TestConcentrationEffects:
    def test_high_conc_higher_peak_than_low(self):
        r_low = _sim(drug_conc_mg_L=0.1, ec50_mg_L=1.0)
        r_high = _sim(drug_conc_mg_L=10.0, ec50_mg_L=1.0)
        assert r_high.peak_response_pct > r_low.peak_response_pct

    def test_above_ec50_gives_nonzero_response(self):
        r = _sim(drug_conc_mg_L=5.0, ec50_mg_L=1.0)
        assert r.peak_response_pct > 0.0

    def test_at_ec50_half_max_occupancy(self):
        """At ec50, Hill occupancy = 0.5; second messenger and response should be nonzero."""
        r = _sim(drug_conc_mg_L=1.0, ec50_mg_L=1.0, k_desensitize_per_h=0.0)
        assert r.peak_response_pct > 0.0

    def test_time_to_peak_positive_for_nonzero_drug(self):
        r = _sim(drug_conc_mg_L=2.0, ec50_mg_L=1.0)
        assert r.time_to_peak_h > 0.0


# ---------------------------------------------------------------------------
# Desensitization
# ---------------------------------------------------------------------------


class TestDesensitization:
    def test_high_desensitize_rate_increases_desensitization(self):
        r_low = _sim(k_desensitize_per_h=0.01, drug_conc_mg_L=5.0)
        r_high = _sim(k_desensitize_per_h=2.0, drug_conc_mg_L=5.0)
        assert r_high.receptor_desensitization_pct > r_low.receptor_desensitization_pct

    def test_high_desensitize_rate_lower_sustained_response(self):
        """High desensitization → reduced long-term response."""
        r_low = _sim(k_desensitize_per_h=0.01, drug_conc_mg_L=5.0, t_end_h=48.0)
        r_high = _sim(k_desensitize_per_h=5.0, drug_conc_mg_L=5.0, t_end_h=48.0)
        assert r_high.sustained_response_pct < r_low.sustained_response_pct

    def test_zero_desensitize_no_downregulation(self):
        r = _sim(k_desensitize_per_h=0.0, drug_conc_mg_L=2.0)
        assert r.receptor_desensitization_pct == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    def test_result_is_frozen_dataclass(self):
        r = _sim()
        with pytest.raises((AttributeError, TypeError)):
            r.peak_response_pct = 999.0  # type: ignore[misc]

    def test_times_length_matches_steps(self):
        r = _sim(t_end_h=10.0, dt_h=0.5)
        expected_len = int(10.0 / 0.5) + 1
        assert len(r.times_h) == expected_len

    def test_all_lists_same_length(self):
        r = _sim(t_end_h=12.0, dt_h=0.25)
        assert len(r.times_h) == len(r.receptor_occupancy_pct)
        assert len(r.times_h) == len(r.second_messenger_norm)
        assert len(r.times_h) == len(r.response_norm)

    def test_times_start_at_zero(self):
        r = _sim()
        assert r.times_h[0] == 0.0

    def test_receptor_occupancy_between_0_and_100(self):
        r = _sim(drug_conc_mg_L=100.0)
        assert all(0.0 <= v <= 100.0 for v in r.receptor_occupancy_pct)

    def test_second_messenger_nonnegative(self):
        r = _sim()
        assert all(v >= 0.0 for v in r.second_messenger_norm)

    def test_response_norm_nonnegative(self):
        r = _sim()
        assert all(v >= 0.0 for v in r.response_norm)


# ---------------------------------------------------------------------------
# compare_signal_drugs
# ---------------------------------------------------------------------------


class TestCompareSignalDrugs:
    def _drugs(self):
        return [
            {"name": "DrugLow", "conc_mg_L": 0.1, "ec50_mg_L": 1.0},
            {"name": "DrugHigh", "conc_mg_L": 10.0, "ec50_mg_L": 1.0},
            {"name": "DrugMed", "conc_mg_L": 1.0, "ec50_mg_L": 1.0},
        ]

    def test_returns_sorted_by_peak_descending(self):
        results = compare_signal_drugs(self._drugs())
        peaks = [r.peak_response_pct for r in results]
        assert peaks == sorted(peaks, reverse=True)

    def test_length_matches_input(self):
        results = compare_signal_drugs(self._drugs())
        assert len(results) == 3

    def test_all_results_are_signal_transduction_result(self):
        results = compare_signal_drugs(self._drugs())
        assert all(isinstance(r, SignalTransductionResult) for r in results)

    def test_high_conc_drug_is_first(self):
        results = compare_signal_drugs(self._drugs())
        assert results[0].drug_name == "DrugHigh"

    def test_kwargs_forwarded_in_compare(self):
        drugs = [
            {"name": "A", "conc_mg_L": 5.0, "ec50_mg_L": 1.0, "t_end_h": 6.0},
            {"name": "B", "conc_mg_L": 2.0, "ec50_mg_L": 1.0, "t_end_h": 6.0},
        ]
        results = compare_signal_drugs(drugs)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_conc_raises(self):
        with pytest.raises(ValueError, match="drug_conc"):
            simulate_signal_cascade("Drug", -1.0, 1.0)

    def test_zero_ec50_raises(self):
        with pytest.raises(ValueError, match="ec50"):
            simulate_signal_cascade("Drug", 1.0, 0.0)

    def test_negative_ec50_raises(self):
        with pytest.raises(ValueError, match="ec50"):
            simulate_signal_cascade("Drug", 1.0, -0.5)
