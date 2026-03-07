"""Tests for Antibody-Drug Conjugate (ADC) PK model (Phase 240)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.adc_pk import ADCPKResult, compare_dar, simulate_adc_pk


# ---------------------------------------------------------------------------
# simulate_adc_pk tests
# ---------------------------------------------------------------------------


class TestSimulateADCPK:
    def test_iv_bolus_adc_peaks_at_t0(self):
        result = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6)
        # IV bolus: ADC concentration is highest at t=0
        assert result.tmax_adc_h == pytest.approx(0.0, abs=1.5)

    def test_payload_peaks_later_than_adc(self):
        result = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6)
        # Free payload peaks after ADC (formed from ADC degradation)
        assert result.tmax_payload_h > result.tmax_adc_h

    def test_auc_adc_positive(self):
        result = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6)
        assert result.auc_adc > 0.0

    def test_auc_payload_positive(self):
        result = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6)
        assert result.auc_payload > 0.0

    def test_cmax_adc_positive(self):
        result = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6)
        assert result.cmax_adc > 0.0

    def test_cmax_payload_positive(self):
        result = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6, dar=4, k_deconj_per_h=0.01)
        assert result.cmax_payload > 0.0

    def test_t_half_adc_reasonable(self):
        # IgG-like antibody half-life should be > 100h (several days)
        result = simulate_adc_pk(
            "T-DM1",
            dose_mg_per_kg=3.6,
            cl_adc_L_per_h=0.01,
            vd_adc_L=4.0,
            k_deconj_per_h=0.001,
            t_end_h=2000.0,
        )
        assert result.t_half_adc_h > 24.0

    def test_payload_exposure_ratio_positive(self):
        result = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6)
        assert result.payload_exposure_ratio > 0.0

    def test_higher_k_deconj_yields_higher_payload(self):
        r_slow = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6, k_deconj_per_h=0.001)
        r_fast = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6, k_deconj_per_h=0.05)
        assert r_fast.cmax_payload > r_slow.cmax_payload

    def test_result_type(self):
        result = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6)
        assert isinstance(result, ADCPKResult)

    def test_times_list(self):
        result = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6)
        assert isinstance(result.times_h, list)
        assert len(result.times_h) > 0

    def test_c_adc_list(self):
        result = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6)
        assert isinstance(result.c_adc_mg_L, list)
        assert len(result.c_adc_mg_L) == len(result.times_h)

    def test_c_payload_list(self):
        result = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6)
        assert isinstance(result.c_payload_mg_L, list)
        assert len(result.c_payload_mg_L) == len(result.times_h)

    def test_dose_scales_cmax(self):
        r_low = simulate_adc_pk("T-DM1", dose_mg_per_kg=1.8)
        r_high = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6)
        assert r_high.cmax_adc > r_low.cmax_adc

    # Validation error tests
    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg_per_kg"):
            simulate_adc_pk("T-DM1", dose_mg_per_kg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg_per_kg"):
            simulate_adc_pk("T-DM1", dose_mg_per_kg=-1.0)

    def test_invalid_dar_zero_raises(self):
        with pytest.raises(ValueError, match="dar"):
            simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6, dar=0)

    def test_invalid_dar_too_high_raises(self):
        with pytest.raises(ValueError, match="dar"):
            simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6, dar=9)

    def test_invalid_bw_raises(self):
        with pytest.raises(ValueError, match="bw_kg"):
            simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6, bw_kg=0.0)

    def test_invalid_k_deconj_raises(self):
        with pytest.raises(ValueError, match="k_deconj"):
            simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6, k_deconj_per_h=0.0)


# ---------------------------------------------------------------------------
# compare_dar tests
# ---------------------------------------------------------------------------


class TestCompareDAR:
    def test_returns_list(self):
        results = compare_dar("T-DM1", dose_mg_per_kg=3.6, dar_values=[2, 4, 8])
        assert isinstance(results, list)
        assert len(results) == 3

    def test_sorted_by_auc_payload_descending(self):
        results = compare_dar("T-DM1", dose_mg_per_kg=3.6, dar_values=[2, 4, 8])
        auc_payloads = [r.auc_payload for r in results]
        assert auc_payloads == sorted(auc_payloads, reverse=True)

    def test_higher_dar_higher_payload_exposure(self):
        results = compare_dar("T-DM1", dose_mg_per_kg=3.6, dar_values=[2, 8])
        dar2 = next(r for r in results if r.dar == 2)
        dar8 = next(r for r in results if r.dar == 8)
        # Higher DAR → faster deconjugation → more payload
        assert dar8.auc_payload > dar2.auc_payload

    def test_empty_dar_values_raises(self):
        with pytest.raises(ValueError, match="dar_values"):
            compare_dar("T-DM1", dose_mg_per_kg=3.6, dar_values=[])

    def test_invalid_dar_in_list_raises(self):
        with pytest.raises(ValueError, match="dar"):
            compare_dar("T-DM1", dose_mg_per_kg=3.6, dar_values=[2, 0])

    def test_result_elements_type(self):
        results = compare_dar("T-DM1", dose_mg_per_kg=3.6, dar_values=[4])
        assert isinstance(results[0], ADCPKResult)

    def test_dar_values_preserved_in_results(self):
        results = compare_dar("T-DM1", dose_mg_per_kg=3.6, dar_values=[2, 4, 8])
        dar_set = {r.dar for r in results}
        assert dar_set == {2, 4, 8}
