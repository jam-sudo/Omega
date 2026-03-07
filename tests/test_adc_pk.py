"""Tests for Antibody-Drug Conjugate (ADC) PK model (Phase 355)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.adc_pk import ADCPKResult, compare_linker_stability, simulate_adc_pk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_result(**kwargs) -> ADCPKResult:
    return simulate_adc_pk("T-DM1", total_antibody_dose_mg=100.0, **kwargs)


# ---------------------------------------------------------------------------
# Basic simulation tests
# ---------------------------------------------------------------------------


class TestSimulateADCPK:
    def test_returns_adc_pk_result(self):
        result = _default_result()
        assert isinstance(result, ADCPKResult)

    def test_frozen_dataclass(self):
        result = _default_result()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "other"  # type: ignore[misc]

    def test_drug_name_preserved(self):
        result = simulate_adc_pk("Kadcyla", total_antibody_dose_mg=200.0)
        assert result.drug_name == "Kadcyla"

    def test_dose_preserved(self):
        result = simulate_adc_pk("T-DM1", total_antibody_dose_mg=150.0)
        assert result.total_antibody_dose_mg == pytest.approx(150.0)

    def test_dar_initial_preserved(self):
        result = simulate_adc_pk("T-DM1", total_antibody_dose_mg=100.0, dar_initial=8.0)
        assert result.dar_initial == pytest.approx(8.0)

    def test_times_list_type(self):
        result = _default_result()
        assert isinstance(result.times_h, list)
        assert len(result.times_h) > 0

    def test_times_start_at_zero(self):
        result = _default_result()
        assert result.times_h[0] == pytest.approx(0.0)

    def test_times_end_at_t_end(self):
        result = simulate_adc_pk("X", total_antibody_dose_mg=100.0, t_end_h=336.0)
        assert result.times_h[-1] == pytest.approx(336.0)

    def test_concentration_list_lengths_match(self):
        result = _default_result()
        n = len(result.times_h)
        assert len(result.c_total_antibody_mg_L) == n
        assert len(result.c_conjugated_mg_L) == n
        assert len(result.c_free_payload_mg_L) == n
        assert len(result.c_unconjugated_ab_mg_L) == n
        assert len(result.dar_mean_over_time) == n

    def test_c_conjugated_starts_high(self):
        result = _default_result()
        # IV bolus: conjugated ADC peaks at t=0
        assert result.c_conjugated_mg_L[0] > 0.0

    def test_c_unconjugated_starts_zero(self):
        result = _default_result()
        # No naked antibody at t=0
        assert result.c_unconjugated_ab_mg_L[0] == pytest.approx(0.0)

    def test_c_free_payload_starts_zero(self):
        result = _default_result()
        assert result.c_free_payload_mg_L[0] == pytest.approx(0.0)

    def test_c_total_antibody_decreasing(self):
        result = simulate_adc_pk("T-DM1", total_antibody_dose_mg=100.0, t_end_h=200.0, dt_h=10.0)
        total = result.c_total_antibody_mg_L
        # Should generally decrease (monotone check over coarse steps)
        assert total[0] > total[-1]

    def test_auc_conjugated_positive(self):
        result = _default_result()
        assert result.auc_conjugated > 0.0

    def test_auc_total_antibody_positive(self):
        result = _default_result()
        assert result.auc_total_antibody > 0.0

    def test_auc_free_payload_positive(self):
        result = _default_result(k_deconj_per_day=0.5)
        assert result.auc_free_payload > 0.0

    def test_t_half_antibody_greater_than_payload(self):
        # Antibody CL is low (0.2 L/day = 0.0083 L/h); payload CL is high (20 L/h)
        result = simulate_adc_pk(
            "T-DM1",
            total_antibody_dose_mg=100.0,
            cl_antibody_L_per_day=0.2,
            vd_antibody_L=3.0,
            cl_payload_L_per_h=20.0,
            vd_payload_L=100.0,
        )
        assert result.t_half_antibody_h > result.t_half_payload_h

    def test_payload_antibody_auc_ratio_positive(self):
        result = _default_result(k_deconj_per_day=0.3)
        assert result.payload_antibody_auc_ratio > 0.0

    def test_fraction_deconj_168h_greater_than_24h(self):
        result = _default_result()
        assert result.fraction_deconjugated_by_168h > result.fraction_deconjugated_by_24h

    def test_fraction_deconj_in_range(self):
        result = _default_result()
        assert 0.0 <= result.fraction_deconjugated_by_24h <= 1.0
        assert 0.0 <= result.fraction_deconjugated_by_168h <= 1.0

    def test_higher_k_deconj_more_deconjugation_at_24h(self):
        r_slow = _default_result(k_deconj_per_day=0.01)
        r_fast = _default_result(k_deconj_per_day=0.5)
        assert r_fast.fraction_deconjugated_by_24h > r_slow.fraction_deconjugated_by_24h

    def test_zero_deconjugation_no_payload(self):
        # k_deconj = 0 means no payload released
        result = simulate_adc_pk(
            "T-DM1",
            total_antibody_dose_mg=100.0,
            k_deconj_per_day=0.0,
        )
        assert result.c_unconjugated_ab_mg_L[-1] == pytest.approx(0.0, abs=1e-10)
        assert result.auc_free_payload == pytest.approx(0.0, abs=1e-6)

    def test_zero_deconjugation_fraction_zero(self):
        result = simulate_adc_pk("T-DM1", total_antibody_dose_mg=100.0, k_deconj_per_day=0.0)
        assert result.fraction_deconjugated_by_24h == pytest.approx(0.0, abs=1e-10)

    def test_dar_mean_decreasing_for_positive_k_deconj(self):
        result = _default_result(k_deconj_per_day=0.2)
        dar = result.dar_mean_over_time
        # DAR should decrease from dar_initial
        assert dar[0] > dar[-1]
        assert dar[0] == pytest.approx(result.dar_initial, rel=1e-6)

    def test_dar_constant_for_zero_deconjugation(self):
        result = simulate_adc_pk("T-DM1", total_antibody_dose_mg=100.0, k_deconj_per_day=0.0)
        dar = result.dar_mean_over_time
        assert dar[0] == pytest.approx(dar[-1], rel=1e-6)

    def test_notes_nonempty(self):
        result = _default_result()
        assert len(result.notes) > 0

    def test_higher_dose_higher_auc(self):
        r_low = simulate_adc_pk("T-DM1", total_antibody_dose_mg=50.0)
        r_high = simulate_adc_pk("T-DM1", total_antibody_dose_mg=200.0)
        assert r_high.auc_conjugated > r_low.auc_conjugated

    def test_t_half_antibody_analytical(self):
        # ke_ab = cl_ab_per_h / vd_ab; t½ = ln2 / ke_ab
        cl_day = 0.2
        vd = 3.0
        ke = (cl_day / 24.0) / vd
        expected_t_half = math.log(2) / ke
        result = simulate_adc_pk("T-DM1", total_antibody_dose_mg=100.0,
                                  cl_antibody_L_per_day=cl_day, vd_antibody_L=vd)
        assert result.t_half_antibody_h == pytest.approx(expected_t_half, rel=1e-6)

    def test_t_half_payload_analytical(self):
        cl_pl = 20.0
        vd_pl = 100.0
        ke_pl = cl_pl / vd_pl
        expected = math.log(2) / ke_pl
        result = simulate_adc_pk("T-DM1", total_antibody_dose_mg=100.0,
                                  cl_payload_L_per_h=cl_pl, vd_payload_L=vd_pl)
        assert result.t_half_payload_h == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestSimulateADCPKValidation:
    def test_invalid_dose_zero_raises(self):
        with pytest.raises(ValueError, match="total_antibody_dose_mg"):
            simulate_adc_pk("X", total_antibody_dose_mg=0.0)

    def test_invalid_dose_negative_raises(self):
        with pytest.raises(ValueError, match="total_antibody_dose_mg"):
            simulate_adc_pk("X", total_antibody_dose_mg=-10.0)

    def test_invalid_cl_antibody_raises(self):
        with pytest.raises(ValueError, match="cl_antibody_L_per_day"):
            simulate_adc_pk("X", total_antibody_dose_mg=100.0, cl_antibody_L_per_day=0.0)

    def test_invalid_vd_antibody_raises(self):
        with pytest.raises(ValueError, match="vd_antibody_L"):
            simulate_adc_pk("X", total_antibody_dose_mg=100.0, vd_antibody_L=-1.0)

    def test_invalid_dar_too_low_raises(self):
        with pytest.raises(ValueError, match="dar_initial"):
            simulate_adc_pk("X", total_antibody_dose_mg=100.0, dar_initial=0.5)

    def test_invalid_k_deconj_negative_raises(self):
        with pytest.raises(ValueError, match="k_deconj_per_day"):
            simulate_adc_pk("X", total_antibody_dose_mg=100.0, k_deconj_per_day=-0.1)

    def test_invalid_cl_payload_raises(self):
        with pytest.raises(ValueError, match="cl_payload_L_per_h"):
            simulate_adc_pk("X", total_antibody_dose_mg=100.0, cl_payload_L_per_h=0.0)

    def test_invalid_vd_payload_raises(self):
        with pytest.raises(ValueError, match="vd_payload_L"):
            simulate_adc_pk("X", total_antibody_dose_mg=100.0, vd_payload_L=0.0)

    def test_invalid_mw_antibody_raises(self):
        with pytest.raises(ValueError, match="mw_antibody_kDa"):
            simulate_adc_pk("X", total_antibody_dose_mg=100.0, mw_antibody_kDa=0.0)

    def test_invalid_mw_payload_raises(self):
        with pytest.raises(ValueError, match="mw_payload_Da"):
            simulate_adc_pk("X", total_antibody_dose_mg=100.0, mw_payload_Da=-1.0)


# ---------------------------------------------------------------------------
# compare_linker_stability tests
# ---------------------------------------------------------------------------


class TestCompareLinkerStability:
    def test_returns_list(self):
        results = compare_linker_stability("T-DM1", total_antibody_dose_mg=100.0)
        assert isinstance(results, list)
        assert len(results) == 4  # default 4 k_deconj values

    def test_sorted_by_auc_conjugated_descending(self):
        results = compare_linker_stability("T-DM1", total_antibody_dose_mg=100.0)
        aucs = [r.auc_conjugated for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_custom_k_deconj_values(self):
        k_vals = [0.01, 0.1, 0.5]
        results = compare_linker_stability("T-DM1", total_antibody_dose_mg=100.0,
                                           k_deconj_values=k_vals)
        assert len(results) == 3

    def test_most_stable_first(self):
        # Lowest k_deconj → highest conjugated AUC → should be first
        results = compare_linker_stability("T-DM1", total_antibody_dose_mg=100.0,
                                           k_deconj_values=[0.01, 0.1, 0.5])
        # First result should have lowest k_deconj (highest conjugated AUC)
        # Last should have highest k_deconj
        assert results[0].auc_conjugated >= results[-1].auc_conjugated

    def test_all_results_are_adc_pk_result(self):
        results = compare_linker_stability("T-DM1", total_antibody_dose_mg=100.0)
        for r in results:
            assert isinstance(r, ADCPKResult)

    def test_single_k_deconj_value(self):
        results = compare_linker_stability("T-DM1", total_antibody_dose_mg=100.0,
                                           k_deconj_values=[0.05])
        assert len(results) == 1

    def test_empty_k_deconj_raises(self):
        with pytest.raises(ValueError):
            compare_linker_stability("T-DM1", total_antibody_dose_mg=100.0, k_deconj_values=[])
