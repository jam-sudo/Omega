"""Tests for Antibody-Drug Conjugate (ADC) PK model — Phase 497."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.adc_pk import (
    ADCPKResult,
    calculate_dar_distribution,
    linker_stability,
    simulate_adc_pk,
    therapeutic_index_adc,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_result(**kwargs) -> ADCPKResult:
    return simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6, **kwargs)


# ---------------------------------------------------------------------------
# simulate_adc_pk — basic correctness
# ---------------------------------------------------------------------------


class TestSimulateADCPKBasic:
    def test_returns_adc_pk_result(self):
        result = _default_result()
        assert isinstance(result, ADCPKResult)

    def test_drug_name_preserved(self):
        result = simulate_adc_pk("Kadcyla", dose_mg_per_kg=3.6)
        assert result.drug_name == "Kadcyla"

    def test_dar_preserved(self):
        result = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6, dar=8)
        assert result.dar == 8

    def test_times_start_at_zero(self):
        result = _default_result()
        assert result.times_days[0] == pytest.approx(0.0)

    def test_times_length_reasonable(self):
        result = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6, t_end_days=10.0, dt_days=0.1)
        # 10/0.1 + 1 = 101 points
        assert len(result.times_days) == pytest.approx(101, abs=2)

    def test_concentration_lists_same_length(self):
        result = _default_result()
        n = len(result.times_days)
        assert len(result.c_adc_mg_L) == n
        assert len(result.c_antibody_free_mg_L) == n
        assert len(result.c_payload_mg_L) == n

    def test_c_adc_starts_high_at_t0(self):
        result = _default_result()
        # IV bolus: ADC peaks at t=0
        assert result.c_adc_mg_L[0] > 0.0

    def test_c_antibody_free_starts_zero(self):
        result = _default_result()
        assert result.c_antibody_free_mg_L[0] == pytest.approx(0.0)

    def test_c_payload_starts_zero(self):
        result = _default_result()
        assert result.c_payload_mg_L[0] == pytest.approx(0.0)

    def test_adc_concentration_decreases(self):
        result = _default_result()
        assert result.c_adc_mg_L[0] > result.c_adc_mg_L[-1]

    def test_cmax_adc_positive(self):
        result = _default_result()
        assert result.cmax_adc_mg_L > 0.0

    def test_cmax_adc_equals_initial(self):
        # For IV bolus, Cmax_ADC should be at t=0
        result = _default_result()
        assert result.cmax_adc_mg_L == pytest.approx(result.c_adc_mg_L[0])

    def test_cmax_payload_positive(self):
        result = _default_result()
        assert result.cmax_payload_mg_L > 0.0

    def test_auc_adc_positive(self):
        result = _default_result()
        assert result.auc_adc_mg_day_L > 0.0

    def test_auc_payload_positive(self):
        result = _default_result()
        assert result.auc_payload_mg_day_L > 0.0

    def test_t_half_adc_positive(self):
        result = _default_result()
        assert result.t_half_adc_days > 0.0

    def test_notes_nonempty(self):
        result = _default_result()
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# DAR and deconjugation effects
# ---------------------------------------------------------------------------


class TestDAREffects:
    def test_higher_dar_higher_payload_cmax(self):
        r_low = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6, dar=2)
        r_high = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6, dar=8)
        assert r_high.cmax_payload_mg_L > r_low.cmax_payload_mg_L

    def test_higher_k_deconj_faster_payload_release(self):
        r_slow = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6, k_deconj_per_day=0.01)
        r_fast = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6, k_deconj_per_day=0.5)
        assert r_fast.cmax_payload_mg_L > r_slow.cmax_payload_mg_L

    def test_payload_cmax_after_adc_cmax(self):
        # For IV bolus, ADC Cmax is at t=0; payload accumulates over time
        result = _default_result()
        adc_tmax_idx = result.c_adc_mg_L.index(result.cmax_adc_mg_L)
        payload_tmax_idx = result.c_payload_mg_L.index(result.cmax_payload_mg_L)
        assert payload_tmax_idx >= adc_tmax_idx

    def test_auc_payload_auc_adc_ratio_scales_with_dar(self):
        r_dar4 = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6, dar=4)
        r_dar8 = simulate_adc_pk("T-DM1", dose_mg_per_kg=3.6, dar=8)
        ratio_dar4 = r_dar4.auc_payload_mg_day_L / r_dar4.auc_adc_mg_day_L
        ratio_dar8 = r_dar8.auc_payload_mg_day_L / r_dar8.auc_adc_mg_day_L
        assert ratio_dar8 > ratio_dar4

    def test_two_x_dose_two_x_cmax_payload(self):
        r1 = simulate_adc_pk("T-DM1", dose_mg_per_kg=2.0)
        r2 = simulate_adc_pk("T-DM1", dose_mg_per_kg=4.0)
        assert r2.cmax_payload_mg_L == pytest.approx(2.0 * r1.cmax_payload_mg_L, rel=0.05)


# ---------------------------------------------------------------------------
# Analytical t_half verification
# ---------------------------------------------------------------------------


class TestAnalyticalTHalf:
    def test_t_half_adc_analytical(self):
        cl = 0.3
        vd = 3.0
        k_deconj = 0.1
        ke_adc = cl / vd
        k_total = k_deconj + ke_adc
        expected = math.log(2) / k_total
        result = simulate_adc_pk(
            "T-DM1",
            dose_mg_per_kg=3.6,
            cl_adc_L_per_day=cl,
            vd_adc_L=vd,
            k_deconj_per_day=k_deconj,
        )
        assert result.t_half_adc_days == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestSimulateADCPKValidation:
    def test_invalid_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg_per_kg"):
            simulate_adc_pk("X", dose_mg_per_kg=0.0)

    def test_invalid_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg_per_kg"):
            simulate_adc_pk("X", dose_mg_per_kg=-1.0)

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError, match="weight_kg"):
            simulate_adc_pk("X", dose_mg_per_kg=3.6, weight_kg=0.0)

    def test_invalid_dar_raises(self):
        with pytest.raises(ValueError, match="dar"):
            simulate_adc_pk("X", dose_mg_per_kg=3.6, dar=0)

    def test_invalid_cl_adc_raises(self):
        with pytest.raises(ValueError, match="cl_adc_L_per_day"):
            simulate_adc_pk("X", dose_mg_per_kg=3.6, cl_adc_L_per_day=0.0)

    def test_invalid_vd_adc_raises(self):
        with pytest.raises(ValueError, match="vd_adc_L"):
            simulate_adc_pk("X", dose_mg_per_kg=3.6, vd_adc_L=-1.0)

    def test_invalid_k_deconj_negative_raises(self):
        with pytest.raises(ValueError, match="k_deconj_per_day"):
            simulate_adc_pk("X", dose_mg_per_kg=3.6, k_deconj_per_day=-0.1)

    def test_invalid_cl_payload_raises(self):
        with pytest.raises(ValueError, match="cl_payload_L_per_day"):
            simulate_adc_pk("X", dose_mg_per_kg=3.6, cl_payload_L_per_day=0.0)

    def test_invalid_vd_payload_raises(self):
        with pytest.raises(ValueError, match="vd_payload_L"):
            simulate_adc_pk("X", dose_mg_per_kg=3.6, vd_payload_L=0.0)

    def test_invalid_t_end_raises(self):
        with pytest.raises(ValueError, match="t_end_days"):
            simulate_adc_pk("X", dose_mg_per_kg=3.6, t_end_days=0.0)

    def test_invalid_dt_raises(self):
        with pytest.raises(ValueError, match="dt_days"):
            simulate_adc_pk("X", dose_mg_per_kg=3.6, dt_days=0.0)


# ---------------------------------------------------------------------------
# calculate_dar_distribution
# ---------------------------------------------------------------------------


class TestCalculateDARDistribution:
    def test_returns_dict_with_required_keys(self):
        result = calculate_dar_distribution(4.0, 0.1, 7.0)
        assert "mean_dar" in result
        assert "fraction_remaining" in result
        assert "fraction_deconjugated" in result

    def test_fractions_sum_to_one(self):
        result = calculate_dar_distribution(4.0, 0.1, 7.0)
        assert result["fraction_remaining"] + result["fraction_deconjugated"] == pytest.approx(1.0)

    def test_zero_time_no_deconjugation(self):
        result = calculate_dar_distribution(4.0, 0.1, 0.0)
        assert result["mean_dar"] == pytest.approx(4.0)
        assert result["fraction_deconjugated"] == pytest.approx(0.0)

    def test_mean_dar_decreases_over_time(self):
        r_early = calculate_dar_distribution(4.0, 0.2, 1.0)
        r_late = calculate_dar_distribution(4.0, 0.2, 10.0)
        assert r_late["mean_dar"] < r_early["mean_dar"]

    def test_higher_k_deconj_lower_mean_dar(self):
        r_slow = calculate_dar_distribution(4.0, 0.05, 7.0)
        r_fast = calculate_dar_distribution(4.0, 0.5, 7.0)
        assert r_fast["mean_dar"] < r_slow["mean_dar"]

    def test_invalid_initial_dar(self):
        with pytest.raises(ValueError, match="initial_dar"):
            calculate_dar_distribution(-1.0, 0.1, 7.0)

    def test_invalid_k_deconj(self):
        with pytest.raises(ValueError, match="k_deconj"):
            calculate_dar_distribution(4.0, -0.1, 7.0)

    def test_invalid_t_days(self):
        with pytest.raises(ValueError, match="t_days"):
            calculate_dar_distribution(4.0, 0.1, -1.0)


# ---------------------------------------------------------------------------
# therapeutic_index_adc
# ---------------------------------------------------------------------------


class TestTherapeuticIndexADC:
    def test_returns_float(self):
        result = therapeutic_index_adc(0.05, 0.01, 0.1)
        assert isinstance(result, float)

    def test_ti_ratio_correct(self):
        result = therapeutic_index_adc(0.05, 0.01, 0.1)
        assert result == pytest.approx(10.0)

    def test_zero_target_conc_raises(self):
        with pytest.raises(ValueError, match="target_conc_mg_L"):
            therapeutic_index_adc(0.05, 0.0, 0.1)

    def test_zero_toxic_conc_raises(self):
        with pytest.raises(ValueError, match="toxic_conc_mg_L"):
            therapeutic_index_adc(0.05, 0.01, 0.0)

    def test_negative_cmax_raises(self):
        with pytest.raises(ValueError, match="cmax_payload_mg_L"):
            therapeutic_index_adc(-0.1, 0.01, 0.1)


# ---------------------------------------------------------------------------
# linker_stability
# ---------------------------------------------------------------------------


class TestLinkerStability:
    def test_returns_float_in_range(self):
        for lt in ("cleavable", "non_cleavable", "disulfide"):
            result = linker_stability(lt)
            assert 0.0 <= result <= 1.0

    def test_non_cleavable_most_stable(self):
        s_nc = linker_stability("non_cleavable")
        s_cl = linker_stability("cleavable")
        s_ds = linker_stability("disulfide")
        assert s_nc >= s_cl
        assert s_nc >= s_ds

    def test_disulfide_less_stable_at_low_ph(self):
        s_neutral = linker_stability("disulfide", ph=7.4)
        s_acidic = linker_stability("disulfide", ph=5.0)
        assert s_acidic < s_neutral

    def test_cleavable_less_stable_at_low_ph(self):
        s_neutral = linker_stability("cleavable", ph=7.4)
        s_acidic = linker_stability("cleavable", ph=5.0)
        assert s_acidic < s_neutral

    def test_higher_temperature_lower_stability(self):
        s_37 = linker_stability("cleavable", temperature_C=37.0)
        s_42 = linker_stability("cleavable", temperature_C=42.0)
        assert s_42 < s_37

    def test_invalid_linker_type_raises(self):
        with pytest.raises(ValueError, match="linker_type"):
            linker_stability("unknown_linker")

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature_C"):
            linker_stability("cleavable", temperature_C=0.0)
