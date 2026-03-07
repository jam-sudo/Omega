"""Tests for Phase 297: Nasal Drug Delivery PK (core/nasal_pk.py)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.nasal_pk import (
    NasalPKResult,
    compare_nasal_vs_oral,
    simulate_nasal_pk,
)


# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=1.0,
    fraction_absorbed=0.8,
    ka_nasal_per_h=6.0,
    k_mucociliary_per_h=20.0,
    cl_L_per_h=5.0,
    vd_L=50.0,
    lag_time_h=0.0,
    t_end_h=12.0,
    dt_h=0.05,
)


def default_result(**overrides) -> NasalPKResult:
    params = {**DEFAULTS, **overrides}
    return simulate_nasal_pk(**params)


# ---------------------------------------------------------------------------
# Return type and dataclass
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_nasal_pk_result(self):
        result = default_result()
        assert isinstance(result, NasalPKResult)

    def test_frozen_dataclass(self):
        result = default_result()
        with pytest.raises((AttributeError, TypeError)):
            result.cmax = 999.0  # type: ignore[misc]

    def test_field_types(self):
        result = default_result()
        assert isinstance(result.drug_name, str)
        assert isinstance(result.dose_mg, float)
        assert isinstance(result.fraction_absorbed, float)
        assert isinstance(result.ka_nasal_per_h, float)
        assert isinstance(result.k_mucociliary_per_h, float)
        assert isinstance(result.times_h, list)
        assert isinstance(result.c_plasma, list)
        assert isinstance(result.cmax, float)
        assert isinstance(result.tmax_h, float)
        assert isinstance(result.auc, float)
        assert isinstance(result.f_bioavailable, float)
        assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_nasal_pk("X", dose_mg=0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_nasal_pk("X", dose_mg=-1.0)

    def test_fraction_absorbed_zero_raises(self):
        with pytest.raises(ValueError, match="fraction_absorbed"):
            simulate_nasal_pk("X", dose_mg=1.0, fraction_absorbed=0.0)

    def test_fraction_absorbed_above_one_raises(self):
        with pytest.raises(ValueError, match="fraction_absorbed"):
            simulate_nasal_pk("X", dose_mg=1.0, fraction_absorbed=1.1)

    def test_ka_zero_raises(self):
        with pytest.raises(ValueError, match="ka_nasal_per_h"):
            simulate_nasal_pk("X", dose_mg=1.0, ka_nasal_per_h=0.0)

    def test_kmc_negative_raises(self):
        with pytest.raises(ValueError, match="k_mucociliary_per_h"):
            simulate_nasal_pk("X", dose_mg=1.0, k_mucociliary_per_h=-1.0)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_nasal_pk("X", dose_mg=1.0, cl_L_per_h=0.0)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_nasal_pk("X", dose_mg=1.0, vd_L=0.0)

    def test_lag_negative_raises(self):
        with pytest.raises(ValueError, match="lag_time_h"):
            simulate_nasal_pk("X", dose_mg=1.0, lag_time_h=-0.1)

    def test_t_end_zero_raises(self):
        with pytest.raises(ValueError, match="t_end_h"):
            simulate_nasal_pk("X", dose_mg=1.0, t_end_h=0.0)

    def test_dt_zero_raises(self):
        with pytest.raises(ValueError, match="dt_h"):
            simulate_nasal_pk("X", dose_mg=1.0, dt_h=0.0)


# ---------------------------------------------------------------------------
# Basic physics
# ---------------------------------------------------------------------------

class TestBasicPhysics:
    def test_c_plasma_initially_zero(self):
        result = default_result()
        assert result.c_plasma[0] == pytest.approx(0.0, abs=1e-12)

    def test_cmax_positive(self):
        result = default_result()
        assert result.cmax > 0.0

    def test_auc_positive(self):
        result = default_result()
        assert result.auc > 0.0

    def test_tmax_in_range(self):
        result = default_result()
        assert 0.0 <= result.tmax_h <= DEFAULTS["t_end_h"]

    def test_times_length_matches_c_plasma(self):
        result = default_result()
        assert len(result.times_h) == len(result.c_plasma)

    def test_plasma_non_negative(self):
        result = default_result()
        assert all(c >= 0.0 for c in result.c_plasma)

    def test_cmax_matches_max_c_plasma(self):
        result = default_result()
        assert result.cmax == pytest.approx(max(result.c_plasma), rel=1e-9)

    def test_drug_name_stored(self):
        result = simulate_nasal_pk("Ketamine", dose_mg=5.0)
        assert result.drug_name == "Ketamine"

    def test_dose_stored(self):
        result = simulate_nasal_pk("X", dose_mg=2.5)
        assert result.dose_mg == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# f_bioavailable
# ---------------------------------------------------------------------------

class TestFBioavailable:
    def test_f_bioavailable_range(self):
        result = default_result()
        assert 0.0 <= result.f_bioavailable <= 1.0

    def test_f_bioavailable_formula(self):
        # f = ka / (ka + k_mc)
        ka = 6.0
        kmc = 20.0
        result = default_result(ka_nasal_per_h=ka, k_mucociliary_per_h=kmc)
        expected = ka / (ka + kmc)
        assert result.f_bioavailable == pytest.approx(expected, rel=1e-6)

    def test_zero_mcc_gives_high_f_bioavailable(self):
        result = default_result(k_mucociliary_per_h=0.0)
        assert result.f_bioavailable == pytest.approx(1.0, rel=1e-6)

    def test_high_mcc_reduces_f_bioavailable(self):
        low = default_result(k_mucociliary_per_h=1.0)
        high = default_result(k_mucociliary_per_h=50.0)
        assert high.f_bioavailable < low.f_bioavailable


# ---------------------------------------------------------------------------
# Effect of parameters
# ---------------------------------------------------------------------------

class TestParameterEffects:
    def test_larger_dose_gives_proportional_cmax(self):
        r1 = simulate_nasal_pk("X", dose_mg=1.0)
        r2 = simulate_nasal_pk("X", dose_mg=2.0)
        assert r2.cmax == pytest.approx(r1.cmax * 2.0, rel=0.01)

    def test_larger_dose_gives_proportional_auc(self):
        r1 = simulate_nasal_pk("X", dose_mg=1.0)
        r2 = simulate_nasal_pk("X", dose_mg=4.0)
        assert r2.auc == pytest.approx(r1.auc * 4.0, rel=0.01)

    def test_faster_ka_gives_higher_cmax(self):
        slow = default_result(ka_nasal_per_h=2.0)
        fast = default_result(ka_nasal_per_h=12.0)
        assert fast.cmax > slow.cmax

    def test_faster_ka_gives_earlier_tmax(self):
        slow = default_result(ka_nasal_per_h=2.0)
        fast = default_result(ka_nasal_per_h=15.0)
        assert fast.tmax_h <= slow.tmax_h

    def test_lag_time_delays_tmax(self):
        no_lag = default_result(lag_time_h=0.0)
        with_lag = default_result(lag_time_h=0.5)
        assert with_lag.tmax_h >= no_lag.tmax_h

    def test_larger_vd_reduces_cmax(self):
        r1 = default_result(vd_L=20.0)
        r2 = default_result(vd_L=100.0)
        assert r2.cmax < r1.cmax

    def test_fraction_absorbed_scales_cmax(self):
        r1 = default_result(fraction_absorbed=0.5)
        r2 = default_result(fraction_absorbed=1.0)
        assert r2.cmax == pytest.approx(r1.cmax * 2.0, rel=0.02)


# ---------------------------------------------------------------------------
# Lag time
# ---------------------------------------------------------------------------

class TestLagTime:
    def test_lag_time_plasma_flat_during_lag(self):
        result = simulate_nasal_pk("X", dose_mg=1.0, lag_time_h=1.0, dt_h=0.05)
        times = result.times_h
        # Find index just before lag ends
        idx_before_lag = next(i for i, t in enumerate(times) if t >= 0.95) - 1
        if idx_before_lag > 0:
            # Plasma should be essentially zero during lag
            assert all(c < 1e-12 for c in result.c_plasma[:idx_before_lag])

    def test_lag_time_zero_same_as_default(self):
        r1 = simulate_nasal_pk("X", dose_mg=1.0, lag_time_h=0.0)
        r2 = simulate_nasal_pk("X", dose_mg=1.0)
        assert r1.cmax == pytest.approx(r2.cmax, rel=1e-9)


# ---------------------------------------------------------------------------
# compare_nasal_vs_oral
# ---------------------------------------------------------------------------

class TestCompareNasalVsOral:
    def test_returns_dict(self):
        result = compare_nasal_vs_oral(
            drug_name="X",
            dose_mg=1.0,
            ka_oral_per_h=1.0,
            ka_nasal_per_h=6.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
        )
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = compare_nasal_vs_oral(
            drug_name="X",
            dose_mg=1.0,
            ka_oral_per_h=1.0,
            ka_nasal_per_h=6.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
        )
        assert "nasal" in result
        assert "oral_cmax" in result
        assert "oral_tmax_h" in result
        assert "oral_auc" in result
        assert "nasal_vs_oral_cmax_ratio" in result
        assert "nasal_vs_oral_auc_ratio" in result

    def test_nasal_result_is_nasal_pk_result(self):
        result = compare_nasal_vs_oral(
            drug_name="X",
            dose_mg=1.0,
            ka_oral_per_h=1.0,
            ka_nasal_per_h=6.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
        )
        assert isinstance(result["nasal"], NasalPKResult)

    def test_invalid_f_oral_raises(self):
        with pytest.raises(ValueError):
            compare_nasal_vs_oral(
                drug_name="X",
                dose_mg=1.0,
                ka_oral_per_h=1.0,
                ka_nasal_per_h=6.0,
                cl_L_per_h=5.0,
                vd_L=50.0,
                f_oral=0.0,
            )

    def test_nasal_faster_onset_than_oral(self):
        """With higher ka, nasal tmax should be earlier."""
        result = compare_nasal_vs_oral(
            drug_name="X",
            dose_mg=1.0,
            ka_oral_per_h=0.5,  # slow oral absorption
            ka_nasal_per_h=10.0,  # fast nasal
            cl_L_per_h=5.0,
            vd_L=50.0,
            f_oral=1.0,
            f_nasal=1.0,
            k_mucociliary_per_h=0.0,  # no MCC loss
        )
        assert result["nasal"].tmax_h < result["oral_tmax_h"]

    def test_oral_values_positive(self):
        result = compare_nasal_vs_oral(
            drug_name="X",
            dose_mg=1.0,
            ka_oral_per_h=1.0,
            ka_nasal_per_h=6.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
        )
        assert result["oral_cmax"] > 0.0
        assert result["oral_auc"] > 0.0


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------

class TestNotes:
    def test_notes_is_string(self):
        result = default_result()
        assert isinstance(result.notes, str)

    def test_high_mcc_note(self):
        result = default_result(ka_nasal_per_h=2.0, k_mucociliary_per_h=50.0)
        assert "MCC" in result.notes or "mucociliary" in result.notes.lower()

    def test_lag_note_when_present(self):
        result = simulate_nasal_pk("X", dose_mg=1.0, lag_time_h=0.5)
        assert "lag" in result.notes.lower() or "Lag" in result.notes
