"""Tests for Phase 306 — Receptor Desensitization Model."""

from __future__ import annotations

import math

import numpy as np
import pytest

from omega_pbpk.clinical.receptor_desensitization import (
    DesensitizationResult,
    simulate_desensitization,
    tachyphylaxis_index,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs) -> DesensitizationResult:
    params = dict(
        drug_name="beta_agonist",
        dose_mg=10.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ec50_mg_L=0.1,
        emax=100.0,
        k_desens_per_h=0.2,
        k_resens_per_h=0.05,
        n_doses=5,
        interval_h=8.0,
        ka_per_h=1.5,
        dt_h=0.1,
    )
    params.update(kwargs)
    return simulate_desensitization(**params)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default(dose_mg=0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default(dose_mg=-5.0)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            _default(cl_L_per_h=0.0)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            _default(vd_L=0.0)

    def test_ec50_zero_raises(self):
        with pytest.raises(ValueError, match="ec50_mg_L"):
            _default(ec50_mg_L=0.0)

    def test_k_desens_zero_raises(self):
        with pytest.raises(ValueError, match="k_desens_per_h"):
            _default(k_desens_per_h=0.0)

    def test_k_resens_zero_raises(self):
        with pytest.raises(ValueError, match="k_resens_per_h"):
            _default(k_resens_per_h=0.0)

    def test_n_doses_zero_raises(self):
        with pytest.raises(ValueError, match="n_doses"):
            _default(n_doses=0)


# ---------------------------------------------------------------------------
# tachyphylaxis_index function
# ---------------------------------------------------------------------------


class TestTachyphylaxisIndex:
    def test_single_dose_returns_zero(self):
        assert tachyphylaxis_index([80.0]) == pytest.approx(0.0)

    def test_no_reduction_returns_zero(self):
        assert tachyphylaxis_index([80.0, 80.0, 80.0]) == pytest.approx(0.0)

    def test_complete_loss_returns_hundred(self):
        assert tachyphylaxis_index([80.0, 0.0]) == pytest.approx(100.0)

    def test_fifty_percent_reduction(self):
        assert tachyphylaxis_index([80.0, 40.0]) == pytest.approx(50.0)

    def test_zero_first_peak_returns_zero(self):
        assert tachyphylaxis_index([0.0, 10.0]) == pytest.approx(0.0)

    def test_empty_list_returns_zero(self):
        assert tachyphylaxis_index([]) == pytest.approx(0.0)

    def test_result_clamped_to_100(self):
        # Should not exceed 100 even with negative last peak
        ti = tachyphylaxis_index([100.0, -10.0])
        assert ti == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Single dose (baseline)
# ---------------------------------------------------------------------------


class TestSingleDose:
    def setup_method(self):
        self.result = _default(n_doses=1, interval_h=8.0)

    def test_peak_effects_has_one_entry(self):
        assert len(self.result.peak_effects) == 1

    def test_tachyphylaxis_index_zero_for_single_dose(self):
        assert self.result.tachyphylaxis_index == pytest.approx(0.0)

    def test_r_active_starts_at_one(self):
        assert self.result.r_active[0] == pytest.approx(1.0)

    def test_c_drug_starts_at_zero(self):
        assert self.result.c_drug[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Multi-dose desensitization
# ---------------------------------------------------------------------------


class TestMultiDoseDesensitization:
    def setup_method(self):
        # High desensitization rate to ensure visible tachyphylaxis
        self.result = _default(
            n_doses=5,
            k_desens_per_h=1.0,
            k_resens_per_h=0.02,
            dose_mg=50.0,
            ec50_mg_L=0.05,
        )

    def test_r_active_decreases_during_dosing(self):
        # R_active should decrease from dose 1 to last dose
        arr = np.array(self.result.r_active)
        # R at last dose time should be less than at first dose time
        last_dose_t = 4 * 8.0  # 4th interval for 5 doses
        first_t_idx = 0
        last_t_idx = int(round(last_dose_t / 0.1))
        assert arr[last_t_idx] < arr[first_t_idx]

    def test_peak_effects_decline_with_tachyphylaxis(self):
        # Peak effects should decrease over doses with high desensitization
        effects = self.result.peak_effects
        assert effects[0] >= effects[-1], "First dose effect should be >= last dose effect"

    def test_tachyphylaxis_index_positive(self):
        assert self.result.tachyphylaxis_index > 0.0

    def test_tachyphylaxis_index_below_100(self):
        assert self.result.tachyphylaxis_index <= 100.0

    def test_n_doses_preserved(self):
        assert self.result.n_doses == 5

    def test_peak_effects_length_equals_n_doses(self):
        assert len(self.result.peak_effects) == 5


# ---------------------------------------------------------------------------
# Resensitization
# ---------------------------------------------------------------------------


class TestResensitization:
    def test_r_active_recovers_after_last_dose(self):
        # With fast resensitization, R should recover after last dose
        result = _default(
            n_doses=3,
            k_desens_per_h=2.0,
            k_resens_per_h=0.5,
            interval_h=4.0,
            t_end_h=100.0,
        )
        arr_r = np.array(result.r_active)
        # Find minimum R during dosing window and check recovery afterward
        r_min = float(np.min(arr_r))
        r_final = arr_r[-1]
        assert r_final > r_min + 0.01  # some recovery occurred

    def test_time_to_50pct_resensitization_positive(self):
        result = _default(n_doses=3, k_desens_per_h=1.0, k_resens_per_h=0.1, t_end_h=100.0)
        assert result.time_to_50pct_resensitization_h >= 0.0

    def test_time_to_50pct_resensitization_finite(self):
        result = _default(
            n_doses=3,
            k_desens_per_h=1.0,
            k_resens_per_h=0.1,
            t_end_h=200.0,
        )
        assert math.isfinite(result.time_to_50pct_resensitization_h)

    def test_faster_resensitization_shorter_recovery_time(self):
        res_fast = _default(
            n_doses=3,
            k_desens_per_h=1.0,
            k_resens_per_h=0.5,
            t_end_h=200.0,
        )
        res_slow = _default(
            n_doses=3,
            k_desens_per_h=1.0,
            k_resens_per_h=0.1,
            t_end_h=200.0,
        )
        assert res_fast.time_to_50pct_resensitization_h <= res_slow.time_to_50pct_resensitization_h


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


class TestResultStructure:
    def setup_method(self):
        self.result = _default()

    def test_result_is_frozen_dataclass(self):
        assert isinstance(self.result, DesensitizationResult)
        with pytest.raises((AttributeError, TypeError)):
            self.result.drug_name = "other"  # type: ignore[misc]

    def test_arrays_same_length(self):
        n = len(self.result.times_h)
        assert len(self.result.c_drug) == n
        assert len(self.result.r_active) == n
        assert len(self.result.effect) == n

    def test_r_active_always_between_0_and_1(self):
        arr = np.array(self.result.r_active)
        assert np.all(arr >= 0.0)
        assert np.all(arr <= 1.0 + 1e-9)

    def test_c_drug_non_negative(self):
        arr = np.array(self.result.c_drug)
        assert np.all(arr >= 0.0)

    def test_notes_is_string(self):
        assert isinstance(self.result.notes, str)

    def test_drug_name_preserved(self):
        result = _default(drug_name="salbutamol")
        assert result.drug_name == "salbutamol"

    def test_interval_h_preserved(self):
        result = _default(interval_h=12.0)
        assert result.interval_h == pytest.approx(12.0)
