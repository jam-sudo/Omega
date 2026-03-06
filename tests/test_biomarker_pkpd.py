"""Tests for Phase 234 — Biomarker PK/PD Model."""

import math

import numpy as np
import pytest

from omega_pbpk.clinical.biomarker_pkpd import (
    BiomarkerResult,
    simulate_biomarker_pkpd,
    time_to_effect,
)


# ---------------------------------------------------------------------------
# Helper: default simulation
# ---------------------------------------------------------------------------


def _default(**kwargs):
    params = dict(
        drug_name="statin",
        biomarker_name="LDL-C",
        dose_mg=20.0,
        cl_L_per_h=5.0,
        vd_L=100.0,
        kin_per_h=10.0,
        kout_per_h=0.1,
        imax_or_smax=0.8,
        ic50_or_sc50_mg_L=0.5,
        model="I",
        route="oral",
        ka_per_h=1.0,
        t_end_h=720.0,
        dt_h=2.0,
    )
    params.update(kwargs)
    return simulate_biomarker_pkpd(**params)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="model"):
            _default(model="V")

    def test_imax_greater_than_one_raises(self):
        with pytest.raises(ValueError, match="Imax"):
            _default(model="I", imax_or_smax=1.5)

    def test_imax_zero_raises_for_inhibition(self):
        with pytest.raises(ValueError, match="Imax"):
            _default(model="I", imax_or_smax=0.0)

    def test_imax_negative_raises(self):
        with pytest.raises(ValueError, match="Imax"):
            _default(model="III", imax_or_smax=-0.1)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            _default(cl_L_per_h=0.0)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            _default(vd_L=0.0)

    def test_kin_zero_raises(self):
        with pytest.raises(ValueError, match="kin_per_h"):
            _default(kin_per_h=0.0)

    def test_kout_zero_raises(self):
        with pytest.raises(ValueError, match="kout_per_h"):
            _default(kout_per_h=0.0)

    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default(dose_mg=0.0)

    def test_ic50_zero_raises(self):
        with pytest.raises(ValueError, match="ic50_or_sc50_mg_L"):
            _default(ic50_or_sc50_mg_L=0.0)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            _default(route="inhaled")

    def test_smax_negative_raises_for_stimulation(self):
        with pytest.raises(ValueError, match="Smax"):
            _default(model="II", imax_or_smax=-0.5)


# ---------------------------------------------------------------------------
# Model I: Inhibit production → biomarker decreases
# ---------------------------------------------------------------------------


class TestModelI:
    def setup_method(self):
        self.result = _default(model="I", imax_or_smax=0.9)

    def test_biomarker_decreases_below_baseline(self):
        b0 = self.result.b0_baseline
        assert self.result.b_min < b0

    def test_pct_change_at_end_negative(self):
        assert self.result.pct_change_at_end < 0.0

    def test_b0_equals_kin_over_kout(self):
        b0_expected = 10.0 / 0.1  # kin/kout
        assert abs(self.result.b0_baseline - b0_expected) < 1e-9


# ---------------------------------------------------------------------------
# Model II: Stimulate production → biomarker increases
# ---------------------------------------------------------------------------


class TestModelII:
    def setup_method(self):
        self.result = _default(model="II", imax_or_smax=1.5)

    def test_biomarker_increases_above_baseline(self):
        b0 = self.result.b0_baseline
        assert self.result.b_max > b0

    def test_pct_change_at_end_positive(self):
        assert self.result.pct_change_at_end > 0.0

    def test_smax_greater_than_one_accepted(self):
        # Smax > 1 is valid for stimulation models
        assert self.result.dose_mg == 20.0


# ---------------------------------------------------------------------------
# Model III: Inhibit degradation → biomarker increases
# ---------------------------------------------------------------------------


class TestModelIII:
    def setup_method(self):
        self.result = _default(model="III", imax_or_smax=0.8)

    def test_biomarker_increases_from_baseline(self):
        b0 = self.result.b0_baseline
        # Inhibiting degradation should increase biomarker
        assert self.result.b_max > b0

    def test_pct_change_at_end_positive(self):
        assert self.result.pct_change_at_end > 0.0


# ---------------------------------------------------------------------------
# Model IV: Stimulate degradation → biomarker decreases
# ---------------------------------------------------------------------------


class TestModelIV:
    def setup_method(self):
        self.result = _default(model="IV", imax_or_smax=2.0)

    def test_biomarker_decreases_from_baseline(self):
        b0 = self.result.b0_baseline
        assert self.result.b_min < b0

    def test_pct_change_at_end_negative(self):
        assert self.result.pct_change_at_end < 0.0


# ---------------------------------------------------------------------------
# Baseline verification
# ---------------------------------------------------------------------------


class TestBaseline:
    def test_b0_baseline_kin_over_kout(self):
        result = _default(kin_per_h=5.0, kout_per_h=0.5)
        assert abs(result.b0_baseline - 10.0) < 1e-9

    def test_biomarker_starts_at_baseline(self):
        result = _default()
        assert abs(result.biomarker_values[0] - result.b0_baseline) < 1e-6


# ---------------------------------------------------------------------------
# time_to_effect
# ---------------------------------------------------------------------------


class TestTimeToEffect:
    def test_time_to_decrease_model_i(self):
        result = _default(model="I", imax_or_smax=0.9, t_end_h=1440.0)
        tte = time_to_effect(result, target_pct_change=-20.0)
        assert math.isfinite(tte)
        assert tte > 0.0

    def test_time_to_increase_model_ii(self):
        result = _default(model="II", imax_or_smax=2.0, t_end_h=720.0)
        tte = time_to_effect(result, target_pct_change=20.0)
        assert math.isfinite(tte)
        assert tte > 0.0

    def test_unachievable_target_returns_inf(self):
        result = _default(model="I", imax_or_smax=0.5, t_end_h=720.0)
        # Target is 200% increase — impossible with an inhibition model
        tte = time_to_effect(result, target_pct_change=200.0)
        assert math.isinf(tte)


# ---------------------------------------------------------------------------
# Result field types
# ---------------------------------------------------------------------------


class TestResultFieldTypes:
    def setup_method(self):
        self.result = _default()

    def test_result_is_dataclass(self):
        assert isinstance(self.result, BiomarkerResult)

    def test_times_is_array(self):
        assert isinstance(self.result.times_h, np.ndarray)

    def test_biomarker_values_is_array(self):
        assert isinstance(self.result.biomarker_values, np.ndarray)

    def test_b0_baseline_is_float(self):
        assert isinstance(self.result.b0_baseline, float)

    def test_pct_change_at_end_is_float(self):
        assert isinstance(self.result.pct_change_at_end, float)

    def test_time_to_nadir_is_float(self):
        assert isinstance(self.result.time_to_nadir_h, float)

    def test_time_to_peak_is_float(self):
        assert isinstance(self.result.time_to_peak_h, float)

    def test_notes_is_list(self):
        assert isinstance(self.result.notes, list)

    def test_model_field_matches_input(self):
        assert self.result.model == "I"
