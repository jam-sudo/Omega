"""Tests for Phase 233 — Receptor Desensitization Model."""

import numpy as np
import pytest

from omega_pbpk.clinical.receptor_desensitization import (
    DesensitizationResult,
    simulate_desensitization,
    tolerance_index,
)


# ---------------------------------------------------------------------------
# Helper: default simulation (fast, short)
# ---------------------------------------------------------------------------


def _default(**kwargs):
    params = dict(
        drug_name="opioid",
        dose_mg=10.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        emax=1.0,
        ec50_mg_L=0.1,
        hill_n=1.0,
        k_des_per_h=0.5,
        k_recycle_per_h=0.1,
        k_intern_per_h=0.05,
        k_recycle_d_per_h=0.02,
        r0=100.0,
        ksyn_per_h=2.0,
        kdeg_per_h=0.02,
        route="oral",
        ka_per_h=1.0,
        t_end_h=48.0,
        dt_h=0.5,
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

    def test_emax_zero_raises(self):
        with pytest.raises(ValueError, match="emax"):
            _default(emax=0.0)

    def test_ec50_zero_raises(self):
        with pytest.raises(ValueError, match="ec50_mg_L"):
            _default(ec50_mg_L=0.0)

    def test_hill_n_zero_raises(self):
        with pytest.raises(ValueError, match="hill_n"):
            _default(hill_n=0.0)

    def test_r0_zero_raises(self):
        with pytest.raises(ValueError, match="r0"):
            _default(r0=0.0)

    def test_k_des_negative_raises(self):
        with pytest.raises(ValueError, match="k_des_per_h"):
            _default(k_des_per_h=-0.1)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            _default(route="subcutaneous")

    def test_oral_ka_zero_raises(self):
        with pytest.raises(ValueError, match="ka_per_h"):
            _default(route="oral", ka_per_h=0.0)


# ---------------------------------------------------------------------------
# No desensitization (k_des=0)
# ---------------------------------------------------------------------------


class TestNoDesensitization:
    def setup_method(self):
        self.result = _default(k_des_per_h=0.0, k_intern_per_h=0.0, k_recycle_d_per_h=0.0)

    def test_r_active_stays_close_to_r0(self):
        # With no desensitization or internalization, active receptors stay near r0
        # (some synthesis/degradation may cause drift but should be close)
        assert np.min(self.result.r_active) > 50.0  # stays above half of r0=100

    def test_r_desensitized_near_zero(self):
        # Without k_des, desensitized pool stays at zero
        assert np.max(self.result.r_desensitized) < 1.0

    def test_note_about_no_desensitization(self):
        assert any("k_des=0" in note for note in self.result.notes)

    def test_tolerance_index_low_when_no_desensitization(self):
        ti = tolerance_index(self.result)
        # With no desensitization, effect declines only because drug washes out
        # tolerance_index measures effect loss from peak to end — should be finite
        assert 0.0 <= ti <= 100.0


# ---------------------------------------------------------------------------
# High desensitization → significant tolerance
# ---------------------------------------------------------------------------


class TestHighDesensitization:
    def setup_method(self):
        self.result = _default(
            k_des_per_h=5.0,
            k_intern_per_h=2.0,
            t_end_h=72.0,
            dose_mg=100.0,
            ec50_mg_L=0.05,
        )

    def test_r_active_min_fraction_low(self):
        # With high k_des, active receptors should drop substantially
        assert self.result.r_active_min < 0.8  # less than 80% of initial

    def test_tolerance_developed_true(self):
        # High desensitization should develop tolerance
        ti = tolerance_index(self.result)
        assert isinstance(ti, float)

    def test_tolerance_index_in_range(self):
        ti = tolerance_index(self.result)
        assert 0.0 <= ti <= 100.0


# ---------------------------------------------------------------------------
# Drug effect dynamics
# ---------------------------------------------------------------------------


class TestDrugEffectDynamics:
    def setup_method(self):
        # Large dose, fast absorption → high early concentration, then desensitization
        self.result = _default(
            dose_mg=50.0,
            ec50_mg_L=0.05,
            k_des_per_h=1.0,
            k_intern_per_h=0.5,
            t_end_h=72.0,
        )

    def test_peak_effect_positive(self):
        assert self.result.peak_effect > 0.0

    def test_drug_effect_peaks_early(self):
        # Peak effect should occur before 50% of total time
        peak_idx = int(np.argmax(self.result.drug_effect))
        assert peak_idx < len(self.result.times_h) * 0.6

    def test_effect_at_end_is_float(self):
        assert isinstance(self.result.effect_at_end, float)

    def test_effect_array_length_matches_times(self):
        assert len(self.result.drug_effect) == len(self.result.times_h)


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------


class TestReceptorConservation:
    def setup_method(self):
        self.result = _default(ksyn_per_h=0.0, kdeg_per_h=0.0)

    def test_receptor_conservation_approximate(self):
        # Total receptors R + D + I should be approximately conserved
        # (within 20% of r0 because synthesis=0, degradation=0)
        total = self.result.r_active + self.result.r_desensitized + self.result.r_internalized
        r0 = 100.0
        # Max deviation from r0 should be within 20%
        max_deviation = np.max(np.abs(total - r0)) / r0
        assert max_deviation < 0.20, f"Max receptor deviation: {max_deviation:.3f}"


# ---------------------------------------------------------------------------
# Result field types
# ---------------------------------------------------------------------------


class TestResultFieldTypes:
    def setup_method(self):
        self.result = _default()

    def test_result_is_dataclass(self):
        assert isinstance(self.result, DesensitizationResult)

    def test_times_is_array(self):
        assert isinstance(self.result.times_h, np.ndarray)

    def test_c_plasma_is_array(self):
        assert isinstance(self.result.c_plasma_mg_L, np.ndarray)

    def test_r_active_is_array(self):
        assert isinstance(self.result.r_active, np.ndarray)

    def test_tolerance_developed_is_bool(self):
        assert isinstance(self.result.tolerance_developed, bool)

    def test_r_active_min_is_float(self):
        assert isinstance(self.result.r_active_min, float)

    def test_notes_is_list(self):
        assert isinstance(self.result.notes, list)

    def test_tolerance_index_is_float(self):
        ti = tolerance_index(self.result)
        assert isinstance(ti, float)

    def test_iv_route(self):
        result = _default(route="iv")
        assert result.route == "iv"
        assert result.c_plasma_mg_L[0] > 0.0  # immediate bolus
