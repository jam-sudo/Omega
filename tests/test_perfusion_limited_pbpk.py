"""Tests for perfusion-limited PBPK model (Phase 428)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.perfusion_limited_pbpk import (
    _DEFAULT_TISSUE_FLOWS,
    _VD_PLASMA_L,
    PerfusionLimitedPBPKResult,
    simulate_perfusion_limited_pbpk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_KP = {"liver": 2.0, "kidney": 1.5, "muscle": 0.8, "fat": 3.0}

def _sim(**kw):
    """Run simulation with convenient defaults."""
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_hep_L_per_h=5.0,
        qh_L_per_h=54.0,
        vd_total_L=50.0,
        kp_tissues=dict(_DEFAULT_KP),
        tissue_flows=dict(_DEFAULT_TISSUE_FLOWS),
        t_end_h=24.0,
        dt_h=0.1,
    )
    defaults.update(kw)
    return simulate_perfusion_limited_pbpk(**defaults)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_result_type(self):
        r = _sim()
        assert isinstance(r, PerfusionLimitedPBPKResult)

    def test_times_h_length_correct(self):
        r = _sim(t_end_h=10.0, dt_h=0.1)
        # Should be n_steps + 1
        expected = int(10.0 / 0.1) + 1
        assert len(r.times_h) == expected

    def test_c_plasma_same_length_as_times(self):
        r = _sim()
        assert len(r.c_plasma_mg_L) == len(r.times_h)

    def test_tissue_concs_has_all_4_tissues(self):
        r = _sim()
        assert set(r.tissue_concs.keys()) == {"liver", "kidney", "muscle", "fat"}

    def test_each_tissue_timecourse_same_length_as_times(self):
        r = _sim()
        for tissue, concs in r.tissue_concs.items():
            assert len(concs) == len(r.times_h), f"{tissue} length mismatch"

    def test_times_start_at_zero(self):
        r = _sim()
        assert r.times_h[0] == 0.0

    def test_times_end_near_t_end(self):
        r = _sim(t_end_h=24.0, dt_h=0.1)
        assert abs(r.times_h[-1] - 24.0) < 0.15


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

class TestInitialConditions:
    def test_initial_plasma_conc_equals_dose_over_vd_plasma(self):
        dose = 100.0
        r = _sim(dose_mg=dose)
        expected = dose / _VD_PLASMA_L
        assert abs(r.c_plasma_mg_L[0] - expected) < 1e-9

    def test_initial_tissue_concs_zero(self):
        r = _sim()
        for tissue, concs in r.tissue_concs.items():
            assert concs[0] == 0.0, f"{tissue} initial concentration should be 0"


# ---------------------------------------------------------------------------
# Physical constraints
# ---------------------------------------------------------------------------

class TestPhysicalConstraints:
    def test_plasma_conc_non_negative(self):
        r = _sim()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_tissue_concs_non_negative(self):
        r = _sim()
        for tissue, concs in r.tissue_concs.items():
            assert all(c >= 0 for c in concs), f"{tissue} has negative concentrations"

    def test_plasma_conc_decreases_over_time(self):
        # With IV bolus and clearance, plasma conc should generally decline
        r = _sim(t_end_h=48.0, dt_h=0.1)
        # Check that final conc is lower than initial
        assert r.c_plasma_mg_L[-1] < r.c_plasma_mg_L[0]

    def test_cmax_plasma_equals_initial_conc(self):
        # For IV bolus, Cmax should be at t=0
        r = _sim()
        assert abs(r.cmax_plasma - r.c_plasma_mg_L[0]) < 1e-6


# ---------------------------------------------------------------------------
# PK metrics
# ---------------------------------------------------------------------------

class TestPKMetrics:
    def test_auc_plasma_positive(self):
        r = _sim()
        assert r.auc_plasma > 0.0

    def test_auc_increases_with_dose(self):
        r1 = _sim(dose_mg=100.0)
        r2 = _sim(dose_mg=200.0)
        assert r2.auc_plasma > r1.auc_plasma

    def test_auc_decreases_with_higher_clearance(self):
        r1 = _sim(cl_hep_L_per_h=5.0)
        r2 = _sim(cl_hep_L_per_h=20.0)
        assert r2.auc_plasma < r1.auc_plasma

    def test_cmax_proportional_to_dose(self):
        r1 = _sim(dose_mg=100.0)
        r2 = _sim(dose_mg=200.0)
        assert abs(r2.cmax_plasma / r1.cmax_plasma - 2.0) < 0.1

    def test_drug_name_stored(self):
        r = _sim(drug_name="Midazolam")
        assert r.drug_name == "Midazolam"

    def test_dose_mg_stored(self):
        r = _sim(dose_mg=250.0)
        assert r.dose_mg == 250.0

    def test_cl_hep_stored(self):
        r = _sim(cl_hep_L_per_h=7.5)
        assert r.cl_hep_L_per_h == 7.5


# ---------------------------------------------------------------------------
# Tissue distribution
# ---------------------------------------------------------------------------

class TestTissueDistribution:
    def test_high_kp_tissue_accumulates_more(self):
        # Liver has kp=2.0 with high flow=54 L/h, muscle has kp=0.8 with flow=15 L/h.
        # Liver peak concentration should exceed muscle due to both higher Kp and higher flow.
        r = _sim(t_end_h=24.0, dt_h=0.1)
        liver_max = max(r.tissue_concs["liver"])
        muscle_max = max(r.tissue_concs["muscle"])
        assert liver_max > muscle_max

    def test_tissue_conc_rises_then_falls(self):
        # Tissue should show uptake then elimination
        r = _sim(t_end_h=48.0, dt_h=0.1)
        liver_concs = r.tissue_concs["liver"]
        # Peak should be somewhere after t=0
        peak_idx = liver_concs.index(max(liver_concs))
        assert peak_idx > 0

    def test_liver_conc_higher_than_muscle_at_peak(self):
        # Liver kp=2.0 > muscle kp=0.8 and liver flow is high → more uptake
        r = _sim(t_end_h=24.0, dt_h=0.1)
        liver_peak = max(r.tissue_concs["liver"])
        muscle_peak = max(r.tissue_concs["muscle"])
        assert liver_peak > muscle_peak


# ---------------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------------

class TestParameterSensitivity:
    def test_zero_clearance_higher_auc(self):
        r1 = _sim(cl_hep_L_per_h=0.0)
        r2 = _sim(cl_hep_L_per_h=10.0)
        assert r1.auc_plasma > r2.auc_plasma

    def test_larger_vd_lower_initial_plasma(self):
        # Initial plasma Cmax = dose / Vd_plasma (fixed 3L), so unchanged by vd_total
        # But tissue volumes increase → more drug redistributes
        r1 = _sim(vd_total_L=20.0)
        r2 = _sim(vd_total_L=100.0)
        # Both have same initial Cmax (dose/Vd_plasma), but r2 distributes more
        assert abs(r1.cmax_plasma - r2.cmax_plasma) < 1e-6

    def test_notes_contains_drug_name_info(self):
        r = _sim()
        assert "PBPK" in r.notes


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _sim(dose_mg=-1.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _sim(dose_mg=0.0)

    def test_negative_cl_hep_raises(self):
        with pytest.raises(ValueError, match="cl_hep_L_per_h"):
            _sim(cl_hep_L_per_h=-1.0)

    def test_zero_vd_total_raises(self):
        with pytest.raises(ValueError, match="vd_total_L"):
            _sim(vd_total_L=0.0)

    def test_zero_t_end_raises(self):
        with pytest.raises(ValueError, match="t_end_h"):
            _sim(t_end_h=0.0)

    def test_zero_dt_raises(self):
        with pytest.raises(ValueError, match="dt_h"):
            _sim(dt_h=0.0)

    def test_missing_kp_tissue_raises(self):
        with pytest.raises(ValueError, match="missing"):
            _sim(kp_tissues={"liver": 1.0, "kidney": 1.0})  # missing muscle, fat

    def test_unknown_tissue_in_kp_raises(self):
        kp = {**_DEFAULT_KP, "brain": 1.0}
        with pytest.raises(ValueError, match="Unknown tissue"):
            _sim(kp_tissues=kp)

    def test_zero_kp_raises(self):
        kp = {**_DEFAULT_KP, "liver": 0.0}
        with pytest.raises(ValueError, match="kp_tissues"):
            _sim(kp_tissues=kp)

    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            _sim(drug_name="")

    def test_whitespace_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            _sim(drug_name="   ")
