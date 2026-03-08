"""Tests for sublingual rapid absorption model (phase 647)."""

import pytest

from omega_pbpk.core.sublingual_rapid_p647 import (
    SublingualRapidResult,
    simulate_sublingual_rapid,
)

# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    drug_name="nitroglycerin",
    dose_mg=0.4,
    logP=1.62,
    cl_sys_L_per_h=50.0,
    vd_sys_L=200.0,
)


def _run(**overrides):
    kw = {**DEFAULTS, **overrides}
    return simulate_sublingual_rapid(**kw)


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------
class TestStructure:
    def test_return_type(self):
        r = _run()
        assert isinstance(r, SublingualRapidResult)

    def test_list_lengths_match(self):
        r = _run()
        assert len(r.times_h) == len(r.c_plasma_mg_L) == len(r.c_sublingual_mg)

    def test_list_lengths_expected(self):
        r = _run(t_end_h=2.0, dt_h=0.01)
        assert len(r.times_h) == int(2.0 / 0.01) + 1

    def test_non_negative_concentrations(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)
        assert all(c >= 0 for c in r.c_sublingual_mg)

    def test_drug_name_stored(self):
        r = _run(drug_name="fentanyl")
        assert r.drug_name == "fentanyl"

    def test_dose_stored(self):
        r = _run()
        assert r.dose_mg == 0.4

    def test_logP_stored(self):
        r = _run()
        assert r.logP == 1.62


# ---------------------------------------------------------------------------
# Pharmacokinetic behaviour
# ---------------------------------------------------------------------------
class TestPKBehaviour:
    def test_higher_logP_faster_onset(self):
        r_low = _run(logP=0.5)
        r_high = _run(logP=4.0)
        assert r_high.t_onset_h <= r_low.t_onset_h

    def test_higher_logP_higher_ka(self):
        r_low = _run(logP=0.5)
        r_high = _run(logP=4.0)
        assert r_high.ka_sublingual_per_h > r_low.ka_sublingual_per_h

    def test_cmax_positive(self):
        r = _run()
        assert r.cmax_mg_L > 0

    def test_tmax_positive(self):
        r = _run()
        assert r.tmax_h > 0

    def test_onset_before_peak(self):
        r = _run()
        # onset should occur before or at tmax
        assert r.t_onset_h <= r.tmax_h

    def test_auc_positive(self):
        r = _run()
        assert r.auc_mg_h_per_L > 0

    def test_dose_proportionality_cmax(self):
        r1 = _run(dose_mg=0.4)
        r2 = _run(dose_mg=0.8)
        ratio = r2.cmax_mg_L / r1.cmax_mg_L
        assert abs(ratio - 2.0) < 0.1

    def test_dose_proportionality_auc(self):
        r1 = _run(dose_mg=0.4, t_end_h=8.0)
        r2 = _run(dose_mg=0.8, t_end_h=8.0)
        ratio = r2.auc_mg_h_per_L / r1.auc_mg_h_per_L
        assert abs(ratio - 2.0) < 0.15


# ---------------------------------------------------------------------------
# Bioavailability
# ---------------------------------------------------------------------------
class TestBioavailability:
    def test_in_range(self):
        r = _run()
        assert 0 <= r.bioavailability_pct <= 100

    def test_value_consistent_with_fsl(self):
        r = _run(logP=3.0)
        # f_sl = 0.6 + 3.0*0.05 = 0.75 → 75%
        assert abs(r.bioavailability_pct - 75.0) < 1.0


# ---------------------------------------------------------------------------
# Ka clamping
# ---------------------------------------------------------------------------
class TestKaClamping:
    def test_ka_lower_bound(self):
        r = _run(logP=-5.0)
        assert r.ka_sublingual_per_h >= 0.5

    def test_ka_upper_bound(self):
        r = _run(logP=20.0)
        assert r.ka_sublingual_per_h <= 15.0


# ---------------------------------------------------------------------------
# Therapeutic window
# ---------------------------------------------------------------------------
class TestTherapeuticWindow:
    def test_mec_above_cmax_onset_equals_tend(self):
        r = _run(mec_mg_L=1e6)
        assert r.t_onset_h == 4.0  # default t_end_h

    def test_time_in_window_non_negative(self):
        r = _run()
        assert r.time_in_therapeutic_window_h >= 0

    def test_mtc_reduces_window(self):
        r_no_mtc = _run(mec_mg_L=1e-6)
        r_with_mtc = _run(mec_mg_L=1e-6, mtc_mg_L=r_no_mtc.cmax_mg_L * 0.5)
        assert r_with_mtc.time_in_therapeutic_window_h <= r_no_mtc.time_in_therapeutic_window_h

    def test_t_offset_zero_when_never_above_mec(self):
        r = _run(mec_mg_L=1e6)
        assert r.t_offset_h == 0.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
class TestValidation:
    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-1)

    def test_zero_clearance(self):
        with pytest.raises(ValueError, match="cl_sys"):
            _run(cl_sys_L_per_h=0)

    def test_zero_vd(self):
        with pytest.raises(ValueError, match="vd_sys"):
            _run(vd_sys_L=0)

    def test_zero_mec(self):
        with pytest.raises(ValueError, match="mec"):
            _run(mec_mg_L=0)


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------
class TestNotes:
    def test_notes_present(self):
        r = _run()
        assert "ka_sl" in r.notes
