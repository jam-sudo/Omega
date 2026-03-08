"""Tests for biliary excretion and enterohepatic circulation (phase 649)."""

import pytest

from omega_pbpk.core.enterohepatic_p649 import (
    EnterohepatiCResult,
    simulate_enterohepatic,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BASE = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
    f_bile=0.3,
)


def _run(**overrides):
    kw = {**BASE, **overrides}
    return simulate_enterohepatic(**kw)


# ---------------------------------------------------------------------------
# Return type & basic structure
# ---------------------------------------------------------------------------
class TestBasicStructure:
    def test_return_type(self):
        r = _run()
        assert isinstance(r, EnterohepatiCResult)

    def test_list_lengths_match(self):
        r = _run()
        assert len(r.times_h) == len(r.c_plasma_mg_L)
        assert len(r.times_h) == len(r.c_gut_mg)
        assert len(r.times_h) == len(r.c_bile_mg)

    def test_times_start_at_zero(self):
        r = _run()
        assert r.times_h[0] == 0.0

    def test_non_negative_plasma(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_non_negative_gut(self):
        r = _run()
        assert all(c >= 0 for c in r.c_gut_mg)

    def test_non_negative_bile(self):
        r = _run()
        assert all(c >= 0 for c in r.c_bile_mg)

    def test_drug_name_preserved(self):
        r = _run(drug_name="Morphine")
        assert r.drug_name == "Morphine"

    def test_dose_preserved(self):
        r = _run()
        assert r.dose_mg == 100.0


# ---------------------------------------------------------------------------
# EHC behaviour
# ---------------------------------------------------------------------------
class TestEHCBehaviour:
    def test_no_bile_no_secondary_peaks(self):
        """f_bile=0 means no enterohepatic recycling → no secondary peaks."""
        r = _run(f_bile=0.0)
        assert r.n_secondary_peaks == 0

    def test_bile_gives_secondary_peaks(self):
        """f_bile=0.5 should produce at least one secondary peak."""
        r = _run(f_bile=0.5, t_end_h=48.0)
        assert r.n_secondary_peaks >= 1

    def test_higher_bile_more_excretion(self):
        r_low = _run(f_bile=0.1)
        r_high = _run(f_bile=0.5)
        assert r_high.bile_excretion_pct > r_low.bile_excretion_pct

    def test_ehc_fraction_matches_f_bile(self):
        r = _run(f_bile=0.4)
        assert r.ehc_fraction == pytest.approx(0.4)

    def test_bile_excretion_pct_equals_f_bile_times_100(self):
        r = _run(f_bile=0.35)
        assert r.bile_excretion_pct == pytest.approx(35.0)

    def test_gallbladder_emptying_bumps(self):
        """Periodic gallbladder emptying should cause concentration bumps."""
        r = _run(f_bile=0.5, t_gallbladder_h=6.0, t_end_h=48.0)
        # After each emptying event, plasma should rise for a bit
        # Check there are at least some local increases after initial decline
        increases = 0
        for i in range(1, len(r.c_plasma_mg_L)):
            if r.c_plasma_mg_L[i] > r.c_plasma_mg_L[i - 1] and r.times_h[i] > 5.0:
                increases += 1
        assert increases > 0


# ---------------------------------------------------------------------------
# Pharmacokinetic metrics
# ---------------------------------------------------------------------------
class TestPKMetrics:
    def test_auc_positive(self):
        r = _run()
        assert r.auc_mg_h_per_L > 0

    def test_cmax_positive(self):
        r = _run()
        assert r.cmax_mg_L > 0

    def test_tmax_positive(self):
        r = _run()
        assert r.tmax_h > 0

    def test_t_half_positive(self):
        r = _run()
        assert r.t_half_h > 0

    def test_t_half_value(self):
        """t_half should be 0.693 / k_el."""
        k_el = 5.0 / 50.0
        expected = 0.693 / k_el
        r = _run()
        assert r.t_half_h == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------
class TestDoseProportionality:
    def test_double_dose_doubles_cmax(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        assert r2.cmax_mg_L == pytest.approx(2.0 * r1.cmax_mg_L, rel=0.05)

    def test_double_dose_doubles_auc(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        assert r2.auc_mg_h_per_L == pytest.approx(2.0 * r1.auc_mg_h_per_L, rel=0.05)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
class TestValidation:
    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-1)

    def test_zero_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=0)

    def test_negative_clearance(self):
        with pytest.raises(ValueError, match="cl_sys"):
            _run(cl_sys_L_per_h=-1)

    def test_negative_vd(self):
        with pytest.raises(ValueError, match="vd_sys"):
            _run(vd_sys_L=-1)

    def test_f_bile_too_high(self):
        with pytest.raises(ValueError, match="f_bile"):
            _run(f_bile=1.0)

    def test_f_bile_negative(self):
        with pytest.raises(ValueError, match="f_bile"):
            _run(f_bile=-0.1)

    def test_gallbladder_interval_zero(self):
        with pytest.raises(ValueError, match="t_gallbladder"):
            _run(t_gallbladder_h=0)


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------
class TestNotes:
    def test_notes_not_empty(self):
        r = _run()
        assert len(r.notes) > 0
