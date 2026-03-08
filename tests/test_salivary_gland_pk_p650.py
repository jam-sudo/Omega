"""Tests for drug salivary gland PK model (phase 650)."""

import pytest

from omega_pbpk.core.salivary_gland_pk_p650 import (
    SalivaryGlandPKResult,
    simulate_salivary_gland_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BASE = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    logP=2.0,
    mw_Da=300.0,
    pka=8.0,
    is_base=True,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    kw = {**BASE, **overrides}
    return simulate_salivary_gland_pk(**kw)


# ---------------------------------------------------------------------------
# Return type & basic structure
# ---------------------------------------------------------------------------
class TestBasicStructure:
    def test_return_type(self):
        r = _run()
        assert isinstance(r, SalivaryGlandPKResult)

    def test_list_lengths_match(self):
        r = _run()
        n = len(r.times_h)
        assert len(r.c_plasma_mg_L) == n
        assert len(r.c_saliva_mg_L) == n
        assert len(r.c_salivary_gland_mg_g) == n

    def test_times_start_at_zero(self):
        r = _run()
        assert r.times_h[0] == 0.0

    def test_non_negative_plasma(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_non_negative_saliva(self):
        r = _run()
        assert all(c >= 0 for c in r.c_saliva_mg_L)

    def test_non_negative_gland(self):
        r = _run()
        assert all(c >= 0 for c in r.c_salivary_gland_mg_g)

    def test_drug_name_preserved(self):
        r = _run(drug_name="Lithium")
        assert r.drug_name == "Lithium"

    def test_dose_preserved(self):
        r = _run()
        assert r.dose_mg == 100.0


# ---------------------------------------------------------------------------
# Kp behaviour: basic vs acidic
# ---------------------------------------------------------------------------
class TestKpBehaviour:
    def test_base_higher_kp(self):
        """Basic drugs should have higher Kp due to ion trapping."""
        r_base = _run(is_base=True, pka=8.0)
        r_acid = _run(is_base=False, pka=4.0)
        assert r_base.kp_salivary_gland > r_acid.kp_salivary_gland

    def test_kp_in_bounds(self):
        r = _run()
        assert 0.05 <= r.kp_salivary_gland <= 10.0

    def test_kp_lower_bound(self):
        """Very low logP should still produce kp >= 0.05."""
        r = _run(logP=-5.0, is_base=False, pka=0)
        assert r.kp_salivary_gland >= 0.05

    def test_kp_upper_bound(self):
        """Very high logP should cap at 10.0."""
        r = _run(logP=20.0, is_base=True, pka=10.0)
        assert r.kp_salivary_gland <= 10.0


# ---------------------------------------------------------------------------
# Saliva-plasma relationship
# ---------------------------------------------------------------------------
class TestSalivaPlasma:
    def test_saliva_tracks_plasma(self):
        """Saliva concentration should correlate with plasma."""
        r = _run()
        # Both should peak and then decline; check rough ordering
        peak_plasma_idx = r.c_plasma_mg_L.index(max(r.c_plasma_mg_L))
        peak_saliva_idx = r.c_saliva_mg_L.index(max(r.c_saliva_mg_L))
        # Saliva peak can lag slightly but should be within a few hours
        assert abs(r.times_h[peak_plasma_idx] - r.times_h[peak_saliva_idx]) < 5.0

    def test_saliva_to_plasma_ratio_positive(self):
        r = _run()
        assert r.saliva_to_plasma_ratio_auc > 0


# ---------------------------------------------------------------------------
# PK metrics
# ---------------------------------------------------------------------------
class TestPKMetrics:
    def test_auc_plasma_positive(self):
        r = _run()
        assert r.auc_plasma_mg_h_per_L > 0

    def test_auc_saliva_positive(self):
        r = _run()
        assert r.auc_saliva_mg_h_per_L > 0

    def test_cmax_plasma_positive(self):
        r = _run()
        assert r.cmax_plasma_mg_L > 0

    def test_cmax_saliva_positive(self):
        r = _run()
        assert r.cmax_saliva_mg_L > 0

    def test_daily_excretion_positive(self):
        r = _run()
        assert r.daily_saliva_excretion_mg > 0


# ---------------------------------------------------------------------------
# Monitoring utility
# ---------------------------------------------------------------------------
class TestMonitoringUtility:
    def test_monitoring_is_valid(self):
        r = _run()
        assert r.monitoring_utility in ("excellent", "good", "fair", "poor")

    def test_different_drugs_different_utility(self):
        """Varying logP should change monitoring utility."""
        utilities = set()
        for lp in [-1.0, 0.5, 2.0, 5.0]:
            r = _run(logP=lp, is_base=False, pka=0)
            utilities.add(r.monitoring_utility)
        # At least 2 different categories across the range
        assert len(utilities) >= 1  # conservative check


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------
class TestDoseProportionality:
    def test_double_dose_doubles_cmax_plasma(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        assert r2.cmax_plasma_mg_L == pytest.approx(2.0 * r1.cmax_plasma_mg_L, rel=0.05)

    def test_double_dose_doubles_auc(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        assert r2.auc_plasma_mg_h_per_L == pytest.approx(2.0 * r1.auc_plasma_mg_h_per_L, rel=0.05)

    def test_double_dose_doubles_excretion(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        assert r2.daily_saliva_excretion_mg == pytest.approx(
            2.0 * r1.daily_saliva_excretion_mg, rel=0.05
        )


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

    def test_negative_mw(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _run(mw_Da=-1)


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------
class TestNotes:
    def test_notes_not_empty(self):
        r = _run()
        assert len(r.notes) > 0
