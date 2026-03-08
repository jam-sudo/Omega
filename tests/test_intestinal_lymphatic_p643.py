"""Tests for Phase 643 — Intestinal Lymphatic Absorption."""

from __future__ import annotations

import pytest

from omega_pbpk.core.intestinal_lymphatic_p643 import (
    IntestinalLymphaticResult,
    simulate_intestinal_lymphatic,
)

# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------
_DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    logP=4.0,
    mw_Da=400.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    kw = {**_DEFAULTS, **overrides}
    return simulate_intestinal_lymphatic(**kw)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------
class TestReturnType:
    def test_returns_dataclass(self):
        r = _run()
        assert isinstance(r, IntestinalLymphaticResult)

    def test_drug_name(self):
        r = _run(drug_name="Alpha")
        assert r.drug_name == "Alpha"

    def test_dose_stored(self):
        r = _run(dose_mg=200.0)
        assert r.dose_mg == 200.0

    def test_logP_stored(self):
        r = _run(logP=5.0)
        assert r.logP == 5.0


# ---------------------------------------------------------------------------
# List lengths
# ---------------------------------------------------------------------------
class TestListLengths:
    def test_default_lengths(self):
        r = _run()
        n = len(r.times_h)
        assert n > 0
        assert len(r.c_plasma_mg_L) == n
        assert len(r.c_lymph_mg_L) == n
        assert len(r.c_portal_mg_L) == n

    def test_custom_t_end(self):
        r = _run(t_end_h=10.0, dt_h=0.5)
        assert len(r.times_h) == 21  # 0..10 step 0.5


# ---------------------------------------------------------------------------
# Non-negative concentrations
# ---------------------------------------------------------------------------
class TestNonNegative:
    def test_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0.0 for c in r.c_plasma_mg_L)

    def test_lymph_non_negative(self):
        r = _run()
        assert all(c >= 0.0 for c in r.c_lymph_mg_L)

    def test_portal_non_negative(self):
        r = _run()
        assert all(c >= 0.0 for c in r.c_portal_mg_L)


# ---------------------------------------------------------------------------
# LogP < 3 => lymphatic_fraction == 0
# ---------------------------------------------------------------------------
class TestLowLogP:
    def test_logP_below_3(self):
        r = _run(logP=2.0)
        assert r.lymphatic_fraction == 0.0

    def test_logP_at_3(self):
        r = _run(logP=3.0)
        assert r.lymphatic_fraction == 0.0

    def test_portal_fraction_is_one(self):
        r = _run(logP=2.0)
        assert r.portal_fraction == 1.0


# ---------------------------------------------------------------------------
# LogP > 5 => lymphatic_fraction > 0
# ---------------------------------------------------------------------------
class TestHighLogP:
    def test_logP_above_5(self):
        r = _run(logP=5.5)
        assert r.lymphatic_fraction > 0.0

    def test_logP_5(self):
        r = _run(logP=5.0)
        assert r.lymphatic_fraction > 0.0


# ---------------------------------------------------------------------------
# Lipid formulation effect
# ---------------------------------------------------------------------------
class TestLipidFormulation:
    def test_lipid_increases_fraction(self):
        r_no = _run(logP=4.0, lipid_formulation=False)
        r_yes = _run(logP=4.0, lipid_formulation=True)
        assert r_yes.lymphatic_fraction > r_no.lymphatic_fraction

    def test_lipid_no_effect_low_logP(self):
        r = _run(logP=2.0, lipid_formulation=True)
        assert r.lymphatic_fraction == 0.0


# ---------------------------------------------------------------------------
# Higher logP → higher lymph_to_portal_auc_ratio
# ---------------------------------------------------------------------------
class TestLymphPortalRatio:
    def test_higher_logP_higher_ratio(self):
        r1 = _run(logP=3.5)
        r2 = _run(logP=5.0)
        assert r2.lymph_to_portal_auc_ratio > r1.lymph_to_portal_auc_ratio

    def test_zero_ratio_low_logP(self):
        r = _run(logP=2.0)
        assert r.lymph_to_portal_auc_ratio == 0.0


# ---------------------------------------------------------------------------
# First pass avoidance
# ---------------------------------------------------------------------------
class TestFirstPassAvoidance:
    def test_range(self):
        r = _run(logP=5.0)
        assert 0.0 <= r.first_pass_avoidance_pct <= 100.0

    def test_zero_for_low_logP(self):
        r = _run(logP=2.0)
        assert r.first_pass_avoidance_pct == 0.0


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------
class TestDoseProportionality:
    def test_double_dose_doubles_cmax(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
        assert abs(ratio - 2.0) < 0.05

    def test_double_dose_doubles_auc(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.auc_plasma_mg_h_per_L / r1.auc_plasma_mg_h_per_L
        assert abs(ratio - 2.0) < 0.05


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
class TestValidation:
    def test_zero_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=0.0)

    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-10.0)

    def test_zero_clearance(self):
        with pytest.raises(ValueError, match="cl_sys_L_per_h"):
            _run(cl_sys_L_per_h=0.0)

    def test_zero_vd(self):
        with pytest.raises(ValueError, match="vd_sys_L"):
            _run(vd_sys_L=0.0)
