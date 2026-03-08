"""Tests for Phase 867 — Cochlear Drug Distribution."""

import math
import pytest
from omega_pbpk.core.cochlear_pk_p867 import CochlearPKResult, simulate_cochlear_pk

# Default parameters
DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=10.0,
    logP=2.0,
    mw_Da=300.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    params = {**DEFAULTS, **overrides}
    return simulate_cochlear_pk(**params)


# --- Return type and structure ---

class TestReturnType:
    def test_returns_cochlear_pk_result(self):
        r = _run()
        assert isinstance(r, CochlearPKResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_dose_mg(self):
        r = _run()
        assert r.dose_mg == 10.0

    def test_route_default(self):
        r = _run()
        assert r.route == "systemic"

    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list) and len(r.times_h) > 1

    def test_plasma_list(self):
        r = _run()
        assert isinstance(r.c_plasma_mg_L, list)
        assert len(r.c_plasma_mg_L) == len(r.times_h)

    def test_perilymph_list(self):
        r = _run()
        assert isinstance(r.c_perilymph_mg_L, list)
        assert len(r.c_perilymph_mg_L) == len(r.times_h)

    def test_endolymph_list(self):
        r = _run()
        assert isinstance(r.c_endolymph_mg_L, list)
        assert len(r.c_endolymph_mg_L) == len(r.times_h)


# --- Systemic route behavior ---

class TestSystemicRoute:
    def test_plasma_higher_than_perilymph(self):
        r = _run(route="systemic")
        assert r.cmax_plasma_mg_L > r.cmax_perilymph_mg_L

    def test_perilymph_higher_than_endolymph(self):
        r = _run(route="systemic")
        assert r.cmax_perilymph_mg_L > r.cmax_endolymph_mg_L

    def test_endolymph_positive(self):
        r = _run(route="systemic")
        assert r.cmax_endolymph_mg_L > 0


# --- Intratympanic route behavior ---

class TestIntratympanicRoute:
    def test_perilymph_starts_high(self):
        r = _run(route="intratympanic")
        assert r.c_perilymph_mg_L[0] > 0

    def test_plasma_negligible(self):
        r = _run(route="intratympanic")
        # Plasma exposure should be very small relative to perilymph
        assert r.cmax_plasma_mg_L < r.cmax_perilymph_mg_L * 0.01

    def test_endolymph_gets_drug(self):
        r = _run(route="intratympanic")
        assert r.cmax_endolymph_mg_L > 0


# --- Non-negative concentrations ---

class TestNonNegative:
    def test_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_perilymph_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_perilymph_mg_L)

    def test_endolymph_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_endolymph_mg_L)


# --- BLB factor ---

class TestBLBFactor:
    def test_blb_in_range(self):
        r = _run()
        assert 0.01 <= r.blood_labyrinth_barrier_factor <= 0.8

    def test_blb_low_logP(self):
        r = _run(logP=-2.0)
        assert r.blood_labyrinth_barrier_factor == pytest.approx(0.01, abs=0.005)

    def test_blb_high_mw_reduces(self):
        r_low = _run(mw_Da=200.0)
        r_high = _run(mw_Da=1000.0)
        assert r_low.blood_labyrinth_barrier_factor > r_high.blood_labyrinth_barrier_factor


# --- Half-life ---

class TestHalfLife:
    def test_t_half_positive(self):
        r = _run()
        assert r.t_half_perilymph_h > 0

    def test_t_half_value(self):
        r = _run()
        expected = math.log(2) / 0.1
        assert r.t_half_perilymph_h == pytest.approx(expected, rel=0.01)


# --- Dose linearity ---

class TestDoseLinearity:
    def test_double_dose_doubles_cmax_plasma(self):
        r1 = _run(dose_mg=10.0)
        r2 = _run(dose_mg=20.0)
        assert r2.cmax_plasma_mg_L == pytest.approx(2 * r1.cmax_plasma_mg_L, rel=0.05)

    def test_double_dose_doubles_cmax_perilymph(self):
        r1 = _run(dose_mg=10.0)
        r2 = _run(dose_mg=20.0)
        assert r2.cmax_perilymph_mg_L == pytest.approx(2 * r1.cmax_perilymph_mg_L, rel=0.05)


# --- Validation errors ---

class TestValidation:
    def test_zero_dose(self):
        with pytest.raises(ValueError):
            _run(dose_mg=0)

    def test_negative_dose(self):
        with pytest.raises(ValueError):
            _run(dose_mg=-1)

    def test_bad_route(self):
        with pytest.raises(ValueError):
            _run(route="intravenous")

    def test_zero_cl(self):
        with pytest.raises(ValueError):
            _run(cl_sys_L_per_h=0)

    def test_zero_vd(self):
        with pytest.raises(ValueError):
            _run(vd_sys_L=0)


# --- Notes ---

class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str) and len(r.notes) > 0
