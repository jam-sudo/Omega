"""Tests for Phase 171 — Drug Efflux at Blood-Brain Barrier."""

import math

import pytest

from omega_pbpk.core.bbb_efflux import (
    BBBEffluxResult,
    compare_efflux_inhibition,
    simulate_bbb_efflux,
)


# ── Default parameters ──────────────────────────────────────────────

DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cl_L_per_h=5.0,
    vd_plasma_L=50.0,
)


def _run(**overrides):
    kw = {**DEFAULTS, **overrides}
    return simulate_bbb_efflux(**kw)


# ── Basic functionality ─────────────────────────────────────────────

class TestBasicSimulation:
    def test_returns_result(self):
        r = _run()
        assert isinstance(r, BBBEffluxResult)

    def test_drug_name_preserved(self):
        r = _run(drug_name="Donepezil")
        assert r.drug_name == "Donepezil"

    def test_dose_preserved(self):
        r = _run(dose_mg=200.0)
        assert r.dose_mg == 200.0

    def test_times_start_at_zero(self):
        r = _run()
        assert r.times_h[0] == 0.0

    def test_times_end_near_t_end(self):
        r = _run(t_end_h=12.0)
        assert r.times_h[-1] == pytest.approx(12.0, abs=0.02)

    def test_plasma_starts_positive(self):
        r = _run()
        assert r.c_plasma_mg_L[0] > 0

    def test_brain_starts_at_zero(self):
        r = _run()
        assert r.c_brain_mg_L[0] == 0.0

    def test_cmax_plasma_equals_initial(self):
        r = _run()
        assert r.cmax_plasma == pytest.approx(DEFAULTS["dose_mg"] / DEFAULTS["vd_plasma_L"], rel=0.01)

    def test_auc_plasma_positive(self):
        r = _run()
        assert r.auc_plasma > 0

    def test_auc_brain_positive(self):
        r = _run()
        assert r.auc_brain > 0


# ── P-gp effects ────────────────────────────────────────────────────

class TestPgpEffects:
    def test_kp_brain_less_than_1_with_pgp(self):
        r = _run(k_pgp_efflux_per_h=0.5, pgp_inhibition_fraction=0.0)
        assert r.kp_brain < 1.0

    def test_higher_inhibition_higher_brain_exposure(self):
        r_low = _run(pgp_inhibition_fraction=0.0)
        r_high = _run(pgp_inhibition_fraction=0.8)
        assert r_high.auc_brain > r_low.auc_brain

    def test_full_inhibition_maximum_brain_exposure(self):
        r_none = _run(pgp_inhibition_fraction=0.0)
        r_half = _run(pgp_inhibition_fraction=0.5)
        r_full = _run(pgp_inhibition_fraction=1.0)
        assert r_full.auc_brain > r_half.auc_brain > r_none.auc_brain

    def test_pgp_contribution_positive(self):
        r = _run(k_pgp_efflux_per_h=0.5)
        assert r.pgp_contribution > 0

    def test_pgp_contribution_zero_when_no_pgp(self):
        r = _run(k_pgp_efflux_per_h=0.0)
        assert r.pgp_contribution == pytest.approx(0.0, abs=1e-6)

    def test_inhibition_fraction_stored(self):
        r = _run(pgp_inhibition_fraction=0.3)
        assert r.pgp_inhibition_fraction == 0.3


# ── Linear PK ────────────────────────────────────────────────────────

class TestLinearPK:
    def test_double_dose_doubles_auc_plasma(self):
        r1 = _run(dose_mg=50.0)
        r2 = _run(dose_mg=100.0)
        assert r2.auc_plasma == pytest.approx(2.0 * r1.auc_plasma, rel=0.05)

    def test_double_dose_doubles_auc_brain(self):
        r1 = _run(dose_mg=50.0)
        r2 = _run(dose_mg=100.0)
        assert r2.auc_brain == pytest.approx(2.0 * r1.auc_brain, rel=0.05)

    def test_kp_brain_independent_of_dose(self):
        r1 = _run(dose_mg=50.0)
        r2 = _run(dose_mg=200.0)
        assert r1.kp_brain == pytest.approx(r2.kp_brain, rel=0.05)


# ── Validation ───────────────────────────────────────────────────────

class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-10)

    def test_zero_clearance_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            _run(cl_L_per_h=0)

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError, match="vd_plasma_L"):
            _run(vd_plasma_L=-1)

    def test_inhibition_out_of_range_raises(self):
        with pytest.raises(ValueError, match="pgp_inhibition_fraction"):
            _run(pgp_inhibition_fraction=1.5)


# ── compare_efflux_inhibition ────────────────────────────────────────

class TestCompare:
    def test_returns_correct_count(self):
        fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = compare_efflux_inhibition(
            drug_name="TestDrug", dose_mg=100, inhibition_fractions=fracs,
            cl_L_per_h=5.0, vd_plasma_L=50.0,
        )
        assert len(results) == 5

    def test_monotonically_increasing_brain_auc(self):
        fracs = [0.0, 0.5, 1.0]
        results = compare_efflux_inhibition(
            drug_name="TestDrug", dose_mg=100, inhibition_fractions=fracs,
            cl_L_per_h=5.0, vd_plasma_L=50.0,
        )
        aucs = [r.auc_brain for r in results]
        assert aucs[0] < aucs[1] < aucs[2]

    def test_empty_fractions_raises(self):
        with pytest.raises(ValueError, match="inhibition_fractions"):
            compare_efflux_inhibition(
                drug_name="TestDrug", dose_mg=100, inhibition_fractions=[],
                cl_L_per_h=5.0, vd_plasma_L=50.0,
            )


# ── Notes ─────────────────────────────────────────────────────────────

class TestNotes:
    def test_inhibition_note_present(self):
        r = _run(pgp_inhibition_fraction=0.5)
        assert any("inhibited" in n.lower() for n in r.notes)

    def test_no_inhibition_no_note(self):
        r = _run(pgp_inhibition_fraction=0.0)
        assert not any("inhibited" in n.lower() for n in r.notes)
