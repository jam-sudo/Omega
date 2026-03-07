"""Tests for biofilm PK model (Phase 217)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.biofilm_pk import (
    BiofilmPKResult,
    assess_biofilm_eradication,
    simulate_biofilm_pk,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _default_iv(**kwargs):
    """Return a standard IV simulation result with optional overrides."""
    params = dict(
        drug_name="ciprofloxacin",
        dose_mg=500.0,
        cl_L_per_h=15.0,
        vd_L=200.0,
        k_pen_per_h=0.05,
        mic_mg_L=0.5,
        route="iv",
        kill_rate_per_h=0.5,
        kg_bacteria_per_h=0.05,
        t_end_h=24.0,
        dt_h=0.1,
    )
    params.update(kwargs)
    return simulate_biofilm_pk(**params)


# ── Input validation ──────────────────────────────────────────────────────────


class TestInputValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_iv(dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_iv(dose_mg=-100.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            _default_iv(cl_L_per_h=0.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            _default_iv(vd_L=0.0)

    def test_zero_kpen_raises(self):
        with pytest.raises(ValueError, match="k_pen_per_h"):
            _default_iv(k_pen_per_h=0.0)

    def test_zero_mic_raises(self):
        with pytest.raises(ValueError, match="mic_mg_L"):
            _default_iv(mic_mg_L=0.0)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            _default_iv(route="subcutaneous")


# ── Result field types ────────────────────────────────────────────────────────


class TestResultTypes:
    def test_result_is_dataclass(self):
        r = _default_iv()
        assert isinstance(r, BiofilmPKResult)

    def test_drug_name_str(self):
        r = _default_iv()
        assert isinstance(r.drug_name, str)

    def test_dose_mg_float(self):
        r = _default_iv()
        assert isinstance(r.dose_mg, float)

    def test_lists_are_lists(self):
        r = _default_iv()
        assert isinstance(r.times_h, list)
        assert isinstance(r.c_plasma_mg_L, list)
        assert isinstance(r.c_biofilm_surface_mg_L, list)
        assert isinstance(r.c_biofilm_core_mg_L, list)
        assert isinstance(r.bacteria_norm, list)

    def test_scalar_fields_are_floats(self):
        r = _default_iv()
        assert isinstance(r.cmax_plasma, float)
        assert isinstance(r.cmax_biofilm_core, float)
        assert isinstance(r.t_above_mic_plasma_h, float)
        assert isinstance(r.t_above_mic_biofilm_h, float)
        assert isinstance(r.penetration_ratio, float)
        assert isinstance(r.log_kill_bacteria, float)

    def test_eradication_bool(self):
        r = _default_iv()
        assert isinstance(r.eradication, bool)

    def test_notes_str(self):
        r = _default_iv()
        assert isinstance(r.notes, str)


# ── Basic correctness ─────────────────────────────────────────────────────────


class TestBasicCorrectness:
    def test_bacteria_starts_at_one(self):
        r = _default_iv()
        assert abs(r.bacteria_norm[0] - 1.0) < 1e-9

    def test_plasma_concentration_positive(self):
        r = _default_iv()
        assert r.cmax_plasma > 0

    def test_high_dose_penetration_positive(self):
        """High dose should achieve positive biofilm core concentrations."""
        r = _default_iv(dose_mg=2000.0)
        assert r.cmax_biofilm_core > 0
        assert r.penetration_ratio > 0

    def test_penetration_ratio_nonnegative(self):
        r = _default_iv()
        assert r.penetration_ratio >= 0

    def test_log_kill_nonnegative(self):
        r = _default_iv()
        assert r.log_kill_bacteria >= 0

    def test_list_lengths_consistent(self):
        r = _default_iv(t_end_h=10.0, dt_h=0.1)
        n = len(r.times_h)
        assert len(r.c_plasma_mg_L) == n
        assert len(r.c_biofilm_surface_mg_L) == n
        assert len(r.c_biofilm_core_mg_L) == n
        assert len(r.bacteria_norm) == n


# ── Biofilm penetration mechanics ─────────────────────────────────────────────


class TestBiofilmPenetration:
    def test_low_kpen_results_in_lower_core_conc(self):
        """Slow penetration → much lower core conc than plasma."""
        r_slow = _default_iv(k_pen_per_h=0.001, dose_mg=1000.0)
        r_fast = _default_iv(k_pen_per_h=2.0, dose_mg=1000.0)
        # Slow penetration → lower core concentration
        assert r_slow.cmax_biofilm_core < r_fast.cmax_biofilm_core

    def test_penetration_ratio_less_than_one(self):
        """Biofilm core concentration typically lower than plasma."""
        r = _default_iv(k_pen_per_h=0.05)
        assert r.penetration_ratio < 1.0

    def test_biofilm_protected_compared_to_plasma(self):
        """Time above MIC in biofilm core <= time in plasma."""
        r = _default_iv(dose_mg=1000.0, k_pen_per_h=0.05, mic_mg_L=0.1)
        # Plasma should exceed MIC for at least as long as biofilm core
        assert r.t_above_mic_biofilm_h <= r.t_above_mic_plasma_h + 0.5  # small tolerance for dt


# ── Kill kinetics ─────────────────────────────────────────────────────────────


class TestKillKinetics:
    def test_high_dose_kills_bacteria(self):
        """Very high dose should reduce bacteria significantly."""
        r = _default_iv(
            dose_mg=10000.0,
            mic_mg_L=0.1,
            kill_rate_per_h=2.0,
            kg_bacteria_per_h=0.01,
            t_end_h=48.0,
            k_pen_per_h=0.5,
        )
        assert r.log_kill_bacteria > 0.5

    def test_eradication_very_high_dose(self):
        """Very high dose with effective kill rate should achieve eradication."""
        r = simulate_biofilm_pk(
            drug_name="tobramycin",
            dose_mg=50000.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
            k_pen_per_h=1.0,
            mic_mg_L=0.01,
            kill_rate_per_h=5.0,
            kg_bacteria_per_h=0.02,
            t_end_h=72.0,
            dt_h=0.05,
        )
        assert r.eradication is True

    def test_subtherapeutic_dose_no_eradication(self):
        """Dose far below MIC should not eradicate bacteria."""
        r = _default_iv(dose_mg=10.0, mic_mg_L=10.0)
        assert r.eradication is False


# ── Oral route ────────────────────────────────────────────────────────────────


class TestOralRoute:
    def test_oral_route_works(self):
        r = simulate_biofilm_pk(
            drug_name="amoxicillin",
            dose_mg=500.0,
            cl_L_per_h=10.0,
            vd_L=30.0,
            k_pen_per_h=0.05,
            mic_mg_L=1.0,
            route="oral",
            ka_per_h=1.5,
            t_end_h=12.0,
        )
        assert r.route == "oral"
        assert r.cmax_plasma > 0


# ── assess_biofilm_eradication ────────────────────────────────────────────────


class TestAssessBiofilmEradication:
    def test_returns_list(self):
        results = assess_biofilm_eradication(
            drug_name="ciprofloxacin",
            dose_mg=500.0,
            cl_L_per_h=15.0,
            vd_L=200.0,
            mic_mg_L=0.5,
            doses_mg=[100.0, 500.0, 1000.0],
            k_pen_per_h=0.1,
            t_end_h=24.0,
        )
        assert isinstance(results, list)
        assert len(results) == 3

    def test_sorted_by_log_kill_descending(self):
        results = assess_biofilm_eradication(
            drug_name="ciprofloxacin",
            dose_mg=500.0,
            cl_L_per_h=15.0,
            vd_L=200.0,
            mic_mg_L=0.5,
            doses_mg=[100.0, 500.0, 2000.0, 5000.0],
            k_pen_per_h=0.1,
            kill_rate_per_h=0.5,
            t_end_h=48.0,
        )
        log_kills = [r.log_kill_bacteria for r in results]
        assert log_kills == sorted(log_kills, reverse=True)

    def test_all_results_are_biofilm_pk_result(self):
        results = assess_biofilm_eradication(
            drug_name="test",
            dose_mg=100.0,
            cl_L_per_h=10.0,
            vd_L=50.0,
            mic_mg_L=1.0,
            doses_mg=[50.0, 200.0],
            k_pen_per_h=0.05,
        )
        for r in results:
            assert isinstance(r, BiofilmPKResult)
