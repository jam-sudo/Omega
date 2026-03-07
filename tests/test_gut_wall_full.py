"""Tests for core/gut_wall_full.py — Phase 353."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.core.gut_wall_full import (
    GutWallFullResult,
    gut_availability_sensitivity,
    simulate_gut_wall_full,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

BASE_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    peff_cm_per_s=2e-4,
    solubility_mg_mL=5.0,
    cl_hepatic_L_per_h=20.0,
    vd_L=50.0,
)


def _run(**override):
    kw = dict(BASE_KWARGS)
    kw.update(override)
    return simulate_gut_wall_full(**kw)


# ---------------------------------------------------------------------------
# Basic sanity checks
# ---------------------------------------------------------------------------


class TestBasicSanity:
    def test_returns_dataclass(self):
        r = _run()
        assert isinstance(r, GutWallFullResult)

    def test_frozen_dataclass(self):
        r = _run()
        with pytest.raises((AttributeError, TypeError)):
            r.fa_total = 0.9  # type: ignore[misc]

    def test_drug_name_preserved(self):
        r = _run(drug_name="Midazolam")
        assert r.drug_name == "Midazolam"

    def test_dose_preserved(self):
        r = _run(dose_mg=200.0)
        assert r.dose_mg == pytest.approx(200.0)

    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)


# ---------------------------------------------------------------------------
# Range constraints
# ---------------------------------------------------------------------------


class TestRanges:
    def test_fa_total_in_01(self):
        r = _run()
        assert 0.0 <= r.fa_total <= 1.0

    def test_fg_total_in_01(self):
        r = _run()
        assert 0.0 <= r.fg_total <= 1.0

    def test_fh_in_01(self):
        r = _run()
        assert 0.01 <= r.fh <= 1.0

    def test_f_oral_in_01(self):
        r = _run()
        assert 0.0 <= r.f_oral_total <= 1.0

    def test_f_oral_le_fa(self):
        """Extractions only reduce bioavailability."""
        r = _run()
        assert r.f_oral_total <= r.fa_total + 1e-9

    def test_segment_eg_in_01(self):
        r = _run()
        for eg in (r.eg_duodenum, r.eg_jejunum, r.eg_ileum, r.eg_colon):
            assert 0.0 <= eg <= 1.0

    def test_segment_fa_nonneg(self):
        r = _run()
        for fa in (r.fa_duodenum, r.fa_jejunum, r.fa_ileum, r.fa_colon):
            assert fa >= 0.0


# ---------------------------------------------------------------------------
# Permeability effect
# ---------------------------------------------------------------------------


class TestPermeabilityEffect:
    def test_high_peff_high_fa(self):
        high = _run(peff_cm_per_s=1e-3)
        low = _run(peff_cm_per_s=1e-5)
        assert high.fa_total > low.fa_total

    def test_very_high_peff_fa_near_1(self):
        r = _run(peff_cm_per_s=1e-2)
        assert r.fa_total > 0.9


# ---------------------------------------------------------------------------
# Solubility / dissolution effect
# ---------------------------------------------------------------------------


class TestDissolutionEffect:
    def test_low_solubility_reduces_fa(self):
        high_sol = _run(solubility_mg_mL=100.0)
        low_sol = _run(solubility_mg_mL=0.01)
        assert high_sol.fa_total > low_sol.fa_total

    def test_high_dose_number_note(self):
        r = _run(dose_mg=10000.0, solubility_mg_mL=0.001)
        assert "dissolution-limited" in r.notes


# ---------------------------------------------------------------------------
# CYP3A4 abundance effect
# ---------------------------------------------------------------------------


class TestCYPEffect:
    def test_high_cyp_lower_fg(self):
        high_cyp = _run(cyp3a4_relative_abundance=[5.0, 10.0, 5.0, 1.0])
        low_cyp = _run(cyp3a4_relative_abundance=[0.1, 0.1, 0.1, 0.01])
        assert high_cyp.fg_total < low_cyp.fg_total

    def test_zero_cyp_fg_near_1(self):
        r = _run(cyp3a4_relative_abundance=[0.0, 0.0, 0.0, 0.0])
        assert r.fg_total > 0.99

    def test_high_cyp_increases_metabolized(self):
        high = _run(cyp3a4_relative_abundance=[5.0, 10.0, 5.0, 1.0])
        low = _run(cyp3a4_relative_abundance=[0.1, 0.1, 0.1, 0.01])
        assert high.f_metabolized_gut > low.f_metabolized_gut


# ---------------------------------------------------------------------------
# P-gp efflux
# ---------------------------------------------------------------------------


class TestPgpEffect:
    def test_higher_pgp_reduces_fa(self):
        low_pgp = _run(pgp_efflux_ratio=1.0)
        high_pgp = _run(pgp_efflux_ratio=5.0)
        assert high_pgp.fa_total < low_pgp.fa_total

    def test_pgp_ratio_1_baseline(self):
        r = _run(pgp_efflux_ratio=1.0)
        assert r.fa_total > 0.0


# ---------------------------------------------------------------------------
# Mass balance relationships
# ---------------------------------------------------------------------------


class TestMassBalance:
    def test_f_metabolized_plus_portal_approx_fa(self):
        r = _run()
        assert r.f_metabolized_gut + r.f_parent_portal == pytest.approx(
            r.fa_total, abs=1e-9
        )

    def test_f_parent_portal_eq_fa_times_fg(self):
        r = _run()
        assert r.f_parent_portal == pytest.approx(r.fa_total * r.fg_total, rel=1e-6)

    def test_f_oral_eq_fa_fg_fh(self):
        r = _run()
        assert r.f_oral_total == pytest.approx(r.fa_total * r.fg_total * r.fh, rel=1e-6)


# ---------------------------------------------------------------------------
# Hepatic availability
# ---------------------------------------------------------------------------


class TestHepaticAvailability:
    def test_fh_minimum_clip(self):
        """CL >> blood flow → fh clipped to 0.01."""
        r = _run(cl_hepatic_L_per_h=1000.0, hepatic_blood_flow_L_per_h=90.0)
        assert r.fh == pytest.approx(0.01)

    def test_fh_maximum_clip(self):
        """Zero hepatic CL → fh = 1.0."""
        r = _run(cl_hepatic_L_per_h=0.0)
        assert r.fh == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-10.0)

    def test_zero_peff_raises(self):
        with pytest.raises(ValueError, match="peff"):
            _run(peff_cm_per_s=0.0)

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError, match="solubility"):
            _run(solubility_mg_mL=0.0)

    def test_wrong_cyp_length_raises(self):
        with pytest.raises(ValueError, match="4 elements"):
            _run(cyp3a4_relative_abundance=[1.0, 2.0])

    def test_negative_cyp_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _run(cyp3a4_relative_abundance=[1.0, -1.0, 1.0, 0.1])

    def test_pgp_less_than_1_raises(self):
        with pytest.raises(ValueError, match="pgp_efflux_ratio"):
            _run(pgp_efflux_ratio=0.5)


# ---------------------------------------------------------------------------
# Gut wall CL
# ---------------------------------------------------------------------------


class TestGutWallCL:
    def test_cl_gut_wall_nonneg(self):
        r = _run()
        assert r.cl_gut_wall_L_per_h >= 0.0

    def test_cl_gut_wall_increases_with_cyp(self):
        high = _run(cyp3a4_relative_abundance=[5.0, 10.0, 5.0, 1.0])
        low = _run(cyp3a4_relative_abundance=[0.1, 0.1, 0.1, 0.01])
        assert high.cl_gut_wall_L_per_h > low.cl_gut_wall_L_per_h


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


class TestSensitivity:
    def test_returns_correct_length(self):
        results = gut_availability_sensitivity(
            **BASE_KWARGS, cyp3a4_fold_changes=[0.5, 1.0, 2.0, 5.0]
        )
        assert len(results) == 4

    def test_sorted_descending_fg(self):
        results = gut_availability_sensitivity(**BASE_KWARGS)
        fg_values = [r.fg_total for r in results]
        assert fg_values == sorted(fg_values, reverse=True)

    def test_default_fold_changes_length(self):
        results = gut_availability_sensitivity(**BASE_KWARGS)
        assert len(results) == 4  # default [0.5, 1.0, 2.0, 5.0]

    def test_high_fold_change_lowest_fg(self):
        results = gut_availability_sensitivity(
            **BASE_KWARGS, cyp3a4_fold_changes=[0.1, 5.0]
        )
        # sorted descending: first has higher fg, last has lower
        assert results[0].fg_total >= results[-1].fg_total

    def test_all_results_are_dataclass(self):
        results = gut_availability_sensitivity(**BASE_KWARGS)
        assert all(isinstance(r, GutWallFullResult) for r in results)
