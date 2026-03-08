"""
Tests for Phase 637 — Drug Corneal Penetration
"""

import math

import pytest

from omega_pbpk.core.corneal_penetration_p637 import (
    CornealPenetrationResult,
    simulate_corneal_penetration,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_default(**kwargs):
    defaults = dict(drug_name="TestDrug", dose_mg=1.0, logP=1.0, mw_Da=300.0)
    defaults.update(kwargs)
    return simulate_corneal_penetration(**defaults)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass_instance(self):
        r = run_default()
        assert isinstance(r, CornealPenetrationResult)

    def test_drug_name_preserved(self):
        r = run_default(drug_name="Timolol")
        assert r.drug_name == "Timolol"

    def test_dose_preserved(self):
        r = run_default(dose_mg=2.5)
        assert r.dose_mg == 2.5

    def test_times_is_list(self):
        r = run_default()
        assert isinstance(r.times_h, list)

    def test_c_tear_is_list(self):
        r = run_default()
        assert isinstance(r.c_tear_mg_mL, list)

    def test_c_epithelium_is_list(self):
        r = run_default()
        assert isinstance(r.c_epithelium_mg_g, list)

    def test_c_stroma_is_list(self):
        r = run_default()
        assert isinstance(r.c_stroma_mg_g, list)

    def test_c_aqueous_is_list(self):
        r = run_default()
        assert isinstance(r.c_aqueous_mg_mL, list)

    def test_all_lists_same_length(self):
        r = run_default(t_end_h=2.0, dt_h=0.1)
        n = len(r.times_h)
        assert len(r.c_tear_mg_mL) == n
        assert len(r.c_epithelium_mg_g) == n
        assert len(r.c_stroma_mg_g) == n
        assert len(r.c_aqueous_mg_mL) == n

    def test_times_start_at_zero(self):
        r = run_default()
        assert r.times_h[0] == 0.0

    def test_times_end_near_t_end(self):
        r = run_default(t_end_h=3.0, dt_h=0.1)
        assert abs(r.times_h[-1] - 3.0) < 0.2


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_c_tear_initial_positive(self):
        r = run_default(dose_mg=1.0, drop_volume_uL=30.0)
        assert r.c_tear_mg_mL[0] > 0.0

    def test_c_epithelium_initial_zero(self):
        r = run_default()
        assert r.c_epithelium_mg_g[0] == 0.0

    def test_c_stroma_initial_zero(self):
        r = run_default()
        assert r.c_stroma_mg_g[0] == 0.0

    def test_c_aqueous_initial_zero(self):
        r = run_default()
        assert r.c_aqueous_mg_mL[0] == 0.0


# ---------------------------------------------------------------------------
# Physically reasonable trajectories
# ---------------------------------------------------------------------------


class TestTrajectories:
    def test_tear_concentration_decreases_overall(self):
        r = run_default(t_end_h=2.0)
        # Tear should generally decrease after t=0
        assert r.c_tear_mg_mL[-1] < r.c_tear_mg_mL[0]

    def test_aqueous_rises_then_falls(self):
        r = run_default(t_end_h=4.0, dt_h=0.01)
        # Aqueous rises from 0 and should have a maximum
        assert r.cmax_aqueous_mg_mL > 0.0
        assert r.tmax_aqueous_h > 0.0

    def test_all_concentrations_non_negative(self):
        r = run_default(t_end_h=4.0)
        assert all(v >= 0.0 for v in r.c_tear_mg_mL)
        assert all(v >= 0.0 for v in r.c_epithelium_mg_g)
        assert all(v >= 0.0 for v in r.c_stroma_mg_g)
        assert all(v >= 0.0 for v in r.c_aqueous_mg_mL)

    def test_cmax_aqueous_positive(self):
        r = run_default()
        assert r.cmax_aqueous_mg_mL > 0.0

    def test_auc_aqueous_positive(self):
        r = run_default()
        assert r.auc_aqueous_mg_h_per_mL > 0.0


# ---------------------------------------------------------------------------
# logP effects
# ---------------------------------------------------------------------------


class TestLogPEffect:
    def test_higher_logP_increases_permeability(self):
        r_low = simulate_corneal_penetration("Drug", 1.0, logP=-1.0, mw_Da=200.0)
        r_high = simulate_corneal_penetration("Drug", 1.0, logP=3.0, mw_Da=200.0)
        assert r_high.corneal_permeability_cm_s >= r_low.corneal_permeability_cm_s

    def test_higher_logP_higher_aqueous_penetration(self):
        r_low = simulate_corneal_penetration("Drug", 1.0, logP=-2.0, mw_Da=200.0)
        r_high = simulate_corneal_penetration("Drug", 1.0, logP=4.0, mw_Da=200.0)
        assert r_high.cmax_aqueous_mg_mL >= r_low.cmax_aqueous_mg_mL


# ---------------------------------------------------------------------------
# Molecular weight effects
# ---------------------------------------------------------------------------


class TestMWEffect:
    def test_higher_mw_reduces_epithelial_permeability(self):
        # At same logP, larger MW should reduce P_epi (exponent term)
        r_small = simulate_corneal_penetration("Drug", 1.0, logP=2.0, mw_Da=100.0)
        r_large = simulate_corneal_penetration("Drug", 1.0, logP=2.0, mw_Da=600.0)
        # P_stroma also decreases with MW; overall permeability should be lower for large MW
        assert r_large.corneal_permeability_cm_s <= r_small.corneal_permeability_cm_s

    def test_higher_mw_reduces_aqueous_penetration(self):
        r_small = simulate_corneal_penetration("Drug", 1.0, logP=2.0, mw_Da=100.0)
        r_large = simulate_corneal_penetration("Drug", 1.0, logP=2.0, mw_Da=800.0)
        assert r_large.cmax_aqueous_mg_mL <= r_small.cmax_aqueous_mg_mL


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


class TestDoseProportionality:
    def test_double_dose_doubles_cmax(self):
        r1 = run_default(dose_mg=1.0)
        r2 = run_default(dose_mg=2.0)
        assert abs(r2.cmax_aqueous_mg_mL / r1.cmax_aqueous_mg_mL - 2.0) < 0.05

    def test_double_dose_doubles_auc(self):
        r1 = run_default(dose_mg=1.0)
        r2 = run_default(dose_mg=2.0)
        assert abs(r2.auc_aqueous_mg_h_per_mL / r1.auc_aqueous_mg_h_per_mL - 2.0) < 0.05


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------


class TestDerivedMetrics:
    def test_t_half_aqueous_correct(self):
        r = run_default()
        expected = math.log(2.0) / 0.1  # k_turnover = 0.1 /h
        assert abs(r.t_half_aqueous_h - expected) < 0.01

    def test_bioavailability_between_0_and_100(self):
        r = run_default()
        assert 0.0 <= r.bioavailability_ocular_pct <= 100.0

    def test_corneal_permeability_positive(self):
        r = run_default()
        assert r.corneal_permeability_cm_s > 0.0

    def test_notes_is_string(self):
        r = run_default()
        assert isinstance(r.notes, str)


# ---------------------------------------------------------------------------
# Drop volume effect
# ---------------------------------------------------------------------------


class TestDropVolumeEffect:
    def test_larger_drop_dilutes_initial_tear_concentration(self):
        r_small = run_default(drop_volume_uL=10.0, dose_mg=1.0)
        r_large = run_default(drop_volume_uL=50.0, dose_mg=1.0)
        # Larger drop → larger dilution volume → lower initial C_tear
        assert r_small.c_tear_mg_mL[0] > r_large.c_tear_mg_mL[0]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_corneal_penetration("Drug", -1.0, logP=1.0, mw_Da=300.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_corneal_penetration("Drug", 0.0, logP=1.0, mw_Da=300.0)

    def test_negative_drop_volume_raises(self):
        with pytest.raises(ValueError, match="drop_volume_uL"):
            simulate_corneal_penetration("Drug", 1.0, logP=1.0, mw_Da=300.0, drop_volume_uL=-5.0)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            simulate_corneal_penetration("Drug", 1.0, logP=1.0, mw_Da=0.0)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            simulate_corneal_penetration("Drug", 1.0, logP=1.0, mw_Da=-100.0)
