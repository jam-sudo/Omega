"""Tests for pulmonary drug disposition model (phase 654)."""

from __future__ import annotations

import dataclasses

import pytest

from omega_pbpk.core.pulmonary_disposition_p654 import (
    PulmonaryDispositionResult,
    simulate_pulmonary_disposition,
)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    route="iv",
    logP=2.0,
    mw_Da=400.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
    is_base=True,
    t_end_h=24.0,
    dt_h=0.1,
)


def _sim(**overrides):
    params = {**DEFAULTS, **overrides}
    return simulate_pulmonary_disposition(**params)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestResultType:
    def test_return_type(self):
        r = _sim()
        assert isinstance(r, PulmonaryDispositionResult)

    def test_is_dataclass(self):
        r = _sim()
        assert dataclasses.is_dataclass(r)

    def test_not_frozen(self):
        """Result has list fields so should NOT be frozen."""
        r = _sim()
        r.notes = "modified"  # should not raise


# ---------------------------------------------------------------------------
# List fields
# ---------------------------------------------------------------------------


class TestListFields:
    def test_list_lengths_consistent(self):
        r = _sim()
        n = len(r.times_h)
        assert len(r.c_plasma_mg_L) == n
        assert len(r.c_lung_tissue_mg_g) == n
        assert len(r.c_elf_mg_L) == n
        assert len(r.c_macrophage_mg_g) == n

    def test_list_non_empty(self):
        r = _sim()
        assert len(r.times_h) > 0

    def test_non_negative_concentrations(self):
        r = _sim()
        assert all(c >= 0 for c in r.c_plasma_mg_L)
        assert all(c >= 0 for c in r.c_lung_tissue_mg_g)
        assert all(c >= 0 for c in r.c_elf_mg_L)
        assert all(c >= 0 for c in r.c_macrophage_mg_g)


# ---------------------------------------------------------------------------
# Kp and partitioning
# ---------------------------------------------------------------------------


class TestPartitioning:
    def test_kp_lung_range(self):
        r = _sim()
        assert 0.5 <= r.kp_lung <= 20.0

    def test_basic_drug_higher_kp(self):
        r_base = _sim(is_base=True)
        r_acid = _sim(is_base=False)
        assert r_base.kp_lung > r_acid.kp_lung

    def test_kp_lung_low_logP_clamped(self):
        r = _sim(logP=-2.0, is_base=False)
        assert r.kp_lung >= 0.5

    def test_kp_lung_high_logP_clamped(self):
        r = _sim(logP=30.0, is_base=True)
        assert r.kp_lung <= 20.0


# ---------------------------------------------------------------------------
# Route effects
# ---------------------------------------------------------------------------


class TestRouteEffects:
    def test_inhaled_higher_lung_conc(self):
        """Inhaled route should produce higher initial lung concentrations."""
        r_oral = _sim(route="oral")
        r_inh = _sim(route="inhaled")
        assert r_inh.cmax_lung_mg_g > r_oral.cmax_lung_mg_g

    def test_iv_nonzero_plasma(self):
        r = _sim(route="iv")
        assert r.cmax_plasma_mg_L > 0

    def test_oral_absorption(self):
        r = _sim(route="oral")
        assert r.cmax_plasma_mg_L > 0

    def test_inhaled_plasma_nonzero(self):
        r = _sim(route="inhaled")
        assert r.cmax_plasma_mg_L > 0


# ---------------------------------------------------------------------------
# AUC and ratios
# ---------------------------------------------------------------------------


class TestAUCRatios:
    def test_lung_to_plasma_auc_ratio_positive(self):
        r = _sim()
        assert r.lung_to_plasma_auc_ratio > 0

    def test_elf_to_plasma_ratio_positive(self):
        r = _sim()
        assert r.elf_to_plasma_ratio > 0

    def test_macrophage_accumulation_fold_gte_one_for_base(self):
        r = _sim(is_base=True)
        assert r.macrophage_accumulation_fold >= 1.0

    def test_cmax_lung_positive(self):
        r = _sim()
        assert r.cmax_lung_mg_g > 0


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


class TestDoseProportionality:
    def test_dose_proportionality_iv(self):
        r1 = _sim(route="iv", dose_mg=100.0)
        r2 = _sim(route="iv", dose_mg=200.0)
        ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
        assert 1.9 < ratio < 2.1

    def test_dose_proportionality_oral(self):
        r1 = _sim(route="oral", dose_mg=100.0)
        r2 = _sim(route="oral", dose_mg=200.0)
        ratio = r2.auc_plasma_mg_h_per_L / r1.auc_plasma_mg_h_per_L
        assert 1.9 < ratio < 2.1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _sim(dose_mg=-10)

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError, match="cl_sys_L_per_h"):
            _sim(cl_sys_L_per_h=-1)

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError, match="vd_sys_L"):
            _sim(vd_sys_L=-1)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            _sim(route="topical")

    def test_route_case_sensitive(self):
        with pytest.raises(ValueError, match="route"):
            _sim(route="IV")
