"""Phase 12 — Parent-metabolite PBPK tests.

Tests cover:
- MetaboliteSpec dataclass
- simulate_metabolite: basic functionality, mass balance, AUC relationships
- simulate_all_metabolites: multi-metabolite handling
- MetaboliteResult.pk: Cmax, Tmax, AUC, t½
- Physics checks: metabolite appears after parent, declines when parent clears
- Active metabolite reaches Cmax later than parent
"""

import math
import pytest
import numpy as np

from omega_pbpk.core.metabolite import (
    MetaboliteResult,
    MetaboliteSpec,
    simulate_all_metabolites,
    simulate_metabolite,
)
from omega_pbpk.core.body import WholeBodyPBPK, IDX_MET_HEPATIC
from omega_pbpk.drugs.drug import Drug


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_parent_iv(dose=100.0, clint=8.0, clr=0.5):
    """Run parent IV simulation and return (result, parent_time, clh_rate)."""
    drug = Drug(
        name="Parent",
        mw=400.0,
        logP=2.0,
        fup=0.3,
        rbp=1.0,
        clint_hepatic_L_per_h=clint,
        clr_L_per_h=clr,
    )
    model = WholeBodyPBPK(drug, body_weight=70.0)
    model.setup_iv(dose_mg=dose)
    result = model.simulate(t_end_h=24.0)

    # Compute instantaneous CLh rate from cumulative MET_HEPATIC state
    met_hepatic = result.amounts[:, IDX_MET_HEPATIC]
    t = result.time_h
    dt = np.diff(t, prepend=t[0] - (t[1] - t[0]))
    dt[dt <= 0] = 0.1
    clh_rate = np.diff(met_hepatic, prepend=0.0) / np.maximum(dt, 1e-12)
    clh_rate = np.maximum(clh_rate, 0.0)

    return result, t, clh_rate


# ---------------------------------------------------------------------------
# MetaboliteSpec
# ---------------------------------------------------------------------------


class TestMetaboliteSpec:
    def test_default_creation(self):
        met = MetaboliteSpec(name="M1")
        assert met.name == "M1"
        assert met.fm_parent == 1.0
        assert met.clint_met_L_per_h == 0.0
        assert not met.is_active

    def test_active_toxic_flags(self):
        met = MetaboliteSpec(name="ActiveTox", is_active=True, is_toxic=True)
        assert met.is_active
        assert met.is_toxic

    def test_fm_parent_is_float(self):
        met = MetaboliteSpec(name="M2", fm_parent=0.4)
        assert met.fm_parent == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# simulate_metabolite basic tests
# ---------------------------------------------------------------------------


class TestSimulateMetabolite:
    def setup_method(self):
        """Run parent simulation once and reuse for multiple tests."""
        self.result, self.t, self.clh_rate = _run_parent_iv(dose=100.0)
        self.met_spec = MetaboliteSpec(
            name="M1",
            fm_parent=0.7,
            fup=0.4,
            rbp=1.0,
            clint_met_L_per_h=2.0,
            clr_met_L_per_h=1.0,
            kp_liver=1.5,
            kp_rest=0.8,
            vd_L_per_kg=0.8,
        )
        self.met_result = simulate_metabolite(
            self.met_spec, self.t, self.clh_rate, body_weight=70.0
        )

    def test_returns_metabolite_result(self):
        assert isinstance(self.met_result, MetaboliteResult)

    def test_correct_name(self):
        assert self.met_result.name == "M1"

    def test_time_array_preserved(self):
        assert len(self.met_result.time_h) == len(self.t)

    def test_amounts_shape(self):
        # 3 compartments: blood, liver, rest
        assert self.met_result.amounts_mg.shape[1] == 3

    def test_plasma_conc_non_negative(self):
        assert np.all(self.met_result.plasma_conc_mg_L >= 0)

    def test_plasma_conc_rises_then_falls(self):
        cp = self.met_result.plasma_conc_mg_L
        tmax_idx = np.argmax(cp)
        # Should have clear peak in middle of simulation
        assert 0 < tmax_idx < len(cp) - 1

    def test_auc_positive(self):
        assert self.met_result.pk["AUC_mg_h_L"] > 0

    def test_metabolite_cmax_reasonable(self):
        """Metabolite Cmax should be > 0 and < dose (in reasonable units)."""
        cmax = self.met_result.pk["Cmax_mg_L"]
        assert 0 < cmax < 50.0  # reasonable for 100 mg dose

    def test_metabolite_tmax_after_time_zero(self):
        assert self.met_result.pk["Tmax_h"] > 0.0

    def test_spec_preserved(self):
        assert self.met_result.spec is self.met_spec


class TestMetaboliteResultPK:
    def test_pk_has_all_keys(self):
        _, t, clh_rate = _run_parent_iv(dose=100.0)
        met = MetaboliteSpec(name="M1", fm_parent=1.0, fup=0.5, rbp=1.0, kp_rest=1.0)
        result = simulate_metabolite(met, t, clh_rate)
        for key in ("Cmax_mg_L", "Tmax_h", "AUC_mg_h_L", "half_life_h"):
            assert key in result.pk

    def test_half_life_positive(self):
        """Half-life should be finite for a metabolite with significant elimination.

        Use a 48h simulation to ensure enough terminal phase data for regression.
        """
        drug = Drug(
            name="Parent",
            mw=400.0,
            logP=2.0,
            fup=0.3,
            rbp=1.0,
            clint_hepatic_L_per_h=8.0,
            clr_L_per_h=0.5,
        )
        model = WholeBodyPBPK(drug)
        model.setup_iv(100.0)
        result = model.simulate(t_end_h=48.0)
        t = result.time_h
        met_hepatic = result.amounts[:, IDX_MET_HEPATIC]
        dt = np.diff(t, prepend=t[0] - (t[1] - t[0]))
        dt[dt <= 0] = 0.1
        clh_rate = np.maximum(np.diff(met_hepatic, prepend=0.0) / np.maximum(dt, 1e-12), 0.0)

        met = MetaboliteSpec(
            name="M1", fm_parent=1.0, fup=0.5, rbp=1.0, clint_met_L_per_h=3.0, kp_rest=1.0
        )
        met_result = simulate_metabolite(met, t, clh_rate)
        hl = met_result.pk["half_life_h"]
        assert math.isfinite(hl)
        assert hl > 0


# ---------------------------------------------------------------------------
# Physics checks
# ---------------------------------------------------------------------------


class TestMetabolitePhysics:
    def test_higher_fm_gives_higher_metabolite_auc(self):
        _, t, clh_rate = _run_parent_iv(dose=100.0)
        base_kwargs = dict(fup=0.5, rbp=1.0, kp_liver=1.0, kp_rest=1.0)

        m_low = MetaboliteSpec(name="Low", fm_parent=0.2, **base_kwargs)
        m_high = MetaboliteSpec(name="High", fm_parent=0.8, **base_kwargs)

        r_low = simulate_metabolite(m_low, t, clh_rate)
        r_high = simulate_metabolite(m_high, t, clh_rate)

        assert r_high.pk["AUC_mg_h_L"] > r_low.pk["AUC_mg_h_L"]

    def test_higher_met_clint_gives_lower_met_auc(self):
        _, t, clh_rate = _run_parent_iv(dose=100.0)
        base_kwargs = dict(fm_parent=0.8, fup=0.5, rbp=1.0, kp_liver=1.0, kp_rest=1.0)

        m_slow = MetaboliteSpec(name="Slow", clint_met_L_per_h=0.5, **base_kwargs)
        m_fast = MetaboliteSpec(name="Fast", clint_met_L_per_h=20.0, **base_kwargs)

        r_slow = simulate_metabolite(m_slow, t, clh_rate)
        r_fast = simulate_metabolite(m_fast, t, clh_rate)

        assert r_slow.pk["AUC_mg_h_L"] > r_fast.pk["AUC_mg_h_L"]

    def test_metabolite_cmax_later_than_parent_tmax(self):
        """Metabolite plasma Cmax should occur after or at parent Tmax for IV."""
        parent_result, t, clh_rate = _run_parent_iv(dose=100.0, clint=3.0)
        parent_cp = parent_result.plasma_concentration()
        parent_tmax = t[np.argmax(parent_cp)]

        met = MetaboliteSpec(name="M1", fm_parent=0.9, fup=0.4, rbp=1.0, kp_liver=1.0, kp_rest=0.8)
        met_result = simulate_metabolite(met, t, clh_rate)
        met_tmax = met_result.pk["Tmax_h"]

        # For IV, parent starts declining immediately; metabolite peaks later
        assert met_tmax >= parent_tmax

    def test_all_amounts_non_negative(self):
        _, t, clh_rate = _run_parent_iv(dose=100.0)
        met = MetaboliteSpec(name="M1", fm_parent=0.8, fup=0.3, rbp=1.0, kp_liver=1.5, kp_rest=1.0)
        result = simulate_metabolite(met, t, clh_rate)
        assert np.all(result.amounts_mg >= 0)

    def test_fm_zero_gives_no_metabolite(self):
        """fm_parent = 0 → no metabolite formed → concentrations ≈ 0."""
        _, t, clh_rate = _run_parent_iv(dose=100.0)
        met = MetaboliteSpec(name="Null", fm_parent=0.0, fup=0.5, rbp=1.0)
        result = simulate_metabolite(met, t, clh_rate)
        assert np.max(result.plasma_conc_mg_L) < 1e-6


# ---------------------------------------------------------------------------
# simulate_all_metabolites
# ---------------------------------------------------------------------------


class TestSimulateAllMetabolites:
    def test_returns_list_of_results(self):
        parent_result, t, _ = _run_parent_iv(dose=100.0)
        met_hepatic = parent_result.amounts[:, IDX_MET_HEPATIC]
        metabolites = [
            MetaboliteSpec(name="M1", fm_parent=0.6, fup=0.4, rbp=1.0),
            MetaboliteSpec(name="M2", fm_parent=0.3, fup=0.2, rbp=0.8),
        ]
        results = simulate_all_metabolites(metabolites, t, met_hepatic)
        assert len(results) == 2

    def test_metabolite_names_preserved(self):
        parent_result, t, _ = _run_parent_iv(dose=100.0)
        met_hepatic = parent_result.amounts[:, IDX_MET_HEPATIC]
        metabolites = [
            MetaboliteSpec(name="Alpha", fm_parent=0.5, fup=0.5, rbp=1.0),
            MetaboliteSpec(name="Beta", fm_parent=0.5, fup=0.5, rbp=1.0),
        ]
        results = simulate_all_metabolites(metabolites, t, met_hepatic)
        names = [r.name for r in results]
        assert names == ["Alpha", "Beta"]

    def test_individual_auc_proportional_to_fm(self):
        parent_result, t, _ = _run_parent_iv(dose=100.0)
        met_hepatic = parent_result.amounts[:, IDX_MET_HEPATIC]
        m1 = MetaboliteSpec(name="M1", fm_parent=0.2, fup=0.5, rbp=1.0, kp_rest=1.0)
        m2 = MetaboliteSpec(name="M2", fm_parent=0.6, fup=0.5, rbp=1.0, kp_rest=1.0)
        r1, r2 = simulate_all_metabolites([m1, m2], t, met_hepatic)
        # Same PK parameters, fm ratio 0.2:0.6 = 1:3 → AUC ratio should be ~1:3
        ratio = r2.pk["AUC_mg_h_L"] / max(r1.pk["AUC_mg_h_L"], 1e-12)
        assert ratio == pytest.approx(3.0, rel=0.15)

    def test_empty_list_returns_empty(self):
        parent_result, t, _ = _run_parent_iv(dose=100.0)
        met_hepatic = parent_result.amounts[:, IDX_MET_HEPATIC]
        results = simulate_all_metabolites([], t, met_hepatic)
        assert results == []
