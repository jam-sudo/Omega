"""Tests for multi-organ clearance model (Phase 218)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.clearance_organs import (
    ClearanceBreakdown,
    MultiOrganCLResult,
    organ_clearance,
    simulate_multiorgancl,
    total_clearance,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _default_sim(**kwargs):
    """Return a default multi-organ simulation with optional overrides."""
    params = dict(
        drug_name="warfarin",
        dose_mg=10.0,
        hepatic_q_L_per_h=90.0,
        hepatic_clint_L_per_h=20.0,
        renal_q_L_per_h=72.0,
        renal_clint_L_per_h=2.0,
        pulmonary_q_L_per_h=360.0,
        pulmonary_clint_L_per_h=0.5,
        vd_L=10.0,
        fup=0.1,
        route="iv",
        t_end_h=24.0,
        dt_h=0.05,
    )
    params.update(kwargs)
    return simulate_multiorgancl(**params)


# ── organ_clearance ───────────────────────────────────────────────────────────


class TestOrganClearance:
    def test_zero_clint_gives_zero_cl(self):
        cl = organ_clearance(q_L_per_h=90.0, cl_int_L_per_h=0.0, fup=0.5)
        assert cl == 0.0

    def test_very_high_clint_approaches_flow(self):
        """When CLint >> Q, CL_organ → Q (flow-limited)."""
        q = 90.0
        cl = organ_clearance(q_L_per_h=q, cl_int_L_per_h=1e6, fup=1.0)
        assert abs(cl - q) / q < 0.001

    def test_moderate_clint_between_zero_and_flow(self):
        q = 90.0
        cl = organ_clearance(q_L_per_h=q, cl_int_L_per_h=30.0, fup=0.5)
        assert 0 < cl < q

    def test_flow_limited_extraction_ratio_high(self):
        """Very high CLint → EH > 0.7."""
        q = 90.0
        cl = organ_clearance(q_L_per_h=q, cl_int_L_per_h=1000.0, fup=1.0)
        eh = cl / q
        assert eh > 0.7

    def test_fup_scaling(self):
        """Higher fup → higher clearance (more free drug to metabolize)."""
        cl_low = organ_clearance(90.0, 50.0, 0.1)
        cl_high = organ_clearance(90.0, 50.0, 0.5)
        assert cl_high > cl_low

    def test_zero_flow_raises(self):
        with pytest.raises(ValueError, match="q_L_per_h"):
            organ_clearance(q_L_per_h=0.0, cl_int_L_per_h=10.0, fup=0.5)

    def test_negative_clint_raises(self):
        with pytest.raises(ValueError, match="cl_int_L_per_h"):
            organ_clearance(q_L_per_h=90.0, cl_int_L_per_h=-1.0, fup=0.5)

    def test_invalid_fup_raises(self):
        with pytest.raises(ValueError, match="fup"):
            organ_clearance(q_L_per_h=90.0, cl_int_L_per_h=10.0, fup=0.0)


# ── total_clearance ───────────────────────────────────────────────────────────


class TestTotalClearance:
    def test_returns_clearance_breakdown(self):
        result = total_clearance(
            hepatic_q=90.0,
            hepatic_clint=20.0,
            renal_q=72.0,
            renal_clint=2.0,
            pulmonary_q=360.0,
            pulmonary_clint=0.5,
            fup=0.1,
        )
        assert isinstance(result, ClearanceBreakdown)

    def test_fe_sum_approximately_one(self):
        """Fractional excretions must sum to 1 (within floating point)."""
        result = total_clearance(
            hepatic_q=90.0,
            hepatic_clint=50.0,
            renal_q=72.0,
            renal_clint=10.0,
            pulmonary_q=360.0,
            pulmonary_clint=1.0,
            fup=0.5,
        )
        fe_sum = result.fe_hepatic + result.fe_renal + result.fe_pulmonary
        assert abs(fe_sum - 1.0) < 1e-9

    def test_zero_renal_clint_gives_zero_fe_renal(self):
        result = total_clearance(
            hepatic_q=90.0,
            hepatic_clint=50.0,
            renal_q=72.0,
            renal_clint=0.0,
            pulmonary_q=360.0,
            pulmonary_clint=0.0,
            fup=0.5,
        )
        assert result.fe_renal == 0.0
        assert result.cl_renal == 0.0

    def test_hepatic_dominant_extraction(self):
        """High hepatic CLint → eh > 0.5."""
        result = total_clearance(
            hepatic_q=90.0,
            hepatic_clint=500.0,
            renal_q=72.0,
            renal_clint=0.0,
            pulmonary_q=360.0,
            pulmonary_clint=0.0,
            fup=1.0,
        )
        assert result.eh > 0.5

    def test_extraction_ratios_in_zero_one(self):
        result = total_clearance(
            hepatic_q=90.0,
            hepatic_clint=20.0,
            renal_q=72.0,
            renal_clint=5.0,
            pulmonary_q=360.0,
            pulmonary_clint=2.0,
            fup=0.3,
        )
        assert 0.0 <= result.eh <= 1.0
        assert 0.0 <= result.er <= 1.0
        assert 0.0 <= result.ep <= 1.0

    def test_total_cl_equals_sum_of_organ_cls(self):
        result = total_clearance(
            hepatic_q=90.0,
            hepatic_clint=30.0,
            renal_q=72.0,
            renal_clint=8.0,
            pulmonary_q=360.0,
            pulmonary_clint=1.0,
            fup=0.2,
        )
        expected = result.cl_hepatic + result.cl_renal + result.cl_pulmonary
        assert abs(result.cl_total - expected) < 1e-9


# ── simulate_multiorgancl ─────────────────────────────────────────────────────


class TestSimulateMultiOrganCL:
    def test_returns_result_type(self):
        r = _default_sim()
        assert isinstance(r, MultiOrganCLResult)

    def test_clearance_breakdown_in_result(self):
        r = _default_sim()
        assert isinstance(r.clearance, ClearanceBreakdown)

    def test_renal_clint_zero_gives_zero_fe_renal(self):
        r = _default_sim(renal_clint_L_per_h=0.0)
        assert r.clearance.fe_renal == 0.0

    def test_hepatic_dominant_gives_high_eh(self):
        r = _default_sim(
            hepatic_clint_L_per_h=500.0,
            renal_clint_L_per_h=0.0,
            pulmonary_clint_L_per_h=0.0,
            fup=1.0,
        )
        assert r.clearance.eh > 0.5

    def test_plasma_concentration_positive(self):
        r = _default_sim()
        assert r.cmax > 0

    def test_auc_inversely_proportional_to_cl(self):
        """Higher CLint → higher CL → lower AUC."""
        r_low = _default_sim(
            hepatic_clint_L_per_h=5.0, renal_clint_L_per_h=1.0, pulmonary_clint_L_per_h=0.1
        )
        r_high = _default_sim(
            hepatic_clint_L_per_h=200.0, renal_clint_L_per_h=50.0, pulmonary_clint_L_per_h=5.0
        )
        assert r_low.auc > r_high.auc

    def test_higher_fup_increases_total_cl(self):
        """2× fup → higher CL (more free drug available for elimination)."""
        r_low_fup = _default_sim(fup=0.1)
        r_high_fup = _default_sim(fup=0.2)
        assert r_high_fup.clearance.cl_total > r_low_fup.clearance.cl_total

    def test_t_half_formula(self):
        """t½ ≈ 0.693 * Vd / CL_total."""
        r = _default_sim(vd_L=20.0)
        expected_t_half = 0.693 * 20.0 / r.clearance.cl_total
        assert abs(r.t_half_h - expected_t_half) / expected_t_half < 0.01

    def test_oral_route_works(self):
        r = _default_sim(route="oral", ka_per_h=1.2)
        assert r.route == "oral"
        assert r.cmax > 0

    def test_times_list_length(self):
        r = _default_sim(t_end_h=10.0, dt_h=0.1)
        assert len(r.times_h) == len(r.c_plasma_mg_L)

    def test_notes_is_str(self):
        r = _default_sim()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0

    # Input validation

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_sim(dose_mg=0.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            _default_sim(vd_L=0.0)

    def test_zero_hepatic_q_raises(self):
        with pytest.raises(ValueError, match="hepatic_q"):
            _default_sim(hepatic_q_L_per_h=0.0)

    def test_zero_renal_q_raises(self):
        with pytest.raises(ValueError, match="renal_q"):
            _default_sim(renal_q_L_per_h=0.0)

    def test_zero_pulmonary_q_raises(self):
        with pytest.raises(ValueError, match="pulmonary_q"):
            _default_sim(pulmonary_q_L_per_h=0.0)

    def test_invalid_fup_raises(self):
        with pytest.raises(ValueError, match="fup"):
            _default_sim(fup=0.0)

    def test_negative_hepatic_clint_raises(self):
        with pytest.raises(ValueError, match="hepatic_clint"):
            _default_sim(hepatic_clint_L_per_h=-1.0)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            _default_sim(route="inhalation")
