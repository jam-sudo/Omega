"""Tests for core/rbc_binding.py — Phase 332."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.rbc_binding import (
    RBCPartitionResult,
    RBCPKResult,
    blood_to_plasma_ratio,
    cl_blood_to_plasma,
    fu_blood_from_bp,
    rbc_partitioning,
    simulate_rbc_pk,
)

# ---------------------------------------------------------------------------
# blood_to_plasma_ratio
# ---------------------------------------------------------------------------


class TestBloodToPlasmaRatio:
    def test_equal_fractions_gives_one(self):
        # fu_plasma == fu_blood → Kp_rbc = 1 → B/P = 1
        bp = blood_to_plasma_ratio(fu_plasma=0.5, fu_blood=0.5, hematocrit=0.45)
        assert math.isclose(bp, 1.0, rel_tol=1e-6)

    def test_drug_in_rbcs(self):
        # fu_blood < fu_plasma → Kp_rbc > 1 → B/P > 1
        bp = blood_to_plasma_ratio(fu_plasma=0.9, fu_blood=0.3, hematocrit=0.45)
        assert bp > 1.0

    def test_drug_excluded_from_rbcs(self):
        # fu_blood > fu_plasma → Kp_rbc < 1 → B/P < 1
        bp = blood_to_plasma_ratio(fu_plasma=0.2, fu_blood=0.8, hematocrit=0.45)
        assert bp < 1.0

    def test_formula_correctness(self):
        fu_p = 0.6
        fu_b = 0.3
        hct = 0.45
        kp_rbc = fu_p / fu_b
        expected = (1.0 - hct) + hct * kp_rbc
        result = blood_to_plasma_ratio(fu_p, fu_b, hct)
        assert math.isclose(result, expected, rel_tol=1e-10)

    def test_hematocrit_effect(self):
        # Higher hematocrit amplifies RBC effect
        bp_low_hct = blood_to_plasma_ratio(0.9, 0.3, hematocrit=0.3)
        bp_high_hct = blood_to_plasma_ratio(0.9, 0.3, hematocrit=0.6)
        assert bp_high_hct > bp_low_hct

    def test_invalid_fu_plasma_zero(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            blood_to_plasma_ratio(0.0, 0.5)

    def test_invalid_fu_blood_zero(self):
        with pytest.raises(ValueError, match="fu_blood"):
            blood_to_plasma_ratio(0.5, 0.0)

    def test_invalid_fu_plasma_gt_one(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            blood_to_plasma_ratio(1.1, 0.5)

    def test_invalid_hematocrit_low(self):
        with pytest.raises(ValueError, match="hematocrit"):
            blood_to_plasma_ratio(0.5, 0.5, hematocrit=0.2)

    def test_invalid_hematocrit_high(self):
        with pytest.raises(ValueError, match="hematocrit"):
            blood_to_plasma_ratio(0.5, 0.5, hematocrit=0.7)


# ---------------------------------------------------------------------------
# fu_blood_from_bp
# ---------------------------------------------------------------------------


class TestFuBloodFromBp:
    def test_bp_one_returns_fu_plasma(self):
        result = fu_blood_from_bp(bp_ratio=1.0, fu_plasma=0.5)
        assert math.isclose(result, 0.5, rel_tol=1e-6)

    def test_bp_two_halves_fu(self):
        result = fu_blood_from_bp(bp_ratio=2.0, fu_plasma=0.8)
        assert math.isclose(result, 0.4, rel_tol=1e-6)

    def test_clamped_above_one(self):
        # If bp_ratio < fu_plasma, fu_blood could exceed 1; should be clamped
        result = fu_blood_from_bp(bp_ratio=0.5, fu_plasma=0.9)
        assert result <= 1.0

    def test_positive_result(self):
        result = fu_blood_from_bp(2.0, 0.5)
        assert result > 0

    def test_invalid_bp_zero(self):
        with pytest.raises(ValueError, match="bp_ratio"):
            fu_blood_from_bp(0.0, 0.5)

    def test_invalid_fu_plasma_gt_one(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            fu_blood_from_bp(1.0, 1.5)


# ---------------------------------------------------------------------------
# rbc_partitioning
# ---------------------------------------------------------------------------


class TestRBCPartitioning:
    def test_neutral_compound(self):
        result = rbc_partitioning(logP=2.0, charge="neutral")
        assert isinstance(result, RBCPartitionResult)
        assert result.kp_rbc > 0

    def test_basic_higher_kp_than_neutral(self):
        neutral = rbc_partitioning(logP=2.0, charge="neutral")
        basic = rbc_partitioning(logP=2.0, charge="basic")
        assert basic.kp_rbc > neutral.kp_rbc

    def test_acidic_lower_kp_than_neutral(self):
        neutral = rbc_partitioning(logP=2.0, charge="neutral")
        acidic = rbc_partitioning(logP=2.0, charge="acidic")
        assert acidic.kp_rbc < neutral.kp_rbc

    def test_kp_rbc_clamped_min(self):
        # Very negative logP, acidic → would give very small Kp_rbc
        result = rbc_partitioning(logP=-10.0, charge="acidic")
        assert result.kp_rbc >= 0.1

    def test_kp_rbc_clamped_max(self):
        # Very large logP, basic → would give very large Kp_rbc
        result = rbc_partitioning(logP=20.0, charge="basic")
        assert result.kp_rbc <= 20.0

    def test_bp_ratio_formula(self):
        result = rbc_partitioning(logP=1.5, charge="neutral", hematocrit=0.45)
        expected_bp = (1.0 - 0.45) + 0.45 * result.kp_rbc
        assert math.isclose(result.bp_ratio, expected_bp, rel_tol=1e-6)

    def test_pka_included_in_notes(self):
        result = rbc_partitioning(logP=2.0, pka=8.5, charge="basic")
        assert "8.5" in result.notes

    def test_frozen_dataclass(self):
        result = rbc_partitioning(1.0)
        with pytest.raises((AttributeError, TypeError)):
            result.kp_rbc = 5.0  # type: ignore[misc]

    def test_invalid_charge(self):
        with pytest.raises(ValueError, match="charge"):
            rbc_partitioning(logP=2.0, charge="zwitterion")

    def test_invalid_hematocrit(self):
        with pytest.raises(ValueError, match="hematocrit"):
            rbc_partitioning(logP=2.0, hematocrit=0.1)

    def test_fu_blood_between_zero_and_one(self):
        result = rbc_partitioning(logP=2.0, charge="basic")
        assert 0.0 < result.fu_blood <= 1.0

    def test_high_logp_increases_kp(self):
        r1 = rbc_partitioning(logP=1.0, charge="neutral")
        r2 = rbc_partitioning(logP=3.0, charge="neutral")
        assert r2.kp_rbc >= r1.kp_rbc


# ---------------------------------------------------------------------------
# cl_blood_to_plasma
# ---------------------------------------------------------------------------


class TestClBloodToPlasma:
    def test_bp_one_no_change(self):
        cl_p = cl_blood_to_plasma(cl_blood_L_per_h=10.0, bp_ratio=1.0)
        assert math.isclose(cl_p, 10.0, rel_tol=1e-6)

    def test_bp_two_halves_cl(self):
        cl_p = cl_blood_to_plasma(cl_blood_L_per_h=10.0, bp_ratio=2.0)
        assert math.isclose(cl_p, 5.0, rel_tol=1e-6)

    def test_bp_below_one_increases_cl(self):
        cl_p = cl_blood_to_plasma(cl_blood_L_per_h=10.0, bp_ratio=0.5)
        assert math.isclose(cl_p, 20.0, rel_tol=1e-6)

    def test_invalid_cl_zero(self):
        with pytest.raises(ValueError, match="cl_blood_L_per_h"):
            cl_blood_to_plasma(0.0, 1.0)

    def test_invalid_bp_zero(self):
        with pytest.raises(ValueError, match="bp_ratio"):
            cl_blood_to_plasma(10.0, 0.0)


# ---------------------------------------------------------------------------
# simulate_rbc_pk
# ---------------------------------------------------------------------------


class TestSimulateRBCPK:
    def _run(self, route="oral", bp_ratio=1.0):
        return simulate_rbc_pk(
            drug_name="TestDrug",
            dose_mg=100.0,
            cl_plasma_L_per_h=5.0,
            vd_plasma_L=50.0,
            bp_ratio=bp_ratio,
            ka_per_h=1.5,
            route=route,
            t_end_h=24.0,
            dt_h=0.1,
        )

    def test_returns_result_type(self):
        result = self._run()
        assert isinstance(result, RBCPKResult)

    def test_frozen_dataclass(self):
        result = self._run()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_times_length_consistent(self):
        result = self._run()
        assert len(result.times_h) == len(result.c_plasma) == len(result.c_blood)

    def test_cmax_blood_equals_cmax_plasma_times_bp(self):
        bp = 1.5
        result = self._run(bp_ratio=bp)
        # cmax_blood should be approximately cmax_plasma * B/P
        assert math.isclose(result.cmax_blood, result.cmax_plasma * bp, rel_tol=0.05)

    def test_auc_blood_equals_auc_plasma_times_bp(self):
        bp = 2.0
        result = self._run(bp_ratio=bp)
        assert math.isclose(result.auc_blood, result.auc_plasma * bp, rel_tol=1e-4)

    def test_positive_concentrations(self):
        result = self._run()
        assert all(c >= 0 for c in result.c_plasma)
        assert all(c >= 0 for c in result.c_blood)

    def test_iv_route(self):
        result = self._run(route="iv")
        assert result.cmax_plasma > 0

    def test_iv_cmax_at_t0(self):
        result = self._run(route="iv", bp_ratio=1.0)
        # For IV, peak at t=0
        assert result.c_plasma[0] == result.cmax_plasma

    def test_oral_cmax_not_at_t0(self):
        result = self._run(route="oral")
        # For oral, C(0) = 0
        assert result.c_plasma[0] == 0.0
        assert result.cmax_plasma > 0

    def test_bp_ratio_stored(self):
        result = self._run(bp_ratio=1.8)
        assert result.bp_ratio == 1.8

    def test_notes_nonempty(self):
        result = self._run()
        assert len(result.notes) > 0

    def test_invalid_cl_zero(self):
        with pytest.raises(ValueError, match="cl_plasma_L_per_h"):
            simulate_rbc_pk("D", 100.0, 0.0, 50.0, 1.0)

    def test_invalid_vd_zero(self):
        with pytest.raises(ValueError, match="vd_plasma_L"):
            simulate_rbc_pk("D", 100.0, 5.0, 0.0, 1.0)

    def test_invalid_bp_zero(self):
        with pytest.raises(ValueError, match="bp_ratio"):
            simulate_rbc_pk("D", 100.0, 5.0, 50.0, 0.0)

    def test_drug_name_stored(self):
        result = simulate_rbc_pk("Warfarin", 5.0, 0.2, 10.0, 0.8)
        assert result.drug_name == "Warfarin"
