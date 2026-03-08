"""Tests for Phase 1131 — Hypothalamic PK model."""

import math

import pytest

from omega_pbpk.core.hypothalamic_pk import (
    HypothalamicPKResult,
    _compute_kp_brain,
    simulate_hypothalamic_pk,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def default_result(**kwargs) -> HypothalamicPKResult:
    params = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        logP=2.0,
        cl_L_per_h=5.0,
        vd_plasma_L=10.0,
        t_end_h=12.0,
        dt_h=0.05,
    )
    params.update(kwargs)
    return simulate_hypothalamic_pk(**params)


# ---------------------------------------------------------------------------
# kp_brain computation
# ---------------------------------------------------------------------------


class TestKpBrain:
    def test_logp2_gives_one(self):
        # 10^(0.3*(2-2)) = 10^0 = 1.0
        assert math.isclose(_compute_kp_brain(2.0), 1.0, rel_tol=1e-6)

    def test_low_logp_clamped(self):
        # Very low logP should be clamped to 0.05
        assert _compute_kp_brain(-5.0) == pytest.approx(0.05)

    def test_high_logp_clamped(self):
        # Very high logP should be clamped to 5.0
        assert _compute_kp_brain(10.0) == pytest.approx(5.0)

    def test_intermediate_logp(self):
        # logP=4 → 10^(0.3*2) = 10^0.6 ≈ 3.981
        expected = min(5.0, 10 ** (0.3 * (4.0 - 2)))
        assert _compute_kp_brain(4.0) == pytest.approx(expected, rel=1e-6)

    def test_logp_zero(self):
        # logP=0 → 10^(0.3*(0-2)) = 10^(-0.6) ≈ 0.251
        expected = max(0.05, 10 ** (0.3 * (0.0 - 2.0)))
        assert _compute_kp_brain(0.0) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_c_plasma_t0(self):
        r = default_result(dose_mg=100.0, vd_plasma_L=10.0)
        assert r.c_plasma_mg_L[0] == pytest.approx(10.0, rel=1e-6)

    def test_c_bbb_t0_zero(self):
        r = default_result()
        assert r.c_bbb_mg_L[0] == pytest.approx(0.0)

    def test_c_hypo_t0_zero(self):
        r = default_result()
        assert r.c_hypo_mg_L[0] == pytest.approx(0.0)

    def test_time_starts_at_zero(self):
        r = default_result()
        assert r.times_h[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


class TestDynamics:
    def test_c_plasma_declines(self):
        r = default_result()
        # Plasma should decrease monotonically (IV bolus, no re-input)
        assert r.c_plasma_mg_L[-1] < r.c_plasma_mg_L[0]

    def test_c_hypo_rises_then_falls(self):
        r = default_result()
        hypo = r.c_hypo_mg_L
        _ = hypo.index(max(hypo))
        # After t=0 (where hypo=0), it must rise
        assert max(hypo) > 0
        # And the final value should be less than peak
        assert hypo[-1] < max(hypo)

    def test_c_hypo_non_negative(self):
        r = default_result()
        assert all(c >= 0 for c in r.c_hypo_mg_L)

    def test_c_plasma_non_negative(self):
        r = default_result()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_c_bbb_non_negative(self):
        r = default_result()
        assert all(c >= 0 for c in r.c_bbb_mg_L)

    def test_cmax_hypo_positive(self):
        r = default_result()
        assert r.cmax_hypo_mg_L > 0

    def test_tmax_hypo_positive(self):
        r = default_result()
        assert r.tmax_hypo_h > 0

    def test_auc_hypo_positive(self):
        r = default_result()
        assert r.auc_hypo_mg_h_per_L > 0

    def test_auc_plasma_positive(self):
        r = default_result()
        assert r.auc_plasma_mg_h_per_L > 0


# ---------------------------------------------------------------------------
# LogP effect on Kp and penetration
# ---------------------------------------------------------------------------


class TestLogPEffect:
    def test_higher_logp_higher_kp_brain(self):
        r_low = default_result(logP=0.0)
        r_high = default_result(logP=5.0)
        assert r_high.kp_brain > r_low.kp_brain

    def test_higher_logp_higher_hypo_auc(self):
        r_low = default_result(logP=0.0)
        r_high = default_result(logP=5.0)
        assert r_high.auc_hypo_mg_h_per_L > r_low.auc_hypo_mg_h_per_L

    def test_hypo_to_plasma_ratio_less_than_one(self):
        # Drug should be more concentrated in plasma than hypothalamus overall
        r = default_result(logP=2.0)
        assert r.hypo_to_plasma_auc_ratio < 1.0


# ---------------------------------------------------------------------------
# CNS penetration classification
# ---------------------------------------------------------------------------


class TestCNSClassification:
    def test_poor_penetration_label(self):
        # Low logP → low kp_brain → low ratio → "poor"
        r = default_result(logP=-5.0)
        assert r.cns_penetration_class == "poor"

    def test_classification_in_valid_set(self):
        r = default_result(logP=2.0)
        assert r.cns_penetration_class in {"poor", "moderate", "good"}


# ---------------------------------------------------------------------------
# Dose scaling
# ---------------------------------------------------------------------------


class TestDoseScaling:
    def test_double_dose_doubles_cmax(self):
        r1 = default_result(dose_mg=100.0)
        r2 = default_result(dose_mg=200.0)
        assert r2.cmax_hypo_mg_L == pytest.approx(r1.cmax_hypo_mg_L * 2, rel=1e-3)

    def test_double_dose_doubles_auc(self):
        r1 = default_result(dose_mg=100.0)
        r2 = default_result(dose_mg=200.0)
        assert r2.auc_hypo_mg_h_per_L == pytest.approx(r1.auc_hypo_mg_h_per_L * 2, rel=1e-3)


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


class TestResultStructure:
    def test_result_type(self):
        r = default_result()
        assert isinstance(r, HypothalamicPKResult)

    def test_drug_name_preserved(self):
        r = default_result(drug_name="DrugX")
        assert r.drug_name == "DrugX"

    def test_times_and_concentrations_same_length(self):
        r = default_result()
        assert len(r.times_h) == len(r.c_plasma_mg_L)
        assert len(r.times_h) == len(r.c_bbb_mg_L)
        assert len(r.times_h) == len(r.c_hypo_mg_L)

    def test_times_monotonic(self):
        r = default_result()
        for i in range(1, len(r.times_h)):
            assert r.times_h[i] > r.times_h[i - 1]

    def test_notes_non_empty(self):
        r = default_result()
        assert len(r.notes) > 0

    def test_kp_brain_in_result(self):
        r = default_result(logP=2.0)
        assert r.kp_brain == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_hypothalamic_pk("D", -1.0, 2.0)

    def test_zero_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_hypothalamic_pk("D", 0.0, 2.0)

    def test_negative_cl(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_hypothalamic_pk("D", 100.0, 2.0, cl_L_per_h=-1.0)

    def test_zero_cl(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_hypothalamic_pk("D", 100.0, 2.0, cl_L_per_h=0.0)

    def test_negative_vd(self):
        with pytest.raises(ValueError, match="vd_plasma_L"):
            simulate_hypothalamic_pk("D", 100.0, 2.0, vd_plasma_L=-5.0)

    def test_logp_too_low(self):
        with pytest.raises(ValueError, match="logP"):
            simulate_hypothalamic_pk("D", 100.0, -6.0)

    def test_logp_too_high(self):
        with pytest.raises(ValueError, match="logP"):
            simulate_hypothalamic_pk("D", 100.0, 11.0)
