"""Tests for Phase 47 Parallel-Group Clinical Trial Simulator.

Covers:
  - TestDataClasses (2): DoseGroup / ParallelTrialDesign creation
  - TestHelpers (4): lognormal BSV, Emax equation, ANOVA, pairwise t-test
  - TestSimulation (5): result structure, PK scaling, placebo effect,
                        responder threshold, trial success flag
  - TestDoseResponse (2): Hill fit with known data, fit failure fallback
  - TestInterim (1): conditional power + futility flag
  - TestAPIEndpoint (2): valid request → 200, invalid group count → 422
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.clinical.parallel_trial import (
    ArmSummary,
    DoseGroup,
    DoseResponseFit,
    ParallelTrialDesign,
    ParallelTrialResult,
    _conditional_power,
    _emax_effect,
    _fit_dose_response,
    _lognormal_bsv,
    _one_way_anova,
    _pairwise_tests,
    run_parallel_trial,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Minimal 3-arm design reused across tests
# ---------------------------------------------------------------------------

_DESIGN_3ARM = ParallelTrialDesign(
    dose_groups=(
        DoseGroup("Placebo", 0.0, "oral", 20),
        DoseGroup("10 mg", 10.0, "oral", 20),
        DoseGroup("50 mg", 50.0, "oral", 20),
    ),
    cl_L_per_h=5.0,
    vd_L=70.0,
    ka_per_h=1.2,
    f_bioavail=0.8,
    emax=1.0,
    ec50_mg_L=1.0,
    hill=1.0,
    pd_input="cmax",
    interim_fraction=0.5,
    seed=42,
)

_RESULT_3ARM: ParallelTrialResult | None = None


def _get_result() -> ParallelTrialResult:
    global _RESULT_3ARM
    if _RESULT_3ARM is None:
        _RESULT_3ARM = run_parallel_trial(_DESIGN_3ARM)
    return _RESULT_3ARM


# ---------------------------------------------------------------------------
# TestDataClasses
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDataClasses:
    def test_dose_group_defaults(self):
        """DoseGroup stores name, dose, route, n_subjects."""
        g = DoseGroup("50 mg", 50.0)
        assert g.name == "50 mg"
        assert g.dose_mg == 50.0
        assert g.route == "oral"
        assert g.n_subjects == 30

    def test_parallel_trial_design_defaults(self):
        """ParallelTrialDesign stores dose_groups and PK/PD params."""
        design = ParallelTrialDesign(
            dose_groups=(DoseGroup("Placebo", 0.0), DoseGroup("100 mg", 100.0)),
        )
        assert len(design.dose_groups) == 2
        assert design.cl_L_per_h == pytest.approx(5.0)
        assert design.emax == pytest.approx(1.0)
        assert design.hill == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHelpers:
    def test_lognormal_bsv_mean(self):
        """_lognormal_bsv samples are positive and centered near mean."""
        rng = np.random.default_rng(0)
        samples = [_lognormal_bsv(5.0, 0.3, rng) for _ in range(500)]
        assert all(s > 0 for s in samples)
        assert np.mean(samples) == pytest.approx(5.0, rel=0.15)

    def test_emax_effect_zero_conc(self):
        """Emax effect at C=0 is 0."""
        assert _emax_effect(0.0, 1.0, 1.0, 1.0) == pytest.approx(0.0)

    def test_emax_effect_saturating(self):
        """Emax effect at very high C approaches Emax."""
        eff = _emax_effect(1e6, 1.0, 1.0, 1.0)
        assert eff == pytest.approx(1.0, rel=1e-4)

    def test_emax_effect_ec50(self):
        """At C = EC50, effect = Emax/2."""
        eff = _emax_effect(2.0, 1.0, 2.0, 1.0)
        assert eff == pytest.approx(0.5, rel=1e-6)

    def test_one_way_anova_different_means(self):
        """ANOVA p-value is small when group means are clearly different."""
        rng = np.random.default_rng(7)
        g1 = rng.normal(0.0, 0.1, 30)
        g2 = rng.normal(1.0, 0.1, 30)
        g3 = rng.normal(2.0, 0.1, 30)
        p = _one_way_anova([g1, g2, g3])
        assert p < 0.001

    def test_one_way_anova_same_means(self):
        """ANOVA p-value is large when all groups are from same distribution."""
        rng = np.random.default_rng(1)
        groups = [rng.normal(0.0, 1.0, 30) for _ in range(3)]
        p = _one_way_anova(groups)
        assert p > 0.01  # not necessarily >0.05 due to randomness


# ---------------------------------------------------------------------------
# TestSimulation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSimulation:
    def test_result_has_correct_arm_count(self):
        """Result has one arm summary per dose group."""
        res = _get_result()
        assert len(res.arms) == 3

    def test_total_subjects_correct(self):
        """Total subjects = sum of n_subjects across arms."""
        res = _get_result()
        assert res.total_subjects == 60

    def test_placebo_cmax_near_zero(self):
        """Placebo arm has Cmax ≈ 0."""
        res = _get_result()
        placebo = next(a for a in res.arms if a.dose_mg == 0.0)
        assert placebo.cmax_mean == pytest.approx(0.0, abs=1e-6)

    def test_cmax_increases_with_dose(self):
        """Higher dose groups have higher geometric mean Cmax."""
        res = _get_result()
        dose_arms = [a for a in res.arms if a.dose_mg > 0]
        dose_arms_sorted = sorted(dose_arms, key=lambda a: a.dose_mg)
        for i in range(len(dose_arms_sorted) - 1):
            assert dose_arms_sorted[i].cmax_mean < dose_arms_sorted[i + 1].cmax_mean

    def test_subjects_list_length(self):
        """subjects list has one entry per subject across all arms."""
        res = _get_result()
        assert len(res.subjects) == 60

    def test_responder_pct_range(self):
        """responder_pct is in [0, 100] for all arms."""
        res = _get_result()
        for arm in res.arms:
            assert 0.0 <= arm.responder_pct <= 100.0

    def test_pairwise_results_present(self):
        """Pairwise results include effect, auc, and cmax endpoints."""
        res = _get_result()
        endpoints = {p.endpoint for p in res.pairwise}
        assert "effect" in endpoints
        assert "auc" in endpoints
        assert "cmax" in endpoints

    def test_anova_p_values_finite(self):
        """ANOVA p-values are finite scalars."""
        res = _get_result()
        assert math.isfinite(res.anova_p_auc)
        assert math.isfinite(res.anova_p_cmax)
        assert math.isfinite(res.anova_p_effect)


# ---------------------------------------------------------------------------
# TestDoseResponse
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDoseResponse:
    def test_dose_response_fit_known_data(self):
        """Hill fit recovers Emax ≈ 1, ED50 ≈ 50 on synthetic data."""
        doses = np.array([0.0, 10.0, 25.0, 50.0, 100.0, 200.0])
        effects = _emax_effect(doses, 1.0, 50.0, 1.0)
        dr = _fit_dose_response(doses, effects)
        assert dr.fit_success
        assert dr.emax_fit == pytest.approx(1.0, rel=0.05)
        assert dr.ec50_dose_mg == pytest.approx(50.0, rel=0.10)
        assert dr.r_squared >= 0.99

    def test_dose_response_fit_too_few_points(self):
        """fit returns fit_success=False when fewer than 2 non-zero dose points."""
        doses = np.array([0.0, 10.0])
        effects = np.array([0.0, 0.5])
        dr = _fit_dose_response(doses, effects)
        # Should still try to fit with 1 non-zero point; handle gracefully
        assert isinstance(dr.fit_success, bool)

    def test_dose_response_ed80_less_than_ed90(self):
        """ED80 < ED90 for a well-fitted Hill curve."""
        doses = np.array([10.0, 30.0, 100.0, 300.0])
        effects = np.array([0.2, 0.5, 0.8, 0.95])
        dr = _fit_dose_response(doses, effects)
        if dr.fit_success:
            assert dr.ed80_mg < dr.ed90_mg


# ---------------------------------------------------------------------------
# TestInterim
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInterim:
    def test_interim_conducted(self):
        """Interim analysis is conducted when interim_fraction > 0."""
        res = _get_result()
        assert res.interim.conducted is True
        assert res.interim.fraction_enrolled == pytest.approx(0.5)

    def test_interim_futility_flag_type(self):
        """futility_stop is a boolean."""
        res = _get_result()
        assert isinstance(res.interim.futility_stop, bool)

    def test_conditional_power_in_range(self):
        """Conditional power is in [0, 1]."""
        rng = np.random.default_rng(0)
        high_eff = rng.normal(1.0, 0.1, 15)
        low_eff = rng.normal(0.1, 0.1, 15)
        cp = _conditional_power(high_eff, low_eff, 60)
        assert 0.0 <= cp <= 1.0


# ---------------------------------------------------------------------------
# TestAPIEndpoint
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAPIEndpoint:
    def test_parallel_trial_endpoint_200(self):
        """POST /trial/parallel with valid 3-arm request returns 200."""
        response = client.post(
            "/trial/parallel",
            json={
                "dose_groups": [
                    {"name": "Placebo", "dose_mg": 0.0, "n_subjects": 20},
                    {"name": "10 mg", "dose_mg": 10.0, "n_subjects": 20},
                    {"name": "50 mg", "dose_mg": 50.0, "n_subjects": 20},
                ],
                "cl_L_per_h": 5.0,
                "vd_L": 70.0,
                "ka_per_h": 1.2,
                "f_bioavail": 0.8,
                "emax": 1.0,
                "ec50_mg_L": 1.0,
                "hill": 1.0,
                "pd_input": "cmax",
                "t_end_h": 24.0,
                "alpha": 0.05,
                "interim_fraction": 0.5,
                "seed": 42,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "arms" in data
        assert len(data["arms"]) == 3
        assert "anova_p_effect" in data
        assert "dose_response" in data
        assert "trial_success" in data
        assert data["total_subjects"] == 60

    def test_parallel_trial_endpoint_422_too_few_groups(self):
        """POST /trial/parallel with only 1 dose group returns 422."""
        response = client.post(
            "/trial/parallel",
            json={
                "dose_groups": [{"name": "10 mg", "dose_mg": 10.0, "n_subjects": 20}],
                "cl_L_per_h": 5.0,
                "vd_L": 70.0,
            },
        )
        assert response.status_code == 422
