"""Tests for Phase 32 ER/MR Formulation Release Models.

Covers:
  - TestReleaseModels (5): analytical release model correctness
  - TestCompositeRelease (3): multi-phase formulation behaviour
  - TestFormulationPK (3): formulation-driven PK properties
  - TestFormulationValidation (1): validation catches bad inputs
  - TestFormulationAPI (2): POST /simulate/formulation endpoint contract
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.core.formulation import (
    FormulationSpec,
    ReleasePhase,
    ReleaseResult,
    release_profile,
    simulate_pk_1cpt,
    validate_formulation,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ir_spec(dose_mg: float = 100.0) -> FormulationSpec:
    """Immediate-release formulation."""
    return FormulationSpec(
        name="IR tablet",
        release_phases=[ReleasePhase(fraction=1.0, model="immediate")],
        total_dose_mg=dose_mg,
    )


def _er_zero_order_spec(dose_mg: float = 100.0, T_h: float = 8.0) -> FormulationSpec:
    """Zero-order extended-release formulation."""
    return FormulationSpec(
        name="ER zero-order",
        release_phases=[
            ReleasePhase(
                fraction=1.0,
                model="zero_order",
                params={"T_release_h": T_h},
            )
        ],
        total_dose_mg=dose_mg,
    )


# ---------------------------------------------------------------------------
# TestReleaseModels (5 tests)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReleaseModels:
    def test_immediate_release_all_at_t0(self):
        """100% released instantly for IR formulation."""
        spec = _ir_spec()
        rp = release_profile(spec, t_end_h=4.0, dt_h=0.01)
        assert isinstance(rp, ReleaseResult)
        # All cumulative fraction values should be 1.0 (even at t=0)
        assert rp.cumulative_fraction_released[0] == 1.0
        assert rp.cumulative_fraction_released[-1] == 1.0
        # Every point should be fully released
        assert np.all(rp.cumulative_fraction_released >= 0.999)

    def test_zero_order_constant_rate(self):
        """Linear release: F(4) = 0.5 for T_release = 8h."""
        spec = _er_zero_order_spec(T_h=8.0)
        rp = release_profile(spec, t_end_h=12.0, dt_h=0.01)
        # Find index closest to t=4h
        idx_4 = np.argmin(np.abs(rp.time_h - 4.0))
        f_at_4 = rp.cumulative_fraction_released[idx_4]
        assert abs(f_at_4 - 0.5) < 0.02, f"Expected ~0.5, got {f_at_4}"
        # At t >= 8h should be fully released
        idx_8 = np.argmin(np.abs(rp.time_h - 8.0))
        f_at_8 = rp.cumulative_fraction_released[idx_8]
        assert f_at_8 >= 0.99

    def test_first_order_exponential(self):
        """F(t) = 1 - exp(-k*t) pattern for first-order release."""
        k = 0.5  # per hour
        spec = FormulationSpec(
            name="first-order",
            release_phases=[
                ReleasePhase(
                    fraction=1.0,
                    model="first_order",
                    params={"k1_per_h": k},
                )
            ],
            total_dose_mg=100.0,
        )
        rp = release_profile(spec, t_end_h=10.0, dt_h=0.01)
        # Check at t=2h: expected F = 1 - exp(-1.0) ≈ 0.6321
        idx_2 = np.argmin(np.abs(rp.time_h - 2.0))
        expected = 1.0 - math.exp(-k * 2.0)
        actual = rp.cumulative_fraction_released[idx_2]
        assert abs(actual - expected) < 0.01, f"Expected {expected:.4f}, got {actual:.4f}"

    def test_weibull_shape_parameter(self):
        """beta=1 ≈ first-order; beta=2 gives sigmoidal profile."""
        lam = 4.0
        # beta=1 (exponential-like)
        spec_b1 = FormulationSpec(
            name="Weibull b=1",
            release_phases=[
                ReleasePhase(
                    fraction=1.0,
                    model="weibull",
                    params={"lambda_h": lam, "beta": 1.0},
                )
            ],
            total_dose_mg=100.0,
        )
        # beta=2 (sigmoidal)
        spec_b2 = FormulationSpec(
            name="Weibull b=2",
            release_phases=[
                ReleasePhase(
                    fraction=1.0,
                    model="weibull",
                    params={"lambda_h": lam, "beta": 2.0},
                )
            ],
            total_dose_mg=100.0,
        )
        rp1 = release_profile(spec_b1, t_end_h=12.0, dt_h=0.01)
        rp2 = release_profile(spec_b2, t_end_h=12.0, dt_h=0.01)

        # beta=1 Weibull should match first-order at the same scale:
        # F(t) = 1 - exp(-(t/lambda)^beta) with beta=1 → 1 - exp(-t/lambda)
        idx_2 = np.argmin(np.abs(rp1.time_h - 2.0))
        expected_b1 = 1.0 - math.exp(-(2.0 / lam))
        assert abs(rp1.cumulative_fraction_released[idx_2] - expected_b1) < 0.01

        # At early time (t=1h), beta=2 should release LESS than beta=1
        # because sigmoidal starts slower
        idx_1 = np.argmin(np.abs(rp1.time_h - 1.0))
        assert rp2.cumulative_fraction_released[idx_1] < rp1.cumulative_fraction_released[idx_1]

    def test_higuchi_sqrt_time(self):
        """F(t) = kH * sqrt(t), capped at 1.0."""
        kH = 0.4  # per sqrt(h)
        spec = FormulationSpec(
            name="Higuchi",
            release_phases=[
                ReleasePhase(
                    fraction=1.0,
                    model="higuchi",
                    params={"kH_per_sqrt_h": kH},
                )
            ],
            total_dose_mg=100.0,
        )
        rp = release_profile(spec, t_end_h=10.0, dt_h=0.01)

        # At t=4h: F = 0.4 * sqrt(4) = 0.4 * 2 = 0.8
        idx_4 = np.argmin(np.abs(rp.time_h - 4.0))
        expected = kH * math.sqrt(4.0)
        actual = rp.cumulative_fraction_released[idx_4]
        assert abs(actual - expected) < 0.02, f"Expected {expected:.3f}, got {actual:.3f}"

        # At t=9h: F = 0.4 * 3 = 1.2 → capped at 1.0
        idx_9 = np.argmin(np.abs(rp.time_h - 9.0))
        assert rp.cumulative_fraction_released[idx_9] <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# TestCompositeRelease (3 tests)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompositeRelease:
    def test_ir_er_composite(self):
        """30% IR + 70% ER: biphasic release profile."""
        spec = FormulationSpec(
            name="IR/ER combo",
            release_phases=[
                ReleasePhase(fraction=0.3, model="immediate", site="stomach"),
                ReleasePhase(
                    fraction=0.7,
                    model="zero_order",
                    params={"T_release_h": 8.0},
                    site="jejunum1",
                ),
            ],
            total_dose_mg=100.0,
        )
        rp = release_profile(spec, t_end_h=12.0, dt_h=0.01)

        # At t≈0, IR contributes 0.3 immediately
        assert rp.cumulative_fraction_released[0] >= 0.29

        # At t=4h, ER should have contributed ~0.7 * 0.5 = 0.35 → total ~0.65
        idx_4 = np.argmin(np.abs(rp.time_h - 4.0))
        f_4 = rp.cumulative_fraction_released[idx_4]
        assert 0.55 < f_4 < 0.75, f"Expected ~0.65, got {f_4}"

        # At end, should reach ~1.0
        assert rp.cumulative_fraction_released[-1] >= 0.99

    def test_multi_site_release(self):
        """Phases targeting different GI sites."""
        spec = FormulationSpec(
            name="Multi-site",
            release_phases=[
                ReleasePhase(
                    fraction=0.4,
                    model="immediate",
                    site="stomach",
                ),
                ReleasePhase(
                    fraction=0.3,
                    model="zero_order",
                    params={"T_release_h": 4.0},
                    lag_h=1.0,
                    site="jejunum1",
                ),
                ReleasePhase(
                    fraction=0.3,
                    model="first_order",
                    params={"k1_per_h": 0.5},
                    lag_h=3.0,
                    site="ileum1",
                ),
            ],
            total_dose_mg=100.0,
        )
        rp = release_profile(spec, t_end_h=16.0, dt_h=0.01)
        # Should reach full release eventually
        assert rp.cumulative_fraction_released[-1] >= 0.95
        # Profile should be monotonically non-decreasing (checked in TestFormulationPK)
        diffs = np.diff(rp.cumulative_fraction_released)
        assert np.all(diffs >= -1e-6), "Profile should be monotonically non-decreasing"

    def test_fractions_must_sum_to_one(self):
        """Validation catches fractions != 1.0."""
        spec = FormulationSpec(
            name="Bad fractions",
            release_phases=[
                ReleasePhase(fraction=0.3, model="immediate"),
                ReleasePhase(
                    fraction=0.3,
                    model="zero_order",
                    params={"T_release_h": 4.0},
                ),
            ],
            total_dose_mg=100.0,
        )
        warnings = validate_formulation(spec)
        assert any("sum" in w.lower() or "fraction" in w.lower() for w in warnings), (
            f"Expected fraction sum warning, got: {warnings}"
        )


# ---------------------------------------------------------------------------
# TestFormulationPK (3 tests)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFormulationPK:
    def test_er_vs_ir_cmax_lower(self):
        """ER formulation has lower Cmax than IR."""
        ke = 0.1
        vd = 50.0
        ir_spec = _ir_spec(dose_mg=100.0)
        er_spec = _er_zero_order_spec(dose_mg=100.0, T_h=8.0)

        _, conc_ir = simulate_pk_1cpt(ir_spec, ke=ke, vd=vd, t_end_h=24.0, dt_h=0.01)
        _, conc_er = simulate_pk_1cpt(er_spec, ke=ke, vd=vd, t_end_h=24.0, dt_h=0.01)

        cmax_ir = float(np.max(conc_ir))
        cmax_er = float(np.max(conc_er))
        assert cmax_er < cmax_ir, f"ER Cmax ({cmax_er:.4f}) should be < IR Cmax ({cmax_ir:.4f})"

    def test_er_vs_ir_auc_similar(self):
        """AUC within 15% — same bioavailability means similar total exposure."""
        from omega_pbpk._compat import np_trapz

        ke = 0.1
        vd = 50.0
        ir_spec = _ir_spec(dose_mg=100.0)
        er_spec = _er_zero_order_spec(dose_mg=100.0, T_h=8.0)

        t_ir, conc_ir = simulate_pk_1cpt(ir_spec, ke=ke, vd=vd, t_end_h=48.0, dt_h=0.01)
        t_er, conc_er = simulate_pk_1cpt(er_spec, ke=ke, vd=vd, t_end_h=48.0, dt_h=0.01)

        auc_ir = float(np_trapz(conc_ir, t_ir))
        auc_er = float(np_trapz(conc_er, t_er))

        # AUC should be within 15%
        ratio = auc_er / auc_ir if auc_ir > 0 else 0.0
        assert 0.85 <= ratio <= 1.15, f"AUC ratio {ratio:.3f} outside [0.85, 1.15]"

    def test_release_profile_shape(self):
        """Release profile is monotonically increasing in [0, 1]."""
        spec = _er_zero_order_spec(dose_mg=100.0, T_h=8.0)
        rp = release_profile(spec, t_end_h=12.0, dt_h=0.01)

        # Bounded in [0, 1]
        assert np.all(rp.cumulative_fraction_released >= -1e-9)
        assert np.all(rp.cumulative_fraction_released <= 1.0 + 1e-9)

        # Monotonically non-decreasing
        diffs = np.diff(rp.cumulative_fraction_released)
        assert np.all(diffs >= -1e-6), "Cumulative fraction must be monotonically non-decreasing"


# ---------------------------------------------------------------------------
# TestFormulationValidation (1 test)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormulationValidation:
    def test_validate_catches_errors(self):
        """Negative params, missing keys flagged."""
        spec = FormulationSpec(
            name="Bad spec",
            release_phases=[
                # Missing required param for zero_order
                ReleasePhase(fraction=0.5, model="zero_order", params={}),
                # Negative param value
                ReleasePhase(
                    fraction=0.5,
                    model="first_order",
                    params={"k1_per_h": -1.0},
                ),
            ],
            total_dose_mg=100.0,
        )
        warnings = validate_formulation(spec)
        # Should catch missing T_release_h
        assert any("T_release_h" in w for w in warnings), (
            f"Expected missing param warning: {warnings}"
        )
        # Should catch negative k1_per_h
        assert any("k1_per_h" in w for w in warnings), (
            f"Expected negative param warning: {warnings}"
        )


# ---------------------------------------------------------------------------
# TestFormulationAPI (2 tests)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormulationAPI:
    def test_post_simulate_formulation_200(self):
        """POST returns 200 with valid response."""
        payload = {
            "name": "ER test",
            "release_phases": [
                {
                    "fraction": 0.3,
                    "model": "immediate",
                    "params": {},
                    "lag_h": 0.0,
                    "site": "stomach",
                },
                {
                    "fraction": 0.7,
                    "model": "zero_order",
                    "params": {"T_release_h": 8.0},
                    "lag_h": 0.0,
                    "site": "jejunum1",
                },
            ],
            "total_dose_mg": 100.0,
            "ke": 0.1,
            "vd_L": 42.0,
            "t_end_h": 24.0,
            "dt_h": 0.05,
        }
        r = client.post("/simulate/formulation", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert "formulation_name" in body
        assert "pk_summary" in body
        assert "release_profile" in body
        assert "plasma_profile" in body
        assert body["pk_summary"]["cmax_mg_L"] > 0
        assert body["pk_summary"]["auc_mg_h_L"] > 0
        assert isinstance(body["release_profile"]["time_h"], list)
        assert isinstance(body["plasma_profile"]["conc_mg_L"], list)

    def test_post_simulate_formulation_422(self):
        """Bad params returns 422."""
        payload = {
            "name": "bad",
            "release_phases": [
                {
                    "fraction": 1.0,
                    "model": "INVALID_MODEL",
                    "params": {},
                }
            ],
            "total_dose_mg": 100.0,
        }
        r = client.post("/simulate/formulation", json=payload)
        assert r.status_code == 422
