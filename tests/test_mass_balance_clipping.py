"""Tests for negative-state detection and mass balance after clipping removal.

Verifies that after ODE clipping removal (Task #1):
1. Normal simulations produce no negative states (all states >= -1e-6)
2. The post-solve warning fires when solver returns negative states
3. Mass balance holds (±0.5%) for all 5 reference compounds
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from omega_pbpk.core.body import N_STATES, WholeBodyPBPK
from omega_pbpk.drugs.caffeine import CAFFEINE
from omega_pbpk.drugs.metformin import METFORMIN
from omega_pbpk.drugs.midazolam import MIDAZOLAM
from omega_pbpk.drugs.propranolol import PROPRANOLOL
from omega_pbpk.drugs.warfarin import WARFARIN


# ---------------------------------------------------------------------------
# Test 1: Normal caffeine simulation has no significantly negative states
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestNoNegativeStatesNormal:
    def test_caffeine_iv_no_negative_states(self) -> None:
        """A normal caffeine IV simulation should have all states >= -1e-6."""
        model = WholeBodyPBPK(CAFFEINE)
        model.setup_iv(dose_mg=200.0)
        result = model.simulate(t_end_h=24.0, dt_h=0.1)

        min_per_state = np.min(result.amounts, axis=0)
        for idx in range(N_STATES):
            assert min_per_state[idx] >= -1e-6, (
                f"State[{idx}] went negative: min={min_per_state[idx]:.3e} mg"
            )

    def test_caffeine_oral_no_negative_states(self) -> None:
        """A normal caffeine oral simulation should have all states >= -1e-6."""
        model = WholeBodyPBPK(CAFFEINE)
        model.setup_oral(dose_mg=100.0)
        result = model.simulate(t_end_h=24.0, dt_h=0.1)

        min_per_state = np.min(result.amounts, axis=0)
        for idx in range(N_STATES):
            assert min_per_state[idx] >= -1e-6, (
                f"State[{idx}] went negative: min={min_per_state[idx]:.3e} mg"
            )


# ---------------------------------------------------------------------------
# Test 2: Post-solve warning fires when solver returns negative states
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestNegativeStateWarning:
    def test_warning_on_negative_state(self, caplog: pytest.LogCaptureFixture) -> None:
        """When a state goes below -1e-6, logger.warning should fire."""
        model = WholeBodyPBPK(CAFFEINE)
        model.setup_iv(dose_mg=200.0)

        # Save original _rhs so we can wrap it
        original_rhs = model._rhs

        def _rhs_with_negative(t: float, y: np.ndarray) -> np.ndarray:
            """Wrapper that injects a negative into state 3 (brain)."""
            dydt = original_rhs(t, y)
            # Push brain state strongly negative to trigger warning
            dydt[3] -= 1e6
            return dydt

        model._rhs = _rhs_with_negative  # type: ignore[assignment]

        with caplog.at_level(logging.WARNING, logger="omega_pbpk.core.body"):
            model.simulate(t_end_h=0.5, dt_h=0.1)

        # Verify warning mentions state index 3
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        found = any("state[3]" in msg for msg in warning_msgs)
        assert found, f"Expected warning about state[3] but got: {warning_msgs}"

    def test_no_warning_when_states_clean(self, caplog: pytest.LogCaptureFixture) -> None:
        """Normal simulation should not trigger negative-state warning."""
        model = WholeBodyPBPK(CAFFEINE)
        model.setup_iv(dose_mg=200.0)

        with caplog.at_level(logging.WARNING, logger="omega_pbpk.core.body"):
            model.simulate(t_end_h=24.0, dt_h=0.5)

        neg_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "Negative state detected" in r.message
        ]
        assert len(neg_warnings) == 0, (
            f"Unexpected negative-state warnings: {[r.message for r in neg_warnings]}"
        )


# ---------------------------------------------------------------------------
# Test 3: Mass balance ±0.5% for all reference compounds
# ---------------------------------------------------------------------------
REFERENCE_COMPOUNDS = [
    ("caffeine", CAFFEINE, 200.0),
    ("midazolam", MIDAZOLAM, 5.0),
    ("warfarin", WARFARIN, 5.0),
    ("metformin", METFORMIN, 500.0),
    ("propranolol", PROPRANOLOL, 40.0),
]


@pytest.mark.integration
class TestMassBalanceAllCompounds:
    @pytest.mark.parametrize("name,drug,dose_mg", REFERENCE_COMPOUNDS)
    def test_iv_mass_balance(self, name: str, drug: object, dose_mg: float) -> None:
        """IV mass balance must hold at ±0.5% for all reference compounds."""
        model = WholeBodyPBPK(drug)  # type: ignore[arg-type]
        model.setup_iv(dose_mg)
        result = model.simulate(t_end_h=24.0, dt_h=0.1)
        mb = result.mass_balance()

        for i, total in enumerate(mb):
            deviation = abs(total - dose_mg) / dose_mg
            assert deviation < 0.005, (
                f"[{name}] Mass balance violated at t={result.time_h[i]:.1f}h: "
                f"{total:.6f} mg (expected {dose_mg} mg, deviation {deviation:.4%})"
            )

    @pytest.mark.parametrize("name,drug,dose_mg", REFERENCE_COMPOUNDS)
    def test_oral_mass_balance(self, name: str, drug: object, dose_mg: float) -> None:
        """Oral mass balance must hold at ±0.5% for all reference compounds."""
        model = WholeBodyPBPK(drug)  # type: ignore[arg-type]
        model.setup_oral(dose_mg)
        result = model.simulate(t_end_h=24.0, dt_h=0.1)
        mb = result.mass_balance()

        for i, total in enumerate(mb):
            deviation = abs(total - dose_mg) / dose_mg
            assert deviation < 0.005, (
                f"[{name}] Oral mass balance violated at t={result.time_h[i]:.1f}h: "
                f"{total:.6f} mg (expected {dose_mg} mg, deviation {deviation:.4%})"
            )
