"""Tests for adaptive MCMC proposal scaling (Robbins-Monro)."""

from __future__ import annotations

import numpy as np
import pytest

from omega_pbpk.calibration import calibrate


def _gaussian_log_prob(x: np.ndarray) -> float:
    """1-D standard Gaussian: log p(x) = -0.5 * x^2."""
    return float(-0.5 * x[0] ** 2)


class TestAdaptiveMCMC:
    """Tests for the ``calibrate()`` function with Robbins-Monro adaptation."""

    def test_acceptance_rate_in_range_adaptive(self) -> None:
        """With adaptive=True on a 1-D Gaussian, acceptance rate should be in [0.15, 0.45]."""
        result = calibrate(
            log_prob=_gaussian_log_prob,
            x0=np.array([0.0]),
            n_steps=2000,
            proposal_sd=1.0,
            seed=42,
            adaptive=True,
            adapt_interval=50,
        )
        assert 0.15 <= result.acceptance_rate <= 0.45, (
            f"acceptance_rate={result.acceptance_rate:.3f} outside [0.15, 0.45]"
        )

    def test_adaptive_false_proposal_sd_fixed(self) -> None:
        """With adaptive=False, proposal_sd must remain unchanged."""
        initial_sd = 0.5
        result = calibrate(
            log_prob=_gaussian_log_prob,
            x0=np.array([0.0]),
            n_steps=500,
            proposal_sd=initial_sd,
            seed=123,
            adaptive=False,
        )
        assert result.final_proposal_sd == pytest.approx(initial_sd), (
            f"proposal_sd changed from {initial_sd} to {result.final_proposal_sd} "
            "with adaptive=False"
        )

    def test_proposal_sd_stops_after_adapt_end(self) -> None:
        """Proposal SD should stop changing after adapt_end steps."""
        adapt_end = 200

        # Run with a short adapt_end; then check that the SD recorded at
        # adapt_end matches the final SD (no further adaptation).
        result_short = calibrate(
            log_prob=_gaussian_log_prob,
            x0=np.array([0.0]),
            n_steps=1000,
            proposal_sd=0.1,
            seed=99,
            adaptive=True,
            adapt_interval=50,
            adapt_end=adapt_end,
        )

        # Run again with the same seed but 2x as many post-adapt steps.
        # The final_proposal_sd should be identical because adaptation
        # stopped at the same adapt_end.
        result_long = calibrate(
            log_prob=_gaussian_log_prob,
            x0=np.array([0.0]),
            n_steps=2000,
            proposal_sd=0.1,
            seed=99,
            adaptive=True,
            adapt_interval=50,
            adapt_end=adapt_end,
        )

        assert result_short.final_proposal_sd == pytest.approx(result_long.final_proposal_sd), (
            f"proposal_sd differs between n_steps=1000 ({result_short.final_proposal_sd:.6f}) "
            f"and n_steps=2000 ({result_long.final_proposal_sd:.6f}) "
            f"despite same adapt_end={adapt_end}"
        )

    def test_result_has_posterior_samples(self) -> None:
        """CalibrationResult should contain posterior samples with expected columns."""
        result = calibrate(
            log_prob=_gaussian_log_prob,
            x0=np.array([0.0]),
            n_steps=200,
            seed=7,
            adaptive=False,
        )
        assert len(result.posterior_samples) == 200
        assert "x0" in result.posterior_samples.columns
        assert "log_posterior" in result.posterior_samples.columns

    def test_final_proposal_sd_populated(self) -> None:
        """final_proposal_sd should be a positive number."""
        result = calibrate(
            log_prob=_gaussian_log_prob,
            x0=np.array([0.0]),
            n_steps=300,
            proposal_sd=0.5,
            seed=1,
            adaptive=True,
            adapt_interval=50,
        )
        assert result.final_proposal_sd > 0
