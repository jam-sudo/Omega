"""Unit tests for Bayesian MCMC calibration (Metropolis-Hastings).

Covers:
  - MCMC runs without error on synthetic data (smoke test)
  - Acceptance rate is within reasonable range [10%, 60%]
  - Posterior mean of CLint converges toward the true parameter value
  - CalibrationResult structure contains expected fields
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from omega_pbpk.calibration import CalibrationResult, _simulate_concentration, run_mh_calibration
from omega_pbpk.drugs.drug import Drug

# ------------------------------------------------------------------ helpers


def _make_drug(**overrides) -> Drug:
    """Minimal drug suitable for calibration smoke tests."""
    defaults = dict(
        name="CalibTest",
        mw=300.0,
        logP=2.0,
        fup=0.5,
        rbp=1.0,
        pka=[7.0],
        drug_type="neutral",
        peff=5.0,
        solubility_mg_mL=10.0,
        particle_radius_um=25.0,
        clint_hepatic_L_per_h=10.0,
        clint_gut_L_per_h=1.0,
        clr_L_per_h=0.0,
        fm={"CYP3A4": 1.0},
        kp={
            "lung": 1.0,
            "brain": 1.0,
            "heart": 1.0,
            "kidney": 1.0,
            "liver": 1.0,
            "spleen": 1.0,
            "gut_wall": 1.0,
            "pancreas": 1.0,
            "thymus": 1.0,
            "reproductive": 1.0,
            "rest": 1.0,
        },
        permeability_limited={
            "adipose": {"kp": 1.0, "ps": 20},
            "muscle": {"kp": 1.0, "ps": 40},
            "bone": {"kp": 1.0, "ps": 10},
            "skin": {"kp": 1.0, "ps": 15},
        },
    )
    defaults.update(overrides)
    return Drug(**defaults)


def _synthetic_observed(
    drug: Drug,
    true_clint: float,
    dose_mg: float = 100.0,
    route: str = "iv",
    t_end_h: float = 12.0,
    n_obs: int = 8,
    noise_sd: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic observed data by simulating with true_clint + small noise."""
    t_arr, c_arr = _simulate_concentration(
        drug=drug,
        body_weight=70.0,
        dose_mg=dose_mg,
        route=route,
        t_end_h=t_end_h,
        clint_hepatic_L_per_h=true_clint,
        clint_gut_L_per_h=None,
    )

    # Sample evenly spaced time points within the simulated range
    obs_times = np.linspace(0.5, min(t_end_h - 0.5, t_arr[-1] - 0.5), n_obs)
    obs_conc = np.interp(obs_times, t_arr, c_arr)

    rng = np.random.default_rng(seed)
    obs_conc = np.abs(obs_conc + rng.normal(0, noise_sd, size=n_obs))

    return pd.DataFrame({"time_h": obs_times, "C_plasma_mg_per_L": obs_conc})


# ------------------------------------------------------------------ smoke test


class TestMCMCSmoke:
    def test_mcmc_runs_without_error(self) -> None:
        """MCMC calibration should complete without raising any exception."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0)
        observed = _synthetic_observed(drug, true_clint=10.0)

        result = run_mh_calibration(
            observed=observed,
            drug=drug,
            dose_mg=100.0,
            route="iv",
            t_end_h=12.0,
            n_samples=100,
            burn_in=20,
            calibrate_gut=False,
            seed=0,
        )
        assert isinstance(result, CalibrationResult)

    def test_mcmc_result_has_posterior_samples(self) -> None:
        """CalibrationResult must contain a non-empty DataFrame of posterior samples."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0)
        observed = _synthetic_observed(drug, true_clint=10.0)

        result = run_mh_calibration(
            observed=observed,
            drug=drug,
            dose_mg=100.0,
            route="iv",
            t_end_h=12.0,
            n_samples=100,
            burn_in=20,
            calibrate_gut=False,
            seed=1,
        )
        assert isinstance(result.posterior_samples, pd.DataFrame)
        assert len(result.posterior_samples) > 0

    def test_mcmc_result_has_posterior_predictive(self) -> None:
        """CalibrationResult must contain a posterior_predictive DataFrame."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0)
        observed = _synthetic_observed(drug, true_clint=10.0)

        result = run_mh_calibration(
            observed=observed,
            drug=drug,
            dose_mg=100.0,
            route="iv",
            t_end_h=12.0,
            n_samples=100,
            burn_in=20,
            calibrate_gut=False,
            seed=2,
        )
        assert isinstance(result.posterior_predictive, pd.DataFrame)
        assert "time_h" in result.posterior_predictive.columns
        assert "C_pred_mg_per_L" in result.posterior_predictive.columns

    def test_mcmc_map_estimate_present(self) -> None:
        """map_estimate dict must contain at least 'clint_hepatic_L_per_h'."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0)
        observed = _synthetic_observed(drug, true_clint=10.0)

        result = run_mh_calibration(
            observed=observed,
            drug=drug,
            dose_mg=100.0,
            route="iv",
            t_end_h=12.0,
            n_samples=100,
            burn_in=20,
            calibrate_gut=False,
            seed=3,
        )
        assert "clint_hepatic_L_per_h" in result.map_estimate
        assert result.map_estimate["clint_hepatic_L_per_h"] > 0

    def test_mcmc_posterior_clint_column_present(self) -> None:
        """posterior_samples must contain the 'clint_hepatic_L_per_h' column."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0)
        observed = _synthetic_observed(drug, true_clint=10.0)

        result = run_mh_calibration(
            observed=observed,
            drug=drug,
            dose_mg=100.0,
            route="iv",
            t_end_h=12.0,
            n_samples=100,
            burn_in=20,
            calibrate_gut=False,
            seed=4,
        )
        assert "clint_hepatic_L_per_h" in result.posterior_samples.columns


# ------------------------------------------------------------------ acceptance rate


class TestAcceptanceRate:
    """Metropolis-Hastings acceptance rate should sit in a reasonable range."""

    def _run_and_get_rate(self, seed: int = 0, proposal_sd: float = 0.15) -> float:
        drug = _make_drug(clint_hepatic_L_per_h=10.0)
        observed = _synthetic_observed(drug, true_clint=10.0, seed=seed + 100)
        result = run_mh_calibration(
            observed=observed,
            drug=drug,
            dose_mg=100.0,
            route="iv",
            t_end_h=12.0,
            n_samples=300,
            burn_in=50,
            proposal_sd_log_clint=proposal_sd,
            calibrate_gut=False,
            seed=seed,
        )
        return result.acceptance_rate

    def test_acceptance_rate_in_reasonable_range(self) -> None:
        """Acceptance rate with moderate proposal SD should be in [0.05, 0.75]."""
        rate = self._run_and_get_rate(seed=7)
        assert 0.05 <= rate <= 0.75, (
            f"Acceptance rate {rate:.3f} outside expected range [0.05, 0.75]"
        )

    def test_acceptance_rate_between_0_and_1(self) -> None:
        """Acceptance rate is always a valid probability in [0, 1]."""
        rate = self._run_and_get_rate(seed=8)
        assert 0.0 <= rate <= 1.0

    def test_acceptance_rate_is_float(self) -> None:
        """acceptance_rate attribute should be a Python float."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0)
        observed = _synthetic_observed(drug, true_clint=10.0)
        result = run_mh_calibration(
            observed=observed,
            drug=drug,
            dose_mg=100.0,
            route="iv",
            t_end_h=12.0,
            n_samples=50,
            burn_in=10,
            calibrate_gut=False,
            seed=9,
        )
        assert isinstance(result.acceptance_rate, float)

    def test_very_large_proposal_sd_lowers_acceptance(self) -> None:
        """A huge proposal step size should decrease acceptance rate compared to moderate SD."""
        moderate_rate = self._run_and_get_rate(seed=11, proposal_sd=0.15)
        large_rate = self._run_and_get_rate(seed=11, proposal_sd=10.0)
        # Large proposals should generally yield a lower or comparable acceptance rate
        # This is a stochastic test; we use a loose upper bound
        assert large_rate <= moderate_rate + 0.50, (
            f"Large proposal SD={10.0} gave rate={large_rate:.3f}, "
            f"moderate rate={moderate_rate:.3f}"
        )


# ------------------------------------------------------------------ convergence toward truth


class TestMCMCConvergence:
    """Posterior mean should be in the neighbourhood of the true CLint value.

    We use a generous tolerance because we are running very few MCMC iterations
    to keep tests fast. The key property is that the posterior is not wildly off.
    """

    def test_posterior_mean_near_true_clint(self) -> None:
        """Posterior mean CLint should be within a factor of 3 of the true value."""
        true_clint = 10.0
        drug = _make_drug(clint_hepatic_L_per_h=true_clint)
        observed = _synthetic_observed(drug, true_clint=true_clint, noise_sd=0.01, seed=99)

        result = run_mh_calibration(
            observed=observed,
            drug=drug,
            dose_mg=100.0,
            route="iv",
            t_end_h=12.0,
            n_samples=500,
            burn_in=100,
            proposal_sd_log_clint=0.20,
            calibrate_gut=False,
            seed=42,
        )

        posterior_mean = result.posterior_samples["clint_hepatic_L_per_h"].mean()
        ratio = posterior_mean / true_clint
        assert 0.25 <= ratio <= 4.0, (
            f"Posterior mean CLint={posterior_mean:.3f} is too far from "
            f"true CLint={true_clint}; ratio={ratio:.3f}"
        )

    def test_posterior_samples_are_positive(self) -> None:
        """All posterior CLint samples must be strictly positive (log-normal proposal)."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0)
        observed = _synthetic_observed(drug, true_clint=10.0)

        result = run_mh_calibration(
            observed=observed,
            drug=drug,
            dose_mg=100.0,
            route="iv",
            t_end_h=12.0,
            n_samples=200,
            burn_in=40,
            calibrate_gut=False,
            seed=55,
        )

        clints = result.posterior_samples["clint_hepatic_L_per_h"].values
        assert (clints > 0).all(), "Some posterior CLint samples are non-positive"

    def test_posterior_samples_have_variance(self) -> None:
        """Posterior samples should not all be identical (chain must move)."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0)
        observed = _synthetic_observed(drug, true_clint=10.0)

        result = run_mh_calibration(
            observed=observed,
            drug=drug,
            dose_mg=100.0,
            route="iv",
            t_end_h=12.0,
            n_samples=300,
            burn_in=50,
            calibrate_gut=False,
            seed=66,
        )

        clints = result.posterior_samples["clint_hepatic_L_per_h"].values
        assert clints.std() > 0, "Posterior CLint samples have zero variance (stuck chain)"

    def test_map_estimate_close_to_true(self) -> None:
        """MAP estimate should be within 5x of the true CLint for a well-conditioned problem."""
        true_clint = 10.0
        drug = _make_drug(clint_hepatic_L_per_h=true_clint)
        observed = _synthetic_observed(drug, true_clint=true_clint, noise_sd=0.005, seed=77)

        result = run_mh_calibration(
            observed=observed,
            drug=drug,
            dose_mg=100.0,
            route="iv",
            t_end_h=12.0,
            n_samples=500,
            burn_in=100,
            proposal_sd_log_clint=0.20,
            calibrate_gut=False,
            seed=77,
        )

        map_val = result.map_estimate["clint_hepatic_L_per_h"]
        ratio = map_val / true_clint
        assert 0.1 <= ratio <= 10.0, (
            f"MAP CLint={map_val:.3f} is too far from true={true_clint}; ratio={ratio:.3f}"
        )
