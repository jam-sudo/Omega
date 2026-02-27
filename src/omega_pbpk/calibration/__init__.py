"""Bayesian parameter calibration via Metropolis-Hastings MCMC.

Fits PBPK model parameters (CLint, ka-related absorption scaling, Kp) to
observed clinical concentration-time data. Uses log-normal priors and
Gaussian likelihood on plasma concentrations.

Adapted for the v0.9 35-state WholeBodyPBPK engine.

Convergence diagnostics
-----------------------
Two diagnostics are computed from the post-burn-in chain and reported in
``CalibrationResult.convergence``:

* **Effective Sample Size (ESS)** — Estimates the number of independent draws
  equivalent to the correlated MCMC chain using the monotone positive
  autocorrelation sequence estimator (Geyer 1992).  A rule of thumb: ESS ≥ 100
  per parameter is considered adequate for posterior summaries.

* **Split-chain R̂ (Rhat)** — The single post-burn-in chain is split into two
  equal halves and R̂ is computed from the resulting two pseudo-chains
  (Vehtari et al. 2021, *Bayesian Analysis* 16(2):667-718).  R̂ < 1.05 is
  conventionally treated as evidence of convergence; R̂ > 1.1 indicates
  potential non-convergence.

References:
    Geyer CJ. Markov chain Monte Carlo maximum likelihood. 1992.
    Vehtari A et al. Bayesian Analysis. 2021;16(2):667-718.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from omega_pbpk.core.body import WholeBodyPBPK
from omega_pbpk.drugs.drug import Drug


def _ess_single(chain: NDArray[np.float64]) -> float:
    """Estimate ESS for a 1-D chain using Geyer's monotone positive sequence.

    Args:
        chain: 1-D array of posterior samples.

    Returns:
        ESS estimate (capped at len(chain)).
    """
    n = len(chain)
    if n < 4:
        return float(n)

    # Normalised autocorrelation via FFT
    c = chain - chain.mean()
    # Use next power-of-two for efficiency
    n_fft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(c, n=n_fft)
    acf_raw = np.fft.irfft(f * np.conj(f))[:n]
    acf = acf_raw / acf_raw[0]  # normalise so acf[0] = 1

    # Geyer's monotone positive sequence estimator
    # Sum consecutive pairs (Gamma_k = acf[2k] + acf[2k+1]) until non-positive
    rho_sum = 1.0  # lag 0 contributes 1
    k = 1
    prev_gamma = float("inf")
    while k + 1 < n:
        gamma = acf[k] + acf[k + 1]
        if gamma <= 0.0 or gamma > prev_gamma:
            break
        rho_sum += 2.0 * gamma
        prev_gamma = gamma
        k += 2

    ess = n / max(rho_sum, 1e-12)
    return float(min(ess, float(n)))


def _rhat_split(chain: NDArray[np.float64]) -> float:
    """Compute split-chain R̂ for a 1-D chain.

    Splits the chain into two equal halves (discarding the last sample if
    odd-length) and applies the standard between/within-chain variance
    estimator of Gelman & Rubin (1992) as updated in Vehtari et al. (2021).

    Args:
        chain: 1-D array of posterior samples (post-burn-in).

    Returns:
        R̂ value.  Returns NaN if the chain has < 4 samples.
    """
    n = len(chain)
    if n < 4:
        return float("nan")

    half = n // 2
    chains = np.stack([chain[:half], chain[n - half :]])  # shape (2, half)
    m, n_each = chains.shape

    chain_means = chains.mean(axis=1)          # (m,)
    grand_mean = chain_means.mean()

    # Between-chain variance B (scaled by n_each)
    B = n_each * np.var(chain_means, ddof=1)

    # Within-chain variance W
    W = np.mean(np.var(chains, axis=1, ddof=1))

    if W < 1e-15:
        return 1.0

    var_hat = (1.0 - 1.0 / n_each) * W + B / n_each
    return float(np.sqrt(var_hat / W))


def compute_convergence_diagnostics(
    posterior_samples: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Compute ESS and split-chain R̂ for each parameter column.

    Args:
        posterior_samples: DataFrame of post-burn-in samples (rows = draws,
            columns = parameters).  Columns named 'iter' and 'log_posterior'
            are excluded from parameter-level diagnostics.

    Returns:
        Nested dict ``{param: {"ess": float, "rhat": float}}``.
    """
    skip = {"iter", "log_posterior"}
    diagnostics: dict[str, dict[str, float]] = {}
    for col in posterior_samples.columns:
        if col in skip:
            continue
        chain = posterior_samples[col].to_numpy(dtype=np.float64)
        diagnostics[col] = {
            "ess": _ess_single(chain),
            "rhat": _rhat_split(chain),
        }
    return diagnostics


@dataclass(frozen=True)
class CalibrationResult:
    """Results from Bayesian MCMC calibration.

    Attributes:
        posterior_samples: DataFrame of accepted parameter samples after burn-in.
        posterior_predictive: Predicted concentration-time from the last accepted sample.
        acceptance_rate: Fraction of proposals accepted over all iterations.
        map_estimate: Maximum a-posteriori parameter values.
        convergence: Per-parameter convergence diagnostics with keys 'ess'
            (Effective Sample Size) and 'rhat' (split-chain R̂).
            Rule of thumb: ESS ≥ 100 and rhat < 1.05 indicate adequate convergence.
    """

    posterior_samples: pd.DataFrame
    posterior_predictive: pd.DataFrame
    acceptance_rate: float
    map_estimate: dict[str, float] = field(default_factory=dict)
    convergence: dict[str, dict[str, float]] = field(default_factory=dict)


def _log_likelihood(
    observed_time_h: NDArray[np.float64],
    observed_conc: NDArray[np.float64],
    predicted_time_h: NDArray[np.float64],
    predicted_conc: NDArray[np.float64],
    sigma: float,
) -> float:
    """Gaussian log-likelihood with interpolation of predicted on observed times."""
    predicted_on_obs = np.interp(observed_time_h, predicted_time_h, predicted_conc)
    residual = observed_conc - predicted_on_obs
    n = float(residual.size)
    norm_term = -0.5 * n * np.log(2.0 * np.pi * sigma * sigma)
    sq_term = -0.5 * np.sum((residual / sigma) ** 2)
    return float(norm_term + sq_term)


def _log_prior_lognormal(value: float, mean_log: float, sd_log: float) -> float:
    """Log-normal prior density in log space."""
    if value <= 0.0:
        return float("-inf")
    term = (np.log(value) - mean_log) / sd_log
    return float(-np.log(value * sd_log * np.sqrt(2.0 * np.pi)) - 0.5 * term * term)


def _simulate_concentration(
    drug: Drug,
    body_weight: float,
    dose_mg: float,
    route: str,
    t_end_h: float,
    clint_hepatic_L_per_h: float,
    clint_gut_L_per_h: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run a single simulation with modified clearance parameters.

    Returns (time_array, plasma_concentration_array).
    """
    # Build modified drug with sampled parameters
    drug_dict = {**drug.__dict__}
    drug_dict["clint_hepatic_L_per_h"] = clint_hepatic_L_per_h
    if clint_gut_L_per_h is not None:
        drug_dict["clint_gut_L_per_h"] = clint_gut_L_per_h
    sampled_drug = Drug(**drug_dict)

    model = WholeBodyPBPK(sampled_drug, body_weight=body_weight)
    if route == "iv":
        model.setup_iv(dose_mg)
    else:
        model.setup_oral(dose_mg)

    result = model.simulate(t_end_h=t_end_h)
    return result.time_h.astype(np.float64), result.plasma_concentration().astype(np.float64)


def run_mh_calibration(
    observed: pd.DataFrame,
    drug: Drug,
    dose_mg: float,
    route: str = "oral",
    body_weight: float = 70.0,
    t_end_h: float = 24.0,
    n_samples: int = 2000,
    burn_in: int = 500,
    proposal_sd_log_clint: float = 0.15,
    proposal_sd_log_gut: float = 0.15,
    observation_sigma: float = 0.1,
    calibrate_gut: bool = True,
    seed: int | None = None,
) -> CalibrationResult:
    """Run Metropolis-Hastings MCMC calibration against observed data.

    The observed DataFrame must have columns 'time_h' and 'C_plasma_mg_per_L'.

    Calibrates:
      - clint_hepatic_L_per_h (hepatic intrinsic clearance, in vivo)
      - clint_gut_L_per_h (gut intrinsic clearance, in vivo) if calibrate_gut=True

    Args:
        observed: DataFrame with 'time_h' and 'C_plasma_mg_per_L' columns.
        drug: Drug compound to calibrate.
        dose_mg: Administered dose (mg).
        route: 'oral' or 'iv'.
        body_weight: Subject body weight (kg).
        t_end_h: Simulation end time (h).
        n_samples: Total MCMC iterations (including burn-in).
        burn_in: Number of initial samples to discard.
        proposal_sd_log_clint: Proposal standard deviation in log-space for CLint.
        proposal_sd_log_gut: Proposal standard deviation in log-space for gut CLint.
        observation_sigma: Assumed observation noise SD (mg/L).
        calibrate_gut: Whether to also calibrate gut CLint.
        seed: Random seed for reproducibility.

    Returns:
        CalibrationResult with posterior samples and diagnostics.
    """
    rng = np.random.default_rng(seed)

    obs_time = observed["time_h"].to_numpy(dtype=np.float64)
    obs_conc = observed["C_plasma_mg_per_L"].to_numpy(dtype=np.float64)

    # Initial values from drug
    current_clint = max(drug.clint_scaled_L_per_h, 1.0)
    current_gut = max(drug.gut_clint_scaled_L_per_h, 1.0) if calibrate_gut else None

    # Prior means in log-space (centered on initial IVIVE estimates)
    prior_mean_log_clint = np.log(current_clint)
    prior_mean_log_gut = np.log(current_gut) if current_gut else 0.0

    # Initial simulation
    pred_time, pred_conc = _simulate_concentration(
        drug, body_weight, dose_mg, route, t_end_h, current_clint, current_gut
    )

    current_log_post = _log_likelihood(obs_time, obs_conc, pred_time, pred_conc, observation_sigma)
    current_log_post += _log_prior_lognormal(current_clint, prior_mean_log_clint, 1.0)
    if calibrate_gut and current_gut is not None:
        current_log_post += _log_prior_lognormal(current_gut, prior_mean_log_gut, 1.0)

    kept_rows: list[dict[str, float]] = []
    accepted = 0
    best_log_post = current_log_post
    best_params: dict[str, float] = {
        "clint_hepatic_L_per_h": current_clint,
    }
    if calibrate_gut:
        best_params["clint_gut_L_per_h"] = current_gut or 0.0

    for i in range(n_samples):
        # Propose new parameters in log-space
        proposal_clint = float(
            np.exp(np.log(current_clint) + rng.normal(0.0, proposal_sd_log_clint))
        )

        proposal_gut = None
        if calibrate_gut and current_gut is not None:
            proposal_gut = float(np.exp(np.log(current_gut) + rng.normal(0.0, proposal_sd_log_gut)))

        try:
            prop_time, prop_conc = _simulate_concentration(
                drug, body_weight, dose_mg, route, t_end_h, proposal_clint, proposal_gut
            )
        except Exception:
            # Simulation failed — reject proposal
            continue

        proposal_log_post = _log_likelihood(
            obs_time, obs_conc, prop_time, prop_conc, observation_sigma
        )
        proposal_log_post += _log_prior_lognormal(proposal_clint, prior_mean_log_clint, 1.0)
        if calibrate_gut and proposal_gut is not None:
            proposal_log_post += _log_prior_lognormal(proposal_gut, prior_mean_log_gut, 1.0)

        # Metropolis acceptance criterion
        log_alpha = proposal_log_post - current_log_post
        if np.log(rng.uniform()) < log_alpha:
            current_clint = proposal_clint
            current_gut = proposal_gut
            current_log_post = proposal_log_post
            pred_time, pred_conc = prop_time, prop_conc
            accepted += 1

            if current_log_post > best_log_post:
                best_log_post = current_log_post
                best_params = {"clint_hepatic_L_per_h": current_clint}
                if calibrate_gut and current_gut is not None:
                    best_params["clint_gut_L_per_h"] = current_gut

        if i >= burn_in:
            row: dict[str, float] = {
                "iter": float(i),
                "clint_hepatic_L_per_h": current_clint,
                "log_posterior": current_log_post,
            }
            if calibrate_gut and current_gut is not None:
                row["clint_gut_L_per_h"] = current_gut
            kept_rows.append(row)

    posterior_samples = pd.DataFrame(kept_rows)
    posterior_predictive = pd.DataFrame(
        {
            "time_h": pred_time,
            "C_pred_mg_per_L": pred_conc,
        }
    )
    acceptance_rate = accepted / max(n_samples, 1)
    convergence = compute_convergence_diagnostics(posterior_samples)

    return CalibrationResult(
        posterior_samples=posterior_samples,
        posterior_predictive=posterior_predictive,
        acceptance_rate=acceptance_rate,
        map_estimate=best_params,
        convergence=convergence,
    )


__all__ = [
    "CalibrationResult",
    "run_mh_calibration",
    "compute_convergence_diagnostics",
]
