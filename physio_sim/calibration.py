from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from physio_sim.config import CompoundConfig, SubjectConfig
from physio_sim.pbpk.solver import simulate


@dataclass(frozen=True)
class CalibrationResult:
    posterior_samples: pd.DataFrame
    posterior_predictive: pd.DataFrame
    acceptance_rate: float


def _log_likelihood(
    observed_time_h: NDArray[np.float64],
    observed_conc: NDArray[np.float64],
    predicted_time_h: NDArray[np.float64],
    predicted_conc: NDArray[np.float64],
    sigma: float,
) -> float:
    predicted_on_obs = np.interp(observed_time_h, predicted_time_h, predicted_conc)
    residual = observed_conc - predicted_on_obs
    n = float(residual.size)
    norm_term = -0.5 * n * np.log(2.0 * np.pi * sigma * sigma)
    sq_term = -0.5 * np.sum((residual / sigma) ** 2)
    return float(norm_term + sq_term)


def _log_prior_lognormal(value: float, mean_log: float, sd_log: float) -> float:
    if value <= 0.0:
        return float("-inf")
    term = (np.log(value) - mean_log) / sd_log
    return float(-np.log(value * sd_log * np.sqrt(2.0 * np.pi)) - 0.5 * term * term)


def _simulate_concentration(
    subject: SubjectConfig,
    compound: CompoundConfig,
    dose_mg: float,
    route: str,
    t_end_h: float,
    dt_out_h: float,
    clint: float,
    ka: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    sampled_compound = compound.model_copy(update={"clint_L_per_h": clint, "ka_per_h": ka})
    timecourse = simulate(
        subject=subject,
        compound=sampled_compound,
        dose_mg=dose_mg,
        route=route,
        t_end_h=t_end_h,
        dt_out_h=dt_out_h,
    ).timecourse
    return timecourse["time_h"].to_numpy(), timecourse["C_plasma_mg_per_L"].to_numpy()


def run_mh_calibration(
    observed: pd.DataFrame,
    subject: SubjectConfig,
    compound: CompoundConfig,
    dose_mg: float,
    route: str,
    t_end_h: float,
    dt_out_h: float,
    n_samples: int = 2000,
    burn_in: int = 500,
    proposal_sd_log_clint: float = 0.15,
    proposal_sd_log_ka: float = 0.15,
    observation_sigma: float = 0.1,
    seed: int | None = None,
) -> CalibrationResult:
    rng = np.random.default_rng(seed)
    obs_time = observed["time_h"].to_numpy(dtype=float)
    obs_conc = observed["C_plasma_mg_per_L"].to_numpy(dtype=float)

    current_clint = max(compound.clint_L_per_h, 1e-6)
    current_ka = max(compound.ka_per_h, 1e-6)

    pred_time, pred_conc = _simulate_concentration(
        subject, compound, dose_mg, route, t_end_h, dt_out_h, current_clint, current_ka
    )
    current_log_post = _log_likelihood(obs_time, obs_conc, pred_time, pred_conc, observation_sigma)
    current_log_post += _log_prior_lognormal(current_clint, np.log(compound.clint_L_per_h), 1.0)
    current_log_post += _log_prior_lognormal(current_ka, np.log(compound.ka_per_h), 1.0)

    kept_rows: list[dict[str, float]] = []
    accepted = 0

    for i in range(n_samples):
        proposal_clint = float(
            np.exp(np.log(current_clint) + rng.normal(0.0, proposal_sd_log_clint))
        )
        proposal_ka = float(np.exp(np.log(current_ka) + rng.normal(0.0, proposal_sd_log_ka)))

        prop_time, prop_conc = _simulate_concentration(
            subject, compound, dose_mg, route, t_end_h, dt_out_h, proposal_clint, proposal_ka
        )

        proposal_log_post = _log_likelihood(
            obs_time, obs_conc, prop_time, prop_conc, observation_sigma
        )
        proposal_log_post += _log_prior_lognormal(
            proposal_clint, np.log(compound.clint_L_per_h), 1.0
        )
        proposal_log_post += _log_prior_lognormal(proposal_ka, np.log(compound.ka_per_h), 1.0)

        accept_prob = min(1.0, float(np.exp(proposal_log_post - current_log_post)))
        if rng.uniform() < accept_prob:
            current_clint = proposal_clint
            current_ka = proposal_ka
            current_log_post = proposal_log_post
            pred_time, pred_conc = prop_time, prop_conc
            accepted += 1

        if i >= burn_in:
            kept_rows.append(
                {
                    "iter": float(i),
                    "CLint_L_per_h": current_clint,
                    "ka_per_h": current_ka,
                    "log_posterior": current_log_post,
                }
            )

    posterior_samples = pd.DataFrame(kept_rows)
    posterior_predictive = pd.DataFrame({"time_h": pred_time, "C_pred_mg_per_L": pred_conc})
    acceptance_rate = accepted / float(n_samples)
    return CalibrationResult(
        posterior_samples=posterior_samples,
        posterior_predictive=posterior_predictive,
        acceptance_rate=acceptance_rate,
    )
