"""Therapeutic Drug Monitoring (TDM) with MAP-Bayesian dose individualization.

Estimates individual PK parameters (CL, Vd, ka) from observed drug
concentrations using Maximum A Posteriori (MAP) Bayesian estimation,
then recommends an individualized dose to achieve target trough/peak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize, minimize_scalar

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservedConcentration:
    """Single TDM observation."""

    time_h: float
    conc_mg_L: float
    dose_number: int = 1
    assay_cv: float = 0.10


@dataclass(frozen=True)
class PopulationPrior:
    """Population PK prior for MAP estimation (log-normal)."""

    cl_mean: float
    cl_cv: float = 0.40
    vd_mean: float = 42.0
    vd_cv: float = 0.30
    ka_mean: float = 1.0
    ka_cv: float = 0.50

    @property
    def omega_cl(self) -> float:
        """Log-normal variance for CL."""
        return np.log(1 + self.cl_cv**2)

    @property
    def omega_vd(self) -> float:
        """Log-normal variance for Vd."""
        return np.log(1 + self.vd_cv**2)

    @property
    def omega_ka(self) -> float:
        """Log-normal variance for ka."""
        return np.log(1 + self.ka_cv**2)


@dataclass(frozen=True)
class TDMResult:
    """Result of TDM Bayesian estimation and dose recommendation."""

    individual_cl: float
    individual_vd: float
    individual_ka: float
    cl_ratio: float
    vd_ratio: float
    ka_ratio: float
    predicted: tuple[float, ...]
    observed: tuple[float, ...]
    residuals: tuple[float, ...]
    objective_function_value: float
    recommended_dose_mg: float
    predicted_trough: float
    predicted_peak: float
    forward_time_h: tuple[float, ...]
    forward_conc_mg_L: tuple[float, ...]
    converged: bool
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1-compartment analytical model (multi-dose superposition)
# ---------------------------------------------------------------------------


def _simulate_individualized(
    cl: float,
    vd: float,
    ka: float,
    dose_mg: float,
    interval_h: float,
    times_h: np.ndarray,
    route: str = "oral",
    n_prev_doses: int = 20,
) -> np.ndarray:
    """Simulate concentration profile using 1-compartment analytical model.

    Uses multi-dose superposition for steady-state approximation.

    Oral: C(t) = (F*D*ka)/(Vd*(ka-ke)) * (exp(-ke*t) - exp(-ka*t))
    IV:   C(t) = (D/Vd) * exp(-ke*t)
    """
    ke = cl / vd
    conc = np.zeros_like(times_h, dtype=float)

    for dose_idx in range(n_prev_doses):
        t_since_dose = times_h - dose_idx * interval_h
        mask = t_since_dose > 0
        t_pos = t_since_dose[mask]

        if route == "iv":
            conc[mask] += (dose_mg / vd) * np.exp(-ke * t_pos)
        else:
            # Oral with first-order absorption
            if abs(ka - ke) < 1e-8:
                # Special case: ka ≈ ke
                conc[mask] += (dose_mg / vd) * ka * t_pos * np.exp(-ke * t_pos)
            else:
                conc[mask] += (
                    (dose_mg * ka) / (vd * (ka - ke)) * (np.exp(-ke * t_pos) - np.exp(-ka * t_pos))
                )

    return np.maximum(conc, 0.0)


# ---------------------------------------------------------------------------
# MAP objective
# ---------------------------------------------------------------------------


def _map_objective(
    log_params: np.ndarray,
    observations: list[ObservedConcentration],
    prior: PopulationPrior,
    dose_mg: float,
    interval_h: float,
    route: str,
) -> float:
    """MAP objective: prior penalty (log-normal) + likelihood (weighted LS).

    Parameters are on log scale: [log(CL), log(Vd), log(ka)].
    """
    cl = np.exp(log_params[0])
    vd = np.exp(log_params[1])
    ka = np.exp(log_params[2])

    # Prior penalty (log-normal)
    prior_penalty = (
        (log_params[0] - np.log(prior.cl_mean)) ** 2 / prior.omega_cl
        + (log_params[1] - np.log(prior.vd_mean)) ** 2 / prior.omega_vd
        + (log_params[2] - np.log(prior.ka_mean)) ** 2 / prior.omega_ka
    )

    # Likelihood penalty (weighted least squares)
    likelihood_penalty = 0.0
    for obs in observations:
        # Time relative to dosing: obs at dose_number-th dose
        t_obs = (obs.dose_number - 1) * interval_h + obs.time_h
        times = np.array([t_obs])
        pred = _simulate_individualized(
            cl, vd, ka, dose_mg, interval_h, times, route, n_prev_doses=obs.dose_number
        )
        pred_val = float(pred[0])
        sigma = max(obs.conc_mg_L * obs.assay_cv, 1e-6)
        likelihood_penalty += ((obs.conc_mg_L - pred_val) / sigma) ** 2

    return prior_penalty + likelihood_penalty


# ---------------------------------------------------------------------------
# Parameter estimation
# ---------------------------------------------------------------------------


def estimate_individual_parameters(
    observations: list[ObservedConcentration],
    prior: PopulationPrior,
    dose_mg: float,
    interval_h: float = 12.0,
    route: str = "oral",
) -> dict:
    """Estimate individual CL, Vd, ka via MAP-Bayesian optimization.

    Returns dict with keys: cl, vd, ka, objective, converged.
    """
    x0 = np.array([np.log(prior.cl_mean), np.log(prior.vd_mean), np.log(prior.ka_mean)])

    result = minimize(
        _map_objective,
        x0,
        args=(observations, prior, dose_mg, interval_h, route),
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8},
    )

    cl = float(np.exp(result.x[0]))
    vd = float(np.exp(result.x[1]))
    ka = float(np.exp(result.x[2]))

    return {
        "cl": cl,
        "vd": vd,
        "ka": ka,
        "objective": float(result.fun),
        "converged": bool(result.success),
    }


# ---------------------------------------------------------------------------
# Dose recommendation
# ---------------------------------------------------------------------------


def recommend_dose(
    cl: float,
    vd: float,
    ka: float,
    target_trough: float,
    interval_h: float = 12.0,
    route: str = "oral",
    dose_range_mg: tuple[float, float] = (10.0, 2000.0),
    target_peak: float | None = None,
    peak_penalty_weight: float = 10.0,
) -> dict:
    """Find dose achieving target trough (and optionally constrain peak).

    Returns dict with keys: dose_mg, predicted_trough, predicted_peak.
    """
    n_ss_doses = 30  # enough for steady state

    def objective(dose: float) -> float:
        # Simulate steady-state profile at candidate dose
        t_trough = np.array([interval_h])
        conc_trough = _simulate_individualized(
            cl, vd, ka, dose, interval_h, t_trough, route, n_prev_doses=n_ss_doses
        )
        trough = float(conc_trough[0])

        cost = (trough - target_trough) ** 2

        if target_peak is not None:
            # Find peak in first interval
            t_profile = np.linspace(0.01, interval_h, 100)
            # Shift to last dose time for steady state
            t_shifted = t_profile + (n_ss_doses - 1) * interval_h
            conc_profile = _simulate_individualized(
                cl, vd, ka, dose, interval_h, t_shifted, route, n_prev_doses=n_ss_doses
            )
            peak = float(np.max(conc_profile))
            if peak > target_peak:
                cost += peak_penalty_weight * (peak - target_peak) ** 2

        return cost

    result = minimize_scalar(
        objective,
        bounds=dose_range_mg,
        method="bounded",
    )

    opt_dose = float(result.x)

    # Compute trough and peak at optimal dose
    t_trough = np.array([interval_h])
    conc_trough = _simulate_individualized(
        cl, vd, ka, opt_dose, interval_h, t_trough, route, n_prev_doses=n_ss_doses
    )
    pred_trough = float(conc_trough[0])

    t_profile = np.linspace(0.01, interval_h, 100)
    t_shifted = t_profile + (n_ss_doses - 1) * interval_h
    conc_profile = _simulate_individualized(
        cl, vd, ka, opt_dose, interval_h, t_shifted, route, n_prev_doses=n_ss_doses
    )
    pred_peak = float(np.max(conc_profile))

    return {
        "dose_mg": round(opt_dose, 2),
        "predicted_trough": round(pred_trough, 6),
        "predicted_peak": round(pred_peak, 6),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_tdm(
    observations: list[ObservedConcentration],
    prior: PopulationPrior,
    dose_mg: float,
    interval_h: float = 12.0,
    route: str = "oral",
    target_trough: float = 1.0,
    target_peak: float | None = None,
    dose_range_mg: tuple[float, float] = (10.0, 2000.0),
    forward_hours: float = 72.0,
) -> TDMResult:
    """Run full TDM workflow: estimate parameters, recommend dose, predict forward.

    Args:
        observations: List of observed concentrations.
        prior: Population PK prior.
        dose_mg: Current dose in mg.
        interval_h: Dosing interval in hours.
        route: 'oral' or 'iv'.
        target_trough: Target trough concentration (mg/L).
        target_peak: Optional target peak ceiling (mg/L).
        dose_range_mg: Search bounds for dose recommendation.
        forward_hours: Duration for forward prediction.

    Returns:
        TDMResult with individual parameters, dose recommendation, and prediction.
    """
    warnings: list[str] = []

    if len(observations) < 2:
        warnings.append("Limited identifiability: fewer than 2 observations")
    if len(observations) == 0:
        warnings.append("No observations provided; returning population prior values")
        # Return prior-based result with no individualization
        rec = recommend_dose(
            prior.cl_mean,
            prior.vd_mean,
            prior.ka_mean,
            target_trough,
            interval_h,
            route,
            dose_range_mg,
            target_peak,
        )
        t_fwd = np.linspace(0, forward_hours, int(forward_hours / 0.1) + 1)
        c_fwd = _simulate_individualized(
            prior.cl_mean,
            prior.vd_mean,
            prior.ka_mean,
            rec["dose_mg"],
            interval_h,
            t_fwd,
            route,
            n_prev_doses=30,
        )
        return TDMResult(
            individual_cl=prior.cl_mean,
            individual_vd=prior.vd_mean,
            individual_ka=prior.ka_mean,
            cl_ratio=1.0,
            vd_ratio=1.0,
            ka_ratio=1.0,
            predicted=(),
            observed=(),
            residuals=(),
            objective_function_value=0.0,
            recommended_dose_mg=rec["dose_mg"],
            predicted_trough=rec["predicted_trough"],
            predicted_peak=rec["predicted_peak"],
            forward_time_h=tuple(float(x) for x in t_fwd),
            forward_conc_mg_L=tuple(float(x) for x in c_fwd),
            converged=True,
            warnings=warnings,
        )

    # 1. Estimate individual parameters
    est = estimate_individual_parameters(observations, prior, dose_mg, interval_h, route)
    ind_cl = est["cl"]
    ind_vd = est["vd"]
    ind_ka = est["ka"]

    # 2. Compute predicted vs observed
    pred_vals = []
    obs_vals = []
    for obs in observations:
        t_obs = (obs.dose_number - 1) * interval_h + obs.time_h
        times = np.array([t_obs])
        pred = _simulate_individualized(
            ind_cl, ind_vd, ind_ka, dose_mg, interval_h, times, route, n_prev_doses=obs.dose_number
        )
        pred_vals.append(float(pred[0]))
        obs_vals.append(obs.conc_mg_L)

    residuals = [o - p for o, p in zip(obs_vals, pred_vals, strict=True)]

    # 3. Recommend dose
    rec = recommend_dose(
        ind_cl,
        ind_vd,
        ind_ka,
        target_trough,
        interval_h,
        route,
        dose_range_mg,
        target_peak,
    )

    # 4. Forward prediction at recommended dose
    t_fwd = np.linspace(0, forward_hours, int(forward_hours / 0.1) + 1)
    c_fwd = _simulate_individualized(
        ind_cl,
        ind_vd,
        ind_ka,
        rec["dose_mg"],
        interval_h,
        t_fwd,
        route,
        n_prev_doses=30,
    )

    if not est["converged"]:
        warnings.append("Optimizer did not converge; results may be unreliable")

    return TDMResult(
        individual_cl=round(ind_cl, 4),
        individual_vd=round(ind_vd, 4),
        individual_ka=round(ind_ka, 4),
        cl_ratio=round(ind_cl / prior.cl_mean, 4),
        vd_ratio=round(ind_vd / prior.vd_mean, 4),
        ka_ratio=round(ind_ka / prior.ka_mean, 4),
        predicted=tuple(round(p, 6) for p in pred_vals),
        observed=tuple(obs_vals),
        residuals=tuple(round(r, 6) for r in residuals),
        objective_function_value=round(est["objective"], 6),
        recommended_dose_mg=rec["dose_mg"],
        predicted_trough=rec["predicted_trough"],
        predicted_peak=rec["predicted_peak"],
        forward_time_h=tuple(round(float(x), 2) for x in t_fwd),
        forward_conc_mg_L=tuple(round(float(x), 6) for x in c_fwd),
        converged=est["converged"],
        warnings=warnings,
    )
