"""PK model comparison — fit 1-cpt and 2-cpt models, compare via AIC/BIC.

Uses grid search to find best-fit parameters, then computes information criteria
for model selection. Lower AIC/BIC indicates a better model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PKModelFitResult:
    model_name: str
    n_parameters: int
    n_observations: int
    rss: float  # residual sum of squares
    rmse: float  # root mean squared error
    r_squared: float  # coefficient of determination
    aic: float  # Akaike Information Criterion
    bic: float  # Bayesian Information Criterion
    akaike_weight: float  # relative probability of model being best (0-1)
    delta_aic: float  # AIC - min_AIC across models
    selected: bool  # True for model with lowest AIC
    notes: str


def _linspace(start: float, stop: float, n: int) -> list:
    """Return n evenly spaced values from start to stop inclusive."""
    if n == 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def _validate_inputs(
    times_h: list,
    concs_mg_L: list,
    dose_mg: float,
) -> None:
    if len(times_h) < 3:
        raise ValueError("At least 3 observations required for model fitting")
    if len(times_h) != len(concs_mg_L):
        raise ValueError("times_h and concs_mg_L must have the same length")
    if any(c < 0 for c in concs_mg_L):
        raise ValueError("All concentrations must be >= 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    for i in range(1, len(times_h)):
        if times_h[i] <= times_h[i - 1]:
            raise ValueError("times_h must be strictly monotonically increasing")


def _compute_metrics(
    obs: list,
    pred: list,
    n_params: int,
) -> tuple:
    """Return (rss, rmse, r_squared, aic, bic)."""
    n = len(obs)
    rss = sum((o - p) ** 2 for o, p in zip(obs, pred))
    rmse = math.sqrt(rss / n)

    mean_obs = sum(obs) / n
    ss_tot = sum((o - mean_obs) ** 2 for o in obs)
    r_squared = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0

    # AIC/BIC using log-likelihood approximation for normally distributed residuals
    if rss <= 0:
        # Perfect fit — use very small rss to avoid log(0)
        rss_for_ic = 1e-12
    else:
        rss_for_ic = rss

    aic = n * math.log(rss_for_ic / n) + 2 * n_params
    bic = n * math.log(rss_for_ic / n) + n_params * math.log(n)

    return rss, rmse, r_squared, aic, bic


def fit_one_compartment(
    times_h: list,
    concs_mg_L: list,
    dose_mg: float,
    route: str = "iv",
) -> PKModelFitResult:
    """Fit a one-compartment PK model to concentration-time data.

    For IV: C(t) = C0 * exp(-ke*t), 2 parameters: C0, ke
    For oral: C(t) = F*D*ka/(Vd*(ka-ke)) * [exp(-ke*t) - exp(-ka*t)], 3 parameters: ke, ka, F
    Uses grid search to minimize RSS.
    """
    _validate_inputs(times_h, concs_mg_L, dose_mg)

    if route not in ("iv", "oral"):
        raise ValueError("route must be 'iv' or 'oral'")

    obs = concs_mg_L
    n = len(obs)
    c0_ref = obs[0] if obs[0] > 0 else dose_mg / 50.0  # fallback estimate

    best_rss = math.inf
    best_pred = None

    if route == "iv":
        # 2 parameters: ke, C0
        n_params = 2
        ke_vals = _linspace(0.01, 2.0, 50)
        c0_vals = _linspace(max(c0_ref * 0.5, 0.01), c0_ref * 2.0, 10)

        for ke in ke_vals:
            for c0 in c0_vals:
                pred = [c0 * math.exp(-ke * t) for t in times_h]
                rss = sum((o - p) ** 2 for o, p in zip(obs, pred))
                if rss < best_rss:
                    best_rss = rss
                    best_pred = pred

    else:
        # 3 parameters: ke, ka, F (Vd estimated from cmax timing)
        n_params = 3
        # Estimate Vd roughly from dose / max_conc
        cmax_approx = max(obs)
        vd_est = dose_mg / cmax_approx if cmax_approx > 0 else 50.0

        ke_vals = _linspace(0.01, 2.0, 15)
        ka_vals = _linspace(0.1, 5.0, 15)
        f_vals = [0.5, 0.8, 1.0]

        for ke in ke_vals:
            for ka in ka_vals:
                if abs(ka - ke) < 1e-6:
                    continue
                for f in f_vals:
                    coeff = f * dose_mg * ka / (vd_est * (ka - ke))
                    pred = [coeff * (math.exp(-ke * t) - math.exp(-ka * t)) for t in times_h]
                    # Clamp negatives (can occur at t=0 if ka>ke)
                    pred = [max(0.0, p) for p in pred]
                    rss = sum((o - p) ** 2 for o, p in zip(obs, pred))
                    if rss < best_rss:
                        best_rss = rss
                        best_pred = pred

    if best_pred is None:
        best_pred = [0.0] * n
        best_rss = sum(o**2 for o in obs)

    rss, rmse, r_squared, aic, bic = _compute_metrics(obs, best_pred, n_params)

    notes = (
        f"One-compartment {route} model. "
        f"Grid search over {'ke, C0' if route == 'iv' else 'ke, ka, F'}. "
        f"Best RSS={rss:.4f}, R²={r_squared:.3f}."
    )

    return PKModelFitResult(
        model_name=f"1-cpt-{route}",
        n_parameters=n_params,
        n_observations=n,
        rss=rss,
        rmse=rmse,
        r_squared=r_squared,
        aic=aic,
        bic=bic,
        akaike_weight=0.0,  # set by compare_pk_models
        delta_aic=0.0,  # set by compare_pk_models
        selected=False,  # set by compare_pk_models
        notes=notes,
    )


def fit_two_compartment(
    times_h: list,
    concs_mg_L: list,
    dose_mg: float,
) -> PKModelFitResult:
    """Fit a two-compartment biexponential IV model.

    C(t) = A * exp(-alpha*t) + B * exp(-beta*t), 4 parameters: A, alpha, B, beta
    Constraint: A + B = C(0) ~ obs[0]
    """
    _validate_inputs(times_h, concs_mg_L, dose_mg)

    obs = concs_mg_L
    n = len(obs)
    c0_ref = obs[0] if obs[0] > 0 else dose_mg / 50.0

    n_params = 4  # A, alpha, B, beta

    alpha_vals = _linspace(0.1, 5.0, 10)
    beta_vals = _linspace(0.01, 1.0, 10)
    # A ranges from c0_ref/2 to 2*c0_ref (5 points)
    a_low = max(c0_ref * 0.1, 0.01)
    a_high = max(c0_ref * 2.0, 0.1)
    a_vals = _linspace(a_low, a_high, 5)

    best_rss = math.inf
    best_pred = None

    for alpha in alpha_vals:
        for beta in beta_vals:
            if beta >= alpha:
                continue  # physical constraint: alpha > beta
            for a_coeff in a_vals:
                b_coeff = c0_ref - a_coeff
                # B can be negative in theory, but we skip those cases
                if b_coeff < 0:
                    continue
                pred = [
                    a_coeff * math.exp(-alpha * t) + b_coeff * math.exp(-beta * t) for t in times_h
                ]
                rss = sum((o - p) ** 2 for o, p in zip(obs, pred))
                if rss < best_rss:
                    best_rss = rss
                    best_pred = pred

    if best_pred is None:
        best_pred = [c0_ref * math.exp(-0.1 * t) for t in times_h]
        best_rss = sum((o - p) ** 2 for o, p in zip(obs, best_pred))

    rss, rmse, r_squared, aic, bic = _compute_metrics(obs, best_pred, n_params)

    notes = (
        f"Two-compartment biexponential IV model. "
        f"Grid search over alpha, beta, A (B = C0 - A). "
        f"Best RSS={rss:.4f}, R²={r_squared:.3f}."
    )

    return PKModelFitResult(
        model_name="2-cpt-iv",
        n_parameters=n_params,
        n_observations=n,
        rss=rss,
        rmse=rmse,
        r_squared=r_squared,
        aic=aic,
        bic=bic,
        akaike_weight=0.0,  # set by compare_pk_models
        delta_aic=0.0,  # set by compare_pk_models
        selected=False,  # set by compare_pk_models
        notes=notes,
    )


def compare_pk_models(
    times_h: list,
    concs_mg_L: list,
    dose_mg: float,
    route: str = "iv",
) -> list:
    """Fit 1-cpt and 2-cpt models, compute Akaike weights, return sorted by AIC ascending.

    Sets delta_aic and selected flags on returned results.
    """
    _validate_inputs(times_h, concs_mg_L, dose_mg)

    result_1cpt = fit_one_compartment(times_h, concs_mg_L, dose_mg, route=route)
    result_2cpt = fit_two_compartment(times_h, concs_mg_L, dose_mg)

    results_raw = [result_1cpt, result_2cpt]
    aic_vals = [r.aic for r in results_raw]
    min_aic = min(aic_vals)

    delta_aics = [a - min_aic for a in aic_vals]

    # Akaike weights: w_i = exp(-0.5 * delta_i) / sum(exp(-0.5 * delta_j))
    raw_weights = [math.exp(-0.5 * d) for d in delta_aics]
    total_weight = sum(raw_weights)
    akaike_weights = [w / total_weight for w in raw_weights]

    # Best model (lowest AIC)
    best_idx = aic_vals.index(min_aic)

    final_results = []
    for i, r in enumerate(results_raw):
        final_results.append(
            PKModelFitResult(
                model_name=r.model_name,
                n_parameters=r.n_parameters,
                n_observations=r.n_observations,
                rss=r.rss,
                rmse=r.rmse,
                r_squared=r.r_squared,
                aic=r.aic,
                bic=r.bic,
                akaike_weight=akaike_weights[i],
                delta_aic=delta_aics[i],
                selected=(i == best_idx),
                notes=r.notes,
            )
        )

    # Sort by AIC ascending
    final_results.sort(key=lambda r: r.aic)
    return final_results
