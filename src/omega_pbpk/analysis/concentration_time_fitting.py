"""Fit pharmacokinetic models to observed concentration-time data — Phase 443.

Provides one-compartment (IV and oral) and two-compartment (IV) model fitting
without scipy ODE solvers. Uses log-linear OLS for IV, grid search for oral
one-compartment, and the method of residuals (feathering) for two-compartment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "OneCptFitResult",
    "TwoCptFitResult",
    "fit_one_compartment",
    "fit_two_compartment_iv",
]

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OneCptFitResult:
    """Result from one-compartment model fitting."""

    route: str
    dose_mg: float
    ke: float
    ka: float | None  # oral only
    vd_L: float
    t_half_h: float
    auc_fit: float
    r_squared: float
    n_points: int
    notes: str


@dataclass(frozen=True)
class TwoCptFitResult:
    """Result from two-compartment IV model fitting via method of residuals."""

    A: float
    B: float
    alpha: float
    beta: float
    cl_L_per_h: float
    vd_central_L: float
    vd_ss_L: float
    auc: float
    t_half_terminal_h: float
    r_squared: float
    n_points: int
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    times_h: list[float],
    concentrations: list[float],
    min_points: int = 3,
) -> None:
    """Validate time and concentration arrays."""
    if not isinstance(times_h, list):
        raise ValueError("times_h must be a list")
    if not isinstance(concentrations, list):
        raise ValueError("concentrations must be a list")
    if len(times_h) < min_points:
        raise ValueError(f"At least {min_points} data points required, got {len(times_h)}")
    if len(times_h) != len(concentrations):
        raise ValueError(
            f"times_h and concentrations must have the same length, "
            f"got {len(times_h)} and {len(concentrations)}"
        )
    for i in range(1, len(times_h)):
        if times_h[i] <= times_h[i - 1]:
            raise ValueError("times_h must be strictly increasing")
    if any(c <= 0.0 for c in concentrations):
        raise ValueError("All concentrations must be positive (> 0)")


def _trapezoidal_auc(times: list[float], conc: list[float]) -> float:
    """Compute AUC using the trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (conc[i] + conc[i - 1]) * (times[i] - times[i - 1])
    return auc


def _r_squared(y_obs: list[float], y_fit: list[float]) -> float:
    """Compute R-squared (coefficient of determination)."""
    n = len(y_obs)
    mean_obs = sum(y_obs) / n
    ss_tot = sum((y - mean_obs) ** 2 for y in y_obs)
    ss_res = sum((y_obs[i] - y_fit[i]) ** 2 for i in range(n))
    if ss_tot == 0.0:
        return 1.0
    r2 = 1.0 - ss_res / ss_tot
    return max(0.0, min(1.0, r2))


def _ols_log_linear(times: list[float], log_conc: list[float]) -> tuple[float, float]:
    """OLS regression of log(C) ~ intercept - slope*t.

    Returns (intercept, slope) where C(t) = exp(intercept) * exp(-slope * t).
    """
    n = len(times)
    sum_t = sum(times)
    sum_y = sum(log_conc)
    sum_tt = sum(t * t for t in times)
    sum_ty = sum(times[i] * log_conc[i] for i in range(n))
    denom = n * sum_tt - sum_t ** 2
    if abs(denom) < 1e-15:
        # All times identical — fall back to mean
        slope = 0.0
        intercept = sum_y / n
    else:
        slope = (n * sum_ty - sum_t * sum_y) / denom
        intercept = (sum_y - slope * sum_t) / n
    return intercept, slope


# ---------------------------------------------------------------------------
# Public fitting functions
# ---------------------------------------------------------------------------


def fit_one_compartment(
    times_h: list[float],
    concentrations: list[float],
    route: str,
    dose_mg: float,
) -> OneCptFitResult:
    """Fit a one-compartment PK model to observed concentration-time data.

    Parameters
    ----------
    times_h:
        Observation times in hours (must be strictly increasing, >0 allowed).
    concentrations:
        Observed plasma concentrations (positive values only).
    route:
        Dosing route: ``"iv"`` or ``"oral"``.
    dose_mg:
        Administered dose in milligrams.

    Returns
    -------
    OneCptFitResult
        Fitted PK parameters and goodness-of-fit metrics.

    Raises
    ------
    ValueError
        If inputs are invalid or route is not ``"iv"`` or ``"oral"``.
    """
    _validate_inputs(times_h, concentrations, min_points=3)
    if route not in ("iv", "oral"):
        raise ValueError(f"route must be 'iv' or 'oral', got {route!r}")
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")

    notes_parts: list[str] = []

    if route == "iv":
        return _fit_one_cpt_iv(times_h, concentrations, dose_mg, notes_parts)
    else:
        return _fit_one_cpt_oral(times_h, concentrations, dose_mg, notes_parts)


def _fit_one_cpt_iv(
    times_h: list[float],
    concentrations: list[float],
    dose_mg: float,
    notes_parts: list[str],
) -> OneCptFitResult:
    """IV one-compartment: C(t) = A * exp(-ke * t). OLS on log scale."""
    log_c = [math.log(c) for c in concentrations]
    intercept, slope = _ols_log_linear(times_h, log_c)

    # slope from OLS is d(log C)/dt; for C = A*exp(-ke*t), slope = -ke
    ke = max(-slope, 1e-6)
    A = math.exp(intercept)  # C0 = Dose/Vd → Vd = Dose/A

    vd_L = dose_mg / max(A, 1e-12)
    t_half_h = math.log(2.0) / ke

    # AUC from trapezoidal + terminal extrapolation
    auc_trap = _trapezoidal_auc(times_h, concentrations)
    # Extrapolate from last point to infinity: C_last / ke
    auc_extrap = concentrations[-1] / ke
    auc_fit = auc_trap + auc_extrap

    # Fitted concentrations for R²
    c_fit = [A * math.exp(-ke * t) for t in times_h]
    r2 = _r_squared(concentrations, c_fit)

    notes = "; ".join(notes_parts) if notes_parts else "IV 1-cpt OLS log-linear fit"
    return OneCptFitResult(
        route="iv",
        dose_mg=dose_mg,
        ke=ke,
        ka=None,
        vd_L=vd_L,
        t_half_h=t_half_h,
        auc_fit=auc_fit,
        r_squared=r2,
        n_points=len(times_h),
        notes=notes,
    )


# Grid for oral fitting
_KE_GRID = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
_KA_GRID = [0.5, 1.0, 2.0, 5.0]


def _fit_one_cpt_oral(
    times_h: list[float],
    concentrations: list[float],
    dose_mg: float,
    notes_parts: list[str],
) -> OneCptFitResult:
    """Oral one-compartment with grid search over ke and ka.

    C(t) = (F * D * ka) / (Vd * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))
    F is fixed at 1; Vd is fitted analytically for each ke/ka pair.
    """
    best_ssr = float("inf")
    best_ke = _KE_GRID[0]
    best_ka = _KA_GRID[-1]
    best_vd = dose_mg  # 1 L default placeholder

    n = len(times_h)

    for ke in _KE_GRID:
        for ka in _KA_GRID:
            if abs(ka - ke) < 1e-8:
                continue  # skip degenerate case
            # C(t) = Vd_scale * shape(t)
            # shape(t) = (ka * D) / (ka - ke) * (exp(-ke*t) - exp(-ka*t))
            # Vd_scale = 1/Vd, so C = (1/Vd) * shape
            # Minimise SSR over Vd analytically: Vd = sum(shape*obs) / sum(obs^2)
            # Actually analytically: C_hat = (1/Vd) * shape
            # SSR = sum((obs - (1/Vd)*shape)^2)
            # d/d(1/Vd) SSR = 0 → (1/Vd) = sum(shape*obs) / sum(shape^2)
            shapes = [
                (ka * dose_mg) / (ka - ke) * (math.exp(-ke * t) - math.exp(-ka * t))
                for t in times_h
            ]
            sum_so = sum(shapes[i] * concentrations[i] for i in range(n))
            sum_ss = sum(s * s for s in shapes)
            if sum_ss < 1e-30:
                continue
            inv_vd = sum_so / sum_ss
            if inv_vd <= 0.0:
                continue
            vd = 1.0 / inv_vd
            c_fit = [s * inv_vd for s in shapes]
            ssr = sum((concentrations[i] - c_fit[i]) ** 2 for i in range(n))
            if ssr < best_ssr:
                best_ssr = ssr
                best_ke = ke
                best_ka = ka
                best_vd = vd

    ke = best_ke
    ka = best_ka
    vd_L = best_vd
    t_half_h = math.log(2.0) / ke

    # Fitted concentrations
    inv_vd = 1.0 / vd_L
    c_fit = [
        (ka * dose_mg) / (ka - ke) * (math.exp(-ke * t) - math.exp(-ka * t)) * inv_vd
        for t in times_h
    ]

    r2 = _r_squared(concentrations, c_fit)

    # AUC: trapezoidal + terminal extrapolation
    auc_trap = _trapezoidal_auc(times_h, concentrations)
    auc_extrap = max(concentrations[-1], 0.0) / ke
    auc_fit = auc_trap + auc_extrap

    notes_parts.append(f"grid ke={ke}, ka={ka}")
    notes = "; ".join(notes_parts)
    return OneCptFitResult(
        route="oral",
        dose_mg=dose_mg,
        ke=ke,
        ka=ka,
        vd_L=vd_L,
        t_half_h=t_half_h,
        auc_fit=auc_fit,
        r_squared=r2,
        n_points=len(times_h),
        notes=notes,
    )


def fit_two_compartment_iv(
    times_h: list[float],
    concentrations: list[float],
    dose_mg: float,
) -> TwoCptFitResult:
    """Fit a two-compartment IV model using the method of residuals (feathering).

    The biexponential model is:

        C(t) = A * exp(-alpha * t) + B * exp(-beta * t)

    where alpha > beta (alpha is the faster distribution phase).

    The method of residuals:
    1. Identify the terminal (beta) phase from the last few log-linear points.
    2. Subtract the terminal phase from early data to obtain the alpha phase.

    Parameters
    ----------
    times_h:
        Observation times in hours.
    concentrations:
        Observed plasma concentrations (positive).
    dose_mg:
        Administered dose in mg (used to derive CL and Vd).

    Returns
    -------
    TwoCptFitResult
        Fitted biexponential PK parameters.

    Raises
    ------
    ValueError
        If inputs are invalid or there are insufficient points for feathering.
    """
    _validate_inputs(times_h, concentrations, min_points=5)
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")

    n = len(times_h)
    notes_parts: list[str] = []

    # --- Step 1: Identify terminal phase using the last N_TERM points ----------
    # Use the last third of points (minimum 3) for terminal phase estimation
    n_term = max(3, n // 3)
    n_term = min(n_term, n - 2)  # leave at least 2 points for alpha phase

    term_times = times_h[-n_term:]
    term_log_c = [math.log(c) for c in concentrations[-n_term:]]

    intercept_b, slope_b = _ols_log_linear(term_times, term_log_c)
    beta = max(-slope_b, 1e-6)
    B = math.exp(intercept_b)

    # --- Step 2: Residuals in early phase → alpha ---------------------------
    n_early = n - n_term
    if n_early < 2:
        # Fall back: use first 2 points
        n_early = 2
        notes_parts.append("fallback early phase (2 pts)")

    early_times = times_h[:n_early]
    early_conc = concentrations[:n_early]
    early_residuals = [
        c - B * math.exp(-beta * t) for c, t in zip(early_conc, early_times, strict=True)
    ]
    # Residuals should be positive if alpha > beta; clip negatives
    early_residuals_pos = [max(r, 1e-12) for r in early_residuals]

    early_log_r = [math.log(r) for r in early_residuals_pos]
    intercept_a, slope_a = _ols_log_linear(early_times, early_log_r)
    alpha = max(-slope_a, beta + 1e-4)  # alpha must be > beta
    A = math.exp(intercept_a)

    # --- Derived PK parameters -----------------------------------------------
    # AUC0-inf for biexponential = A/alpha + B/beta
    auc = A / alpha + B / beta

    # CL = Dose / AUC
    cl_L_per_h = dose_mg / max(auc, 1e-12)

    # Vd_central = Dose / (A + B)
    vd_central_L = dose_mg / max(A + B, 1e-12)

    # Vd_ss derived from macro constants
    # For 2-cpt model: Vd_ss = CL * (A/alpha^2 + B/beta^2) / (A/alpha + B/beta)
    # Alternatively: Vd_ss = CL * MRT where MRT = AUMC/AUC
    # AUMC_inf = A/alpha^2 + B/beta^2
    aumc = A / alpha ** 2 + B / beta ** 2
    mrt = aumc / max(auc, 1e-12)
    vd_ss_L = cl_L_per_h * mrt

    t_half_terminal_h = math.log(2.0) / beta

    # --- Goodness of fit -----------------------------------------------------
    c_fit = [A * math.exp(-alpha * t) + B * math.exp(-beta * t) for t in times_h]
    r2 = _r_squared(concentrations, c_fit)

    notes_parts.append(f"feathering n_term={n_term}")
    notes = "; ".join(notes_parts)

    return TwoCptFitResult(
        A=A,
        B=B,
        alpha=alpha,
        beta=beta,
        cl_L_per_h=cl_L_per_h,
        vd_central_L=vd_central_L,
        vd_ss_L=vd_ss_L,
        auc=auc,
        t_half_terminal_h=t_half_terminal_h,
        r_squared=r2,
        n_points=n,
        notes=notes,
    )
