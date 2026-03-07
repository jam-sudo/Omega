"""Non-Linear Mixed Effects (NLME) PK Model Summary.

Summarize and interpret NLME PK parameter estimates (as produced by
tools such as NONMEM, Monolix, or nlmixr).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NLMEEstimates:
    """Container for parsed NLME model estimates."""

    param_names: tuple[str, ...]
    theta: tuple[float, ...]
    omega_diag: tuple[float, ...]
    sigma_diag: tuple[float, ...]
    omega_cv_pct: tuple[float, ...]
    sigma_cv_pct: tuple[float, ...]


def parse_nlme_estimates(
    theta: list[float],
    omega_diag: list[float],
    sigma_diag: list[float],
    param_names: list[str] | None = None,
) -> NLMEEstimates:
    """Parse raw NLME parameter estimates into a structured summary.

    Parameters
    ----------
    theta:
        Fixed-effect estimates [CL, Vd, ka, ...].
    omega_diag:
        Diagonal of the OMEGA matrix (between-subject variance for each theta).
    sigma_diag:
        Residual error variances [proportional_var, additive_var, ...].
    param_names:
        Optional names for the theta parameters. Defaults to ['theta_1', ...].

    Returns
    -------
    NLMEEstimates
    """
    if not theta:
        raise ValueError("theta must be non-empty")
    if param_names is None:
        param_names = [f"theta_{i + 1}" for i in range(len(theta))]
    if len(param_names) != len(theta):
        raise ValueError("param_names length must match theta length")

    # IIV CV% from log-normal parameterisation: CV% = sqrt(exp(omega) - 1) * 100
    omega_cv_pct = []
    for omega in omega_diag:
        try:
            cv = math.sqrt(math.exp(omega) - 1.0) * 100.0
        except (ValueError, OverflowError):
            cv = float("nan")
        omega_cv_pct.append(cv)

    # Residual error CV% (proportional): sqrt(sigma) * 100
    sigma_cv_pct = []
    for sigma in sigma_diag:
        try:
            cv = math.sqrt(max(sigma, 0.0)) * 100.0
        except ValueError:
            cv = float("nan")
        sigma_cv_pct.append(cv)

    return NLMEEstimates(
        param_names=tuple(param_names),
        theta=tuple(theta),
        omega_diag=tuple(omega_diag),
        sigma_diag=tuple(sigma_diag),
        omega_cv_pct=tuple(omega_cv_pct),
        sigma_cv_pct=tuple(sigma_cv_pct),
    )


def _trapezoidal_auc(times: list[float], conc: list[float]) -> float:
    """Compute AUC using the linear trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (conc[i] + conc[i - 1]) * dt
    return auc


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _cv_pct(values: list[float]) -> float:
    m = _mean(values)
    if m == 0:
        return float("nan")
    return _std(values) / m * 100.0


def _terminal_slope(times: list[float], conc: list[float], n_points: int = 3) -> float:
    """Estimate terminal elimination rate constant (ke) via OLS on log-conc."""
    pts = min(n_points, len(times))
    if pts < 2:
        return float("nan")
    t_tail = times[-pts:]
    c_tail = conc[-pts:]
    # OLS on log-linear: log(c) = log(c0) - ke*t
    log_c = []
    for c in c_tail:
        log_c.append(math.log(c) if c > 0 else float("-inf"))
    # Filter out -inf
    valid = [(t, lc) for t, lc in zip(t_tail, log_c) if math.isfinite(lc)]
    if len(valid) < 2:
        return float("nan")
    t_vals, lc_vals = zip(*valid)
    n = len(t_vals)
    sum_t = sum(t_vals)
    sum_lc = sum(lc_vals)
    sum_t2 = sum(t * t for t in t_vals)
    sum_tlc = sum(t * lc for t, lc in zip(t_vals, lc_vals))
    denom = n * sum_t2 - sum_t**2
    if denom == 0:
        return float("nan")
    slope = (n * sum_tlc - sum_t * sum_lc) / denom
    # slope is negative (ke = -slope)
    ke = -slope
    return ke


def compute_nca_statistics(subject_data: list[dict]) -> dict:
    """Compute NCA summary statistics across subjects.

    Parameters
    ----------
    subject_data:
        List of dicts, each with keys:
        - 'id': subject identifier
        - 'times': list of sampling times (h)
        - 'concentrations': list of observed concentrations (same length as times)
        - 'dose': administered dose

    Returns
    -------
    dict with keys:
        n, cmax_mean, cmax_cv_pct, auc_mean, auc_cv_pct,
        tmax_median, thalf_mean, thalf_cv_pct
    """
    if not subject_data:
        return {
            "n": 0,
            "cmax_mean": float("nan"),
            "cmax_cv_pct": float("nan"),
            "auc_mean": float("nan"),
            "auc_cv_pct": float("nan"),
            "tmax_median": float("nan"),
            "thalf_mean": float("nan"),
            "thalf_cv_pct": float("nan"),
        }

    cmax_list: list[float] = []
    auc_list: list[float] = []
    tmax_list: list[float] = []
    thalf_list: list[float] = []

    for subj in subject_data:
        times = subj["times"]
        conc = subj["concentrations"]
        if not times or not conc or len(times) != len(conc):
            continue

        # Cmax and Tmax
        cmax = max(conc)
        idx_max = conc.index(cmax)
        tmax = times[idx_max]
        cmax_list.append(cmax)
        tmax_list.append(tmax)

        # AUC
        auc = _trapezoidal_auc(times, conc)
        auc_list.append(auc)

        # t_half from terminal slope
        ke = _terminal_slope(times, conc, n_points=3)
        if math.isfinite(ke) and ke > 0:
            thalf = math.log(2) / ke
            thalf_list.append(thalf)

    # Median tmax
    if tmax_list:
        sorted_tmax = sorted(tmax_list)
        n_t = len(sorted_tmax)
        if n_t % 2 == 1:
            tmax_median = sorted_tmax[n_t // 2]
        else:
            tmax_median = (sorted_tmax[n_t // 2 - 1] + sorted_tmax[n_t // 2]) / 2.0
    else:
        tmax_median = float("nan")

    return {
        "n": len(subject_data),
        "cmax_mean": _mean(cmax_list),
        "cmax_cv_pct": _cv_pct(cmax_list),
        "auc_mean": _mean(auc_list),
        "auc_cv_pct": _cv_pct(auc_list),
        "tmax_median": tmax_median,
        "thalf_mean": _mean(thalf_list),
        "thalf_cv_pct": _cv_pct(thalf_list),
    }


def _simulate_1cpt_oral(
    times: list[float], dose: float, ka: float, ke: float, vd: float
) -> list[float]:
    """Analytical 1-compartment oral model concentration at given times."""
    pred = []
    for t in times:
        if ke == ka:
            # Avoid division by zero
            c = dose / vd * ka * t * math.exp(-ke * t)
        else:
            c = dose * ka / (vd * (ka - ke)) * (math.exp(-ke * t) - math.exp(-ka * t))
        pred.append(max(c, 0.0))
    return pred


def _simulate_1cpt_iv(times: list[float], dose: float, ke: float, vd: float) -> list[float]:
    """Analytical 1-compartment IV bolus concentration at given times."""
    pred = []
    for t in times:
        c = dose / vd * math.exp(-ke * t)
        pred.append(max(c, 0.0))
    return pred


def _log_ssr(obs: list[float], pred: list[float]) -> float:
    """Sum of squared residuals on log scale."""
    ssr = 0.0
    for o, p in zip(obs, pred):
        log_o = math.log(o + 1e-10)
        log_p = math.log(p + 1e-10)
        ssr += (log_p - log_o) ** 2
    return ssr


def individual_pk_parameters(
    obs_times: list[float],
    obs_conc: list[float],
    dose: float,
    route: str = "oral",
    ka_init: float = 1.0,
    cl_init: float = 5.0,
    vd_init: float = 50.0,
) -> dict:
    """Estimate individual PK parameters via grid search (no scipy).

    Fits a one-compartment model to individual concentration-time data
    by minimizing SSR on the log-concentration scale.

    Parameters
    ----------
    obs_times:
        Observed sampling times (h).
    obs_conc:
        Observed concentrations (same length as obs_times).
    dose:
        Administered dose (same units as Vd and CL).
    route:
        'oral' or 'iv'. Oral also searches over ka.
    ka_init:
        Initial estimate for ka (1/h). Used to build search grid.
    cl_init:
        Initial estimate for CL (L/h).
    vd_init:
        Initial estimate for Vd (L).

    Returns
    -------
    dict with keys: cl, vd, ka, ssr, ke, t_half
    """
    if not obs_times or not obs_conc:
        raise ValueError("obs_times and obs_conc must be non-empty")
    if len(obs_times) != len(obs_conc):
        raise ValueError("obs_times and obs_conc must have the same length")

    # Build search grids (10 points log-spaced for CL and Vd, 5 for ka)
    def _logspace(start: float, stop: float, n: int) -> list[float]:
        log_start = math.log(start)
        log_stop = math.log(stop)
        step = (log_stop - log_start) / (n - 1) if n > 1 else 0
        return [math.exp(log_start + i * step) for i in range(n)]

    cl_grid = _logspace(cl_init * 0.1, cl_init * 5.0, 10)
    vd_grid = _logspace(vd_init * 0.1, vd_init * 5.0, 10)

    best_ssr = float("inf")
    best_cl = cl_init
    best_vd = vd_init
    best_ka = ka_init if route == "oral" else 0.0

    if route == "oral":
        ka_grid = _logspace(0.1, 5.0, 5)
        for cl in cl_grid:
            for vd in vd_grid:
                ke = cl / vd
                for ka in ka_grid:
                    if ka <= ke:
                        # Avoid numerical issues if ka ≈ ke
                        pass
                    pred = _simulate_1cpt_oral(obs_times, dose, ka, ke, vd)
                    ssr = _log_ssr(obs_conc, pred)
                    if ssr < best_ssr:
                        best_ssr = ssr
                        best_cl = cl
                        best_vd = vd
                        best_ka = ka
    else:
        for cl in cl_grid:
            for vd in vd_grid:
                ke = cl / vd
                pred = _simulate_1cpt_iv(obs_times, dose, ke, vd)
                ssr = _log_ssr(obs_conc, pred)
                if ssr < best_ssr:
                    best_ssr = ssr
                    best_cl = cl
                    best_vd = vd

    best_ke = best_cl / best_vd if best_vd > 0 else 0.0
    t_half = math.log(2) / best_ke if best_ke > 0 else float("inf")

    return {
        "cl": best_cl,
        "vd": best_vd,
        "ka": best_ka,
        "ssr": best_ssr,
        "ke": best_ke,
        "t_half": t_half,
    }
