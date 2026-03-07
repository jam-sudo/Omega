"""Clinical Trial PK Simulation.

Simulates clinical trial PK data with inter-subject variability,
sparse sampling, and regulatory analysis endpoints (NCA, BE power).

Model: 1-compartment oral absorption (forward Euler).
    dC/dt = (F * ka * Dose / Vd) * exp(-ka * t) - (CL/Vd) * C

Population variability:
    CL_i = CL_mean * exp(eta_CL),  eta_CL ~ N(0, omega_CL^2)
    Vd_i = Vd_mean * exp(eta_Vd),  eta_Vd ~ N(0, omega_Vd^2)
    C_obs = C_true * (1 + eps),    eps    ~ N(0, sigma^2)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClinicalTrialPKResult:
    """Result of a simulated clinical trial PK study."""

    study_name: str
    n_subjects: int
    dose_mg: float
    route: str
    dosing_regimen: str  # "single_dose", "multiple_dose"

    # Population parameters
    cl_mean_L_per_h: float
    vd_mean_L: float
    omega_cl_pct: float
    omega_vd_pct: float
    sigma_pct: float

    # Summary statistics (geometric mean across subjects)
    auc_gm_mg_h_per_L: float
    cmax_gm_mg_L: float
    tmax_median_h: float
    t_half_gm_h: float

    # Variability metrics
    auc_cv_pct: float
    cmax_cv_pct: float

    # Confidence intervals (90% CI for GM)
    auc_gm_90ci: tuple[float, float]
    cmax_gm_90ci: tuple[float, float]

    # NCA summary
    auc_individual: list[float]
    cmax_individual: list[float]

    # Power analysis
    n_required_for_be: int

    notes: str


# ---------------------------------------------------------------------------
# Internal simulation helpers
# ---------------------------------------------------------------------------


def _simulate_1cpt_oral(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    f_oral: float,
    n_doses: int,
    interval_h: float,
    t_end_h: float,
    dt_h: float,
    rng: np.random.Generator,
    sigma_pct: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate 1-cpt oral PK for a single subject.

    Returns (t_arr, C_obs_arr) where C_obs has proportional residual noise.
    """
    t_arr = np.arange(0.0, t_end_h + dt_h * 0.5, dt_h)
    n_steps = len(t_arr)
    C = np.zeros(n_steps)
    ke = cl_L_per_h / vd_L

    effective_dose = dose_mg * f_oral  # mg absorbed

    # Dose times
    dose_times = [i * interval_h for i in range(n_doses)]

    for step in range(1, n_steps):
        t = t_arr[step]
        dt = t_arr[step] - t_arr[step - 1]

        # Absorption input: sum over all doses already administered
        absorption_rate = 0.0
        for t_dose in dose_times:
            if t > t_dose:
                elapsed = t - t_dose
                # mg/L per h
                absorption_rate += (effective_dose * ka_per_h / vd_L) * math.exp(
                    -ka_per_h * elapsed
                )

        dCdt = absorption_rate - ke * C[step - 1]
        C[step] = max(0.0, C[step - 1] + dCdt * dt)

    # Add proportional residual error
    sigma = sigma_pct / 100.0
    eps = rng.normal(0.0, sigma, size=n_steps)
    C_obs = C * (1.0 + eps)
    C_obs = np.maximum(0.0, C_obs)

    return t_arr, C_obs


def _compute_nca(
    t_arr: np.ndarray,
    C_obs: np.ndarray,
) -> tuple[float, float, float]:
    """Compute AUC (trapz), Cmax, and Tmax from a concentration profile."""
    auc = float(np.trapz(C_obs, t_arr))
    idx_max = int(np.argmax(C_obs))
    cmax = float(C_obs[idx_max])
    tmax = float(t_arr[idx_max])
    return auc, cmax, tmax


def _geometric_mean(values: np.ndarray) -> float:
    log_vals = np.log(values[values > 0])
    if len(log_vals) == 0:
        return 0.0
    return float(np.exp(np.mean(log_vals)))


def _geometric_cv_pct(values: np.ndarray) -> float:
    """CV% based on arithmetic mean."""
    mu = float(np.mean(values))
    if mu == 0:
        return 0.0
    return float(np.std(values) / mu * 100.0)


def _gm_90ci(
    values: np.ndarray,
) -> tuple[float, float]:
    """90% confidence interval for geometric mean using log-normal assumption."""
    log_vals = np.log(values[values > 0])
    n = len(log_vals)
    if n < 2:
        gm = float(np.exp(np.mean(log_vals))) if n == 1 else 0.0
        return (gm, gm)
    log_mean = float(np.mean(log_vals))
    log_var = float(np.var(log_vals, ddof=1))
    se = math.sqrt(log_var / n)
    z = 1.645  # 90% CI
    lower = math.exp(log_mean - z * se)
    upper = math.exp(log_mean + z * se)
    return (lower, upper)


def _n_required_for_be(omega_cl_pct: float) -> int:
    """Sample size for 80% BE power using the standard formula.

    n = ceil(2 * (z_alpha + z_beta)^2 * sigma_within^2 / delta^2)
    z_alpha = 1.645, z_beta = 0.842, delta = log(1.25)
    """
    z_alpha = 1.645
    z_beta = 0.842
    sigma_within = omega_cl_pct / 100.0
    delta = math.log(1.25)
    n_float = 2.0 * (z_alpha + z_beta) ** 2 * sigma_within**2 / delta**2
    return max(2, math.ceil(n_float))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_clinical_trial_pk(
    study_name: str,
    n_subjects: int,
    dose_mg: float,
    cl_mean_L_per_h: float,
    vd_mean_L: float,
    ka_per_h: float = 1.5,
    f_oral: float = 0.9,
    omega_cl_pct: float = 30.0,
    omega_vd_pct: float = 20.0,
    sigma_pct: float = 15.0,
    route: str = "oral",
    dosing_regimen: str = "single_dose",
    n_doses: int = 1,
    interval_h: float = 24.0,
    seed: int = 42,
    t_end_h: float = 48.0,
    dt_h: float = 0.25,
) -> ClinicalTrialPKResult:
    """Simulate clinical trial PK data with inter-subject variability.

    Parameters
    ----------
    study_name:
        Descriptive name for this trial simulation.
    n_subjects:
        Number of virtual subjects (>= 3).
    dose_mg:
        Dose per administration in mg.
    cl_mean_L_per_h:
        Population mean clearance (L/h).
    vd_mean_L:
        Population mean volume of distribution (L).
    ka_per_h:
        First-order absorption rate constant (1/h).
    f_oral:
        Oral bioavailability fraction (0-1).
    omega_cl_pct:
        IIV CV% for CL (log-normal, %).
    omega_vd_pct:
        IIV CV% for Vd (log-normal, %).
    sigma_pct:
        Proportional residual error CV%.
    route:
        Route of administration ("oral", "iv").
    dosing_regimen:
        "single_dose" or "multiple_dose".
    n_doses:
        Number of doses.
    interval_h:
        Dosing interval in hours.
    seed:
        Random seed for reproducibility.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Euler integration step size (h).

    Returns
    -------
    ClinicalTrialPKResult
    """
    # Validation
    if n_subjects < 3:
        raise ValueError(f"n_subjects must be >= 3, got {n_subjects}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_mean_L_per_h <= 0:
        raise ValueError(f"cl_mean_L_per_h must be > 0, got {cl_mean_L_per_h}")
    if vd_mean_L <= 0:
        raise ValueError(f"vd_mean_L must be > 0, got {vd_mean_L}")
    if sigma_pct < 0:
        raise ValueError(f"sigma_pct must be >= 0, got {sigma_pct}")

    rng = np.random.default_rng(seed)

    omega_cl = omega_cl_pct / 100.0
    omega_vd = omega_vd_pct / 100.0

    auc_list: list[float] = []
    cmax_list: list[float] = []
    tmax_list: list[float] = []

    for _ in range(n_subjects):
        # Individual PK parameters (log-normal IIV)
        cl_i = cl_mean_L_per_h * math.exp(rng.normal(0.0, omega_cl))
        vd_i = vd_mean_L * math.exp(rng.normal(0.0, omega_vd))

        t_arr, C_obs = _simulate_1cpt_oral(
            dose_mg=dose_mg,
            cl_L_per_h=cl_i,
            vd_L=vd_i,
            ka_per_h=ka_per_h,
            f_oral=f_oral,
            n_doses=n_doses,
            interval_h=interval_h,
            t_end_h=t_end_h,
            dt_h=dt_h,
            rng=rng,
            sigma_pct=sigma_pct,
        )

        auc, cmax, tmax = _compute_nca(t_arr, C_obs)
        auc_list.append(max(auc, 1e-9))
        cmax_list.append(max(cmax, 1e-9))
        tmax_list.append(tmax)

    auc_arr = np.array(auc_list)
    cmax_arr = np.array(cmax_list)
    tmax_arr = np.array(tmax_list)

    auc_gm = _geometric_mean(auc_arr)
    cmax_gm = _geometric_mean(cmax_arr)
    tmax_median = float(np.median(tmax_arr))

    # Geometric mean t½ from population mean ke
    ke_mean = cl_mean_L_per_h / vd_mean_L
    t_half_gm = math.log(2.0) / ke_mean

    auc_cv = _geometric_cv_pct(auc_arr)
    cmax_cv = _geometric_cv_pct(cmax_arr)

    auc_90ci = _gm_90ci(auc_arr)
    cmax_90ci = _gm_90ci(cmax_arr)

    n_be = _n_required_for_be(omega_cl_pct)

    notes = (
        f"Simulated {n_subjects} subjects; omega_CL={omega_cl_pct:.1f}%, "
        f"omega_Vd={omega_vd_pct:.1f}%, sigma={sigma_pct:.1f}%; "
        f"regimen={dosing_regimen}, n_doses={n_doses}"
    )

    return ClinicalTrialPKResult(
        study_name=study_name,
        n_subjects=n_subjects,
        dose_mg=dose_mg,
        route=route,
        dosing_regimen=dosing_regimen,
        cl_mean_L_per_h=cl_mean_L_per_h,
        vd_mean_L=vd_mean_L,
        omega_cl_pct=omega_cl_pct,
        omega_vd_pct=omega_vd_pct,
        sigma_pct=sigma_pct,
        auc_gm_mg_h_per_L=auc_gm,
        cmax_gm_mg_L=cmax_gm,
        tmax_median_h=tmax_median,
        t_half_gm_h=t_half_gm,
        auc_cv_pct=auc_cv,
        cmax_cv_pct=cmax_cv,
        auc_gm_90ci=auc_90ci,
        cmax_gm_90ci=cmax_90ci,
        auc_individual=auc_list,
        cmax_individual=cmax_list,
        n_required_for_be=n_be,
        notes=notes,
    )


def compare_dose_groups(
    study_name: str,
    doses_mg: list[float],
    cl_mean: float,
    vd_mean: float,
    n_subjects_per_group: int = 12,
    **kwargs,
) -> list[ClinicalTrialPKResult]:
    """Simulate one trial per dose level and return sorted by dose ascending.

    Parameters
    ----------
    study_name:
        Base name; each group gets a suffix with the dose.
    doses_mg:
        List of dose levels in mg.
    cl_mean:
        Population mean CL (L/h).
    vd_mean:
        Population mean Vd (L).
    n_subjects_per_group:
        Number of virtual subjects per dose group.
    **kwargs:
        Additional keyword arguments forwarded to simulate_clinical_trial_pk.

    Returns
    -------
    List of ClinicalTrialPKResult sorted by dose_mg ascending.
    """
    results: list[ClinicalTrialPKResult] = []
    for i, dose in enumerate(doses_mg):
        # Vary seed per group to avoid identical IIV draws
        seed = kwargs.pop("seed", 42) + i if "seed" in kwargs else 42 + i
        r = simulate_clinical_trial_pk(
            study_name=f"{study_name}_dose{dose:.0f}mg",
            n_subjects=n_subjects_per_group,
            dose_mg=dose,
            cl_mean_L_per_h=cl_mean,
            vd_mean_L=vd_mean,
            seed=seed,
            **kwargs,
        )
        results.append(r)

    results.sort(key=lambda r: r.dose_mg)
    return results
