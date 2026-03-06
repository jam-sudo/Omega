"""Stochastic PK Variability — Monte Carlo population PK simulation.

Generates population PK profiles by sampling individual PK parameters
(CL, Vd, ka) from log-normal distributions, consistent with the NONMEM
population PK framework.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

__all__ = [
    "IndividualPKResult",
    "PopulationPKResult",
    "simulate_population_pk",
    "individual_pk_profiles",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndividualPKResult:
    """PK profile for one simulated individual."""

    subject_id: int
    cl_L_per_h: float
    vd_L: float
    ka_per_h: float
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    times_h: list[float]
    c_plasma_mg_L: list[float]


@dataclass(frozen=True)
class PopulationPKResult:
    """Summary of a population PK simulation."""

    drug_name: str
    dose_mg: float
    n_subjects: int
    times_h: list[float]
    c_p5: list[float]
    c_p25: list[float]
    c_p50: list[float]
    c_p75: list[float]
    c_p95: list[float]
    cmax_p50: float
    tmax_p50: float
    auc_p50: float
    cv_cmax_pct: float
    notes: list[str]


# ---------------------------------------------------------------------------
# Internal: 1-compartment concentration profile
# ---------------------------------------------------------------------------


def _simulate_1cpt(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    route: str,
    ka_per_h: float,
    times: np.ndarray,
) -> np.ndarray:
    """Analytical 1-compartment PK profile.

    IV: C(t) = (Dose/Vd) * exp(-kel*t)
    Oral: C(t) = (F*Dose*ka)/(Vd*(ka-kel)) * (exp(-kel*t) - exp(-ka*t))
    """
    kel = cl_L_per_h / vd_L

    if route == "iv":
        c = (dose_mg / vd_L) * np.exp(-kel * times)
    else:
        # oral (assume F=1 here; variability encoded in CL/Vd)
        if abs(ka_per_h - kel) < 1e-8:
            # Avoid division by zero: use limit
            c = (dose_mg / vd_L) * kel * times * np.exp(-kel * times)
        else:
            c = (
                (dose_mg * ka_per_h)
                / (vd_L * (ka_per_h - kel))
                * (np.exp(-kel * times) - np.exp(-ka_per_h * times))
            )

    return np.maximum(c, 0.0)


def _pk_metrics(
    times: np.ndarray,
    c: np.ndarray,
) -> tuple[float, float, float]:
    """Return (cmax, tmax, auc)."""
    cmax = float(np.max(c))
    tmax = float(times[int(np.argmax(c))])
    auc = float(np_trapz(c, times))
    return cmax, tmax, auc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_population_pk(
    drug_name: str,
    dose_mg: float,
    cl_typical_L_per_h: float,
    vd_typical_L: float,
    route: str = "oral",
    ka_typical_per_h: float = 1.0,
    omega_cl: float = 0.3,
    omega_vd: float = 0.3,
    omega_ka: float = 0.2,
    sigma_prop: float = 0.1,
    n_subjects: int = 100,
    t_end_h: float = 24.0,
    dt_h: float = 0.5,
    seed: int = 42,
) -> PopulationPKResult:
    """Monte Carlo population PK simulation.

    Individual parameters are sampled as:
        CL_i = CL_typ * exp(eta_CL),  eta_CL ~ N(0, omega_cl^2)
        Vd_i = Vd_typ * exp(eta_Vd),  eta_Vd ~ N(0, omega_vd^2)
        ka_i = ka_typ * exp(eta_ka),  eta_ka ~ N(0, omega_ka^2)

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Oral/IV dose in mg.
    cl_typical_L_per_h:
        Typical (population mean) CL (L/h).
    vd_typical_L:
        Typical (population mean) Vd (L).
    route:
        ``"oral"`` or ``"iv"``.
    ka_typical_per_h:
        Typical absorption rate constant (h^-1); only used for oral route.
    omega_cl:
        Between-subject variability standard deviation for CL (log-normal SD).
    omega_vd:
        Between-subject variability SD for Vd.
    omega_ka:
        Between-subject variability SD for ka.
    sigma_prop:
        Proportional residual error (not applied to percentile curves but
        recorded in metadata).
    n_subjects:
        Number of simulated subjects.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Time step (h).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    PopulationPKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_typical_L_per_h <= 0:
        raise ValueError("cl_typical_L_per_h must be > 0")
    if vd_typical_L <= 0:
        raise ValueError("vd_typical_L must be > 0")
    if n_subjects < 2:
        raise ValueError("n_subjects must be >= 2")
    if omega_cl < 0:
        raise ValueError("omega_cl must be >= 0")
    if omega_vd < 0:
        raise ValueError("omega_vd must be >= 0")
    if omega_ka < 0:
        raise ValueError("omega_ka must be >= 0")

    rng = np.random.default_rng(seed)
    times = np.arange(0.0, t_end_h + dt_h, dt_h)
    n_t = len(times)

    # Sample individual parameters
    eta_cl = rng.normal(0.0, omega_cl, n_subjects)
    eta_vd = rng.normal(0.0, omega_vd, n_subjects)
    eta_ka = rng.normal(0.0, omega_ka, n_subjects)

    cls = cl_typical_L_per_h * np.exp(eta_cl)
    vds = vd_typical_L * np.exp(eta_vd)
    kas = ka_typical_per_h * np.exp(eta_ka)

    # Simulate all subjects
    all_conc = np.zeros((n_subjects, n_t))
    cmax_arr = np.zeros(n_subjects)

    for i in range(n_subjects):
        c = _simulate_1cpt(dose_mg, float(cls[i]), float(vds[i]), route, float(kas[i]), times)
        all_conc[i] = c
        cmax_arr[i] = float(np.max(c))

    # Percentile profiles
    c_p5 = np.percentile(all_conc, 5, axis=0)
    c_p25 = np.percentile(all_conc, 25, axis=0)
    c_p50 = np.percentile(all_conc, 50, axis=0)
    c_p75 = np.percentile(all_conc, 75, axis=0)
    c_p95 = np.percentile(all_conc, 95, axis=0)

    cmax_p50 = float(np.max(c_p50))
    tmax_idx = int(np.argmax(c_p50))
    tmax_p50 = float(times[tmax_idx])
    auc_p50 = float(np_trapz(c_p50, times))

    cv_cmax = float(np.std(cmax_arr) / np.mean(cmax_arr) * 100.0) if np.mean(cmax_arr) > 0 else 0.0

    notes: list[str] = []
    if cv_cmax > 50:
        notes.append(f"High Cmax variability: CV={cv_cmax:.1f}%")
    if omega_cl == 0 and omega_vd == 0 and omega_ka == 0:
        notes.append("All omega=0: deterministic simulation (no variability).")

    return PopulationPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        n_subjects=n_subjects,
        times_h=times.tolist(),
        c_p5=c_p5.tolist(),
        c_p25=c_p25.tolist(),
        c_p50=c_p50.tolist(),
        c_p75=c_p75.tolist(),
        c_p95=c_p95.tolist(),
        cmax_p50=cmax_p50,
        tmax_p50=tmax_p50,
        auc_p50=auc_p50,
        cv_cmax_pct=cv_cmax,
        notes=notes,
    )


def individual_pk_profiles(
    drug_name: str,
    dose_mg: float,
    cl_typical_L_per_h: float,
    vd_typical_L: float,
    route: str = "oral",
    ka_typical_per_h: float = 1.0,
    omega_cl: float = 0.3,
    omega_vd: float = 0.3,
    n_subjects: int = 10,
    t_end_h: float = 24.0,
    dt_h: float = 0.5,
    seed: int = 42,
) -> list[IndividualPKResult]:
    """Simulate and return PK profiles for individual subjects.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Dose in mg.
    cl_typical_L_per_h:
        Typical CL (L/h).
    vd_typical_L:
        Typical Vd (L).
    route:
        ``"oral"`` or ``"iv"``.
    ka_typical_per_h:
        Typical ka (h^-1).
    omega_cl:
        BSV SD for CL.
    omega_vd:
        BSV SD for Vd.
    n_subjects:
        Number of subjects to simulate.
    t_end_h:
        End time (h).
    dt_h:
        Time step (h).
    seed:
        Random seed.

    Returns
    -------
    list[IndividualPKResult]
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_typical_L_per_h <= 0:
        raise ValueError("cl_typical_L_per_h must be > 0")
    if vd_typical_L <= 0:
        raise ValueError("vd_typical_L must be > 0")
    if n_subjects < 2:
        raise ValueError("n_subjects must be >= 2")
    if omega_cl < 0:
        raise ValueError("omega_cl must be >= 0")
    if omega_vd < 0:
        raise ValueError("omega_vd must be >= 0")

    rng = np.random.default_rng(seed)
    times = np.arange(0.0, t_end_h + dt_h, dt_h)

    eta_cl = rng.normal(0.0, omega_cl, n_subjects)
    eta_vd = rng.normal(0.0, omega_vd, n_subjects)

    # Fixed ka at typical for this function (no omega_ka parameter)
    ka = ka_typical_per_h

    results: list[IndividualPKResult] = []
    for i in range(n_subjects):
        cl_i = float(cl_typical_L_per_h * np.exp(eta_cl[i]))
        vd_i = float(vd_typical_L * np.exp(eta_vd[i]))
        c = _simulate_1cpt(dose_mg, cl_i, vd_i, route, ka, times)
        cmax, tmax, auc = _pk_metrics(times, c)
        results.append(
            IndividualPKResult(
                subject_id=i,
                cl_L_per_h=cl_i,
                vd_L=vd_i,
                ka_per_h=ka,
                cmax_mg_L=cmax,
                tmax_h=tmax,
                auc_mg_h_per_L=auc,
                times_h=times.tolist(),
                c_plasma_mg_L=c.tolist(),
            )
        )

    return results
