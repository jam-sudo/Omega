"""Pharmacokinetic variability decomposition.

Phase 680: Separates and quantifies sources of PK variability using
log-normal IIV simulation with LCG-based deterministic RNG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PKVariabilityResult",
    "decompose_pk_variability",
    "simulate_iiv_profiles",
    "variability_summary",
]

# ---------------------------------------------------------------------------
# LCG random number generator (pure Python, deterministic)
# ---------------------------------------------------------------------------

_LCG_A = 1664525
_LCG_C = 1013904223
_LCG_M = 2**32


def _lcg_next(state: int) -> tuple[int, float]:
    """Advance LCG state; return (new_state, uniform_0_1)."""
    state = (_LCG_A * state + _LCG_C) % _LCG_M
    return state, state / _LCG_M


def _lcg_normal(state: int) -> tuple[int, float]:
    """Approximate N(0,1) sample scaled from LCG uniform via [-sqrt(3), sqrt(3)] range.

    Uses Box-Muller-like mapping: Z = (u - 0.5) * 2 * sqrt(3)
    This gives a symmetric distribution with mean=0, variance=1 for uniform u in [0,1].
    """
    state, u = _lcg_next(state)
    z = (u - 0.5) * 2.0 * math.sqrt(3.0)
    return state, z


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PKVariabilityResult:
    """Result of PK variability decomposition."""

    n_subjects: int
    cl_cv_pct: float
    vd_cv_pct: float
    auc_cv_pct: float
    cmax_cv_pct: float
    notes: str


# ---------------------------------------------------------------------------
# Helper: arithmetic statistics (pure Python)
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _cv_pct(values: list[float]) -> float:
    m = _mean(values)
    if m == 0.0:
        return 0.0
    return (_std(values) / abs(m)) * 100.0


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Compute p-th percentile (0–100) from a sorted list."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = p / 100.0 * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def simulate_iiv_profiles(
    n_subjects: int,
    cl_mean: float,
    vd_mean: float,
    ka_mean: float,
    omega_cl: float = 0.3,
    omega_vd: float = 0.3,
    omega_ka: float = 0.5,
    dose_mg: float = 100.0,
    seed: int = 42,
    t_end_h: float = 24.0,
    dt_h: float = 0.5,
) -> list[dict]:
    """Generate virtual individual PK profiles with log-normal IIV.

    Parameters
    ----------
    n_subjects:
        Number of virtual subjects (>= 2).
    cl_mean:
        Population mean clearance (L/h).
    vd_mean:
        Population mean volume of distribution (L).
    ka_mean:
        Population mean absorption rate constant (1/h).
    omega_cl, omega_vd, omega_ka:
        Standard deviation of log-normal IIV for CL, Vd, ka.
    dose_mg:
        Oral dose in mg.
    seed:
        LCG seed for reproducibility.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    list[dict]
        Each dict contains: subject_id, CL, Vd, ka, cmax, tmax, auc, t_half.
    """
    if n_subjects < 2:
        raise ValueError(f"n_subjects must be >= 2, got {n_subjects}")
    if cl_mean <= 0:
        raise ValueError(f"cl_mean must be > 0, got {cl_mean}")
    if vd_mean <= 0:
        raise ValueError(f"vd_mean must be > 0, got {vd_mean}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if omega_cl < 0:
        raise ValueError(f"omega_cl must be >= 0, got {omega_cl}")
    if omega_vd < 0:
        raise ValueError(f"omega_vd must be >= 0, got {omega_vd}")

    state = seed
    profiles: list[dict] = []

    n_steps = max(int(t_end_h / dt_h), 1)

    for subj_idx in range(n_subjects):
        # Sample eta from approximate N(0,1) via LCG
        state, z_cl = _lcg_normal(state)
        state, z_vd = _lcg_normal(state)
        state, z_ka = _lcg_normal(state)

        cl_i = cl_mean * math.exp(omega_cl * z_cl)
        vd_i = vd_mean * math.exp(omega_vd * z_vd)
        ka_i = ka_mean * math.exp(omega_ka * z_ka)

        # Forward Euler 1-cpt oral model: dAa/dt = -ka*Aa; dAc/dt = ka*Aa - (CL/Vd)*Ac
        aa = dose_mg  # mg in absorption compartment
        ac = 0.0  # mg in central compartment

        ke = cl_i / vd_i
        cmax = 0.0
        tmax = 0.0
        auc = 0.0
        prev_conc = 0.0

        for step in range(n_steps):
            t = step * dt_h
            conc = ac / vd_i

            # Trapezoidal AUC
            if step > 0:
                auc += 0.5 * (prev_conc + conc) * dt_h

            if conc > cmax:
                cmax = conc
                tmax = t

            # Euler update
            daa = -ka_i * aa
            dac = ka_i * aa - ke * ac
            aa = max(aa + daa * dt_h, 0.0)
            ac = max(ac + dac * dt_h, 0.0)
            prev_conc = conc

        # Add last trapezoid step
        conc_final = ac / vd_i
        auc += 0.5 * (prev_conc + conc_final) * dt_h

        t_half = math.log(2.0) / ke if ke > 0 else float("inf")

        profiles.append(
            {
                "subject_id": subj_idx,
                "CL": cl_i,
                "Vd": vd_i,
                "ka": ka_i,
                "cmax": cmax,
                "tmax": tmax,
                "auc": auc,
                "t_half": t_half,
            }
        )

    return profiles


def variability_summary(profiles: list[dict]) -> dict:
    """Summarize PK variability across virtual subjects.

    Parameters
    ----------
    profiles:
        List of dicts as returned by ``simulate_iiv_profiles``.

    Returns
    -------
    dict
        Summary statistics including CV%, percentiles for AUC and Cmax.
    """
    cl_vals = [p["CL"] for p in profiles]
    vd_vals = [p["Vd"] for p in profiles]
    ka_vals = [p["ka"] for p in profiles]
    cmax_vals = [p["cmax"] for p in profiles]
    auc_vals = [p["auc"] for p in profiles]

    auc_sorted = sorted(auc_vals)
    cmax_sorted = sorted(cmax_vals)
    n = len(profiles)

    return {
        "n_subjects": n,
        "cl_cv_pct": _cv_pct(cl_vals),
        "vd_cv_pct": _cv_pct(vd_vals),
        "ka_cv_pct": _cv_pct(ka_vals),
        "cmax_cv_pct": _cv_pct(cmax_vals),
        "auc_cv_pct": _cv_pct(auc_vals),
        "auc_p5": _percentile(auc_sorted, 5.0),
        "auc_p50": _percentile(auc_sorted, 50.0),
        "auc_p95": _percentile(auc_sorted, 95.0),
        "cmax_p5": _percentile(cmax_sorted, 5.0),
        "cmax_p50": _percentile(cmax_sorted, 50.0),
        "cmax_p95": _percentile(cmax_sorted, 95.0),
    }


def decompose_pk_variability(
    pk_profiles: list[dict],
    param_names: list[str],
) -> PKVariabilityResult:
    """Decompose PK variability across a list of individual profiles.

    Parameters
    ----------
    pk_profiles:
        List of dicts; each should contain the keys listed in ``param_names``
        as well as "CL", "Vd", "auc", "cmax" if present.
    param_names:
        List of parameter names to compute CV% for.

    Returns
    -------
    PKVariabilityResult
        Variability summary for CL, Vd, AUC, Cmax.
    """
    n = len(pk_profiles)
    if n < 2:
        raise ValueError(f"pk_profiles must contain >= 2 subjects, got {n}")

    def _extract_cv(key: str) -> float:
        vals = [float(p[key]) for p in pk_profiles if key in p]
        if len(vals) < 2:
            return 0.0
        return _cv_pct(vals)

    cl_cv = _extract_cv("CL")
    vd_cv = _extract_cv("Vd")
    auc_cv = _extract_cv("auc")
    cmax_cv = _extract_cv("cmax")

    param_summaries = []
    for pname in param_names:
        vals = [float(p[pname]) for p in pk_profiles if pname in p]
        if len(vals) >= 2:
            cv = _cv_pct(vals)
            param_summaries.append(f"{pname}: CV%={cv:.1f}")

    notes = f"n_subjects={n}; params=[{', '.join(param_summaries)}]"

    return PKVariabilityResult(
        n_subjects=n,
        cl_cv_pct=cl_cv,
        vd_cv_pct=vd_cv,
        auc_cv_pct=auc_cv,
        cmax_cv_pct=cmax_cv,
        notes=notes,
    )
