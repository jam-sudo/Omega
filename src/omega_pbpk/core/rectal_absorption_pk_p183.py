"""Rectal drug absorption pharmacokinetics (Phase 183).

Models rectal drug absorption via suppositories or enemas with a 2-compartment
ODE (rectal + systemic). Bioavailability is estimated from physicochemical
properties (logP, MW) because the rectal route partially bypasses hepatic
first-pass metabolism.

ODEs (Forward Euler):
    dA_rect/dt  = -ka_rect * A_rect
    dC_plasma/dt = F_rect * ka_rect * A_rect / Vd - ke * C_plasma

Bioavailability formula:
    F_rect = min(1.0, 0.7 + 0.1 * logP_factor - 0.05 * mw_factor)
    logP_factor = clamp(logP / 3, 0, 1)
    mw_factor   = clamp((mw - 200) / 800, 0, 1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RectalPKResult:
    """Result of a rectal PK simulation (Phase 183)."""

    drug_name: str
    dose_mg: float
    logP: float
    mw_Da: float
    f_rect: float
    ka_rect_per_h: float
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    t_half_h: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def _compute_f_rect(logP: float, mw_Da: float) -> float:
    """Estimate rectal bioavailability from physicochemical properties.

    Parameters
    ----------
    logP:
        Octanol-water partition coefficient.
    mw_Da:
        Molecular weight (Da).

    Returns
    -------
    float in (0, 1]
    """
    logP_factor = _clamp(logP / 3.0, 0.0, 1.0)
    mw_factor = _clamp((mw_Da - 200.0) / 800.0, 0.0, 1.0)
    f_rect = 0.7 + 0.1 * logP_factor - 0.05 * mw_factor
    return min(1.0, max(1e-6, f_rect))  # ensure always in (0, 1]


def _trapz_auc(times: list, concs: list) -> float:
    """Compute AUC using the trapezoidal rule."""
    return sum(
        (concs[i] + concs[i + 1]) / 2.0 * (times[i + 1] - times[i]) for i in range(len(concs) - 1)
    )


def _validate_simulate_inputs(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_rectal_pk(
    drug_name: str,
    dose_mg: float,
    logP: float = 2.0,
    mw_Da: float = 300.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    ka_rect_per_h: float = 1.5,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> RectalPKResult:
    """Simulate rectal drug absorption pharmacokinetics.

    Parameters
    ----------
    drug_name:
        Drug identifier string.
    dose_mg:
        Rectal dose (mg).
    logP:
        Octanol-water partition coefficient (dimensionless).
    mw_Da:
        Molecular weight (Da).
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    ka_rect_per_h:
        First-order rectal absorption rate constant (1/h). Default 1.5 /h.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler step size (h).

    Returns
    -------
    RectalPKResult

    Raises
    ------
    ValueError
        If dose_mg, cl_L_per_h, or vd_L are non-positive.
    """
    _validate_simulate_inputs(dose_mg, cl_L_per_h, vd_L)

    f_rect = _compute_f_rect(logP, mw_Da)
    ke = cl_L_per_h / vd_L
    t_half_h = math.log(2.0) / ke

    n_steps = max(int(math.ceil(t_end_h / dt_h)), 1)

    # Initial state
    a_rect = dose_mg  # mg remaining in rectal compartment
    c_plasma = 0.0  # mg/L

    times_h: list = [0.0]
    c_plasma_list: list = [0.0]

    for _ in range(n_steps):
        # ODE RHS
        dc = f_rect * ka_rect_per_h * a_rect / vd_L - ke * c_plasma
        da = -ka_rect_per_h * a_rect

        # Forward Euler update
        c_plasma = max(c_plasma + dc * dt_h, 0.0)
        a_rect = max(a_rect + da * dt_h, 0.0)

        times_h.append(times_h[-1] + dt_h)
        c_plasma_list.append(c_plasma)

    # PK metrics
    cmax_mg_L = max(c_plasma_list)
    tmax_idx = c_plasma_list.index(cmax_mg_L)
    tmax_h = times_h[tmax_idx]
    auc_mg_h_per_L = _trapz_auc(times_h, c_plasma_list)

    notes = (
        f"Rectal PK (Phase 183): F_rect={f_rect:.3f} estimated from logP={logP}, "
        f"MW={mw_Da} Da. ka_rect={ka_rect_per_h}/h, ke={ke:.4f}/h. "
        f"Cmax={cmax_mg_L:.4f} mg/L at Tmax={tmax_h:.2f} h. "
        f"AUC={auc_mg_h_per_L:.4f} mg·h/L. t½={t_half_h:.2f} h."
    )

    return RectalPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        mw_Da=mw_Da,
        f_rect=f_rect,
        ka_rect_per_h=ka_rect_per_h,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc_mg_h_per_L,
        t_half_h=t_half_h,
        notes=notes,
    )


def compare_suppository_formulations(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    formulations: list,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> list:
    """Compare suppository formulations by simulated AUC.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Dose (mg), same for all formulations.
    logP:
        Octanol-water partition coefficient.
    mw_Da:
        Molecular weight (Da).
    formulations:
        List of dicts, each with keys:
            ``name``        (str)  — formulation label
            ``ka_rect_per_h`` (float) — absorption rate constant (1/h)
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler step size (h).

    Returns
    -------
    list of RectalPKResult sorted by AUC descending.
    """
    results = []
    for form in formulations:
        result = simulate_rectal_pk(
            drug_name=f"{drug_name} [{form['name']}]",
            dose_mg=dose_mg,
            logP=logP,
            mw_Da=mw_Da,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            ka_rect_per_h=float(form["ka_rect_per_h"]),
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_mg_h_per_L, reverse=True)
    return results


__all__ = [
    "RectalPKResult",
    "simulate_rectal_pk",
    "compare_suppository_formulations",
]
