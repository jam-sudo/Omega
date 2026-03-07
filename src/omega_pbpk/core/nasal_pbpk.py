"""Nasal drug absorption PBPK model.

Models the nasal route for both local delivery (e.g., corticosteroids) and
systemic delivery (e.g., sumatriptan, desmopressin).

Model Structure (3-compartment Forward Euler ODE)
-------------------------------------------------
Nasal cavity -> Systemic circulation (direct, no first-pass)
Mucociliary clearance -> Swallowed depot -> Systemic (with oral first-pass)

ODEs:
    dA_nasal/dt = -(ka_nasal + k_mc) * A_nasal
    dA_swallowed/dt = k_mc * A_nasal - ka_sw * A_swallowed
    dC_plasma/dt = (ka_nasal * A_nasal + ka_sw * A_swallowed * f_oral_swallowed) / Vd
                   - ke * C_plasma

References
----------
- Djupesland PG, Drug Deliv Transl Res. 2013 (nasal drug delivery review)
- Illum L, Eur J Pharm Sci. 2003 (nasal absorption)
- Gizurarson S, Am J Drug Deliv. 2012 (mucociliary clearance)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["NasalPKResult", "simulate_nasal_pk", "compare_nasal_routes"]


@dataclass
class NasalPKResult:
    """Result of nasal PK simulation.

    Attributes
    ----------
    drug_name : Name of the drug.
    dose_mg : Administered dose (mg).
    times_h : Simulation time points (h).
    a_nasal_mg : Drug mass in nasal cavity over time (mg).
    c_plasma_mg_L : Plasma concentration over time (mg/L).
    cmax_mg_L : Peak plasma concentration (mg/L).
    tmax_h : Time of Cmax (h).
    auc_mg_h_per_L : AUC from 0 to t_end (mg*h/L), trapezoidal.
    t_half_h : Elimination half-life (h) = ln(2)*Vd/CL.
    f_nasal : Fraction absorbed directly via nasal epithelium.
    f_swallowed : Fraction removed via mucociliary clearance (swallowed).
    f_systemic_total : Total systemic bioavailability.
    notes : Summary text.
    """

    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    a_nasal_mg: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    t_half_h: float = 0.0
    f_nasal: float = 0.0
    f_swallowed: float = 0.0
    f_systemic_total: float = 0.0
    notes: str = ""


def _trapz(y: list, x: list) -> float:
    """Trapezoidal integration."""
    total = 0.0
    for i in range(1, len(x)):
        total += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return total


def simulate_nasal_pk(
    drug_name: str,
    dose_mg: float,
    ka_nasal_per_h: float = 1.0,
    k_mc_per_h: float = 1.0,
    ka_sw_per_h: float = 0.5,
    f_oral_swallowed: float = 0.3,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 8.0,
    dt_h: float = 0.05,
) -> NasalPKResult:
    """Simulate nasal drug pharmacokinetics using Forward Euler ODE.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Nasal dose (mg).
    ka_nasal_per_h : Nasal epithelial absorption rate constant (1/h).
    k_mc_per_h : Mucociliary clearance rate constant (1/h).
    ka_sw_per_h : Absorption rate of swallowed fraction (1/h).
    f_oral_swallowed : Oral bioavailability of swallowed fraction (0-1).
    cl_L_per_h : Systemic clearance (L/h).
    vd_L : Volume of distribution (L).
    t_end_h : Simulation duration (h).
    dt_h : Time step (h).

    Returns
    -------
    NasalPKResult with full time-course and PK summary.

    Raises
    ------
    ValueError : If parameters are invalid.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if ka_nasal_per_h <= 0:
        raise ValueError("ka_nasal_per_h must be > 0")
    if k_mc_per_h <= 0:
        raise ValueError("k_mc_per_h must be > 0")
    if ka_sw_per_h <= 0:
        raise ValueError("ka_sw_per_h must be > 0")
    if not (0 < f_oral_swallowed <= 1):
        raise ValueError("f_oral_swallowed must be in (0, 1]")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    ke = cl_L_per_h / vd_L
    t_half_h = math.log(2.0) * vd_L / cl_L_per_h

    # Fractions
    k_total = ka_nasal_per_h + k_mc_per_h
    f_nasal = ka_nasal_per_h / k_total
    f_swallowed = k_mc_per_h / k_total
    f_systemic_total = f_nasal + f_swallowed * f_oral_swallowed

    # Forward Euler simulation
    n_steps = max(int(t_end_h / dt_h), 1)
    times = [0.0] * (n_steps + 1)
    a_nasal_arr = [0.0] * (n_steps + 1)
    a_swallowed_arr = [0.0] * (n_steps + 1)
    c_plasma_arr = [0.0] * (n_steps + 1)

    a_nasal_arr[0] = dose_mg
    a_swallowed_arr[0] = 0.0
    c_plasma_arr[0] = 0.0

    for i in range(n_steps):
        times[i + 1] = (i + 1) * dt_h

        a_n = a_nasal_arr[i]
        a_sw = a_swallowed_arr[i]
        c_p = c_plasma_arr[i]

        # ODEs
        da_nasal = -(ka_nasal_per_h + k_mc_per_h) * a_n
        da_swallowed = k_mc_per_h * a_n - ka_sw_per_h * a_sw
        dc_plasma = (ka_nasal_per_h * a_n + ka_sw_per_h * a_sw * f_oral_swallowed) / vd_L - ke * c_p

        a_nasal_arr[i + 1] = max(0.0, a_n + da_nasal * dt_h)
        a_swallowed_arr[i + 1] = max(0.0, a_sw + da_swallowed * dt_h)
        c_plasma_arr[i + 1] = max(0.0, c_p + dc_plasma * dt_h)

    # PK metrics
    cmax = max(c_plasma_arr)
    tmax_h = times[c_plasma_arr.index(cmax)]
    auc = _trapz(c_plasma_arr, times)

    notes = (
        f"Nasal PK: {drug_name}, {dose_mg} mg nasal dose. "
        f"f_nasal={f_nasal:.2f}, f_swallowed={f_swallowed:.2f}, "
        f"f_systemic_total={f_systemic_total:.2f}. "
        f"Cmax={cmax:.4f} mg/L at {tmax_h:.2f} h, AUC={auc:.3f} mg*h/L, "
        f"t1/2={t_half_h:.2f} h."
    )

    return NasalPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        a_nasal_mg=a_nasal_arr,
        c_plasma_mg_L=c_plasma_arr,
        cmax_mg_L=cmax,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        t_half_h=t_half_h,
        f_nasal=f_nasal,
        f_swallowed=f_swallowed,
        f_systemic_total=f_systemic_total,
        notes=notes,
    )


def compare_nasal_routes(
    drug_name: str,
    dose_mg: float,
    scenarios: list,
) -> list:
    """Compare multiple nasal delivery scenarios.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose (mg).
    scenarios : List of dicts with optional keys:
        label, ka_nasal_per_h, k_mc_per_h, f_oral_swallowed,
        ka_sw_per_h, cl_L_per_h, vd_L, t_end_h, dt_h

    Returns
    -------
    List of NasalPKResult, one per scenario.
    """
    results = []
    for sc in scenarios:
        result = simulate_nasal_pk(
            drug_name=sc.get("label", drug_name),
            dose_mg=dose_mg,
            ka_nasal_per_h=sc.get("ka_nasal_per_h", 1.0),
            k_mc_per_h=sc.get("k_mc_per_h", 1.0),
            ka_sw_per_h=sc.get("ka_sw_per_h", 0.5),
            f_oral_swallowed=sc.get("f_oral_swallowed", 0.3),
            cl_L_per_h=sc.get("cl_L_per_h", 5.0),
            vd_L=sc.get("vd_L", 50.0),
            t_end_h=sc.get("t_end_h", 8.0),
            dt_h=sc.get("dt_h", 0.05),
        )
        results.append(result)
    return results
