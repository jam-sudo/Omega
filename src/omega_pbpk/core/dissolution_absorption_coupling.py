"""Phase 859 — Dissolution-Absorption Coupling.

Mechanistic model coupling Noyes-Whitney dissolution with intestinal
absorption in a single GI compartment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DissolutionAbsorptionResult:
    """Result of dissolution-absorption simulation."""

    drug_name: str
    dose_mg: float
    k_diss_per_h: float
    solubility_mg_mL: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    a_solid_mg: list[float]
    a_dissolved_mg: list[float]
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    f_dissolved: float
    f_absorbed: float
    dissolution_limited: bool
    notes: str


def simulate_dissolution_absorption(
    drug_name: str,
    dose_mg: float,
    k_diss_per_h: float = 0.5,
    solubility_mg_mL: float = 1.0,
    v_gi_mL: float = 250.0,
    ka_per_h: float = 1.0,
    k_transit_per_h: float = 0.1,
    vd_L: float = 10.0,
    cl_L_per_h: float = 2.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> DissolutionAbsorptionResult:
    """Simulate dissolution-absorption coupling.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in mg.
    k_diss_per_h:
        Dissolution rate constant (1/h).
    solubility_mg_mL:
        Drug solubility in GI fluid (mg/mL).
    v_gi_mL:
        GI fluid volume (mL).
    ka_per_h:
        Intestinal absorption rate constant (1/h).
    k_transit_per_h:
        GI transit rate clearing dissolved drug (1/h).
    vd_L:
        Volume of distribution (L).
    cl_L_per_h:
        Clearance (L/h).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    DissolutionAbsorptionResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if k_diss_per_h <= 0:
        raise ValueError("k_diss_per_h must be > 0")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)
    cs_mg_per_mL = solubility_mg_mL
    # Maximum dissolved drug at solubility (mg)
    cs_total_mg = cs_mg_per_mL * v_gi_mL

    # Initial conditions
    a_solid = dose_mg
    a_dissolved = 0.0
    c_plasma = 0.0

    # Tracking
    n_steps = int(math.ceil(t_end_h / dt_h))
    times_h: list[float] = []
    c_plasma_list: list[float] = []
    a_solid_list: list[float] = []
    a_dissolved_list: list[float] = []

    absorbed_mg = 0.0  # cumulative absorbed amount

    t = 0.0
    for _ in range(n_steps + 1):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        a_solid_list.append(a_solid)
        a_dissolved_list.append(a_dissolved)

        if t >= t_end_h:
            break

        # Dissolution: Noyes-Whitney simplified sink model
        # Rate = k_diss * max(0, Cs*V_gi - A_dissolved)
        sink_capacity = max(0.0, cs_total_mg - a_dissolved)
        dissolution_rate = k_diss_per_h * min(a_solid, sink_capacity)

        # ODE right-hand sides
        d_solid = -dissolution_rate
        d_dissolved = dissolution_rate - ka_per_h * a_dissolved - k_transit_per_h * a_dissolved
        absorption_flux = ka_per_h * a_dissolved  # mg/h
        d_c_plasma = absorption_flux / vd_L - ke * c_plasma  # (mg/h)/L - (1/h)*(mg/L) = mg/L/h

        # Forward Euler step
        a_solid = max(0.0, a_solid + d_solid * dt_h)
        a_dissolved = max(0.0, a_dissolved + d_dissolved * dt_h)
        c_plasma = max(0.0, c_plasma + d_c_plasma * dt_h)

        # Track absorbed amount (integral of absorption flux)
        absorbed_mg += absorption_flux * dt_h

        t = min(t + dt_h, t_end_h)

    # Compute PK metrics
    cmax_mg_L = max(c_plasma_list)
    tmax_h = times_h[c_plasma_list.index(cmax_mg_L)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times_h[i] - times_h[i - 1])

    # Fractions
    total_dissolved_mg = dose_mg - a_solid_list[-1]
    f_dissolved = min(1.0, max(0.0, total_dissolved_mg / dose_mg))
    f_absorbed = min(1.0, max(0.0, absorbed_mg / dose_mg))
    dissolution_limited = f_dissolved < 0.9

    notes = (
        f"Simulation of {drug_name}: dose={dose_mg}mg, k_diss={k_diss_per_h}/h, "
        f"Cs={solubility_mg_mL}mg/mL. "
        f"{'Dissolution-limited absorption.' if dissolution_limited else 'Complete dissolution achieved.'}"
    )

    return DissolutionAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        k_diss_per_h=k_diss_per_h,
        solubility_mg_mL=solubility_mg_mL,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        a_solid_mg=a_solid_list,
        a_dissolved_mg=a_dissolved_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        f_dissolved=f_dissolved,
        f_absorbed=f_absorbed,
        dissolution_limited=dissolution_limited,
        notes=notes,
    )


def screen_particle_sizes(
    drug_name: str,
    dose_mg: float,
    k_diss_values: list[float],
    solubility_mg_mL: float = 1.0,
) -> list[DissolutionAbsorptionResult]:
    """Screen particle sizes by varying dissolution rate constants.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in mg.
    k_diss_values:
        List of dissolution rate constants to screen (1/h).
    solubility_mg_mL:
        Drug solubility in GI fluid (mg/mL).

    Returns
    -------
    List of DissolutionAbsorptionResult sorted by auc_mg_h_per_L descending.
    """
    results = []
    for k_diss in k_diss_values:
        result = simulate_dissolution_absorption(
            drug_name=drug_name,
            dose_mg=dose_mg,
            k_diss_per_h=k_diss,
            solubility_mg_mL=solubility_mg_mL,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_mg_h_per_L, reverse=True)
    return results
