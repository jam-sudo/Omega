"""Intranasal drug delivery pharmacokinetics (Phase 819).

Models nasal absorption with mucociliary clearance competing against systemic
absorption. Drug deposited in the nasal cavity is either absorbed directly into
the systemic circulation (bypassing hepatic first-pass) or cleared to the GI
tract by mucociliary transport, where a fraction may be absorbed orally.

ODE system (Forward Euler)
--------------------------
Compartments:
    nasal_depot (mg) — drug in nasal cavity available for absorption
    swallowed   (mg) — drug entering GI from mucociliary clearance + undeposited
    plasma      (mg/L)

ODEs:
    d(nasal_depot)/dt = -(ka_nasal + mucociliary_rate) * nasal_depot
    d(swallowed)/dt   = mucociliary_rate * nasal_depot
    d(plasma)/dt      = ka_nasal * nasal_depot / vd
                      + KA_GI * f_swallowed_absorbed * swallowed / vd
                      - (cl / vd) * plasma

Initial conditions:
    nasal_depot(0) = dose_mg * f_nasal
    swallowed(0)   = dose_mg * (1 - f_nasal)   # undeposited fraction
    plasma(0)      = 0.0

References
----------
- Djupesland PG, Acta Otolaryngol. 2013;133:S1-15
- Illum L, J Control Release. 2003;87(1-3):187-198
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "IntranasalPBPKResult",
    "simulate_intranasal_pbpk",
    "compare_nasal_oral",
]

# GI absorption rate for swallowed fraction (h^-1) — typical oral ka
_KA_GI_PER_H: float = 0.5


def _trapz(y: list, x: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(len(x) - 1):
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return total


@dataclass(frozen=True)
class IntranasalPBPKResult:
    """Result of intranasal PBPK simulation (Phase 819).

    Attributes
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Total administered dose (mg).
    route : str
        Always "intranasal".
    cmax_mg_L : float
        Peak plasma concentration (mg/L).
    tmax_h : float
        Time of peak plasma concentration (h).
    auc_mg_h_per_L : float
        AUC from 0 to t_end (mg·h/L), trapezoidal.
    bioavailability_nasal : float
        Fraction of dose absorbed via the nasal route (0–1).
    fraction_cleared_mucociliary : float
        Fraction of dose lost from nasal cavity via mucociliary transport.
    fraction_swallowed : float
        Total fraction that ends up in the GI tract (undeposited + mucociliary).
    t_half_h : float
        Systemic half-life from CL/Vd (h).
    times_h : list
        Time points (h).
    c_plasma_mg_L : list
        Plasma concentration at each time point (mg/L).
    notes : str
        Human-readable summary.
    """

    drug_name: str
    dose_mg: float
    route: str
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    bioavailability_nasal: float
    fraction_cleared_mucociliary: float
    fraction_swallowed: float
    t_half_h: float
    times_h: list
    c_plasma_mg_L: list
    notes: str


def simulate_intranasal_pbpk(
    drug_name: str,
    dose_mg: float,
    ka_nasal_per_h: float = 3.0,
    f_nasal: float = 0.7,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    mucociliary_rate_per_h: float = 1.5,
    f_swallowed_absorbed: float = 0.3,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> IntranasalPBPKResult:
    """Simulate intranasal drug delivery pharmacokinetics.

    Parameters
    ----------
    drug_name:
        Name or identifier of the drug.
    dose_mg:
        Total administered dose (mg). Must be > 0.
    ka_nasal_per_h:
        Absorption rate constant from the nasal cavity into systemic
        circulation (h^-1). Default 3.0.
    f_nasal:
        Fraction of dose deposited in the nasal cavity and available for
        nasal absorption (0 to 1). The remainder (1 - f_nasal) is placed
        into the swallowed depot at t = 0. Default 0.7.
    cl_L_per_h:
        Systemic clearance (L/h). Must be > 0. Default 5.0.
    vd_L:
        Volume of distribution (L). Must be > 0. Default 50.0.
    mucociliary_rate_per_h:
        Rate constant for mucociliary clearance of drug from the nasal
        cavity (h^-1). Default 1.5.
    f_swallowed_absorbed:
        Fraction of swallowed drug ultimately absorbed from the GI tract
        (oral bioavailability of swallowed portion). Default 0.3.
    t_end_h:
        Simulation duration (h). Default 12.0.
    dt_h:
        Forward Euler time step (h). Default 0.05.

    Returns
    -------
    IntranasalPBPKResult

    Raises
    ------
    ValueError
        If dose_mg <= 0, cl_L_per_h <= 0, vd_L <= 0, or f_nasal is
        outside [0, 1].
    """
    # --- Validation ---
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0.0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0.0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if not (0.0 <= f_nasal <= 1.0):
        raise ValueError(f"f_nasal must be in [0, 1], got {f_nasal}")

    # --- Derived constants ---
    ke = cl_L_per_h / vd_L  # elimination rate constant (h^-1)

    # --- Initial compartment amounts ---
    nasal_depot = dose_mg * f_nasal
    swallowed = dose_mg * (1.0 - f_nasal)  # undeposited fraction starts in GI
    plasma = 0.0

    # --- Accumulate absorbed amounts for bioavailability ---
    total_nasal_absorbed_mg = 0.0

    # --- Build time-course arrays ---
    n_steps = max(int(math.ceil(t_end_h / dt_h)), 1)
    times: list = []
    c_plasma: list = []

    t = 0.0
    for i in range(n_steps + 1):
        times.append(t)
        c_plasma.append(plasma)

        if i < n_steps:
            # Rates (mg/h or mg/L/h)
            nasal_abs_rate = ka_nasal_per_h * nasal_depot
            _mucociliary_rate_out = mucociliary_rate_per_h * nasal_depot
            gi_abs_rate = _KA_GI_PER_H * swallowed * f_swallowed_absorbed

            d_nasal = -(ka_nasal_per_h + mucociliary_rate_per_h) * nasal_depot
            d_swallowed = mucociliary_rate_per_h * nasal_depot
            d_plasma = (nasal_abs_rate + gi_abs_rate) / vd_L - ke * plasma

            # Forward Euler update
            nasal_depot = max(nasal_depot + d_nasal * dt_h, 0.0)
            swallowed = max(swallowed + d_swallowed * dt_h, 0.0)
            plasma = max(plasma + d_plasma * dt_h, 0.0)

            total_nasal_absorbed_mg += nasal_abs_rate * dt_h
            t = min(t + dt_h, t_end_h)

    # --- PK metrics ---
    auc = _trapz(c_plasma, times)
    cmax = max(c_plasma)
    tmax = times[c_plasma.index(cmax)]

    bioavailability_nasal = min(total_nasal_absorbed_mg / dose_mg, 1.0)

    # fraction_cleared_mucociliary: fraction of the *nasal depot* that was
    # cleared by mucociliary transport = mucociliary_rate / (ka_nasal + mucociliary_rate)
    # ... scaled by the fraction actually deposited nasally (f_nasal)
    ka_total = ka_nasal_per_h + mucociliary_rate_per_h
    if ka_total > 0.0:
        fraction_cleared_mucociliary = mucociliary_rate_per_h / ka_total * f_nasal
    else:
        fraction_cleared_mucociliary = 0.0

    # fraction_swallowed: undeposited portion + mucociliary-cleared nasal portion
    fraction_swallowed = (1.0 - f_nasal) + fraction_cleared_mucociliary

    # Systemic half-life
    t_half = math.log(2.0) / ke if ke > 0.0 else float("inf")

    # --- Notes ---
    notes_parts = [
        (
            f"Intranasal PBPK: f_nasal={f_nasal:.2f}, "
            f"ka_nasal={ka_nasal_per_h:.2f}/h, "
            f"mucociliary={mucociliary_rate_per_h:.2f}/h"
        ),
        (
            f"Nasal bioavailability={bioavailability_nasal:.1%}; "
            f"mucociliary fraction={fraction_cleared_mucociliary:.1%}"
        ),
    ]
    if bioavailability_nasal < 0.3:
        notes_parts.append(
            "Low nasal bioavailability; consider higher ka_nasal or reduced mucociliary rate"
        )
    notes = "; ".join(notes_parts)

    return IntranasalPBPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route="intranasal",
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        bioavailability_nasal=bioavailability_nasal,
        fraction_cleared_mucociliary=fraction_cleared_mucociliary,
        fraction_swallowed=fraction_swallowed,
        t_half_h=t_half,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        notes=notes,
    )


def compare_nasal_oral(
    drug_name: str,
    dose_mg: float,
    ka_nasal_per_h: float = 3.0,
    f_nasal: float = 0.7,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    f_oral: float = 0.5,
    ka_oral_per_h: float = 1.0,
    mucociliary_rate_per_h: float = 1.5,
    f_swallowed_absorbed: float = 0.3,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> dict:
    """Compare intranasal vs oral administration for the same drug and dose.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Dose (mg) — same for both routes.
    ka_nasal_per_h:
        Nasal absorption rate constant (h^-1).
    f_nasal:
        Nasal deposition fraction.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    f_oral:
        Oral bioavailability fraction (0 to 1).
    ka_oral_per_h:
        Oral absorption rate constant (h^-1).
    mucociliary_rate_per_h:
        Mucociliary clearance rate (h^-1).
    f_swallowed_absorbed:
        GI absorption fraction of swallowed drug.
    t_end_h:
        Simulation duration (h).
    dt_h:
        Time step (h).

    Returns
    -------
    dict with keys:
        "intranasal" : IntranasalPBPKResult
        "oral"       : dict with cmax_mg_L, auc_mg_h_per_L, tmax_h,
                       bioavailability, t_half_h
    """
    in_result = simulate_intranasal_pbpk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        ka_nasal_per_h=ka_nasal_per_h,
        f_nasal=f_nasal,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        mucociliary_rate_per_h=mucociliary_rate_per_h,
        f_swallowed_absorbed=f_swallowed_absorbed,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    # Oral simulation: 1-cpt forward Euler
    ke = cl_L_per_h / vd_L
    oral_depot = dose_mg * f_oral
    plasma_oral = 0.0
    oral_times: list = []
    oral_conc: list = []

    n_steps = max(int(math.ceil(t_end_h / dt_h)), 1)
    t = 0.0
    for i in range(n_steps + 1):
        oral_times.append(t)
        oral_conc.append(plasma_oral)
        if i < n_steps:
            abs_rate = ka_oral_per_h * oral_depot
            oral_depot = max(oral_depot - ka_oral_per_h * oral_depot * dt_h, 0.0)
            plasma_oral = max(plasma_oral + (abs_rate / vd_L - ke * plasma_oral) * dt_h, 0.0)
            t = min(t + dt_h, t_end_h)

    oral_auc = _trapz(oral_conc, oral_times)
    oral_cmax = max(oral_conc)
    oral_tmax = oral_times[oral_conc.index(oral_cmax)]
    t_half = math.log(2.0) / ke if ke > 0.0 else float("inf")

    return {
        "intranasal": in_result,
        "oral": {
            "cmax_mg_L": oral_cmax,
            "auc_mg_h_per_L": oral_auc,
            "tmax_h": oral_tmax,
            "bioavailability": f_oral,
            "t_half_h": t_half,
        },
    }
