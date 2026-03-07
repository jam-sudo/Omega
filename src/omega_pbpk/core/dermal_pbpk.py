"""Dermal absorption PBPK model (Phase 820).

Models transdermal/dermal drug absorption through skin layers using a
simplified depot-based approach with a stratum-corneum lag phase.

Model structure
---------------
Compartments:
    skin_depot (mg) — available dose in/on skin after SC retention
    plasma     (mg/L)

The stratum corneum acts as a diffusion barrier characterised by a lag time.
Before lag_time_h, no drug enters the plasma from the skin depot.
After lag_time_h, absorption proceeds at rate ka_dermal = kp * 10 (h^-1),
where the factor 10 converts from cm/h (assuming 0.1 cm vehicle thickness).

ODEs (Forward Euler, post-lag):
    d(skin_depot)/dt = -ka_dermal * skin_depot
    d(plasma)/dt     = ka_dermal * skin_depot / vd - (cl / vd) * plasma

Initial conditions:
    skin_depot(0) = total_dose * (1 - f_sc_retention)
    plasma(0)     = 0.0

References
----------
- Potts RO & Guy RH, Pharm Res. 1992;9(5):663-669 (Potts-Guy equation)
- Hadgraft J, Int J Pharm. 2004;272(1-2):1-11 (skin permeation review)
- Williams AC & Barry BW, Adv Drug Deliv Rev. 2004;56(5):603-618
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DermalPKResult",
    "simulate_dermal_absorption",
    "estimate_kp",
]

# Assumed vehicle thickness (cm) for converting kp (cm/h) to ka (h^-1)
_VEHICLE_THICKNESS_CM: float = 0.1

# Clamping limits for estimate_kp (cm/h)
_KP_MIN_CM_PER_H: float = 1e-6
_KP_MAX_CM_PER_H: float = 0.1


def _trapz(y: list, x: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(len(x) - 1):
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return total


@dataclass(frozen=True)
class DermalPKResult:
    """Result of dermal absorption PBPK simulation (Phase 820).

    Attributes
    ----------
    drug_name : str
        Name of the drug.
    dose_mg_per_cm2 : float
        Applied dose per unit area (mg/cm²).
    application_area_cm2 : float
        Area of skin to which drug is applied (cm²).
    total_dose_mg : float
        Total applied dose (mg) = dose_mg_per_cm2 * application_area_cm2.
    cmax_mg_L : float
        Peak plasma concentration (mg/L).
    tmax_h : float
        Time of peak concentration (h).
    auc_mg_h_per_L : float
        AUC from 0 to t_end (mg·h/L), trapezoidal.
    lag_time_h : float
        Diffusion lag time through the stratum corneum (h).
    steady_state_flux_mg_per_h_per_cm2 : float
        Approximate steady-state flux = kp * dose_mg_per_cm2 (mg/h/cm²).
    systemic_bioavailability : float
        Fraction of total dose reaching systemic circulation.
    stratum_corneum_retention_mg : float
        Drug retained in the stratum corneum at simulation end (mg).
    times_h : list
        Time points (h).
    c_plasma_mg_L : list
        Plasma concentration at each time point (mg/L).
    notes : str
        Human-readable summary.
    """

    drug_name: str
    dose_mg_per_cm2: float
    application_area_cm2: float
    total_dose_mg: float
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    lag_time_h: float
    steady_state_flux_mg_per_h_per_cm2: float
    systemic_bioavailability: float
    stratum_corneum_retention_mg: float
    times_h: list
    c_plasma_mg_L: list
    notes: str


def simulate_dermal_absorption(
    drug_name: str,
    dose_mg_per_cm2: float,
    application_area_cm2: float,
    kp_cm_per_h: float = 0.001,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    lag_time_h: float = 2.0,
    f_sc_retention: float = 0.1,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> DermalPKResult:
    """Simulate transdermal drug absorption using a skin-depot PBPK model.

    Parameters
    ----------
    drug_name:
        Name or identifier of the drug.
    dose_mg_per_cm2:
        Applied dose per unit skin area (mg/cm²). Must be > 0.
    application_area_cm2:
        Skin application area (cm²). Must be > 0.
    kp_cm_per_h:
        Skin permeability coefficient (cm/h). Must be > 0. Default 0.001.
    cl_L_per_h:
        Systemic clearance (L/h). Must be > 0. Default 5.0.
    vd_L:
        Volume of distribution (L). Must be > 0. Default 50.0.
    lag_time_h:
        Diffusion lag time through the stratum corneum (h). Default 2.0.
    f_sc_retention:
        Fraction of total dose retained in the stratum corneum (0 to 1).
        Default 0.1.
    t_end_h:
        Simulation duration (h). Default 24.0.
    dt_h:
        Forward Euler time step (h). Default 0.05.

    Returns
    -------
    DermalPKResult

    Raises
    ------
    ValueError
        If dose_mg_per_cm2 <= 0, kp_cm_per_h <= 0, or
        application_area_cm2 <= 0.
    """
    # --- Validation ---
    if dose_mg_per_cm2 <= 0.0:
        raise ValueError(f"dose_mg_per_cm2 must be > 0, got {dose_mg_per_cm2}")
    if kp_cm_per_h <= 0.0:
        raise ValueError(f"kp_cm_per_h must be > 0, got {kp_cm_per_h}")
    if application_area_cm2 <= 0.0:
        raise ValueError(f"application_area_cm2 must be > 0, got {application_area_cm2}")

    # --- Dose calculations ---
    total_dose_mg = dose_mg_per_cm2 * application_area_cm2
    sc_retention_mg = total_dose_mg * f_sc_retention
    available_dose_mg = total_dose_mg * (1.0 - f_sc_retention)

    # --- Derived absorption rate ---
    # ka_dermal (h^-1): simplified from kp/vehicle_thickness
    ka_dermal = kp_cm_per_h / _VEHICLE_THICKNESS_CM

    ke = cl_L_per_h / vd_L  # elimination rate constant (h^-1)

    # --- Steady-state flux (sink approximation) ---
    steady_state_flux = kp_cm_per_h * dose_mg_per_cm2  # mg/h/cm²

    # --- Forward Euler integration ---
    skin_depot = available_dose_mg
    plasma = 0.0

    n_steps = max(int(math.ceil(t_end_h / dt_h)), 1)
    times: list = []
    c_plasma: list = []

    total_absorbed_mg = 0.0
    t = 0.0

    for i in range(n_steps + 1):
        times.append(t)
        c_plasma.append(plasma)

        if i < n_steps:
            # No absorption before lag_time_h (stratum corneum diffusion lag)
            if t < lag_time_h:
                abs_rate = 0.0
            else:
                abs_rate = ka_dermal * skin_depot

            d_skin = -abs_rate
            d_plasma = abs_rate / vd_L - ke * plasma

            skin_depot = max(skin_depot + d_skin * dt_h, 0.0)
            plasma = max(plasma + d_plasma * dt_h, 0.0)

            total_absorbed_mg += abs_rate * dt_h
            t = min(t + dt_h, t_end_h)

    # --- PK metrics ---
    auc = _trapz(c_plasma, times)
    cmax = max(c_plasma)
    tmax = times[c_plasma.index(cmax)]

    systemic_bioavailability = min(total_absorbed_mg / total_dose_mg, 1.0)

    # Systemic half-life
    # t_half computed via ke below in notes

    # --- Notes ---
    notes_parts = [
        (
            f"Dermal PBPK: kp={kp_cm_per_h:.4f} cm/h, "
            f"area={application_area_cm2:.1f} cm², "
            f"lag={lag_time_h:.1f} h"
        ),
        (
            f"Total dose={total_dose_mg:.3f} mg; "
            f"SC retention={sc_retention_mg:.3f} mg ({f_sc_retention:.0%})"
        ),
        f"Systemic bioavailability={systemic_bioavailability:.1%}",
    ]
    if lag_time_h > 0.5 * t_end_h:
        notes_parts.append("Lag time > 50% of simulation time; extend t_end_h for full profile")
    notes = "; ".join(notes_parts)

    return DermalPKResult(
        drug_name=drug_name,
        dose_mg_per_cm2=dose_mg_per_cm2,
        application_area_cm2=application_area_cm2,
        total_dose_mg=total_dose_mg,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        lag_time_h=lag_time_h,
        steady_state_flux_mg_per_h_per_cm2=steady_state_flux,
        systemic_bioavailability=systemic_bioavailability,
        stratum_corneum_retention_mg=sc_retention_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        notes=notes,
    )


def estimate_kp(logp: float, mw_da: float) -> float:
    """Estimate skin permeability coefficient using the Potts-Guy equation.

    log(Kp [cm/s]) = 0.71 * logP - 0.0061 * MW - 6.3

    Result is converted to cm/h and clamped to [1e-6, 0.1] cm/h.

    Parameters
    ----------
    logp:
        Octanol-water partition coefficient (log units).
    mw_da:
        Molecular weight (Da).

    Returns
    -------
    float
        Permeability coefficient in cm/h, clamped to [1e-6, 0.1].
    """
    log_kp_cm_per_s = 0.71 * logp - 0.0061 * mw_da - 6.3
    kp_cm_per_s = 10.0**log_kp_cm_per_s
    kp_cm_per_h = kp_cm_per_s * 3600.0
    # Clamp to physiologically plausible range
    kp_cm_per_h = max(_KP_MIN_CM_PER_H, min(kp_cm_per_h, _KP_MAX_CM_PER_H))
    return kp_cm_per_h
