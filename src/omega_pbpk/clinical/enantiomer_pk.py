"""Enantiomer pharmacokinetics model for chiral drugs.

Models the pharmacokinetics of chiral drugs where R and S enantiomers have
different PK properties. Chiral inversion can interconvert the two enantiomers.

Model: 2-compartment Forward Euler for each enantiomer (R and S), with optional
chiral inversion between the two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "EnantiomerPKResult",
    "simulate_enantiomer_pk",
    "compare_enantiomers",
]


@dataclass
class EnantiomerPKResult:
    """Result of enantiomer PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    racemate: bool
    times_h: np.ndarray
    c_r_mg_L: np.ndarray
    c_s_mg_L: np.ndarray
    cmax_r: float
    cmax_s: float
    tmax_r_h: float
    tmax_s_h: float
    auc_r: float
    auc_s: float
    auc_ratio_sr: float
    inversion_occurred: bool
    notes: str


def _validate_inputs(
    dose_mg: float,
    cl_r_L_per_h: float,
    vd_r_L: float,
    cl_s_L_per_h: float,
    vd_s_L: float,
    enantiomer_fraction_r: float,
    k_inv_rs_per_h: float,
    k_inv_sr_per_h: float,
) -> None:
    """Validate all inputs for simulate_enantiomer_pk."""
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_r_L_per_h <= 0:
        raise ValueError(f"cl_r_L_per_h must be > 0, got {cl_r_L_per_h}")
    if vd_r_L <= 0:
        raise ValueError(f"vd_r_L must be > 0, got {vd_r_L}")
    if cl_s_L_per_h <= 0:
        raise ValueError(f"cl_s_L_per_h must be > 0, got {cl_s_L_per_h}")
    if vd_s_L <= 0:
        raise ValueError(f"vd_s_L must be > 0, got {vd_s_L}")
    if not (0.0 <= enantiomer_fraction_r <= 1.0):
        raise ValueError(f"enantiomer_fraction_r must be in [0, 1], got {enantiomer_fraction_r}")
    if k_inv_rs_per_h < 0:
        raise ValueError(f"k_inv_rs_per_h must be >= 0, got {k_inv_rs_per_h}")
    if k_inv_sr_per_h < 0:
        raise ValueError(f"k_inv_sr_per_h must be >= 0, got {k_inv_sr_per_h}")


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration compatible with older numpy."""
    try:
        return float(np.trapz(y, x))
    except AttributeError:
        return float(np.trapezoid(y, x))


def simulate_enantiomer_pk(
    drug_name: str,
    dose_mg: float,
    cl_r_L_per_h: float,
    vd_r_L: float,
    cl_s_L_per_h: float,
    vd_s_L: float,
    route: str = "oral",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    racemate: bool = True,
    enantiomer_fraction_r: float = 0.5,
    k_inv_rs_per_h: float = 0.0,
    k_inv_sr_per_h: float = 0.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> EnantiomerPKResult:
    """Simulate PK of a chiral drug with optional chiral inversion.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Total dose in mg. Must be > 0.
    cl_r_L_per_h:
        Clearance of R-enantiomer in L/h. Must be > 0.
    vd_r_L:
        Volume of distribution of R-enantiomer in L. Must be > 0.
    cl_s_L_per_h:
        Clearance of S-enantiomer in L/h. Must be > 0.
    vd_s_L:
        Volume of distribution of S-enantiomer in L. Must be > 0.
    route:
        "oral" or "iv".
    ka_per_h:
        First-order absorption rate constant in 1/h (oral only).
    f_oral:
        Oral bioavailability fraction (0 to 1).
    racemate:
        If True, dose is administered as racemate split by enantiomer_fraction_r.
        If False, pure enantiomer determined by enantiomer_fraction_r.
    enantiomer_fraction_r:
        Fraction of total dose that is R-enantiomer. In range [0, 1].
    k_inv_rs_per_h:
        Rate of chiral inversion R → S in 1/h. Must be >= 0.
    k_inv_sr_per_h:
        Rate of chiral inversion S → R in 1/h. Must be >= 0.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward Euler time step in hours.

    Returns
    -------
    EnantiomerPKResult
    """
    _validate_inputs(
        dose_mg,
        cl_r_L_per_h,
        vd_r_L,
        cl_s_L_per_h,
        vd_s_L,
        enantiomer_fraction_r,
        k_inv_rs_per_h,
        k_inv_sr_per_h,
    )

    # Determine dose of each enantiomer
    dose_r_mg = dose_mg * enantiomer_fraction_r
    dose_s_mg = dose_mg * (1.0 - enantiomer_fraction_r)

    # Bioavailable dose
    if route == "oral":
        dose_r_abs = dose_r_mg * f_oral
        dose_s_abs = dose_s_mg * f_oral
    else:
        # IV: dose enters directly
        dose_r_abs = dose_r_mg
        dose_s_abs = dose_s_mg

    # Elimination rate constants
    ke_r = cl_r_L_per_h / vd_r_L  # 1/h
    ke_s = cl_s_L_per_h / vd_s_L  # 1/h

    # Time vector
    n_steps = int(math.ceil(t_end_h / dt_h))
    times = np.linspace(0.0, n_steps * dt_h, n_steps + 1)

    # State variables:
    # For oral: depot compartment (amount in mg) + central compartment (conc mg/L)
    # For iv: only central compartment

    c_r = np.zeros(n_steps + 1)  # R central conc [mg/L]
    c_s = np.zeros(n_steps + 1)  # S central conc [mg/L]

    if route == "oral":
        depot_r = dose_r_abs  # mg in GI depot
        depot_s = dose_s_abs  # mg in GI depot
        # Initial central concentrations are 0
        c_r[0] = 0.0
        c_s[0] = 0.0
    else:
        # IV bolus: instantaneous distribution into central compartment
        depot_r = 0.0
        depot_s = 0.0
        c_r[0] = dose_r_abs / vd_r_L
        c_s[0] = dose_s_abs / vd_s_L

    for i in range(n_steps):
        cr = c_r[i]
        cs = c_s[i]

        if route == "oral":
            # Depot absorption fluxes (mg/h)
            abs_r = ka_per_h * depot_r
            abs_s = ka_per_h * depot_s
            # Update depot (forward Euler)
            depot_r = depot_r - abs_r * dt_h
            depot_s = depot_s - abs_s * dt_h
            if depot_r < 0.0:
                depot_r = 0.0
            if depot_s < 0.0:
                depot_s = 0.0
            # Absorbed conc flux (mg/L/h) = abs_rate_mg_per_h / Vd
            abs_r_conc = abs_r / vd_r_L
            abs_s_conc = abs_s / vd_s_L
        else:
            abs_r_conc = 0.0
            abs_s_conc = 0.0

        # Chiral inversion fluxes (concentration basis)
        inv_r_to_s = k_inv_rs_per_h * cr  # R→S flux (mg/L/h leaving R)
        inv_s_to_r = k_inv_sr_per_h * cs  # S→R flux (mg/L/h leaving S)

        # ODEs:
        # dCr/dt = abs_r - ke_r*Cr - k_inv_rs*Cr + k_inv_sr*Cs*(Vd_s/Vd_r)
        # dCs/dt = abs_s - ke_s*Cs + k_inv_rs*Cr*(Vd_r/Vd_s) - k_inv_sr*Cs
        # The volume-corrected inversion terms account for conservation of mass:
        # when R converts to S, the mass (mg) is conserved but volumes differ
        dcr_dt = abs_r_conc - ke_r * cr - inv_r_to_s + inv_s_to_r * (vd_s_L / vd_r_L)
        dcs_dt = abs_s_conc - ke_s * cs + inv_r_to_s * (vd_r_L / vd_s_L) - inv_s_to_r

        c_r[i + 1] = max(0.0, cr + dcr_dt * dt_h)
        c_s[i + 1] = max(0.0, cs + dcs_dt * dt_h)

    # Compute PK metrics
    cmax_r = float(np.max(c_r))
    cmax_s = float(np.max(c_s))

    idx_r = int(np.argmax(c_r))
    idx_s = int(np.argmax(c_s))
    tmax_r_h = float(times[idx_r])
    tmax_s_h = float(times[idx_s])

    auc_r = _trapz(c_r, times)
    auc_s = _trapz(c_s, times)

    auc_ratio_sr = auc_s / max(auc_r, 1e-12)
    inversion_occurred = k_inv_rs_per_h > 0 or k_inv_sr_per_h > 0

    notes_parts = []
    if racemate:
        notes_parts.append(
            f"Racemate: {enantiomer_fraction_r * 100:.0f}% R / "
            f"{(1 - enantiomer_fraction_r) * 100:.0f}% S"
        )
    else:
        notes_parts.append(f"Pure enantiomer: {enantiomer_fraction_r * 100:.0f}% R")
    if inversion_occurred:
        notes_parts.append(
            f"Chiral inversion active (k_RS={k_inv_rs_per_h:.3f}, k_SR={k_inv_sr_per_h:.3f} 1/h)"
        )
    notes = "; ".join(notes_parts)

    return EnantiomerPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        racemate=racemate,
        times_h=times,
        c_r_mg_L=c_r,
        c_s_mg_L=c_s,
        cmax_r=cmax_r,
        cmax_s=cmax_s,
        tmax_r_h=tmax_r_h,
        tmax_s_h=tmax_s_h,
        auc_r=auc_r,
        auc_s=auc_s,
        auc_ratio_sr=auc_ratio_sr,
        inversion_occurred=inversion_occurred,
        notes=notes,
    )


def compare_enantiomers(
    drug_name: str,
    dose_mg: float,
    cl_r_L_per_h: float,
    vd_r_L: float,
    cl_s_L_per_h: float,
    vd_s_L: float,
    **kwargs,
) -> dict:
    """Compare R and S enantiomer PK metrics.

    Runs simulate_enantiomer_pk and returns a comparison dictionary.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Total dose in mg.
    cl_r_L_per_h:
        Clearance of R-enantiomer in L/h.
    vd_r_L:
        Volume of distribution of R-enantiomer in L.
    cl_s_L_per_h:
        Clearance of S-enantiomer in L/h.
    vd_s_L:
        Volume of distribution of S-enantiomer in L.
    **kwargs:
        Additional keyword arguments passed to simulate_enantiomer_pk.

    Returns
    -------
    dict
        Dictionary with keys: AUC_R, AUC_S, AUC_ratio, Cmax_R, Cmax_S,
        Tmax_R_h, Tmax_S_h, inversion_occurred, notes.
    """
    result = simulate_enantiomer_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_r_L_per_h=cl_r_L_per_h,
        vd_r_L=vd_r_L,
        cl_s_L_per_h=cl_s_L_per_h,
        vd_s_L=vd_s_L,
        **kwargs,
    )
    return {
        "AUC_R": result.auc_r,
        "AUC_S": result.auc_s,
        "AUC_ratio": result.auc_ratio_sr,
        "Cmax_R": result.cmax_r,
        "Cmax_S": result.cmax_s,
        "Tmax_R_h": result.tmax_r_h,
        "Tmax_S_h": result.tmax_s_h,
        "inversion_occurred": result.inversion_occurred,
        "notes": result.notes,
    }
