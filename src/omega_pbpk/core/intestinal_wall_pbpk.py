"""Intestinal wall first-pass metabolism PBPK model.

Models first-pass metabolism in the intestinal wall (enterocytes) via a
3-compartment system: gut lumen -> enterocyte (gut wall) -> portal vein.

This captures CYP3A4-mediated gut wall extraction, which reduces oral
bioavailability independent of hepatic first-pass metabolism.

References
----------
Paine et al. 1996 (Clin Pharmacol Ther), Kolars et al. 1991 (Lancet):
CYP3A4 in enterocytes can contribute 30-50% extraction for high-clearance drugs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IntestinalWallPKResult:
    """Result of intestinal wall PBPK simulation."""

    drug_name: str
    dose_mg: float
    peff_cm_s: float
    cl_gut_L_per_h: float
    times_h: list[float] = field(default_factory=list)
    a_lumen_mg: list[float] = field(default_factory=list)
    c_gut_mg_L: list[float] = field(default_factory=list)
    c_portal_mg_L: list[float] = field(default_factory=list)
    auc_portal: float = 0.0
    fg_estimated: float = 0.0
    notes: str = ""


def simulate_intestinal_wall_pk(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL: float,
    peff_cm_s: float,
    cl_gut_L_per_h: float,
    vd_gut_L: float,
    cl_portal_L_per_h: float,
    vd_portal_L: float,
    t_end_h: float = 8.0,
    dt_h: float = 0.01,
) -> IntestinalWallPKResult:
    """Simulate intestinal wall first-pass metabolism (3-compartment model).

    3-compartment: gut lumen -> enterocyte (gut wall) -> portal vein.

    Compartment ODEs
    ----------------
    Lumen (amount, mg):
        dA_lumen/dt = -ka * A_lumen

    Enterocyte wall (concentration, mg/L):
        dC_gut/dt = ka * A_lumen / Vd_gut
                  - CL_gut / Vd_gut * C_gut
                  - k_transfer * C_gut

    Portal vein (concentration, mg/L):
        dC_portal/dt = k_transfer * C_gut * Vd_gut / Vd_portal
                     - ke_portal * C_portal

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    dose_mg : float
        Oral dose administered to gut lumen (mg).
    solubility_mg_mL : float
        Aqueous solubility (mg/mL); used for contextual notes.
    peff_cm_s : float
        Effective intestinal permeability (cm/s).
    cl_gut_L_per_h : float
        Gut wall CYP3A4 clearance (L/h); drives fg reduction.
    vd_gut_L : float
        Enterocyte volume of distribution (L).
    cl_portal_L_per_h : float
        Portal vein elimination clearance (L/h).
    vd_portal_L : float
        Portal vein volume of distribution (L).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Forward Euler step size (h).

    Returns
    -------
    IntestinalWallPKResult
    """
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if peff_cm_s <= 0:
        raise ValueError("peff_cm_s must be > 0")
    if cl_gut_L_per_h < 0:
        raise ValueError("cl_gut_L_per_h must be >= 0")
    if vd_gut_L <= 0:
        raise ValueError("vd_gut_L must be > 0")
    if cl_portal_L_per_h < 0:
        raise ValueError("cl_portal_L_per_h must be >= 0")
    if vd_portal_L <= 0:
        raise ValueError("vd_portal_L must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    # Absorption rate constant from Peff (simplified cylindrical tube model)
    # ka = 2 * Peff * 3600 / radius(cm); radius = 0.5 cm for small intestine
    ka = 2.0 * peff_cm_s * 3600.0 / 0.5  # per hour

    # Transit rate from enterocyte to portal (same as ka for simplicity)
    k_transfer = ka

    # Portal elimination rate constant
    ke_portal = cl_portal_L_per_h / vd_portal_L

    # Gut wall metabolism rate constant
    k_gut_meta = cl_gut_L_per_h / vd_gut_L

    # Initial conditions
    a_lumen = dose_mg
    c_gut = 0.0
    c_portal = 0.0

    n_steps = max(int(t_end_h / dt_h), 1)
    times: list[float] = [0.0]
    a_lumen_arr: list[float] = [a_lumen]
    c_gut_arr: list[float] = [c_gut]
    c_portal_arr: list[float] = [c_portal]

    for i in range(n_steps):
        # Derivatives (forward Euler)
        da_lumen = -ka * a_lumen

        dc_gut = (
            ka * a_lumen / vd_gut_L
            - k_gut_meta * c_gut
            - k_transfer * c_gut
        )

        dc_portal = (
            k_transfer * c_gut * vd_gut_L / vd_portal_L
            - ke_portal * c_portal
        )

        # Update state
        a_lumen = max(a_lumen + da_lumen * dt_h, 0.0)
        c_gut = max(c_gut + dc_gut * dt_h, 0.0)
        c_portal = max(c_portal + dc_portal * dt_h, 0.0)

        t_now = (i + 1) * dt_h
        times.append(t_now)
        a_lumen_arr.append(a_lumen)
        c_gut_arr.append(c_gut)
        c_portal_arr.append(c_portal)

    # Trapezoidal AUC for portal vein concentration
    auc_portal = 0.0
    for i in range(1, len(times)):
        auc_portal += 0.5 * (c_portal_arr[i - 1] + c_portal_arr[i]) * (times[i] - times[i - 1])

    # Estimate fg: fraction of dose reaching portal vein intact
    # Total drug delivered to portal = AUC_portal * CL_portal (mass cleared via portal)
    # Mass absorbed into gut wall = dose_mg - a_lumen[-1]
    absorbed_into_gut = dose_mg - a_lumen_arr[-1]
    # Mass reaching portal (surviving gut wall) = AUC_portal * ke_portal * vd_portal
    mass_portal = auc_portal * ke_portal * vd_portal_L
    fg_estimated = float(mass_portal / absorbed_into_gut) if absorbed_into_gut > 1e-9 else 1.0
    fg_estimated = max(0.0, min(1.0, fg_estimated))

    notes_parts = [f"ka={ka:.2f}/h", f"k_transfer={k_transfer:.2f}/h"]
    if cl_gut_L_per_h > 0:
        eg = cl_gut_L_per_h / (cl_gut_L_per_h + k_transfer * vd_gut_L)
        notes_parts.append(f"Eg_approx={eg:.3f}")
    if solubility_mg_mL < 0.1:
        notes_parts.append("low-solubility drug: dissolution may limit absorption")
    notes = "; ".join(notes_parts)

    return IntestinalWallPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        peff_cm_s=peff_cm_s,
        cl_gut_L_per_h=cl_gut_L_per_h,
        times_h=times,
        a_lumen_mg=a_lumen_arr,
        c_gut_mg_L=c_gut_arr,
        c_portal_mg_L=c_portal_arr,
        auc_portal=auc_portal,
        fg_estimated=fg_estimated,
        notes=notes,
    )


def gut_wall_extraction_sensitivity(
    drug_name: str,
    dose_mg: float,
    peff_cm_s: float,
    cl_gut_values: list[float],
    vd_gut: float,
    cl_portal: float,
    vd_portal: float,
    solubility_mg_mL: float = 1.0,
    t_end_h: float = 8.0,
    dt_h: float = 0.01,
) -> list[dict]:
    """Sweep CLgut values and return fg (gut wall survival fraction) for each.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    dose_mg : float
        Oral dose (mg).
    peff_cm_s : float
        Effective intestinal permeability (cm/s).
    cl_gut_values : list[float]
        List of gut wall clearance values (L/h) to evaluate.
    vd_gut : float
        Gut wall volume of distribution (L).
    cl_portal : float
        Portal vein clearance (L/h).
    vd_portal : float
        Portal vein volume of distribution (L).
    solubility_mg_mL : float
        Aqueous solubility (mg/mL).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Forward Euler step size (h).

    Returns
    -------
    list[dict]
        Each entry: {"cl_gut_L_per_h": float, "fg": float, "auc_portal": float}.
    """
    if not cl_gut_values:
        raise ValueError("cl_gut_values must be a non-empty list")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if peff_cm_s <= 0:
        raise ValueError("peff_cm_s must be > 0")
    if vd_gut <= 0:
        raise ValueError("vd_gut must be > 0")
    if cl_portal < 0:
        raise ValueError("cl_portal must be >= 0")
    if vd_portal <= 0:
        raise ValueError("vd_portal must be > 0")

    results = []
    for cl_gut in cl_gut_values:
        if cl_gut < 0:
            raise ValueError(f"cl_gut value {cl_gut} must be >= 0")
        sim = simulate_intestinal_wall_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            solubility_mg_mL=solubility_mg_mL,
            peff_cm_s=peff_cm_s,
            cl_gut_L_per_h=cl_gut,
            vd_gut_L=vd_gut,
            cl_portal_L_per_h=cl_portal,
            vd_portal_L=vd_portal,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(
            {
                "cl_gut_L_per_h": cl_gut,
                "fg": sim.fg_estimated,
                "auc_portal": sim.auc_portal,
            }
        )
    return results


__all__ = [
    "IntestinalWallPKResult",
    "simulate_intestinal_wall_pk",
    "gut_wall_extraction_sensitivity",
]
