"""Hepatic Sinusoid Zonation model (Phase 621).

Models drug metabolism with hepatic zonation (periportal vs. pericentral)
accounting for oxygen and enzyme gradients across three hepatic zones.
Uses tanks-in-series approach with Forward Euler integration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HepaticZonationResult:
    """Result of hepatic zonation simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_portal_mg_L: list  # portal vein input concentration
    c_zone1_mg_L: list  # periportal zone (zone 1)
    c_zone2_mg_L: list  # midzonal (zone 2)
    c_zone3_mg_L: list  # pericentral zone (zone 3, higher CYP3A4)
    c_hepatic_vein_mg_L: list  # hepatic vein output
    extraction_ratio_portal: float  # overall hepatic extraction
    zone_contribution: dict  # {"zone1": float, "zone2": float, "zone3": float}
    auc_portal_mg_h_per_L: float
    auc_hepatic_vein_mg_h_per_L: float
    notes: str


# Enzyme distribution factors per zone (normalized to 1.0 average)
_ENZYME_FACTORS = {
    "cyp1a2": [1.2, 1.0, 0.8],  # zone1, zone2, zone3
    "cyp2e1": [0.3, 0.7, 2.0],  # pericentral
    "cyp3a4": [0.5, 1.0, 1.5],  # pericentral
    "other": [1.0, 1.0, 1.0],
}


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Manual trapezoidal AUC."""
    return sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def simulate_hepatic_zonation(
    drug_name: str,
    dose_mg: float,
    cl_int_portal_L_per_h: float,
    fm_cyp3a4: float,
    fm_cyp2e1: float,
    t_end_h: float = 6.0,
    dt_h: float = 0.02,
) -> HepaticZonationResult:
    """Simulate hepatic zonation pharmacokinetics.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in mg.
    cl_int_portal_L_per_h:
        Intrinsic clearance referenced to portal blood (L/h).
    fm_cyp3a4:
        Fraction of metabolism by CYP3A4 (0 to 1).
    fm_cyp2e1:
        Fraction of metabolism by CYP2E1 (0 to 1).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler step size (h).

    Returns
    -------
    HepaticZonationResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_int_portal_L_per_h <= 0:
        raise ValueError(f"cl_int_portal_L_per_h must be > 0, got {cl_int_portal_L_per_h}")
    if not (0.0 <= fm_cyp3a4 <= 1.0):
        raise ValueError(f"fm_cyp3a4 must be in [0, 1], got {fm_cyp3a4}")
    if not (0.0 <= fm_cyp2e1 <= 1.0):
        raise ValueError(f"fm_cyp2e1 must be in [0, 1], got {fm_cyp2e1}")
    if fm_cyp3a4 + fm_cyp2e1 > 1.0:
        raise ValueError(f"fm_cyp3a4 + fm_cyp2e1 must be <= 1.0, got {fm_cyp3a4 + fm_cyp2e1}")

    # --- PK parameters ---
    Qh = 90.0  # hepatic blood flow (L/h)
    V_liver_L = 1.8  # liver volume (L)
    V_zone = V_liver_L / 3.0  # equal thirds
    ka = 1.2  # absorption rate constant (1/h)
    f_abs = 0.85  # fraction absorbed

    # fm_other = remaining fraction (not CYP3A4 or CYP2E1)
    fm_other = 1.0 - fm_cyp3a4 - fm_cyp2e1

    # Effective CLint per zone:
    # cl_int_z = cl_int_portal * (fm_cyp3a4*cyp3a4_z + fm_cyp2e1*cyp2e1_z + fm_other*1.0) / 3
    cyp3a4_factors = _ENZYME_FACTORS["cyp3a4"]
    cyp2e1_factors = _ENZYME_FACTORS["cyp2e1"]

    cl_int_zones = []
    for z in range(3):
        cl_int_z = (
            cl_int_portal_L_per_h
            * (fm_cyp3a4 * cyp3a4_factors[z] + fm_cyp2e1 * cyp2e1_factors[z] + fm_other * 1.0)
            / 3.0
        )
        cl_int_zones.append(cl_int_z)

    # --- State initialisation ---
    A_gut = dose_mg * f_abs  # gut amount (mg)
    C_z1 = 0.0  # zone 1 concentration (mg/L)
    C_z2 = 0.0  # zone 2 concentration (mg/L)
    C_z3 = 0.0  # zone 3 concentration (mg/L)

    # Output storage
    times_h: list = []
    c_portal: list = []
    c_z1_list: list = []
    c_z2_list: list = []
    c_z3_list: list = []
    c_hv_list: list = []

    # Trapezoid helpers for zone metabolism integrals
    auc_z1 = 0.0
    auc_z2 = 0.0
    auc_z3 = 0.0
    prev_C_z1 = prev_C_z2 = prev_C_z3 = 0.0

    t = 0.0
    step = 0
    while t <= t_end_h + 1e-9:
        # Portal input concentration from gut absorption
        C_in = ka * A_gut / Qh  # mg/L

        # Record
        times_h.append(t)
        c_portal.append(max(0.0, C_in))
        c_z1_list.append(max(0.0, C_z1))
        c_z2_list.append(max(0.0, C_z2))
        c_z3_list.append(max(0.0, C_z3))
        c_hv_list.append(max(0.0, C_z3))

        # Accumulate zone AUC for metabolism fractions (trapezoid)
        if step > 0:
            auc_z1 += 0.5 * (prev_C_z1 + C_z1) * dt_h
            auc_z2 += 0.5 * (prev_C_z2 + C_z2) * dt_h
            auc_z3 += 0.5 * (prev_C_z3 + C_z3) * dt_h

        prev_C_z1 = C_z1
        prev_C_z2 = C_z2
        prev_C_z3 = C_z3

        # --- Forward Euler ---
        dA_gut = -ka * A_gut
        dC_z1 = (Qh / V_zone) * (C_in - C_z1) - (cl_int_zones[0] / V_zone) * C_z1
        dC_z2 = (Qh / V_zone) * (C_z1 - C_z2) - (cl_int_zones[1] / V_zone) * C_z2
        dC_z3 = (Qh / V_zone) * (C_z2 - C_z3) - (cl_int_zones[2] / V_zone) * C_z3

        A_gut = max(0.0, A_gut + dA_gut * dt_h)
        C_z1 = max(0.0, C_z1 + dC_z1 * dt_h)
        C_z2 = max(0.0, C_z2 + dC_z2 * dt_h)
        C_z3 = max(0.0, C_z3 + dC_z3 * dt_h)

        t += dt_h
        step += 1

    # --- Derived outputs ---
    auc_portal = _trapezoidal_auc(times_h, c_portal)
    auc_hv = _trapezoidal_auc(times_h, c_hv_list)

    if auc_portal > 0.0:
        extraction_ratio = max(0.0, min(1.0, (auc_portal - auc_hv) / auc_portal))
    else:
        extraction_ratio = 0.0

    # Zone metabolism contributions: met_z = cl_int_z * AUC_z
    met_z1 = cl_int_zones[0] * auc_z1
    met_z2 = cl_int_zones[1] * auc_z2
    met_z3 = cl_int_zones[2] * auc_z3
    total_met = met_z1 + met_z2 + met_z3

    if total_met > 0.0:
        zone_contribution = {
            "zone1": met_z1 / total_met,
            "zone2": met_z2 / total_met,
            "zone3": met_z3 / total_met,
        }
    else:
        zone_contribution = {"zone1": 1.0 / 3.0, "zone2": 1.0 / 3.0, "zone3": 1.0 / 3.0}

    notes = (
        f"Hepatic zonation model: Qh={Qh} L/h, V_liver={V_liver_L} L, "
        f"CYP3A4 fm={fm_cyp3a4:.2f}, CYP2E1 fm={fm_cyp2e1:.2f}. "
        f"CLint zones (L/h): {[round(c, 3) for c in cl_int_zones]}."
    )

    return HepaticZonationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_portal_mg_L=c_portal,
        c_zone1_mg_L=c_z1_list,
        c_zone2_mg_L=c_z2_list,
        c_zone3_mg_L=c_z3_list,
        c_hepatic_vein_mg_L=c_hv_list,
        extraction_ratio_portal=extraction_ratio,
        zone_contribution=zone_contribution,
        auc_portal_mg_h_per_L=auc_portal,
        auc_hepatic_vein_mg_h_per_L=auc_hv,
        notes=notes,
    )
