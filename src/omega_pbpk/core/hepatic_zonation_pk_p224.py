"""
Phase 224 — Hepatic Zonation PK
Model hepatic drug metabolism considering zonation (periportal vs pericentral).
Pure Python (stdlib: math, dataclasses) only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HepaticZonationResult:
    """Result of hepatic zonation simulation."""

    drug_name: str
    cl_intrinsic_L_per_h: float
    fu_plasma: float
    q_hepatic_L_per_h: float
    e_zone1: float
    e_zone2: float
    e_zone3: float
    e_hepatic_uniform: float
    e_hepatic_zonal: float
    fh_uniform: float
    fh_zonal: float
    cl_hepatic_zonal_L_per_h: float
    notes: str


def _extraction_ratio(
    fu: float,
    cl_int: float,
    q: float,
) -> float:
    """
    Well-stirred model extraction ratio.
    E = fu * CLint / (Q + fu * CLint)
    Clamps result to [0, 1].
    """
    denom = q + fu * cl_int
    if denom <= 0.0:
        return 0.0
    e = (fu * cl_int) / denom
    return max(0.0, min(1.0, e))


def simulate_hepatic_zonation(
    drug_name: str,
    cl_intrinsic_L_per_h: float,
    fu_plasma: float,
    q_hepatic_L_per_h: float = 1.5,
) -> HepaticZonationResult:
    """
    Simulate hepatic zonation PK.

    The liver has 3 zones (periportal→midzonal→pericentral) each receiving
    Q_total/3 blood flow. Zone-specific CLint factors reflect CYP3A4 expression:
      Zone 1 (periportal):  factor = 0.8
      Zone 2 (midzonal):    factor = 1.2
      Zone 3 (pericentral): factor = 1.0

    Parameters
    ----------
    drug_name : str
    cl_intrinsic_L_per_h : float
        Whole-liver intrinsic clearance (L/h). Must be >= 0.
    fu_plasma : float
        Unbound fraction in plasma (0 < fu_plasma <= 1).
    q_hepatic_L_per_h : float
        Hepatic blood flow (L/h). Must be > 0. Default 1.5 L/h.

    Returns
    -------
    HepaticZonationResult

    Raises
    ------
    ValueError
        If CLint < 0, fu_plasma out of (0,1], or Q_hepatic <= 0.
    """
    if cl_intrinsic_L_per_h < 0:
        raise ValueError(f"cl_intrinsic_L_per_h must be >= 0, got {cl_intrinsic_L_per_h}")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if q_hepatic_L_per_h <= 0:
        raise ValueError(f"q_hepatic_L_per_h must be > 0, got {q_hepatic_L_per_h}")

    # --- Uniform (well-stirred) model ---
    e_uniform = _extraction_ratio(fu_plasma, cl_intrinsic_L_per_h, q_hepatic_L_per_h)
    fh_uniform = 1.0 - e_uniform

    # --- Zonal model ---
    # Blood flow per zone = Q_total / 3
    q_zone = q_hepatic_L_per_h / 3.0

    # Zone-specific CLint factors (relative CYP3A4 expression)
    zone_factors = [0.8, 1.2, 1.0]

    # Each zone receives the whole-liver CLint scaled by zone factor
    # (CLint is distributed proportionally, then scaled by expression)
    cl_int_z1 = cl_intrinsic_L_per_h * zone_factors[0]
    cl_int_z2 = cl_intrinsic_L_per_h * zone_factors[1]
    cl_int_z3 = cl_intrinsic_L_per_h * zone_factors[2]

    e_z1 = _extraction_ratio(fu_plasma, cl_int_z1, q_zone)
    e_z2 = _extraction_ratio(fu_plasma, cl_int_z2, q_zone)
    e_z3 = _extraction_ratio(fu_plasma, cl_int_z3, q_zone)

    # Sequential passage: drug that escapes zone 1 enters zone 2, etc.
    # F_zone_i = 1 - E_zone_i
    # Overall Fh = F_zone1 * F_zone2 * F_zone3
    f_z1 = 1.0 - e_z1
    f_z2 = 1.0 - e_z2
    f_z3 = 1.0 - e_z3

    fh_zonal = f_z1 * f_z2 * f_z3
    e_zonal = 1.0 - fh_zonal

    # Hepatic clearance (zonal): CL_hepatic = Q * E_h (well-stirred equivalent)
    cl_hepatic_zonal = q_hepatic_L_per_h * e_zonal

    notes = (
        f"Uniform Fh={fh_uniform:.4f}, Zonal Fh={fh_zonal:.4f}. "
        f"Zone CLint factors: Z1=0.8, Z2=1.2, Z3=1.0. "
        f"CL_hepatic_zonal={cl_hepatic_zonal:.4f} L/h."
    )

    return HepaticZonationResult(
        drug_name=drug_name,
        cl_intrinsic_L_per_h=cl_intrinsic_L_per_h,
        fu_plasma=fu_plasma,
        q_hepatic_L_per_h=q_hepatic_L_per_h,
        e_zone1=e_z1,
        e_zone2=e_z2,
        e_zone3=e_z3,
        e_hepatic_uniform=e_uniform,
        e_hepatic_zonal=e_zonal,
        fh_uniform=fh_uniform,
        fh_zonal=fh_zonal,
        cl_hepatic_zonal_L_per_h=cl_hepatic_zonal,
        notes=notes,
    )


def assess_zone_selectivity(
    drug_name: str,
    cl_intrinsic_L_per_h: float,
    fu_plasma: float,
    cyp_zone_profile: dict[str, float],
    q_hepatic_L_per_h: float = 1.5,
) -> dict[str, Any]:
    """
    Assess zone-specific extraction and selectivity using a custom CYP zone profile.

    Parameters
    ----------
    drug_name : str
    cl_intrinsic_L_per_h : float
        Whole-liver CLint (L/h).
    fu_plasma : float
        Unbound fraction (0 < fu <= 1).
    cyp_zone_profile : dict
        {"zone1": factor, "zone2": factor, "zone3": factor}
        Each factor scales the CLint for that zone.
    q_hepatic_L_per_h : float
        Hepatic blood flow (L/h). Default 1.5.

    Returns
    -------
    dict with keys:
        drug_name, zone1_e, zone2_e, zone3_e,
        dominant_zone, fh_zonal, cl_hepatic_L_per_h, profile_used
    """
    if cl_intrinsic_L_per_h < 0:
        raise ValueError(f"cl_intrinsic_L_per_h must be >= 0, got {cl_intrinsic_L_per_h}")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if q_hepatic_L_per_h <= 0:
        raise ValueError(f"q_hepatic_L_per_h must be > 0, got {q_hepatic_L_per_h}")

    q_zone = q_hepatic_L_per_h / 3.0

    factor1 = cyp_zone_profile.get("zone1", 1.0)
    factor2 = cyp_zone_profile.get("zone2", 1.0)
    factor3 = cyp_zone_profile.get("zone3", 1.0)

    e1 = _extraction_ratio(fu_plasma, cl_intrinsic_L_per_h * factor1, q_zone)
    e2 = _extraction_ratio(fu_plasma, cl_intrinsic_L_per_h * factor2, q_zone)
    e3 = _extraction_ratio(fu_plasma, cl_intrinsic_L_per_h * factor3, q_zone)

    zone_extractions = {"zone1": e1, "zone2": e2, "zone3": e3}
    dominant_zone = max(zone_extractions, key=lambda z: zone_extractions[z])

    fh_zonal = (1.0 - e1) * (1.0 - e2) * (1.0 - e3)
    e_total = 1.0 - fh_zonal
    cl_hepatic = q_hepatic_L_per_h * e_total

    return {
        "drug_name": drug_name,
        "zone1_e": e1,
        "zone2_e": e2,
        "zone3_e": e3,
        "dominant_zone": dominant_zone,
        "fh_zonal": fh_zonal,
        "cl_hepatic_L_per_h": cl_hepatic,
        "profile_used": cyp_zone_profile,
    }
