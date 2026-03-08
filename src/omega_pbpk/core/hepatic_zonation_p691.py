"""Phase 691 — Hepatic Zonation model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class HepaticZonationResult:
    """Result of hepatic zonation simulation."""

    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_portal_mg_L: list = field(default_factory=list)
    c_zone1_mg_L: list = field(default_factory=list)
    c_zone3_mg_L: list = field(default_factory=list)
    c_systemic_mg_L: list = field(default_factory=list)
    fu_zone1: float = 0.0
    fu_zone3: float = 0.0
    clint_zone1_L_per_h: float = 0.0
    clint_zone3_L_per_h: float = 0.0
    extraction_ratio: float = 0.0
    first_pass_loss_pct: float = 0.0
    zonation_index: float = 0.0
    notes: str = ""


def simulate_hepatic_zonation(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_int_L_per_h: float,
    fu_plasma: float,
    vd_sys_L: float,
    cyp_zone1_fraction: float = 0.3,
    cyp_zone3_fraction: float = 0.7,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> HepaticZonationResult:
    """Simulate drug metabolism across periportal and pericentral hepatic zones."""

    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_int_L_per_h <= 0:
        raise ValueError("cl_int_L_per_h must be positive")
    if fu_plasma <= 0 or fu_plasma > 1:
        raise ValueError("fu_plasma must be in (0, 1]")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive")
    if not (0 <= cyp_zone1_fraction <= 1):
        raise ValueError("cyp_zone1_fraction must be in [0, 1]")
    if not (0 <= cyp_zone3_fraction <= 1):
        raise ValueError("cyp_zone3_fraction must be in [0, 1]")
    if abs((cyp_zone1_fraction + cyp_zone3_fraction) - 1.0) > 0.01:
        raise ValueError("cyp_zone1_fraction + cyp_zone3_fraction must sum to ~1.0")

    # Hepatic blood flow
    q_h = 90.0  # L/h

    # Zone-specific intrinsic clearances
    clint_zone1 = cl_int_L_per_h * cyp_zone1_fraction
    clint_zone3 = cl_int_L_per_h * cyp_zone3_fraction

    # Well-stirred model per zone
    cl_zone1 = q_h * fu_plasma * clint_zone1 / (q_h + fu_plasma * clint_zone1)
    cl_zone3 = q_h * fu_plasma * clint_zone3 / (q_h + fu_plasma * clint_zone3)
    cl_total = cl_zone1 + cl_zone3

    # Extraction ratio and first-pass
    extraction_ratio = cl_total / q_h
    first_pass_loss_pct = extraction_ratio * 100.0

    # fu for zones (simplified)
    fu_zone1 = fu_plasma
    fu_zone3 = fu_plasma

    # Zonation index
    zonation_index = clint_zone3 / clint_zone1 if clint_zone1 > 0 else float("inf")

    # Systemic PK parameters
    ka = 1.2  # /h
    f_oral = 0.85
    a_gut_0 = dose_mg * f_oral * (1.0 - extraction_ratio)
    ke = cl_total / vd_sys_L
    portal_volume = 1.5  # L

    # Forward Euler simulation
    n_steps = int(t_end_h / dt_h) + 1
    times_h = []
    c_portal_mg_L = []
    c_zone1_mg_L = []
    c_zone3_mg_L = []
    c_systemic_mg_L = []

    a_gut = dose_mg * f_oral  # gut amount for portal calc (pre-first-pass)
    a_central = 0.0

    for i in range(n_steps):
        t = i * dt_h

        # Portal concentration from gut absorption
        c_portal = ka * a_gut / portal_volume
        if c_portal < 0:
            c_portal = 0.0

        # Zone concentrations (instantaneous snapshots)
        c_z1 = c_portal * (1.0 - cl_zone1 / q_h)
        c_z3 = c_z1 * (1.0 - cl_zone3 / q_h)

        # Systemic concentration
        c_sys = a_central / vd_sys_L
        if c_sys < 0:
            c_sys = 0.0

        times_h.append(round(t, 6))
        c_portal_mg_L.append(c_portal)
        c_zone1_mg_L.append(c_z1)
        c_zone3_mg_L.append(c_z3)
        c_systemic_mg_L.append(c_sys)

        # Euler steps
        da_gut = -ka * a_gut
        da_central = ka * a_gut_0 * math.exp(-ka * t) / a_gut_0 * a_gut_0 if a_gut_0 > 0 else 0.0
        # Simpler: use the post-first-pass gut compartment for systemic
        # dA_central/dt = ka * A_gut_sys - ke * A_central
        # where A_gut_sys tracks post-first-pass amount

        a_gut += da_gut * dt_h

    # Redo systemic simulation cleanly with its own gut compartment
    a_gut_sys = a_gut_0  # post-first-pass
    a_central = 0.0
    c_systemic_mg_L = []

    for _i in range(n_steps):
        c_sys = a_central / vd_sys_L
        if c_sys < 0:
            c_sys = 0.0
        c_systemic_mg_L.append(c_sys)

        da_gut_sys = -ka * a_gut_sys
        da_central = ka * a_gut_sys - ke * a_central
        a_gut_sys += da_gut_sys * dt_h
        a_central += da_central * dt_h

    notes_parts = [
        f"Zone 1 (periportal): CLint={clint_zone1:.2f} L/h",
        f"Zone 3 (pericentral): CLint={clint_zone3:.2f} L/h",
        f"Extraction ratio: {extraction_ratio:.3f}",
        f"First-pass loss: {first_pass_loss_pct:.1f}%",
    ]

    return HepaticZonationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_portal_mg_L=c_portal_mg_L,
        c_zone1_mg_L=c_zone1_mg_L,
        c_zone3_mg_L=c_zone3_mg_L,
        c_systemic_mg_L=c_systemic_mg_L,
        fu_zone1=fu_zone1,
        fu_zone3=fu_zone3,
        clint_zone1_L_per_h=clint_zone1,
        clint_zone3_L_per_h=clint_zone3,
        extraction_ratio=extraction_ratio,
        first_pass_loss_pct=first_pass_loss_pct,
        zonation_index=zonation_index,
        notes="; ".join(notes_parts),
    )
