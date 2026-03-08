"""Phase 340 — Hepatic Zonation Model.

Models drug metabolism across hepatic zones:
  - Zone 1 (periportal): highest CYP1A2 activity
  - Zone 2 (midlobular): intermediate activity
  - Zone 3 (perivenous): highest CYP3A4 and CYP2E1 activity

Drug passes sequentially through portal blood → zone1 → zone2 → zone3 → hepatic vein.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Zone extraction ratios (fraction of hepatic elimination in each zone)
# Format: (R_zone1, R_zone2, R_zone3) — these are fractions of elimination in each zone
_CYP_ZONE_RATIOS: dict[str, tuple[float, float, float]] = {
    "CYP3A4": (0.20, 0.35, 0.45),
    "CYP2E1": (0.15, 0.30, 0.55),
    "CYP1A2": (0.50, 0.30, 0.20),
}

VALID_CYP_ISOFORMS = frozenset(_CYP_ZONE_RATIOS.keys())


@dataclass
class HepaticZonationResult:
    """Result of hepatic zonation simulation."""

    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_zone1_mg_L: list = field(default_factory=list)
    c_zone2_mg_L: list = field(default_factory=list)
    c_zone3_mg_L: list = field(default_factory=list)
    auc_plasma_mg_h_per_L: float = 0.0
    t_half_h: float = 0.0
    zone1_metabolism_pct: float = 0.0
    zone2_metabolism_pct: float = 0.0
    zone3_metabolism_pct: float = 0.0
    overall_hepatic_extraction: float = 0.0
    cyp_isoform: str = ""
    notes: str = ""


def simulate_hepatic_zonation(
    drug_name: str,
    dose_mg: float,
    cl_total_L_per_h: float,
    vd_L: float,
    cyp_isoform: str,
    t_end_h: float,
    dt_h: float,
    f_zone1: float = 0.3,
    f_zone2: float = 0.4,
    f_zone3: float = 0.3,
    blood_flow_L_per_h: float = 90.0,
) -> HepaticZonationResult:
    """Simulate drug metabolism across hepatic zones.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Initial dose in mg (IV bolus into plasma).
    cl_total_L_per_h:
        Total clearance (L/h).
    vd_L:
        Volume of distribution (L).
    cyp_isoform:
        CYP isoform: "CYP3A4", "CYP2E1", or "CYP1A2".
    t_end_h:
        Simulation end time (hours).
    dt_h:
        Time step (hours).
    f_zone1:
        Fraction of hepatic zone 1 contribution (default 0.3).
    f_zone2:
        Fraction of hepatic zone 2 contribution (default 0.4).
    f_zone3:
        Fraction of hepatic zone 3 contribution (default 0.3).
    blood_flow_L_per_h:
        Portal blood flow (L/h, default 90).

    Returns
    -------
    HepaticZonationResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_total_L_per_h <= 0:
        raise ValueError(f"cl_total_L_per_h must be > 0, got {cl_total_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if cyp_isoform not in VALID_CYP_ISOFORMS:
        raise ValueError(
            f"cyp_isoform must be one of {sorted(VALID_CYP_ISOFORMS)}, got '{cyp_isoform}'"
        )
    zone_sum = f_zone1 + f_zone2 + f_zone3
    if abs(zone_sum - 1.0) > 0.01:
        raise ValueError(f"f_zone1 + f_zone2 + f_zone3 must sum to 1.0 (±0.01), got {zone_sum:.4f}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    # Zone extraction ratios from isoform
    R1, R2, R3 = _CYP_ZONE_RATIOS[cyp_isoform]

    # Effective hepatic extraction ratio for the whole liver
    # Based on well-stirred model: E = CL_H / Q_H
    # We use CL_total as hepatic CL (assume hepatic elimination only)
    cl_hepatic = cl_total_L_per_h
    # Overall hepatic extraction ratio
    e_hepatic = cl_hepatic / (blood_flow_L_per_h + cl_hepatic)
    # Clamp to [0, 1)
    e_hepatic = max(0.0, min(0.999, e_hepatic))

    # Zone concentrations: computed from plasma (portal = plasma approximation)
    # C_zone1 = C_plasma (portal)
    # C_zone2 = C_zone1 * (1 - R1)
    # C_zone3 = C_zone2 * (1 - R2)
    # These represent the drug concentration entering/in each zone
    factor_z2 = 1.0 - R1
    factor_z3 = (1.0 - R1) * (1.0 - R2)

    # 1-compartment Forward Euler simulation
    # dC/dt = -CL/Vd * C
    ke = cl_total_L_per_h / vd_L  # elimination rate constant (1/h)
    c0 = dose_mg / vd_L  # initial plasma concentration (mg/L)

    times = []
    c_plasma = []
    c_z1 = []
    c_z2 = []
    c_z3 = []

    c = c0
    n_steps = int(round(t_end_h / dt_h))

    for i in range(n_steps + 1):
        t_now = i * dt_h
        times.append(t_now)
        c_plasma.append(max(0.0, c))
        c_z1.append(max(0.0, c))
        c_z2.append(max(0.0, c * factor_z2))
        c_z3.append(max(0.0, c * factor_z3))

        if i < n_steps:
            # Forward Euler step
            dcdt = -ke * c
            c = c + dcdt * dt_h
            c = max(0.0, c)

    # Trapezoidal AUC for plasma
    auc = sum(
        0.5 * (c_plasma[i] + c_plasma[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    # Half-life from elimination rate
    t_half = math.log(2.0) / ke if ke > 0 else float("inf")

    # Zone metabolism percentages based on CYP isoform extraction ratios R.
    # R values represent the fraction of total hepatic elimination occurring in each zone.
    # Normalize R values so they sum to 100%.
    total_r = R1 + R2 + R3
    if total_r > 0:
        z1_pct = R1 / total_r * 100.0
        z2_pct = R2 / total_r * 100.0
        z3_pct = R3 / total_r * 100.0
    else:
        z1_pct = z2_pct = z3_pct = 100.0 / 3.0

    notes = (
        f"Drug: {drug_name}; CYP isoform: {cyp_isoform}; "
        f"ke={ke:.4f} 1/h; t_half={t_half:.2f} h; "
        f"zone extraction ratios R=({R1},{R2},{R3}); "
        f"hepatic extraction E={e_hepatic:.3f}; "
        f"zone fractions f=({f_zone1},{f_zone2},{f_zone3})."
    )

    return HepaticZonationResult(
        times_h=times,
        c_plasma_mg_L=c_plasma,
        c_zone1_mg_L=c_z1,
        c_zone2_mg_L=c_z2,
        c_zone3_mg_L=c_z3,
        auc_plasma_mg_h_per_L=auc,
        t_half_h=t_half,
        zone1_metabolism_pct=z1_pct,
        zone2_metabolism_pct=z2_pct,
        zone3_metabolism_pct=z3_pct,
        overall_hepatic_extraction=e_hepatic,
        cyp_isoform=cyp_isoform,
        notes=notes,
    )
