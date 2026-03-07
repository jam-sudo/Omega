"""Phase 998 — Zonated hepatic drug metabolism (periportal vs centrilobular zones)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HepaticZonalPKResult:
    """Result of a hepatic zonal PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_portal_mg_L: list
    c_zone1_mg_L: list
    c_zone2_mg_L: list
    c_systemic_mg_L: list
    cmax_systemic_mg_L: float
    auc_systemic_mg_h_per_L: float
    zone1_extraction: float
    zone2_extraction: float
    total_hepatic_extraction: float
    primary_zone_of_metabolism: str
    hepatotoxicity_zone: str
    notes: str


def simulate_hepatic_zonal_pk(
    drug_name: str,
    dose_mg: float,
    clint_zone1_mL_min: float = 20.0,
    clint_zone2_mL_min: float = 30.0,
    fup: float = 0.5,
    q_portal_L_per_h: float = 72.0,
    v_zone1_L: float = 0.75,
    v_zone2_L: float = 0.75,
    vd_systemic_L: float = 50.0,
    cl_systemic_L_per_h: float = 5.0,
    t_end_h: float = 8.0,
    dt_h: float = 0.05,
) -> HepaticZonalPKResult:
    """Simulate zonated hepatic PK using a 2-zone serial liver model.

    Portal blood enters zone 1 (periportal), then zone 2 (centrilobular),
    before draining to the hepatic vein / systemic circulation.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in milligrams. Must be > 0.
    clint_zone1_mL_min:
        Intrinsic clearance in zone 1 (periportal), mL/min.
    clint_zone2_mL_min:
        Intrinsic clearance in zone 2 (centrilobular), mL/min.
    fup:
        Fraction unbound in plasma (0-1).
    q_portal_L_per_h:
        Portal blood flow, L/h. Must be > 0.
    v_zone1_L:
        Volume of zone 1 hepatic tissue, L.
    v_zone2_L:
        Volume of zone 2 hepatic tissue, L.
    vd_systemic_L:
        Systemic volume of distribution, L. Must be > 0.
    cl_systemic_L_per_h:
        Systemic (extra-hepatic) clearance, L/h.
    t_end_h:
        Simulation end time, h.
    dt_h:
        ODE integration step size, h.

    Returns
    -------
    HepaticZonalPKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if q_portal_L_per_h <= 0:
        raise ValueError(f"q_portal_L_per_h must be > 0, got {q_portal_L_per_h}")
    if vd_systemic_L <= 0:
        raise ValueError(f"vd_systemic_L must be > 0, got {vd_systemic_L}")

    # Convert CLint from mL/min to L/h and apply fup (well-stirred model)
    clint_z1_L_h = clint_zone1_mL_min * 0.001 * 60.0 * fup
    clint_z2_L_h = clint_zone2_mL_min * 0.001 * 60.0 * fup

    # Extraction ratios (well-stirred, serial flow Q through both zones)
    q = q_portal_L_per_h
    e_zone1 = clint_z1_L_h / (q + clint_z1_L_h)
    e_zone2 = clint_z2_L_h / (q + clint_z2_L_h)
    e_total = 1.0 - (1.0 - e_zone1) * (1.0 - e_zone2)

    # Gut absorption rate constant (1/h) — first-order
    ka = 1.0  # 1/h, representative oral absorption

    # State variables: A_gut (mg), C_z1 (mg/L), C_z2 (mg/L), C_sys (mg/L)
    a_gut = dose_mg
    c_z1 = 0.0
    c_z2 = 0.0
    c_sys = 0.0

    times: list[float] = []
    c_portal_list: list[float] = []
    c_z1_list: list[float] = []
    c_z2_list: list[float] = []
    c_sys_list: list[float] = []

    n_steps = int(round(t_end_h / dt_h))

    for step in range(n_steps + 1):
        t = step * dt_h
        times.append(t)

        # Approximate portal concentration as absorbed fraction in zone 1 volume
        c_portal = a_gut / v_zone1_L * 0.1

        c_portal_list.append(c_portal)
        c_z1_list.append(c_z1)
        c_z2_list.append(c_z2)
        c_sys_list.append(c_sys)

        if step == n_steps:
            break

        # Forward Euler derivatives
        da_gut_dt = -ka * a_gut

        # Zone 1 (periportal): receives gut input + portal inflow
        dc_z1_dt = (
            ka * a_gut / v_zone1_L - (q / v_zone1_L) * c_z1 - (clint_z1_L_h / v_zone1_L) * c_z1
        )

        # Zone 2 (centrilobular): receives outflow from zone 1 (fraction that escaped)
        dc_z2_dt = (
            q * c_z1 * (1.0 - e_zone1) / v_zone2_L
            - (q / v_zone2_L) * c_z2
            - (clint_z2_L_h / v_zone2_L) * c_z2
        )

        # Systemic: receives outflow from zone 2 (fraction that escaped)
        dc_sys_dt = (
            q * c_z2 * (1.0 - e_zone2) / vd_systemic_L
            - (cl_systemic_L_per_h / vd_systemic_L) * c_sys
        )

        # Update state
        a_gut += da_gut_dt * dt_h
        c_z1 += dc_z1_dt * dt_h
        c_z2 += dc_z2_dt * dt_h
        c_sys += dc_sys_dt * dt_h

        # Floor at 0 (mass conservation guard)
        if a_gut < 0.0:
            a_gut = 0.0
        if c_z1 < 0.0:
            c_z1 = 0.0
        if c_z2 < 0.0:
            c_z2 = 0.0
        if c_sys < 0.0:
            c_sys = 0.0

    # Derived PK metrics from systemic concentrations
    cmax_systemic = max(c_sys_list) if c_sys_list else 0.0

    # Trapezoidal AUC
    auc_systemic = 0.0
    for i in range(1, len(times)):
        auc_systemic += 0.5 * (c_sys_list[i] + c_sys_list[i - 1]) * (times[i] - times[i - 1])

    # Determine primary zone of metabolism
    if e_zone1 > e_zone2 * 1.5:
        primary_zone = "zone1"
    elif e_zone2 > e_zone1 * 1.5:
        primary_zone = "zone2"
    else:
        primary_zone = "distributed"

    # Hepatotoxicity zone mapping
    if primary_zone == "zone1":
        hepatotoxicity_zone = "periportal"
    elif primary_zone == "zone2":
        hepatotoxicity_zone = "centrilobular"
    else:
        hepatotoxicity_zone = "panlobular"

    notes = (
        f"Serial 2-zone liver model. "
        f"CLint zone1={clint_zone1_mL_min} mL/min, "
        f"CLint zone2={clint_zone2_mL_min} mL/min, fup={fup}. "
        f"E_zone1={e_zone1:.3f}, E_zone2={e_zone2:.3f}, E_total={e_total:.3f}. "
        f"Portal flow Q={q_portal_L_per_h} L/h."
    )

    return HepaticZonalPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_portal_mg_L=c_portal_list,
        c_zone1_mg_L=c_z1_list,
        c_zone2_mg_L=c_z2_list,
        c_systemic_mg_L=c_sys_list,
        cmax_systemic_mg_L=cmax_systemic,
        auc_systemic_mg_h_per_L=auc_systemic,
        zone1_extraction=e_zone1,
        zone2_extraction=e_zone2,
        total_hepatic_extraction=e_total,
        primary_zone_of_metabolism=primary_zone,
        hepatotoxicity_zone=hepatotoxicity_zone,
        notes=notes,
    )


__all__ = [
    "HepaticZonalPKResult",
    "simulate_hepatic_zonal_pk",
]
