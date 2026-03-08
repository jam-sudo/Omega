"""
Phase 455 — Portal Vein PK Model
3-compartment model: GI lumen -> portal vein -> systemic
Pure Python, forward Euler ODE, no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PortalVeinPKResult:
    """Result of portal vein PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_portal_mg_L: list
    c_systemic_mg_L: list
    cmax_portal_mg_L: float
    tmax_portal_h: float
    auc_portal_mg_h_per_L: float
    cmax_systemic_mg_L: float
    tmax_systemic_h: float
    auc_systemic_mg_h_per_L: float
    portal_to_systemic_cmax_ratio: float
    f_hepatic: float
    notes: str


def simulate_portal_vein_pk(
    drug_name: str,
    dose_mg: float,
    ka_per_h: float = 1.0,
    cl_hepatic_L_per_h: float = 20.0,
    vd_portal_L: float = 3.0,
    vd_sys_L: float = 50.0,
    f_gut: float = 0.9,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PortalVeinPKResult:
    """Simulate portal vein and systemic PK using a 3-compartment model.

    Compartments:
      - GI lumen (amount_gi_mg): drug absorbs at rate ka
      - Portal vein (amount_portal_mg): receives absorbed drug, cleared by liver
      - Systemic (amount_sys_mg): receives drug that escapes hepatic extraction

    Hepatic extraction fraction E_h = cl_hepatic / (cl_hepatic + Q_portal),
    where Q_portal (portal blood flow) defaults to 72 L/h (human typical).
    Drug entering portal cleared at rate cl_hepatic; remainder enters systemic.

    Args:
        drug_name: Identifier for the drug.
        dose_mg: Oral dose in mg.
        ka_per_h: First-order absorption rate constant (1/h).
        cl_hepatic_L_per_h: Hepatic clearance (L/h).
        vd_portal_L: Volume of distribution of portal compartment (L).
        vd_sys_L: Volume of distribution of systemic compartment (L).
        f_gut: Fraction of dose absorbed from GI (gut availability, 0-1].
        t_end_h: Simulation end time (h).
        dt_h: Forward Euler time step (h).

    Returns:
        PortalVeinPKResult with concentrations and summary metrics.
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if cl_hepatic_L_per_h <= 0:
        raise ValueError("cl_hepatic_L_per_h must be > 0")
    if vd_portal_L <= 0:
        raise ValueError("vd_portal_L must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if not (0 < f_gut <= 1):
        raise ValueError("f_gut must be in (0, 1]")

    # Portal blood flow (L/h) — hepatic portal flow ~72 L/h in humans
    q_portal_L_per_h = 72.0

    # Hepatic extraction ratio
    e_h = cl_hepatic_L_per_h / (cl_hepatic_L_per_h + q_portal_L_per_h)

    # Effective CL from portal compartment (via liver)
    # The portal compartment is cleared at rate = (cl_hepatic + q_portal) equivalent
    # Simplified: drug in portal is removed at rate proportional to cl_hepatic
    # Rate of elimination from portal = cl_hepatic * C_portal
    # Rate of transfer to systemic = (1 - e_h) * cl_hepatic_eff * C_portal
    # We model portal as a mixing + clearance compartment:
    #   dA_portal/dt = ka * f_gut * A_gi - (cl_hepatic / vd_portal) * A_portal
    # Systemic receives fraction (1 - e_h) of the hepatic flux:
    #   dA_sys/dt = (1 - e_h) * (cl_hepatic / vd_portal) * A_portal - (cl_sys / vd_sys) * A_sys
    # Systemic clearance assumed = cl_hepatic (same organ, total body clearance ≈ hepatic)
    cl_sys_L_per_h = cl_hepatic_L_per_h

    # Initial conditions
    a_gi = dose_mg  # all dose in GI at t=0
    a_portal = 0.0
    a_sys = 0.0

    # k values
    k_abs = ka_per_h  # GI -> portal rate
    k_portal = cl_hepatic_L_per_h / vd_portal_L  # elimination from portal (1/h)
    k_sys = cl_sys_L_per_h / vd_sys_L  # elimination from systemic (1/h)
    transfer_fraction = 1.0 - e_h  # fraction reaching systemic from portal

    times_h: list[float] = []
    c_portal: list[float] = []
    c_systemic: list[float] = []

    n_steps = int(t_end_h / dt_h) + 1

    for i in range(n_steps):
        t = i * dt_h
        times_h.append(t)

        c_portal_t = a_portal / vd_portal_L
        c_sys_t = a_sys / vd_sys_L
        c_portal.append(c_portal_t)
        c_systemic.append(c_sys_t)

        # Derivatives
        d_gi = -k_abs * a_gi
        d_portal = k_abs * f_gut * a_gi - k_portal * a_portal
        d_sys = transfer_fraction * k_portal * a_portal - k_sys * a_sys

        # Forward Euler update
        a_gi = a_gi + dt_h * d_gi
        if a_gi < 0:
            a_gi = 0.0
        a_portal = a_portal + dt_h * d_portal
        if a_portal < 0:
            a_portal = 0.0
        a_sys = a_sys + dt_h * d_sys
        if a_sys < 0:
            a_sys = 0.0

    # --- Summary metrics ---
    cmax_portal = max(c_portal)
    tmax_portal = times_h[c_portal.index(cmax_portal)]

    cmax_systemic = max(c_systemic)
    tmax_systemic = times_h[c_systemic.index(cmax_systemic)]

    # Trapezoidal AUC
    auc_portal = sum(
        0.5 * (c_portal[i] + c_portal[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_systemic = sum(
        0.5 * (c_systemic[i] + c_systemic[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    portal_to_systemic_cmax_ratio = (
        cmax_portal / cmax_systemic if cmax_systemic > 0 else float("inf")
    )

    # f_hepatic = fraction escaping hepatic extraction = 1 - E_h
    f_hepatic = 1.0 - e_h

    notes = (
        f"e_h={e_h:.3f}, f_hepatic={f_hepatic:.3f}, k_portal={k_portal:.3f}/h, k_sys={k_sys:.4f}/h"
    )

    return PortalVeinPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_portal_mg_L=c_portal,
        c_systemic_mg_L=c_systemic,
        cmax_portal_mg_L=cmax_portal,
        tmax_portal_h=tmax_portal,
        auc_portal_mg_h_per_L=auc_portal,
        cmax_systemic_mg_L=cmax_systemic,
        tmax_systemic_h=tmax_systemic,
        auc_systemic_mg_h_per_L=auc_systemic,
        portal_to_systemic_cmax_ratio=portal_to_systemic_cmax_ratio,
        f_hepatic=f_hepatic,
        notes=notes,
    )


def compare_hepatic_extraction(
    drug_name: str,
    dose_mg: float,
    ka_per_h: float,
    cl_values_L_per_h: list,
    vd_sys_L: float,
) -> list:
    """Compare portal vein PK across multiple hepatic clearance values.

    Returns list of PortalVeinPKResult sorted by f_hepatic descending
    (highest bioavailability first).
    """
    results = []
    for cl in cl_values_L_per_h:
        r = simulate_portal_vein_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            ka_per_h=ka_per_h,
            cl_hepatic_L_per_h=cl,
            vd_sys_L=vd_sys_L,
        )
        results.append(r)
    results.sort(key=lambda r: r.f_hepatic, reverse=True)
    return results
