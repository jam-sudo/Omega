"""Phase 989 — Portal vein first-pass extraction PK model.

Models first-pass extraction through gut wall and liver using a
3-compartment forward Euler ODE: Gut → Portal vein → Liver → Systemic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PortalVeinPKResult:
    """Result of portal vein PK simulation (mutable — has list fields)."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_portal_mg_L: list
    c_liver_mg_L: list
    c_systemic_mg_L: list
    cmax_portal_mg_L: float
    cmax_systemic_mg_L: float
    tmax_systemic_h: float
    auc_portal_mg_h_per_L: float
    auc_systemic_mg_h_per_L: float
    f_gut: float
    f_hepatic: float
    bioavailability_f: float
    portal_to_systemic_ratio: float
    notes: str


def _trapz(x: list, y: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(x)):
        total += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return total


def simulate_portal_vein_pk(
    drug_name: str,
    dose_mg: float,
    logp: float = 1.5,
    fup: float = 0.5,
    clint_gut_mL_min: float = 10.0,
    clint_liver_mL_min_per_g: float = 40.0,
    v_portal_L: float = 1.5,
    v_liver_L: float = 1.5,
    q_portal_L_per_h: float = 72.0,
    cl_systemic_L_per_h: float = 20.0,
    vd_systemic_L: float = 50.0,
    t_end_h: float = 8.0,
    dt_h: float = 0.05,
) -> PortalVeinPKResult:
    """Simulate portal vein PK with first-pass extraction.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in mg.
    logp:
        Lipophilicity (log P), used for notes only.
    fup:
        Fraction unbound in plasma (0-1).
    clint_gut_mL_min:
        Gut wall intrinsic clearance (mL/min).
    clint_liver_mL_min_per_g:
        Hepatic intrinsic clearance per gram liver (mL/min/g).
    v_portal_L:
        Volume of portal vein compartment (L).
    v_liver_L:
        Volume of liver compartment (L).
    q_portal_L_per_h:
        Portal blood flow (L/h), typically ~72 L/h.
    cl_systemic_L_per_h:
        Systemic clearance (L/h).
    vd_systemic_L:
        Volume of distribution for systemic compartment (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler step size (h).

    Returns
    -------
    PortalVeinPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if v_portal_L <= 0:
        raise ValueError(f"v_portal_L must be > 0, got {v_portal_L}")
    if v_liver_L <= 0:
        raise ValueError(f"v_liver_L must be > 0, got {v_liver_L}")
    if cl_systemic_L_per_h <= 0:
        raise ValueError(f"cl_systemic_L_per_h must be > 0, got {cl_systemic_L_per_h}")
    if vd_systemic_L <= 0:
        raise ValueError(f"vd_systemic_L must be > 0, got {vd_systemic_L}")

    # --- Gut wall extraction ---
    q_gut_L_per_h = 60.0  # gut blood flow (L/h)
    # CLint_gut scaled by fup, converted to L/h
    cl_int_gut_L_per_h = clint_gut_mL_min / 1000.0 * 60.0 * fup
    e_gut = cl_int_gut_L_per_h / (q_gut_L_per_h + cl_int_gut_L_per_h)
    f_gut = 1.0 - e_gut

    # --- Hepatic first-pass (well-stirred model) ---
    # ~1800 g liver weight; convert mL/min/g -> L/h
    cl_int_liver_L_per_h = clint_liver_mL_min_per_g * 1800.0 * 0.001 * 60.0 * fup
    q_liver_L_per_h = q_portal_L_per_h
    e_liver = cl_int_liver_L_per_h / (q_liver_L_per_h + cl_int_liver_L_per_h)
    f_hepatic = 1.0 - e_liver

    # --- Rate constants ---
    ka = 1.0  # first-order absorption from gut (1/h)
    k_portal_drain = q_portal_L_per_h / v_portal_L
    # Liver -> systemic fraction that escapes hepatic extraction
    k_liver_to_systemic = q_portal_L_per_h / v_liver_L * (1.0 - e_liver)
    # Liver elimination loss
    k_liver_loss = q_portal_L_per_h / v_liver_L * e_liver
    k_elim_systemic = cl_systemic_L_per_h / vd_systemic_L

    # --- Initial conditions ---
    a_gut = dose_mg  # amount in gut (mg)
    c_portal = 0.0  # concentration in portal vein (mg/L)
    c_liver = 0.0  # concentration in liver (mg/L)
    c_systemic = 0.0  # concentration in systemic (mg/L)

    # --- Storage ---
    n_steps = int(t_end_h / dt_h) + 1
    times = [0.0] * n_steps
    cp_list = [0.0] * n_steps
    cl_list = [0.0] * n_steps
    cs_list = [0.0] * n_steps

    times[0] = 0.0
    cp_list[0] = c_portal
    cl_list[0] = c_liver
    cs_list[0] = c_systemic

    # --- Forward Euler integration ---
    for i in range(1, n_steps):
        t = i * dt_h

        # Derivatives
        da_gut = -ka * a_gut
        dc_portal = ka * f_gut * a_gut / v_portal_L - k_portal_drain * c_portal
        dc_liver = (
            k_portal_drain * c_portal * v_portal_L / v_liver_L
            - (k_liver_to_systemic + k_liver_loss) * c_liver
        )
        dc_systemic = (
            k_liver_to_systemic * c_liver * v_liver_L / vd_systemic_L - k_elim_systemic * c_systemic
        )

        # Update
        a_gut = max(0.0, a_gut + da_gut * dt_h)
        c_portal = max(0.0, c_portal + dc_portal * dt_h)
        c_liver = max(0.0, c_liver + dc_liver * dt_h)
        c_systemic = max(0.0, c_systemic + dc_systemic * dt_h)

        times[i] = t
        cp_list[i] = c_portal
        cl_list[i] = c_liver
        cs_list[i] = c_systemic

    # --- Derived metrics ---
    cmax_portal = max(cp_list)
    cmax_systemic = max(cs_list)
    tmax_systemic = times[cs_list.index(cmax_systemic)]

    auc_portal = _trapz(times, cp_list)
    auc_systemic = _trapz(times, cs_list)

    bioavailability_f = f_gut * f_hepatic

    portal_to_systemic_ratio = auc_portal / auc_systemic if auc_systemic > 0.0 else 0.0

    notes = (
        f"logP={logp:.2f}, fup={fup:.3f}; "
        f"E_gut={e_gut:.3f}, E_liver={e_liver:.3f}; "
        f"CLint_gut={cl_int_gut_L_per_h:.2f} L/h, "
        f"CLint_liver={cl_int_liver_L_per_h:.2f} L/h"
    )

    return PortalVeinPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_portal_mg_L=cp_list,
        c_liver_mg_L=cl_list,
        c_systemic_mg_L=cs_list,
        cmax_portal_mg_L=cmax_portal,
        cmax_systemic_mg_L=cmax_systemic,
        tmax_systemic_h=tmax_systemic,
        auc_portal_mg_h_per_L=auc_portal,
        auc_systemic_mg_h_per_L=auc_systemic,
        f_gut=f_gut,
        f_hepatic=f_hepatic,
        bioavailability_f=bioavailability_f,
        portal_to_systemic_ratio=portal_to_systemic_ratio,
        notes=notes,
    )
