"""Phase 208 — Portal vein PK model (p208 variant).

Models drug transport through the portal vein and first-pass hepatic
extraction after oral absorption using a 3-compartment Forward Euler ODE.

Compartments
------------
- C_gut (mg/L): drug in gut wall (after GI absorption)
- C_portal (mg/L): drug in portal vein blood
- C_systemic (mg/L): systemic circulation

ODEs
----
dC_gut/dt     = ka * dose_absorbed_mg / Vd_gut
                - k_gut_met * C_gut - k_gut_efflux * C_gut
dC_portal/dt  = k_gut_efflux * C_gut * Vd_gut / Vd_portal
                - k_hep * C_portal
dC_systemic/dt = k_hep * (1 - E_h) * C_portal * Vd_portal / Vd_sys
                 - ke * C_systemic

where
    E_h = fu * CLint / (Q_h + fu * CLint)
    Q_h = 90 L/h
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PortalPKResult:
    """Result of portal vein PK simulation (mutable — has list fields)."""

    drug_name: str
    dose_mg: float
    fu_plasma: float
    cl_intrinsic_L_per_h: float
    e_hepatic: float
    times_h: list
    c_gut_mg_L: list
    c_portal_mg_L: list
    c_systemic_mg_L: list
    cmax_systemic_mg_L: float
    tmax_systemic_h: float
    auc_systemic_mg_h_per_L: float
    f_oral: float
    notes: str


# Fixed physiological parameters
_VD_GUT_L: float = 1.0
_VD_PORTAL_L: float = 1.5
_KA_PER_H: float = 1.0  # absorption rate constant (1/h)
_K_GUT_MET_PER_H: float = 0.1  # gut wall CYP3A4 metabolism (1/h)
_K_GUT_EFFLUX_PER_H: float = 0.5  # gut wall → portal passage (1/h)
_Q_H_L_PER_H: float = 90.0  # hepatic blood flow (L/h)


def _trapz(x: list, y: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(x)):
        total += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return total


def simulate_portal_pk(
    drug_name: str,
    dose_mg: float,
    fu_plasma: float,
    cl_intrinsic_L_per_h: float,
    cl_systemic_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
    notes: str = "",
) -> PortalPKResult:
    """Simulate portal vein PK with first-pass hepatic extraction.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in mg (absorbed into gut wall compartment).
    fu_plasma:
        Fraction unbound in plasma (0 < fu <= 1).
    cl_intrinsic_L_per_h:
        Hepatic intrinsic clearance (L/h).
    cl_systemic_L_per_h:
        Systemic elimination clearance (L/h).
    vd_sys_L:
        Volume of distribution for systemic compartment (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler step size (h).
    notes:
        Optional annotation appended to result notes.

    Returns
    -------
    PortalPKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if cl_intrinsic_L_per_h < 0:
        raise ValueError(f"cl_intrinsic_L_per_h must be >= 0, got {cl_intrinsic_L_per_h}")

    # Hepatic extraction ratio (well-stirred model)
    e_h = (fu_plasma * cl_intrinsic_L_per_h) / (_Q_H_L_PER_H + fu_plasma * cl_intrinsic_L_per_h)
    e_h = max(0.0, min(1.0, e_h))

    # Rate constants
    k_hep = _Q_H_L_PER_H / _VD_PORTAL_L  # portal drain rate (1/h)
    ke = cl_systemic_L_per_h / vd_sys_L  # systemic elimination (1/h)

    # Initial conditions — all drug starts as absorbed amount in gut wall
    dose_absorbed_mg = dose_mg
    c_gut = dose_absorbed_mg / _VD_GUT_L  # initial gut-wall concentration (mg/L)
    c_portal = 0.0
    c_systemic = 0.0

    # Allocate storage
    n_steps = int(t_end_h / dt_h) + 1
    times: list[float] = [0.0] * n_steps
    c_gut_list: list[float] = [0.0] * n_steps
    c_portal_list: list[float] = [0.0] * n_steps
    c_sys_list: list[float] = [0.0] * n_steps

    c_gut_list[0] = c_gut
    c_portal_list[0] = c_portal
    c_sys_list[0] = c_systemic

    # Forward Euler integration
    for i in range(1, n_steps):
        times[i] = i * dt_h

        dc_gut = -_K_GUT_MET_PER_H * c_gut - _K_GUT_EFFLUX_PER_H * c_gut
        dc_portal = _K_GUT_EFFLUX_PER_H * c_gut * _VD_GUT_L / _VD_PORTAL_L - k_hep * c_portal
        dc_systemic = k_hep * (1.0 - e_h) * c_portal * _VD_PORTAL_L / vd_sys_L - ke * c_systemic

        c_gut = max(0.0, c_gut + dc_gut * dt_h)
        c_portal = max(0.0, c_portal + dc_portal * dt_h)
        c_systemic = max(0.0, c_systemic + dc_systemic * dt_h)

        c_gut_list[i] = c_gut
        c_portal_list[i] = c_portal
        c_sys_list[i] = c_systemic

    # Derived metrics
    cmax_systemic = max(c_sys_list)
    tmax_systemic = times[c_sys_list.index(cmax_systemic)]
    auc_systemic = _trapz(times, c_sys_list)

    # Oral bioavailability estimate: fraction that reaches systemic relative to dose
    # fg = fraction escaping gut wall
    fg = _K_GUT_EFFLUX_PER_H / (_K_GUT_EFFLUX_PER_H + _K_GUT_MET_PER_H)
    fh = 1.0 - e_h
    f_oral = fg * fh

    auto_notes = (
        f"E_h={e_h:.3f}, fg={fg:.3f}, fh={fh:.3f}, f_oral={f_oral:.3f}; "
        f"fu={fu_plasma:.3f}, CLint={cl_intrinsic_L_per_h:.2f} L/h"
    )
    if notes:
        auto_notes = auto_notes + "; " + notes

    return PortalPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fu_plasma=fu_plasma,
        cl_intrinsic_L_per_h=cl_intrinsic_L_per_h,
        e_hepatic=e_h,
        times_h=times,
        c_gut_mg_L=c_gut_list,
        c_portal_mg_L=c_portal_list,
        c_systemic_mg_L=c_sys_list,
        cmax_systemic_mg_L=cmax_systemic,
        tmax_systemic_h=tmax_systemic,
        auc_systemic_mg_h_per_L=auc_systemic,
        f_oral=f_oral,
        notes=auto_notes,
    )


def assess_first_pass(result: PortalPKResult) -> dict:
    """Assess first-pass extraction from a simulation result.

    Parameters
    ----------
    result:
        A :class:`PortalPKResult` returned by :func:`simulate_portal_pk`.

    Returns
    -------
    dict with keys:
        - ``hepatic_extraction_ratio``: E_h (0–1)
        - ``fg_gut_bioavailability``: fraction escaping gut wall
        - ``fh_hepatic_bioavailability``: fraction escaping liver (1 - E_h)
        - ``f_oral_total``: fg * fh
        - ``high_first_pass``: True if E_h > 0.7
    """
    e_h = result.e_hepatic
    fg = _K_GUT_EFFLUX_PER_H / (_K_GUT_EFFLUX_PER_H + _K_GUT_MET_PER_H)
    fh = 1.0 - e_h
    f_oral_total = fg * fh

    return {
        "hepatic_extraction_ratio": e_h,
        "fg_gut_bioavailability": fg,
        "fh_hepatic_bioavailability": fh,
        "f_oral_total": f_oral_total,
        "high_first_pass": e_h > 0.7,
    }
