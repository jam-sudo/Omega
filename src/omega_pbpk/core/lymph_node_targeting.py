"""Phase 330 — Lymph Node Drug Targeting Model.

4-compartment Forward Euler ODE:
  injection_site -> lymph_node -> systemic -> (lymph_return is modelled implicitly)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LymphNodeTargetingResult:
    """Result of a lymph node targeting simulation."""

    times_h: list
    c_injection_site_mg_mL: list
    c_lymph_node_mg_mL: list
    c_systemic_mg_L: list
    auc_lymph_node_mg_h_per_mL: float
    auc_systemic_mg_h_per_L: float
    cmax_lymph_node_mg_mL: float
    tmax_lymph_node_h: float
    targeting_efficiency_pct: float
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal AUC integration."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_lymph_node_targeting(
    drug_name: str,
    dose_mg: float,
    k_lymph_per_h: float,
    k_clearance_ln_per_h: float,
    k_systemic_per_h: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    v_ln_mL: float = 0.5,
    t_end_h: float = 48.0,
    dt_h: float = 0.01,
) -> LymphNodeTargetingResult:
    """Simulate targeted drug delivery to lymph nodes.

    Parameters
    ----------
    drug_name:
        Name of the drug/formulation.
    dose_mg:
        Injected dose in mg.
    k_lymph_per_h:
        Rate constant for transfer from injection site to lymph node (1/h).
    k_clearance_ln_per_h:
        Lymph node intrinsic clearance rate constant (1/h).
    k_systemic_per_h:
        Rate constant for exit from lymph node to systemic circulation (1/h).
    cl_sys_L_per_h:
        Systemic clearance in L/h.
    vd_sys_L:
        Systemic volume of distribution in L.
    v_ln_mL:
        Volume of lymph node compartment in mL (default 0.5 mL).
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward Euler time step in hours.

    Returns
    -------
    LymphNodeTargetingResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")
    if k_lymph_per_h <= 0:
        raise ValueError(f"k_lymph_per_h must be > 0, got {k_lymph_per_h}")
    if k_clearance_ln_per_h < 0:
        raise ValueError(f"k_clearance_ln_per_h must be >= 0, got {k_clearance_ln_per_h}")
    if k_systemic_per_h < 0:
        raise ValueError(f"k_systemic_per_h must be >= 0, got {k_systemic_per_h}")
    if v_ln_mL <= 0:
        raise ValueError(f"v_ln_mL must be > 0, got {v_ln_mL}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    # Fixed injection site volume for concentration calculation
    v_injection_mL = 0.5

    # Initial conditions
    # A_site in mg (amount at injection site)
    # C_ln in mg/mL (concentration in lymph node)
    # C_sys in mg/L (systemic concentration)
    a_site = dose_mg
    c_ln = 0.0
    c_sys = 0.0

    times_h: list = []
    c_site_list: list = []
    c_ln_list: list = []
    c_sys_list: list = []

    t = 0.0
    n_steps = int(t_end_h / dt_h) + 1

    for _ in range(n_steps):
        # Record state
        times_h.append(t)
        c_site_list.append(a_site / v_injection_mL)  # mg/mL
        c_ln_list.append(c_ln)
        c_sys_list.append(c_sys)

        # ODEs (Forward Euler)
        # injection_site: dA/dt = -k_lymph * A
        da_site_dt = -k_lymph_per_h * a_site

        # lymph_node: dC/dt = k_lymph * A / v_ln - (k_clearance_ln + k_systemic) * C
        dc_ln_dt = (
            k_lymph_per_h * a_site / v_ln_mL - (k_clearance_ln_per_h + k_systemic_per_h) * c_ln
        )

        # systemic: dC/dt = k_systemic * C_ln * v_ln / vd_sys - (cl_sys / vd_sys) * C_sys
        # k_systemic * c_ln * v_ln_mL gives mg/h (rate from lymph node to systemic)
        # vd_sys_L in L; result in mg/L
        dc_sys_dt = (
            k_systemic_per_h * c_ln * v_ln_mL / vd_sys_L - (cl_sys_L_per_h / vd_sys_L) * c_sys
        )

        # Update state
        a_site += da_site_dt * dt_h
        c_ln += dc_ln_dt * dt_h
        c_sys += dc_sys_dt * dt_h

        # Clamp to non-negative
        a_site = max(0.0, a_site)
        c_ln = max(0.0, c_ln)
        c_sys = max(0.0, c_sys)

        t += dt_h
        if t > t_end_h + dt_h * 0.5:
            break

    # PK metrics for lymph node
    auc_ln = _trapezoidal_auc(times_h, c_ln_list)
    auc_sys = _trapezoidal_auc(times_h, c_sys_list)
    cmax_ln = max(c_ln_list)
    tmax_ln = times_h[c_ln_list.index(cmax_ln)]

    # Targeting efficiency: ratio of lymph node Cmax (mg/mL) vs systemic Cmax (mg/L)
    # They have different units; apply a scaling factor to bring to [0, 100]
    # Interpretation: mg/mL = 1000 mg/L, so lymph node cmax is inherently higher per unit dose
    # Use cmax_ln / (cmax_ln + cmax_sys_in_mg_mL) * 100 as a partition fraction
    cmax_sys = max(c_sys_list)
    # Convert systemic cmax from mg/L to mg/mL for comparison
    cmax_sys_mg_mL = cmax_sys / 1000.0

    if cmax_ln + cmax_sys_mg_mL > 0:
        targeting_efficiency_pct = min(100.0, (cmax_ln / (cmax_ln + cmax_sys_mg_mL)) * 100.0)
    else:
        targeting_efficiency_pct = 0.0

    notes = (
        f"Lymph node targeting simulation for {drug_name}: "
        f"dose={dose_mg} mg, k_lymph={k_lymph_per_h}/h, "
        f"k_clearance_ln={k_clearance_ln_per_h}/h, k_systemic={k_systemic_per_h}/h, "
        f"targeting_efficiency={targeting_efficiency_pct:.1f}%"
    )

    return LymphNodeTargetingResult(
        times_h=times_h,
        c_injection_site_mg_mL=c_site_list,
        c_lymph_node_mg_mL=c_ln_list,
        c_systemic_mg_L=c_sys_list,
        auc_lymph_node_mg_h_per_mL=auc_ln,
        auc_systemic_mg_h_per_L=auc_sys,
        cmax_lymph_node_mg_mL=cmax_ln,
        tmax_lymph_node_h=tmax_ln,
        targeting_efficiency_pct=targeting_efficiency_pct,
        notes=notes,
    )
