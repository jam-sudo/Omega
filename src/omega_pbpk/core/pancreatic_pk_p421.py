"""Pancreatic tissue PK model — Phase 421.

Models drug PK in pancreatic tissue relevant for diabetes drugs (insulin
sensitizers), pancreatic cancer treatments, and digestive enzyme disorders.
2-compartment model: plasma + pancreas with secretion into pancreatic juice.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PancreaticPKResult:
    """Result of pancreatic PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_pancreas_mg_L: list
    c_pancreatic_juice_mg_L: list
    cmax_plasma_mg_L: float
    cmax_pancreas_mg_L: float
    auc_plasma_mg_h_per_L: float
    auc_pancreas_mg_h_per_L: float
    pancreas_to_plasma_ratio: float
    juice_secretion_rate_mg_per_h: float
    t_half_pancreas_h: float
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal AUC calculation."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_pancreatic_pk(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_plasma_L: float = 5.0,
    vd_pancreas_L: float = 0.1,
    k_pt_per_h: float = 0.5,
    k_tp_per_h: float = 0.3,
    k_secretion_per_h: float = 0.2,
    juice_flow_L_per_h: float = 0.05,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> PancreaticPKResult:
    """Simulate drug PK in pancreatic tissue.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg.
    cl_sys_L_per_h:
        Systemic clearance in L/h.
    vd_plasma_L:
        Volume of distribution of plasma compartment in L.
    vd_pancreas_L:
        Volume of distribution of pancreas compartment in L (~100 g tissue).
    k_pt_per_h:
        Plasma-to-pancreas transfer rate constant in 1/h.
    k_tp_per_h:
        Pancreas-to-plasma transfer rate constant in 1/h.
    k_secretion_per_h:
        Pancreatic secretion rate constant into juice in 1/h.
    juice_flow_L_per_h:
        Pancreatic juice flow rate in L/h.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward Euler time step in hours.

    Returns
    -------
    PancreaticPKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_plasma_L <= 0:
        raise ValueError(f"vd_plasma_L must be > 0, got {vd_plasma_L}")
    if vd_pancreas_L <= 0:
        raise ValueError(f"vd_pancreas_L must be > 0, got {vd_pancreas_L}")

    # Initial conditions: IV bolus
    c_plasma = dose_mg / vd_plasma_L
    c_pancreas = 0.0

    times: list = [0.0]
    c_plasma_list: list = [c_plasma]
    c_pancreas_list: list = [c_pancreas]

    # Juice concentration at t=0
    juice_conc_0 = (k_secretion_per_h * c_pancreas * vd_pancreas_L) / max(juice_flow_L_per_h, 1e-12)
    c_juice_list: list = [juice_conc_0]

    max_juice_secretion_rate = 0.0

    n_steps = int(round(t_end_h / dt_h))
    t = 0.0

    for _ in range(n_steps):
        # Derivatives
        # dC_plasma/dt = -CL/Vd_plasma * C_plasma - k_pt * C_plasma + k_tp * C_pancreas * (Vd_pancreas / Vd_plasma)
        d_plasma = (
            -(cl_sys_L_per_h / vd_plasma_L) * c_plasma
            - k_pt_per_h * c_plasma
            + k_tp_per_h * c_pancreas * (vd_pancreas_L / vd_plasma_L)
        )

        # dC_pancreas/dt = k_pt * (Vd_plasma/Vd_pancreas) * C_plasma - k_tp * C_pancreas - k_secretion * C_pancreas
        d_pancreas = (
            k_pt_per_h * (vd_plasma_L / vd_pancreas_L) * c_plasma
            - k_tp_per_h * c_pancreas
            - k_secretion_per_h * c_pancreas
        )

        # Forward Euler update
        c_plasma = max(0.0, c_plasma + dt_h * d_plasma)
        c_pancreas = max(0.0, c_pancreas + dt_h * d_pancreas)
        t += dt_h

        # Juice concentration
        juice_rate_mg_per_h = k_secretion_per_h * c_pancreas * vd_pancreas_L
        c_juice = juice_rate_mg_per_h / max(juice_flow_L_per_h, 1e-12)

        if juice_rate_mg_per_h > max_juice_secretion_rate:
            max_juice_secretion_rate = juice_rate_mg_per_h

        times.append(t)
        c_plasma_list.append(c_plasma)
        c_pancreas_list.append(c_pancreas)
        c_juice_list.append(c_juice)

    # Derived metrics
    cmax_plasma = max(c_plasma_list)
    cmax_pancreas = max(c_pancreas_list)

    auc_plasma = _trapezoidal_auc(times, c_plasma_list)
    auc_pancreas = _trapezoidal_auc(times, c_pancreas_list)

    pancreas_to_plasma_ratio = auc_pancreas / max(auc_plasma, 1e-12)

    k_elim_pancreas = k_tp_per_h + k_secretion_per_h
    t_half_pancreas = 0.693 / k_elim_pancreas if k_elim_pancreas > 0 else float("inf")

    notes = (
        f"{drug_name}: 2-cpt pancreatic PK model. "
        f"Dose {dose_mg} mg IV bolus. "
        f"Pancreas-to-plasma AUC ratio = {pancreas_to_plasma_ratio:.3f}. "
        f"Peak juice secretion rate = {max_juice_secretion_rate:.4f} mg/h."
    )

    return PancreaticPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_pancreas_mg_L=c_pancreas_list,
        c_pancreatic_juice_mg_L=c_juice_list,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_pancreas_mg_L=cmax_pancreas,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_pancreas_mg_h_per_L=auc_pancreas,
        pancreas_to_plasma_ratio=pancreas_to_plasma_ratio,
        juice_secretion_rate_mg_per_h=max_juice_secretion_rate,
        t_half_pancreas_h=t_half_pancreas,
        notes=notes,
    )


def compare_pancreatic_penetration(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    k_pt_values: list,
) -> list:
    """Compare pancreatic penetration across different k_pt values.

    Parameters
    ----------
    drug_name:
        Drug name.
    dose_mg:
        IV bolus dose in mg.
    cl_sys_L_per_h:
        Systemic clearance in L/h.
    k_pt_values:
        List of plasma-to-pancreas transfer rate constants to compare.

    Returns
    -------
    list of PancreaticPKResult sorted by pancreas_to_plasma_ratio descending.
    """
    results = []
    for k_pt in k_pt_values:
        result = simulate_pancreatic_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_sys_L_per_h=cl_sys_L_per_h,
            k_pt_per_h=k_pt,
        )
        results.append(result)

    results.sort(key=lambda r: r.pancreas_to_plasma_ratio, reverse=True)
    return results
