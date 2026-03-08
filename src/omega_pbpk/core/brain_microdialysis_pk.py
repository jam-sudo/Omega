"""Phase 1113 — Brain microdialysis PK model.

Models unbound extracellular drug concentration in brain ECF using a
two-compartment system (plasma + brain ECF) with Forward Euler integration.
"""

from dataclasses import dataclass


@dataclass
class BrainMicrodialysisResult:
    """Result from brain microdialysis PK simulation."""

    drug_name: str
    dose_mg: float
    kp_uu: float
    times_h: list
    c_plasma_mg_L: list
    c_ecf_mg_L: list
    cmax_ecf_mg_L: float
    tmax_ecf_h: float
    auc_ecf_mg_h_per_L: float
    auc_plasma_mg_h_per_L: float
    ecf_to_plasma_auc_ratio: float
    brain_penetration_class: str  # "poor"(<0.1), "moderate"(0.1-0.3), "good"(>0.3)
    notes: str


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal integration."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1])
    return auc


def simulate_brain_microdialysis(
    drug_name: str,
    dose_mg: float,
    ps_mL_per_min: float = 1.0,
    kp_uu: float = 0.3,
    fu_plasma: float = 0.1,
    cl_sys_L_per_h: float = 5.0,
    vd_plasma_L: float = 10.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> BrainMicrodialysisResult:
    """Simulate brain ECF PK via microdialysis two-compartment model.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        IV bolus dose in mg.
    ps_mL_per_min:
        Permeability-surface area product (mL/min).
    kp_uu:
        Unbound brain-to-plasma ratio (target parameter).
    fu_plasma:
        Unbound fraction in plasma (0 < fu_plasma <= 1).
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_plasma_L:
        Plasma volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler step size (h).

    Returns
    -------
    BrainMicrodialysisResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if not (0 < fu_plasma <= 1):
        raise ValueError("fu_plasma must be in (0, 1]")
    if ps_mL_per_min < 0:
        raise ValueError("ps_mL_per_min must be >= 0")
    if kp_uu <= 0:
        raise ValueError("kp_uu must be > 0")

    # --- Constants ---
    v_ecf_L = 0.2  # brain ECF volume (L)
    cl_ecf_L_per_h = 0.01  # ECF bulk flow clearance (L/h)

    # Convert PS from mL/min to L/h
    ps_L_per_h = ps_mL_per_min * 60.0 / 1000.0

    # --- Initial conditions ---
    c_plasma = dose_mg / vd_plasma_L
    c_ecf = 0.0

    times: list = []
    c_plasma_list: list = []
    c_ecf_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(n_steps):
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_ecf_list.append(c_ecf)

        # ODEs (no external input after bolus)
        dc_plasma_dt = -(cl_sys_L_per_h / vd_plasma_L) * c_plasma
        dc_ecf_dt = (
            ps_L_per_h * (fu_plasma * c_plasma - c_ecf / kp_uu) / v_ecf_L
            - (cl_ecf_L_per_h / v_ecf_L) * c_ecf
        )

        c_plasma = max(0.0, c_plasma + dc_plasma_dt * dt_h)
        c_ecf = max(0.0, c_ecf + dc_ecf_dt * dt_h)

        t += dt_h

    # --- Derived metrics ---
    cmax_ecf = max(c_ecf_list)
    tmax_ecf = times[c_ecf_list.index(cmax_ecf)]
    auc_ecf = _trapz(times, c_ecf_list)
    auc_plasma = _trapz(times, c_plasma_list)

    if auc_plasma > 0:
        ecf_to_plasma_ratio = auc_ecf / auc_plasma
    else:
        ecf_to_plasma_ratio = 0.0

    # Brain penetration class based on AUC ratio
    if ecf_to_plasma_ratio < 0.1:
        penetration_class = "poor"
    elif ecf_to_plasma_ratio <= 0.3:
        penetration_class = "moderate"
    else:
        penetration_class = "good"

    notes = (
        f"PS={ps_mL_per_min:.2f} mL/min, Kp_uu={kp_uu:.3f}, "
        f"fu_plasma={fu_plasma:.3f}. "
        f"ECF/Plasma AUC ratio={ecf_to_plasma_ratio:.3f}."
    )

    return BrainMicrodialysisResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        kp_uu=kp_uu,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_ecf_mg_L=c_ecf_list,
        cmax_ecf_mg_L=cmax_ecf,
        tmax_ecf_h=tmax_ecf,
        auc_ecf_mg_h_per_L=auc_ecf,
        auc_plasma_mg_h_per_L=auc_plasma,
        ecf_to_plasma_auc_ratio=ecf_to_plasma_ratio,
        brain_penetration_class=penetration_class,
        notes=notes,
    )
