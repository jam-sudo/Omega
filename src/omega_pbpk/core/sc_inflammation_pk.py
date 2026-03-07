"""Phase 372 — Subcutaneous Absorption with Injection Site Inflammation.

Models SC drug absorption with local injection site inflammation affecting
absorption kinetics. Forward Euler ODE integration.
"""

from dataclasses import dataclass


@dataclass
class SCInflammationPKResult:
    """Result of SC absorption with inflammation simulation."""

    times_h: list
    c_sc_depot_mg_mL: list
    c_systemic_mg_L: list
    k_abs_profile_per_h: list
    auc_systemic_mg_h_per_L: float
    cmax_systemic_mg_L: float
    tmax_systemic_h: float
    bioavailability_pct: float
    inflammation_effect_on_auc_pct: float
    absorption_phase_class: str
    notes: str


def _trapezoidal_auc(times: list, values: list) -> float:
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def _simulate_core(
    dose_mg: float,
    k_abs_normal: float,
    inflammation_factor: float,
    cl_sys: float,
    vd_sys: float,
    t_inflammation_h: float,
    v_sc_mL: float,
    t_end_h: float,
    dt_h: float,
) -> tuple:
    """Run Forward Euler simulation.

    Returns (times, c_sc_depot, c_systemic, k_abs_profile).
    """
    ke = cl_sys / vd_sys  # elimination rate constant (1/h)

    # Initial conditions
    a_sc = dose_mg  # mg in SC depot
    c_sys = 0.0  # mg/L in systemic compartment

    times = []
    c_sc_list = []
    c_sys_list = []
    k_abs_list = []

    t = 0.0
    while t <= t_end_h + 1e-9:
        k_abs_eff = k_abs_normal * inflammation_factor if t <= t_inflammation_h else k_abs_normal

        times.append(t)
        c_sc_list.append(a_sc / v_sc_mL)
        c_sys_list.append(c_sys)
        k_abs_list.append(k_abs_eff)

        # Forward Euler step
        da_sc = -k_abs_eff * a_sc * dt_h
        dc_sys = (k_abs_eff * a_sc / vd_sys - ke * c_sys) * dt_h

        a_sc = max(0.0, a_sc + da_sc)
        c_sys = max(0.0, c_sys + dc_sys)

        t = round(t + dt_h, 10)

    return times, c_sc_list, c_sys_list, k_abs_list


def simulate_sc_inflammation_pk(
    drug_name: str,
    dose_mg: float,
    k_abs_normal_per_h: float,
    inflammation_factor: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    t_inflammation_h: float = 48.0,
    v_sc_mL: float = 3.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> SCInflammationPKResult:
    """Simulate SC absorption with injection site inflammation.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        SC dose in mg (must be > 0).
    k_abs_normal_per_h:
        Normal absorption rate constant (1/h, must be > 0).
    inflammation_factor:
        Multiplier on k_abs during inflammation window.
        >1 = increased vascularization (faster absorption).
        <1 = fibrosis (slower absorption).
        Must be > 0.
    cl_sys_L_per_h:
        Systemic clearance in L/h (must be > 0).
    vd_sys_L:
        Systemic volume of distribution in L (must be > 0).
    t_inflammation_h:
        Duration of inflammation in hours (default 48).
    v_sc_mL:
        Volume of SC compartment in mL (default 3.0).
    t_end_h:
        Simulation end time in hours (default 72).
    dt_h:
        Time step for Forward Euler integration (default 0.1 h).

    Returns
    -------
    SCInflammationPKResult
        Full PK simulation results including inflammation effect.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}.")
    if k_abs_normal_per_h <= 0:
        raise ValueError(f"k_abs_normal_per_h must be > 0, got {k_abs_normal_per_h}.")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}.")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}.")
    if inflammation_factor <= 0:
        raise ValueError(f"inflammation_factor must be > 0, got {inflammation_factor}.")

    # Primary simulation (with inflammation)
    times, c_sc_list, c_sys_list, k_abs_list = _simulate_core(
        dose_mg=dose_mg,
        k_abs_normal=k_abs_normal_per_h,
        inflammation_factor=inflammation_factor,
        cl_sys=cl_sys_L_per_h,
        vd_sys=vd_sys_L,
        t_inflammation_h=t_inflammation_h,
        v_sc_mL=v_sc_mL,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    auc = _trapezoidal_auc(times, c_sys_list)
    cmax = max(c_sys_list)
    tmax = times[c_sys_list.index(cmax)]

    # Bioavailability relative to theoretical infinite AUC = dose / CL
    theoretical_auc = dose_mg / cl_sys_L_per_h
    bioavailability_pct = (auc / theoretical_auc) * 100.0

    # Reference simulation with no inflammation (factor = 1.0)
    _, _, c_sys_ref, _ = _simulate_core(
        dose_mg=dose_mg,
        k_abs_normal=k_abs_normal_per_h,
        inflammation_factor=1.0,
        cl_sys=cl_sys_L_per_h,
        vd_sys=vd_sys_L,
        t_inflammation_h=t_inflammation_h,
        v_sc_mL=v_sc_mL,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )
    auc_ref = _trapezoidal_auc(times, c_sys_ref)

    if auc_ref > 0:
        inflammation_effect_pct = (auc - auc_ref) / auc_ref * 100.0
    else:
        inflammation_effect_pct = 0.0

    # Absorption phase class
    if inflammation_factor > 1.2:
        absorption_phase_class = "accelerated"
    elif inflammation_factor < 0.8:
        absorption_phase_class = "delayed"
    else:
        absorption_phase_class = "normal"

    notes_parts = [
        f"Drug: {drug_name}.",
        f"Inflammation factor: {inflammation_factor:.2f} for first {t_inflammation_h:.1f} h.",
        f"Absorption classified as '{absorption_phase_class}'.",
        f"AUC (inflamed): {auc:.3f} mg*h/L, AUC (normal): {auc_ref:.3f} mg*h/L.",
        f"Inflammation effect on AUC: {inflammation_effect_pct:+.1f}%.",
    ]

    return SCInflammationPKResult(
        times_h=times,
        c_sc_depot_mg_mL=c_sc_list,
        c_systemic_mg_L=c_sys_list,
        k_abs_profile_per_h=k_abs_list,
        auc_systemic_mg_h_per_L=auc,
        cmax_systemic_mg_L=cmax,
        tmax_systemic_h=tmax,
        bioavailability_pct=bioavailability_pct,
        inflammation_effect_on_auc_pct=inflammation_effect_pct,
        absorption_phase_class=absorption_phase_class,
        notes=" ".join(notes_parts),
    )
