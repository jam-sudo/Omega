"""Phase 626 — Drug Spinal Fluid (CSF) Distribution.

Models drug distribution into cerebrospinal fluid (CSF) for CNS drug assessment,
accounting for blood-CSF barrier permeability, plasma protein binding, and CSF
drainage.
"""

from dataclasses import dataclass


@dataclass
class CSFDistributionResult:
    """Result of CSF distribution simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_csf_mg_L: list
    csf_plasma_ratio: list
    cmax_plasma_mg_L: float
    cmax_csf_mg_L: float
    tmax_csf_h: float
    auc_plasma_mg_h_per_L: float
    auc_csf_mg_h_per_L: float
    csf_auc_ratio: float
    bbcsfb_permeability: float
    t_equilibration_h: float
    efficacy_potential: str
    notes: str


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Manual trapezoidal AUC integration."""
    return sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def simulate_csf_distribution(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    psa_A2: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    fu_plasma: float = 0.1,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> CSFDistributionResult:
    """Simulate drug distribution into cerebrospinal fluid.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in mg.
    logP:
        Octanol-water partition coefficient (lipophilicity).
    mw_Da:
        Molecular weight in Daltons.
    psa_A2:
        Polar surface area in Angstrom^2.
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Volume of distribution (L).
    fu_plasma:
        Unbound fraction in plasma (0–1).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Integration step size (h).

    Returns
    -------
    CSFDistributionResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if psa_A2 < 0:
        raise ValueError("psa_A2 must be >= 0")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be in (0, 1]")

    # --- Blood-CSF barrier permeability ---
    psa_factor = max(0.0, 1.0 - psa_A2 / 200.0)
    P_bcsfb = 1e-6 * max(0.01, logP) * (300.0 / max(200.0, mw_Da)) * psa_factor
    P_bcsfb = max(1e-9, min(1e-5, P_bcsfb))

    # Transfer rate to CSF (/h): permeability × 3600 × surface area factor
    k_csf = P_bcsfb * 3600.0 * 0.1
    k_csf = max(0.001, min(0.5, k_csf))

    # --- CSF compartment constants ---
    V_csf = 0.15  # L (150 mL)
    k_drain = 0.14  # /h (CSF drainage ~500 mL/day)

    # --- Oral PK constants ---
    ka = 1.2  # /h
    f_oral = 0.85

    # --- Initial conditions ---
    A_gut = dose_mg * f_oral  # mg
    C_plasma = 0.0  # mg/L
    C_csf = 0.0  # mg/L

    # --- Output arrays ---
    n_steps = int(round(t_end_h / dt_h)) + 1
    times = []
    c_plasma_list = []
    c_csf_list = []
    ratio_list = []

    for step in range(n_steps):
        t = step * dt_h

        # Record state
        times.append(t)
        c_plasma_list.append(max(0.0, C_plasma))
        c_csf_list.append(max(0.0, C_csf))
        ratio = C_csf / max(1e-10, C_plasma)
        ratio_list.append(ratio)

        if step == n_steps - 1:
            break

        # --- ODE derivatives ---
        dA_gut_dt = -ka * A_gut

        dC_plasma_dt = (
            ka * A_gut / vd_sys_L
            - (cl_sys_L_per_h / vd_sys_L) * C_plasma
            - k_csf * fu_plasma * C_plasma
        )

        dC_csf_dt = k_csf * fu_plasma * C_plasma * vd_sys_L / V_csf - k_drain * C_csf

        # --- Forward Euler update ---
        A_gut = max(0.0, A_gut + dA_gut_dt * dt_h)
        C_plasma = max(0.0, C_plasma + dC_plasma_dt * dt_h)
        C_csf = max(0.0, C_csf + dC_csf_dt * dt_h)

    # --- Post-processing ---
    auc_plasma = _trapezoidal_auc(times, c_plasma_list)
    auc_csf = _trapezoidal_auc(times, c_csf_list)

    csf_auc_ratio = auc_csf / auc_plasma if auc_plasma > 0.0 else 0.0

    cmax_plasma = max(c_plasma_list)
    cmax_csf = max(c_csf_list)

    tmax_csf_idx = c_csf_list.index(cmax_csf)
    tmax_csf = times[tmax_csf_idx]

    # Steady-state CSF/plasma ratio (analytical)
    # At steady state: k_csf * fu_plasma * C_plasma * vd_sys / V_csf = k_drain * C_csf
    # → C_csf / C_plasma = k_csf * fu_plasma * vd_sys / (k_drain * V_csf)
    ss_ratio = k_csf * fu_plasma * vd_sys_L / (k_drain * V_csf)

    # t_equilibration: time when csf_plasma_ratio first exceeds 0.9 * csf_auc_ratio
    target_ratio = 0.9 * csf_auc_ratio
    t_equilibration = t_end_h  # default: not reached
    for i, r in enumerate(ratio_list):
        if r >= target_ratio and target_ratio > 0:
            t_equilibration = times[i]
            break

    # Efficacy potential
    if csf_auc_ratio > 0.5:
        efficacy_potential = "high"
    elif csf_auc_ratio > 0.1:
        efficacy_potential = "moderate"
    else:
        efficacy_potential = "low"

    notes = (
        f"Blood-CSF barrier permeability: {P_bcsfb:.2e} cm/s, "
        f"k_csf={k_csf:.4f} /h, k_drain={k_drain} /h, "
        f"ss_ratio={ss_ratio:.3f}"
    )

    return CSFDistributionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_csf_mg_L=c_csf_list,
        csf_plasma_ratio=ratio_list,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_csf_mg_L=cmax_csf,
        tmax_csf_h=tmax_csf,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_csf_mg_h_per_L=auc_csf,
        csf_auc_ratio=csf_auc_ratio,
        bbcsfb_permeability=P_bcsfb,
        t_equilibration_h=t_equilibration,
        efficacy_potential=efficacy_potential,
        notes=notes,
    )
