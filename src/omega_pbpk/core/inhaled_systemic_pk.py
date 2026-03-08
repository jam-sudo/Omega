"""Phase 321 — Inhaled Drug Systemic Absorption Model.

Models systemic absorption of inhaled drugs from lung deposition.
Two-step: (1) pulmonary absorption from lung → systemic;
(2) any swallowed fraction → GI absorption.
Relevant for inhaled corticosteroids (ICS), beta-agonists.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InhaledSystemicPKResult:
    """Result of inhaled systemic PK simulation."""

    drug_name: str
    dose_mg: float
    fine_particle_fraction: float
    lung_deposition_fraction: float
    f_oral_swallowed: float
    times_h: list
    c_systemic_mg_L: list
    auc_systemic_mg_h_per_L: float
    cmax_systemic_mg_L: float
    tmax_systemic_h: float
    lung_dose_mg: float
    swallowed_dose_mg: float
    lung_contribution_pct: float
    notes: str


def _validate_inhaled_params(
    dose_mg: float,
    lung_deposition_fraction: float,
    fine_particle_fraction: float,
    f_oral_swallowed: float,
    ka_lung_per_h: float,
    ka_gi_per_h: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (0 <= lung_deposition_fraction <= 1):
        raise ValueError("lung_deposition_fraction must be in [0, 1]")
    if not (0 <= fine_particle_fraction <= 1):
        raise ValueError("fine_particle_fraction must be in [0, 1]")
    if not (0 < f_oral_swallowed <= 1):
        raise ValueError("f_oral_swallowed must be in (0, 1]")
    if ka_lung_per_h <= 0:
        raise ValueError("ka_lung_per_h must be > 0")
    if ka_gi_per_h <= 0:
        raise ValueError("ka_gi_per_h must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")


def simulate_inhaled_systemic_pk(
    drug_name: str,
    dose_mg: float,
    fine_particle_fraction: float,
    lung_deposition_fraction: float,
    f_oral_swallowed: float,
    ka_lung_per_h: float,
    ka_gi_per_h: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> InhaledSystemicPKResult:
    """Simulate systemic PK of an inhaled drug.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Nominal inhaled dose (mg).
    fine_particle_fraction:
        Fraction of dose as fine particles (deposited in lungs).
        The complement (1 - fine_particle_fraction) is coarse particles
        (deposited in upper airways → swallowed). This parameter is
        stored in the result for reference; lung_deposition_fraction
        controls the actual ODE initial conditions.
    lung_deposition_fraction:
        Fraction of inhaled dose actually reaching the lungs
        (the rest is deposited in the upper airways and swallowed).
    f_oral_swallowed:
        First-pass corrected oral bioavailability of the swallowed fraction.
    ka_lung_per_h:
        Pulmonary absorption rate constant (1/h).
    ka_gi_per_h:
        GI absorption rate constant for swallowed fraction (1/h).
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward-Euler time step (h).

    Returns
    -------
    InhaledSystemicPKResult
    """
    _validate_inhaled_params(
        dose_mg,
        lung_deposition_fraction,
        fine_particle_fraction,
        f_oral_swallowed,
        ka_lung_per_h,
        ka_gi_per_h,
        cl_sys_L_per_h,
        vd_sys_L,
        t_end_h,
    )

    k_elim = cl_sys_L_per_h / vd_sys_L

    lung_dose_mg = dose_mg * lung_deposition_fraction
    swallowed_dose_mg = dose_mg * (1.0 - lung_deposition_fraction)

    # Initial conditions
    a_lung = lung_dose_mg
    a_gut = swallowed_dose_mg
    a_sys = 0.0

    n_steps = max(1, int(t_end_h / dt_h))
    times_h: list = [0.0]
    c_sys_list: list = [a_sys / vd_sys_L]

    for _ in range(n_steps):
        d_lung = -ka_lung_per_h * a_lung
        d_gut = -ka_gi_per_h * a_gut
        d_sys = ka_lung_per_h * a_lung + f_oral_swallowed * ka_gi_per_h * a_gut - k_elim * a_sys

        a_lung = max(0.0, a_lung + d_lung * dt_h)
        a_gut = max(0.0, a_gut + d_gut * dt_h)
        a_sys = max(0.0, a_sys + d_sys * dt_h)

        times_h.append(times_h[-1] + dt_h)
        c_sys_list.append(a_sys / vd_sys_L)

    # AUC via trapezoidal rule
    auc = sum(
        0.5 * (c_sys_list[i] + c_sys_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    cmax = max(c_sys_list)
    tmax = times_h[c_sys_list.index(cmax)]

    # Lung contribution estimate
    denom = lung_dose_mg + f_oral_swallowed * swallowed_dose_mg
    if denom > 0:
        lung_contribution_pct = lung_dose_mg / denom * 100.0
    else:
        lung_contribution_pct = 0.0

    notes = (
        f"Lung dose: {lung_dose_mg:.3f} mg; "
        f"Swallowed dose: {swallowed_dose_mg:.3f} mg; "
        f"f_oral_swallowed: {f_oral_swallowed:.2f}; "
        f"Lung contribution: {lung_contribution_pct:.1f}%"
    )

    return InhaledSystemicPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fine_particle_fraction=fine_particle_fraction,
        lung_deposition_fraction=lung_deposition_fraction,
        f_oral_swallowed=f_oral_swallowed,
        times_h=times_h,
        c_systemic_mg_L=c_sys_list,
        auc_systemic_mg_h_per_L=auc,
        cmax_systemic_mg_L=cmax,
        tmax_systemic_h=tmax,
        lung_dose_mg=lung_dose_mg,
        swallowed_dose_mg=swallowed_dose_mg,
        lung_contribution_pct=lung_contribution_pct,
        notes=notes,
    )


def compare_devices(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    device_list: list,
    ka_lung_per_h: float = 1.5,
    ka_gi_per_h: float = 0.8,
    f_oral_swallowed: float = 0.2,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> list:
    """Compare inhaler devices by systemic AUC.

    Parameters
    ----------
    drug_name:
        Drug name.
    dose_mg:
        Nominal inhaled dose (mg).
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Volume of distribution (L).
    device_list:
        List of dicts with keys:
        - device_name (str)
        - fine_particle_fraction (float)
        - lung_deposition_fraction (float)
    ka_lung_per_h:
        Pulmonary absorption rate (1/h); default 1.5.
    ka_gi_per_h:
        GI absorption rate (1/h); default 0.8.
    f_oral_swallowed:
        Bioavailability of swallowed fraction; default 0.2.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward-Euler step (h).

    Returns
    -------
    list of dicts sorted by auc_systemic descending, each with:
    device_name, auc_systemic_mg_h_per_L, cmax_systemic_mg_L,
    lung_contribution_pct, result (InhaledSystemicPKResult)
    """
    results = []
    for device in device_list:
        res = simulate_inhaled_systemic_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            fine_particle_fraction=device["fine_particle_fraction"],
            lung_deposition_fraction=device["lung_deposition_fraction"],
            f_oral_swallowed=f_oral_swallowed,
            ka_lung_per_h=ka_lung_per_h,
            ka_gi_per_h=ka_gi_per_h,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_sys_L=vd_sys_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(
            {
                "device_name": device["device_name"],
                "auc_systemic_mg_h_per_L": res.auc_systemic_mg_h_per_L,
                "cmax_systemic_mg_L": res.cmax_systemic_mg_L,
                "lung_contribution_pct": res.lung_contribution_pct,
                "result": res,
            }
        )

    results.sort(key=lambda x: x["auc_systemic_mg_h_per_L"], reverse=True)
    return results
