"""Nasal absorption PK for intranasal drug delivery (sprays, drops).

Models rapid nasal absorption bypassing first-pass metabolism, with
mucociliary clearance delivering drug to GI for oral absorption.
"""

from dataclasses import dataclass


@dataclass
class NasalAbsorptionResult:
    drug_name: str
    dose_mg: float
    logP: float
    f_nasal_deposit: float
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    f_nasal_bioavailability_pct: float
    first_pass_bypass_advantage: bool
    notes: str


def _k_abs_nasal_from_logp(logp: float) -> float:
    """Nasal absorption rate constant adjusted for logP."""
    base = 1.2
    if logp < 0:
        return base * 0.4
    if logp <= 3:
        return base * 1.0
    return base * 0.7


def simulate_nasal_absorption_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_L_per_h: float = 5.0,
    vd_L: float = 30.0,
    f_nasal_deposit: float = 0.7,
    t_end_h: float = 8.0,
    dt_h: float = 0.05,
) -> NasalAbsorptionResult:
    """Simulate nasal absorption PK using Forward Euler ODE integration.

    Compartments:
        A_nasal (mg): drug in nasal cavity
        A_mucociliary (mg): drug cleared to GI by mucociliary transport
        C_plasma (mg/L): systemic plasma concentration

    Args:
        drug_name: identifier for the drug
        dose_mg: intranasal dose in mg
        logP: octanol-water partition coefficient
        cl_L_per_h: systemic clearance (L/h)
        vd_L: volume of distribution (L)
        f_nasal_deposit: fraction of dose reaching nasal mucosa (rest swallowed)
        t_end_h: simulation end time (h)
        dt_h: integration time step (h)

    Returns:
        NasalAbsorptionResult with PK metrics and time series.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if logP < -5 or logP > 10:
        raise ValueError("logP must be in [-5, 10]")
    if f_nasal_deposit <= 0 or f_nasal_deposit > 1:
        raise ValueError("f_nasal_deposit must be in (0, 1]")

    k_abs_nasal = _k_abs_nasal_from_logp(logP)
    k_mucociliary = 0.4  # /h
    k_abs_oral = 0.3  # /h GI absorption of swallowed mucociliary fraction

    # Initial conditions
    a_nasal = dose_mg * f_nasal_deposit
    a_muco = 0.0
    c_plasma = 0.0

    # Oral fraction swallowed directly (absorbed at k_abs_oral)
    a_oral_direct = dose_mg * (1.0 - f_nasal_deposit)

    ke = cl_L_per_h / vd_L

    times_h = []
    c_plasma_list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        times_h.append(t)
        c_plasma_list.append(c_plasma)

        # Compute derivatives
        d_a_nasal = -(k_abs_nasal + k_mucociliary) * a_nasal
        d_a_muco = k_mucociliary * a_nasal - k_abs_oral * a_muco
        # oral direct fraction contribution
        d_a_oral_direct = -k_abs_oral * a_oral_direct

        d_c_plasma = (
            k_abs_nasal * a_nasal / vd_L
            + k_abs_oral * a_muco / vd_L
            + k_abs_oral * a_oral_direct / vd_L
            - ke * c_plasma
        )

        # Forward Euler update
        a_nasal = a_nasal + dt_h * d_a_nasal
        a_muco = a_muco + dt_h * d_a_muco
        a_oral_direct = a_oral_direct + dt_h * d_a_oral_direct
        c_plasma = c_plasma + dt_h * d_c_plasma

        if a_nasal < 0:
            a_nasal = 0.0
        if a_muco < 0:
            a_muco = 0.0
        if a_oral_direct < 0:
            a_oral_direct = 0.0
        if c_plasma < 0:
            c_plasma = 0.0

        t += dt_h

    # PK metrics
    cmax = max(c_plasma_list)
    tmax = times_h[c_plasma_list.index(cmax)]

    # Manual trapz AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma_list[i - 1] + c_plasma_list[i]) * (times_h[i] - times_h[i - 1])

    # Bioavailability: AUC_nasal relative to theoretical IV AUC (dose/CL)
    theoretical_iv_auc = dose_mg / cl_L_per_h
    f_nasal_bioavailability_pct = (auc / theoretical_iv_auc) * 100.0

    # Oral equivalent AUC: f_oral=0.6, k_abs_oral=1.0/h
    oral_equivalent_auc = 0.6 * dose_mg / cl_L_per_h
    first_pass_bypass_advantage = auc > oral_equivalent_auc

    logp_desc = (
        "low logP (hydrophilic)"
        if logP < 0
        else ("optimal logP" if logP <= 3 else "high logP (lipophilic)")
    )
    notes = (
        f"k_abs_nasal={k_abs_nasal:.3f}/h ({logp_desc}), "
        f"f_nasal_deposit={f_nasal_deposit:.2f}, "
        f"oral_swallowed_fraction={1 - f_nasal_deposit:.2f}"
    )

    return NasalAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        f_nasal_deposit=f_nasal_deposit,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        f_nasal_bioavailability_pct=f_nasal_bioavailability_pct,
        first_pass_bypass_advantage=first_pass_bypass_advantage,
        notes=notes,
    )
