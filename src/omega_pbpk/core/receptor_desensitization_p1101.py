"""Phase 1101 — Drug Receptor Desensitization Kinetics.

3-state receptor model: Active (Ra), Desensitized (Rd), Resensitized (implicit Rs = 1 - Ra - Rd).
Relevant for beta-agonists, opioids, benzodiazepines, SSRIs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReceptorDesensitizationResult:
    """Result of receptor desensitization simulation."""

    drug_name: str
    dose_mg: float
    tau_h: float
    n_doses: int
    times_h: list
    c_drug_mg_L: list
    ra_fraction: list  # active receptor fraction [0, 1]
    rd_fraction: list  # desensitized fraction [0, 1]
    effect_pct: list  # effect at each time point
    peak_effect_dose1: float  # peak effect after first dose
    peak_effect_doseN: float  # peak effect after nth dose
    tachyphylaxis_ratio: float  # peak_effect_doseN / peak_effect_dose1 (< 1 means tachyphylaxis)
    recovery_time_h: float  # time for Ra to return to >90% after stopping
    notes: str


def simulate_receptor_desensitization(
    drug_name: str,
    dose_mg: float,
    tau_h: float = 8.0,
    n_doses: int = 5,
    ke_drug: float = 0.3,
    vd_L: float = 50.0,
    k_des: float = 0.5,
    k_res: float = 0.1,
    ec50: float = 0.1,
    emax: float = 100.0,
    dt_h: float = 0.02,
) -> ReceptorDesensitizationResult:
    """Simulate receptor desensitization with repeated dosing.

    Uses forward Euler ODE for the 3-state receptor model combined with
    a 1-compartment drug PK model.

    Parameters
    ----------
    drug_name:   compound name
    dose_mg:     dose per administration [mg]
    tau_h:       dosing interval [h]
    n_doses:     number of doses
    ke_drug:     first-order elimination rate constant [1/h]
    vd_L:        volume of distribution [L]
    k_des:       receptor desensitization rate constant [1/h] (when drug present)
    k_res:       receptor resensitization rate constant [1/h]
    ec50:        half-maximal effect concentration [mg/L]
    emax:        maximum effect [%]
    dt_h:        ODE integration step size [h]

    Returns
    -------
    ReceptorDesensitizationResult
    """
    # --- validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if tau_h <= 0:
        raise ValueError("tau_h must be > 0")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")
    if ke_drug <= 0:
        raise ValueError("ke_drug must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if ec50 <= 0:
        raise ValueError("ec50 must be > 0")
    if emax <= 0:
        raise ValueError("emax must be > 0")

    # Total simulation: n_doses dosing intervals + 1 extra for recovery observation
    t_total = (n_doses + 1) * tau_h
    n_steps = int(t_total / dt_h) + 1

    times_h: list = []
    c_drug_mg_L: list = []
    ra_fraction: list = []
    rd_fraction: list = []
    effect_pct: list = []

    # State variables
    c = 0.0  # drug concentration [mg/L]
    ra = 1.0  # active receptor fraction
    rd = 0.0  # desensitized receptor fraction

    # Track doses administered
    doses_given = 0
    # First dose at t=0
    c += dose_mg / vd_L
    doses_given = 1

    # Determine time windows for each dose to compute peak effects
    dose1_end = tau_h  # first dose interval [0, tau_h)
    doseN_start = (n_doses - 1) * tau_h  # last dose interval starts here
    doseN_end = n_doses * tau_h

    peak_effect_dose1 = 0.0
    peak_effect_doseN = 0.0

    for step in range(n_steps):
        t = step * dt_h

        # Compute effect at current state
        eff = emax * ra * c / (ec50 + c) if (ec50 + c) > 0 else 0.0
        eff = max(0.0, min(100.0, eff))

        times_h.append(t)
        c_drug_mg_L.append(c)
        ra_fraction.append(ra)
        rd_fraction.append(rd)
        effect_pct.append(eff)

        # Track peak effects per dose window
        if t < dose1_end:
            if eff > peak_effect_dose1:
                peak_effect_dose1 = eff
        if doseN_start <= t < doseN_end:
            if eff > peak_effect_doseN:
                peak_effect_doseN = eff

        # Forward Euler for ODE
        # Drug PK: dC/dt = -ke * C
        dc_dt = -ke_drug * c

        # Receptor: desensitization rate (drug-dependent)
        des_rate = k_des * ra * c / (ec50 + c) if (ec50 + c) > 0 else 0.0
        dra_dt = k_res * rd - des_rate
        drd_dt = des_rate - k_res * rd

        c = max(0.0, c + dc_dt * dt_h)
        ra = ra + dra_dt * dt_h
        rd = rd + drd_dt * dt_h

        # Clamp receptor fractions to valid range
        ra = max(0.0, min(1.0, ra))
        rd = max(0.0, min(1.0 - ra, rd))

        # Administer next dose at dosing interval
        t_next = (step + 1) * dt_h
        if doses_given < n_doses:
            next_dose_time = doses_given * tau_h
            if t < next_dose_time <= t_next:
                c += dose_mg / vd_L
                doses_given += 1

    # Estimate recovery time: time after last dose for Ra to return to >90%
    # Scan from last dose onward
    recovery_time_h = tau_h  # default if recovery found quickly
    recovery_threshold = 0.9
    found_recovery = False
    for i, t in enumerate(times_h):
        if t >= n_doses * tau_h and ra_fraction[i] >= recovery_threshold:
            recovery_time_h = t - n_doses * tau_h
            found_recovery = True
            break
    if not found_recovery:
        # Extend simulation to find recovery
        c_ext = c_drug_mg_L[-1]
        ra_ext = ra_fraction[-1]
        rd_ext = rd_fraction[-1]
        t_ext = 0.0
        max_recovery_search_h = 48.0
        while t_ext < max_recovery_search_h:
            des_rate_ext = k_des * ra_ext * c_ext / (ec50 + c_ext) if (ec50 + c_ext) > 0 else 0.0
            dc_dt_ext = -ke_drug * c_ext
            dra_dt_ext = k_res * rd_ext - des_rate_ext
            drd_dt_ext = des_rate_ext - k_res * rd_ext
            c_ext = max(0.0, c_ext + dc_dt_ext * dt_h)
            ra_ext = ra_ext + dra_dt_ext * dt_h
            rd_ext = rd_ext + drd_dt_ext * dt_h
            ra_ext = max(0.0, min(1.0, ra_ext))
            rd_ext = max(0.0, min(1.0 - ra_ext, rd_ext))
            t_ext += dt_h
            if ra_ext >= recovery_threshold:
                recovery_time_h = t_ext
                found_recovery = True
                break
        if not found_recovery:
            recovery_time_h = max_recovery_search_h

    # Tachyphylaxis ratio
    if peak_effect_dose1 > 0:
        tachyphylaxis_ratio = peak_effect_doseN / peak_effect_dose1
    else:
        tachyphylaxis_ratio = 1.0

    notes = (
        f"Simulated {n_doses} doses of {dose_mg:.1f} mg every {tau_h:.1f} h. "
        f"Tachyphylaxis ratio: {tachyphylaxis_ratio:.3f}. "
        f"Recovery time: {recovery_time_h:.2f} h."
    )

    return ReceptorDesensitizationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        tau_h=tau_h,
        n_doses=n_doses,
        times_h=times_h,
        c_drug_mg_L=c_drug_mg_L,
        ra_fraction=ra_fraction,
        rd_fraction=rd_fraction,
        effect_pct=effect_pct,
        peak_effect_dose1=peak_effect_dose1,
        peak_effect_doseN=peak_effect_doseN,
        tachyphylaxis_ratio=tachyphylaxis_ratio,
        recovery_time_h=recovery_time_h,
        notes=notes,
    )


def compare_dosing_intervals(
    drug_name: str,
    dose_mg: float,
    tau_list: list,
) -> list:
    """Compare tachyphylaxis across multiple dosing intervals.

    Parameters
    ----------
    drug_name:  compound name
    dose_mg:    dose per administration [mg]
    tau_list:   list of dosing intervals [h] to evaluate

    Returns
    -------
    list of ReceptorDesensitizationResult sorted by tachyphylaxis_ratio descending
    (least tachyphylaxis = largest ratio = best)
    """
    results = []
    for tau in tau_list:
        res = simulate_receptor_desensitization(
            drug_name=drug_name,
            dose_mg=dose_mg,
            tau_h=tau,
        )
        results.append(res)

    # Sort descending by tachyphylaxis_ratio (highest = least tachyphylaxis = best)
    results.sort(key=lambda r: r.tachyphylaxis_ratio, reverse=True)
    return results
