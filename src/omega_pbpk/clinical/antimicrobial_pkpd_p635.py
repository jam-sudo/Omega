"""
Phase 635 — Antimicrobial PK/PD Integration

Integrates PK with PD for antimicrobials including time-kill kinetics
and resistance emergence prediction.
"""

from dataclasses import dataclass


@dataclass
class AntimicrobialPKPDResult:
    drug_name: str
    pathogen: str
    mic_mg_L: float
    pkpd_index: str
    times_h: list
    c_drug_mg_L: list
    log10_cfu: list
    times_above_mic_h: float
    fT_above_mic_pct: float
    auc_mic_ratio: float
    cmax_mic_ratio: float
    predicted_outcome: str
    resistance_risk: str
    optimal_pkpd_target_met: bool
    notes: str


def simulate_antimicrobial_pkpd(
    drug_name: str,
    pathogen: str,
    mic_mg_L: float,
    dose_mg: float,
    vd_L: float,
    cl_L_per_h: float,
    f_oral: float = 0.85,
    ka_per_h: float = 1.2,
    pkpd_index: str = "AUC/MIC",
    tau_h: float = 24.0,
    n_doses: int = 3,
    t_end_h: float = None,
    dt_h: float = 0.1,
) -> AntimicrobialPKPDResult:
    """Simulate antimicrobial PK/PD with time-kill kinetics and resistance prediction."""
    # Validation
    if mic_mg_L <= 0:
        raise ValueError("mic_mg_L must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")
    if pkpd_index not in ("fT>MIC", "AUC/MIC", "Cmax/MIC"):
        raise ValueError("pkpd_index must be one of 'fT>MIC', 'AUC/MIC', 'Cmax/MIC'")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")

    if t_end_h is None:
        t_end_h = n_doses * tau_h + 12.0

    # PD parameters
    E_max = 2.0  # max kill rate log10 CFU/h
    E_c50 = mic_mg_L * 2.0  # concentration for 50% max effect
    k_growth = 0.5  # baseline growth rate log10 CFU/h
    N_max = 9.0  # carrying capacity log10 CFU/mL
    N_init = 6.0  # initial bacterial burden log10 CFU/mL

    # Dosing times
    dosing_times = [i * tau_h for i in range(n_doses)]

    # Simulation
    times_h = []
    c_drug = []
    log10_cfu = []

    t = 0.0
    A_gut = 0.0
    C = 0.0
    N = N_init  # log10 CFU/mL

    # Add first dose
    dose_gut = dose_mg * f_oral
    A_gut += dose_gut

    n_steps = int(round(t_end_h / dt_h))

    times_h.append(t)
    c_drug.append(C)
    log10_cfu.append(N)

    for _step in range(n_steps):
        # Check for doses at this time step (skip first dose, already added)
        for d_idx in range(1, n_doses):
            dose_time = dosing_times[d_idx]
            # Add dose at the step boundary (within dt_h tolerance)
            t_next = t + dt_h
            if t < dose_time <= t_next:
                A_gut += dose_mg * f_oral

        # PK ODEs — forward Euler
        dA_gut = -ka_per_h * A_gut
        dC = ka_per_h * A_gut / vd_L - (cl_L_per_h / vd_L) * C

        A_gut = max(0.0, A_gut + dA_gut * dt_h)
        C = max(0.0, C + dC * dt_h)

        # PD — bactericidal Emax model on log10 CFU
        k_kill = E_max * C / (E_c50 + C)
        dN = k_growth * (1.0 - N / N_max) - k_kill
        N = N + dN * dt_h
        N = max(0.0, min(N_max, N))

        t = t + dt_h
        times_h.append(t)
        c_drug.append(C)
        log10_cfu.append(N)

    # Compute PK/PD indices
    times_above_mic_h = sum(dt_h for c in c_drug if c > mic_mg_L)
    fT_above_mic_pct = min(100.0, times_above_mic_h / (n_doses * tau_h) * 100.0)

    # AUC via trapezoidal
    auc_drug = sum(
        0.5 * (c_drug[i] + c_drug[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_mic_ratio = auc_drug / mic_mg_L

    cmax = max(c_drug)
    cmax_mic_ratio = cmax / mic_mg_L

    # Outcome prediction
    if pkpd_index == "fT>MIC":
        if fT_above_mic_pct > 40.0:
            predicted_outcome = "bactericidal"
        elif fT_above_mic_pct > 20.0:
            predicted_outcome = "bacteriostatic"
        else:
            predicted_outcome = "treatment_failure"
    elif pkpd_index == "AUC/MIC":
        if auc_mic_ratio > 125.0:
            predicted_outcome = "bactericidal"
        elif auc_mic_ratio > 30.0:
            predicted_outcome = "bacteriostatic"
        else:
            predicted_outcome = "treatment_failure"
    else:  # "Cmax/MIC"
        if cmax_mic_ratio > 10.0:
            predicted_outcome = "bactericidal"
        elif cmax_mic_ratio > 4.0:
            predicted_outcome = "bacteriostatic"
        else:
            predicted_outcome = "treatment_failure"

    # Resistance risk
    final_log10_cfu = log10_cfu[-1]
    if final_log10_cfu > 4.0:
        resistance_risk = "high"
    elif final_log10_cfu > 2.0:
        resistance_risk = "moderate"
    else:
        resistance_risk = "low"

    optimal_pkpd_target_met = (
        predicted_outcome in ("bactericidal", "bacteriostatic") and final_log10_cfu < 3.0
    )

    notes = (
        f"Simulated {n_doses} doses of {dose_mg} mg {drug_name} "
        f"against {pathogen} (MIC={mic_mg_L} mg/L). "
        f"PK/PD index: {pkpd_index}. "
        f"Outcome: {predicted_outcome}. "
        f"Final burden: {final_log10_cfu:.2f} log10 CFU/mL."
    )

    return AntimicrobialPKPDResult(
        drug_name=drug_name,
        pathogen=pathogen,
        mic_mg_L=mic_mg_L,
        pkpd_index=pkpd_index,
        times_h=times_h,
        c_drug_mg_L=c_drug,
        log10_cfu=log10_cfu,
        times_above_mic_h=times_above_mic_h,
        fT_above_mic_pct=fT_above_mic_pct,
        auc_mic_ratio=auc_mic_ratio,
        cmax_mic_ratio=cmax_mic_ratio,
        predicted_outcome=predicted_outcome,
        resistance_risk=resistance_risk,
        optimal_pkpd_target_met=optimal_pkpd_target_met,
        notes=notes,
    )
