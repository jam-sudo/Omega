"""Antimicrobial Resistance PK/PD — Phase 1041.

Models PK/PD of antibiotics against resistant and sensitive bacterial populations
using a 1-compartment PK model with forward Euler integration and Hill Emax kill kinetics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _trapz(times: list[float], values: list[float]) -> float:
    """Trapezoidal AUC."""
    if len(times) < 2:
        return 0.0
    auc = 0.0
    for i in range(len(times) - 1):
        auc += 0.5 * (values[i] + values[i + 1]) * (times[i + 1] - times[i])
    return auc


@dataclass
class AntimicrobialPKPDResult:
    drug_name: str
    dose_mg: float
    mic_sensitive_mg_L: float
    mic_resistant_mg_L: float
    times_h: list
    c_plasma_mg_L: list
    n_sensitive: list
    n_resistant: list
    n_total: list
    auc_mic_ratio: float
    cmax_mic_ratio: float
    pct_time_above_mic: float
    eradication_probability: float
    resistance_amplification: bool
    outcome: str
    notes: str


def simulate_antimicrobial_pkpd(
    drug_name: str,
    dose_mg: float,
    dosing_interval_h: float = 24.0,
    n_doses: int = 7,
    mic_sensitive_mg_L: float = 0.5,
    mic_resistant_mg_L: float = 4.0,
    pkpd_type: str = "AUC_MIC",
    cl_L_per_h: float = 5.0,
    vd_L: float = 30.0,
    ka_per_h: float = 1.0,
    n0_sensitive: float = 6.0,
    n0_resistant: float = 4.0,
    mu_max_per_h: float = 0.8,
    n_max: float = 9.0,
) -> AntimicrobialPKPDResult:
    """Simulate antimicrobial PK/PD against sensitive and resistant bacterial populations."""
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")
    if mic_sensitive_mg_L <= 0:
        raise ValueError("mic_sensitive_mg_L must be > 0")
    if mic_resistant_mg_L <= mic_sensitive_mg_L:
        raise ValueError("mic_resistant_mg_L must be > mic_sensitive_mg_L")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if pkpd_type not in {"AUC_MIC", "Cmax_MIC", "fT_MIC"}:
        raise ValueError("pkpd_type must be one of: AUC_MIC, Cmax_MIC, fT_MIC")

    # PK parameters
    ke = cl_L_per_h / vd_L  # elimination rate constant

    # Simulation parameters
    dt = 0.05  # hours
    t_total = dosing_interval_h * n_doses
    n_steps = int(t_total / dt) + 1

    # Determine dosing times
    dose_times = [i * dosing_interval_h for i in range(n_doses)]

    # State variables
    a_gut = 0.0  # drug amount in gut (mg)
    a_plasma = 0.0  # drug amount in plasma (mg)

    # Bacterial log10 CFU/mL
    ns = n0_sensitive
    nr = n0_resistant

    # PD parameters
    kmax = 2.0 * mu_max_per_h
    ec50_s = mic_sensitive_mg_L / 2.0
    ec50_r = mic_resistant_mg_L / 2.0
    hill = 1.0

    # Storage
    times_h: list[float] = []
    c_plasma_list: list[float] = []
    n_sensitive_list: list[float] = []
    n_resistant_list: list[float] = []

    dose_times_set = set()
    for dt_val in dose_times:
        # Find the step index for each dose time
        step_idx = round(dt_val / dt)
        dose_times_set.add(step_idx)

    t = 0.0
    for step in range(n_steps):
        # Add dose at dosing times
        if step in dose_times_set:
            a_gut += dose_mg

        # Current plasma concentration
        c = a_plasma / vd_L

        # Record state
        times_h.append(t)
        c_plasma_list.append(max(0.0, c))
        n_sensitive_list.append(ns)
        n_resistant_list.append(nr)

        # Forward Euler for PK
        da_gut = -ka_per_h * a_gut
        da_plasma = ka_per_h * a_gut - ke * a_plasma

        a_gut += da_gut * dt
        a_plasma += da_plasma * dt
        if a_gut < 0.0:
            a_gut = 0.0
        if a_plasma < 0.0:
            a_plasma = 0.0

        # PD: Hill Emax kill rate
        c_for_pd = max(0.0, c)
        kill_s = kmax * (c_for_pd**hill) / (ec50_s**hill + c_for_pd**hill + 1e-12)
        kill_r = kmax * (c_for_pd**hill) / (ec50_r**hill + c_for_pd**hill + 1e-12)

        # Logistic growth
        capacity_factor_s = max(0.0, 1.0 - (10**ns) / (10**n_max))
        capacity_factor_r = max(0.0, 1.0 - (10**nr) / (10**n_max))

        # ODE in log10 scale
        dns_dt = (mu_max_per_h * capacity_factor_s - kill_s) / math.log(10)
        dnr_dt = (mu_max_per_h * capacity_factor_r - kill_r) / math.log(10)

        ns += dns_dt * dt
        nr += dnr_dt * dt

        # Clamp to detection limits
        ns = max(-2.0, min(n_max, ns))
        nr = max(-2.0, min(n_max, nr))

        t += dt

    # Compute n_total
    n_total_list: list[float] = []
    for i in range(len(n_sensitive_list)):
        n_s_val = n_sensitive_list[i]
        n_r_val = n_resistant_list[i]
        total_actual = (10**n_s_val) + (10**n_r_val)
        if total_actual > 0:
            n_total_list.append(math.log10(total_actual))
        else:
            n_total_list.append(-2.0)

    # Compute summary metrics
    auc_plasma = _trapz(times_h, c_plasma_list)
    auc_mic_ratio = auc_plasma / mic_sensitive_mg_L

    cmax = max(c_plasma_list) if c_plasma_list else 0.0
    cmax_mic_ratio = cmax / mic_sensitive_mg_L

    above_mic_count = sum(1 for c in c_plasma_list if c > mic_sensitive_mg_L)
    pct_time_above_mic = 100.0 * above_mic_count / len(c_plasma_list) if c_plasma_list else 0.0

    # Eradication probability based on pkpd_type
    if pkpd_type == "AUC_MIC":
        ratio = auc_mic_ratio
        if ratio > 400:
            eradication_probability = 0.9
        elif ratio > 100:
            eradication_probability = 0.7
        elif ratio > 25:
            eradication_probability = 0.4
        else:
            eradication_probability = 0.1
    elif pkpd_type == "Cmax_MIC":
        ratio = cmax_mic_ratio
        if ratio > 10:
            eradication_probability = 0.9
        elif ratio > 8:
            eradication_probability = 0.7
        elif ratio > 5:
            eradication_probability = 0.4
        else:
            eradication_probability = 0.1
    else:  # fT_MIC
        pct_frac = pct_time_above_mic / 100.0
        if pct_frac > 0.8:
            eradication_probability = 0.9
        elif pct_frac > 0.6:
            eradication_probability = 0.7
        elif pct_frac > 0.4:
            eradication_probability = 0.4
        else:
            eradication_probability = 0.1

    # Resistance amplification
    resistance_amplification = n_resistant_list[-1] > n0_resistant + 1.0

    # Outcome
    final_ns = n_sensitive_list[-1]
    if final_ns <= -1.0 and not resistance_amplification:
        outcome = "eradication"
    elif final_ns <= 2.0 and not resistance_amplification:
        outcome = "suppression"
    elif resistance_amplification:
        outcome = "amplification"
    else:
        outcome = "failure"

    notes = (
        f"Drug: {drug_name}, Dose: {dose_mg} mg q{dosing_interval_h}h x{n_doses} doses. "
        f"PK/PD type: {pkpd_type}. AUC/MIC={auc_mic_ratio:.1f}, Cmax/MIC={cmax_mic_ratio:.1f}, "
        f"fT>MIC={pct_time_above_mic:.1f}%. "
        f"Sensitive pop final: {final_ns:.2f} log10 CFU/mL. "
        f"Resistant pop final: {n_resistant_list[-1]:.2f} log10 CFU/mL. "
        f"Outcome: {outcome}."
    )

    return AntimicrobialPKPDResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mic_sensitive_mg_L=mic_sensitive_mg_L,
        mic_resistant_mg_L=mic_resistant_mg_L,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        n_sensitive=n_sensitive_list,
        n_resistant=n_resistant_list,
        n_total=n_total_list,
        auc_mic_ratio=auc_mic_ratio,
        cmax_mic_ratio=cmax_mic_ratio,
        pct_time_above_mic=pct_time_above_mic,
        eradication_probability=eradication_probability,
        resistance_amplification=resistance_amplification,
        outcome=outcome,
        notes=notes,
    )
