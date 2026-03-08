"""Phase 703 — Adipose Tissue Sequestration model."""

from dataclasses import dataclass
from math import log


@dataclass
class AdiposeSequestrationResult:
    drug_name: str
    dose_mg: float
    n_doses: int
    times_h: list
    c_plasma_mg_L: list
    c_adipose_mg_g: list
    kp_adipose: float
    cmax_plasma_mg_L: float
    cmax_adipose_mg_g: float
    auc_plasma_mg_h_per_L: float
    auc_adipose_mg_h_per_g: float
    accumulation_ratio: float
    washout_t50_h: float
    steady_state_adipose_mg_g: float
    notes: str


def simulate_adipose_sequestration(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    fat_mass_kg: float = 20.0,
    n_doses: int = 1,
    dosing_interval_h: float = 24.0,
    t_end_h: float = 168.0,
    dt_h: float = 0.1,
) -> AdiposeSequestrationResult:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be positive")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive")
    if fat_mass_kg <= 0:
        raise ValueError("fat_mass_kg must be positive")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")
    if dosing_interval_h <= 0:
        raise ValueError("dosing_interval_h must be positive")

    # Kp adipose
    kp = max(0.5, min(50.0, (logP * 3.0 + 1.0) / max(1.0, mw_Da / 300.0)))

    # Adipose blood flow
    q_fat = 0.05 * fat_mass_kg  # L/h

    # Volumes and rates
    vd_adipose = fat_mass_kg * kp  # L
    k_in_fat = q_fat / vd_sys_L
    k_out_fat = q_fat / vd_adipose
    washout_t50_h = log(2) / k_out_fat

    ka = 1.2  # absorption rate constant
    ke = cl_sys_L_per_h / vd_sys_L

    # State variables
    A_gut = 0.0
    C_plasma = 0.0
    C_adipose = 0.0

    # Determine substep count for stability
    n_substeps = max(1, int(0.1 / dt_h))
    sub_dt = dt_h / n_substeps

    # Collect results
    n_steps = int(t_end_h / dt_h) + 1
    times_h = []
    c_plasma_list = []
    c_adipose_list = []

    # Dose times
    dose_times = set()
    for i in range(n_doses):
        dose_times.add(round(i * dosing_interval_h / dt_h) * dt_h)

    # Track per-dose adipose peaks for accumulation ratio
    first_dose_peak_adipose = 0.0
    last_dose_peak_adipose = 0.0
    first_interval_end = dosing_interval_h if n_doses > 1 else t_end_h
    last_interval_start = (n_doses - 1) * dosing_interval_h

    for step in range(n_steps):
        t = step * dt_h

        # Add dose if at a dosing time
        t_rounded = round(t, 6)
        for dt_val in list(dose_times):
            if abs(t_rounded - dt_val) < dt_h * 0.5:
                A_gut += dose_mg * 0.85
                dose_times.discard(dt_val)

        # Record state
        times_h.append(t)
        c_plasma_list.append(C_plasma)
        c_adipose_mg_g = C_adipose * kp / 1000.0
        c_adipose_list.append(c_adipose_mg_g)

        # Track first dose adipose peak
        if t <= first_interval_end:
            if c_adipose_mg_g > first_dose_peak_adipose:
                first_dose_peak_adipose = c_adipose_mg_g

        # Track last dose adipose peak
        if t >= last_interval_start:
            if c_adipose_mg_g > last_dose_peak_adipose:
                last_dose_peak_adipose = c_adipose_mg_g

        # Forward Euler with substeps
        for _ in range(n_substeps):
            dA_gut = -ka * A_gut
            dC_plasma = (
                ka * A_gut / vd_sys_L - ke * C_plasma - k_in_fat * C_plasma + k_out_fat * C_adipose
            )
            dC_adipose = k_in_fat * C_plasma - k_out_fat * C_adipose

            A_gut += dA_gut * sub_dt
            C_plasma += dC_plasma * sub_dt
            C_adipose += dC_adipose * sub_dt

            if A_gut < 0:
                A_gut = 0.0
            if C_plasma < 0:
                C_plasma = 0.0
            if C_adipose < 0:
                C_adipose = 0.0

    # Cmax
    cmax_plasma = max(c_plasma_list)
    cmax_adipose = max(c_adipose_list)

    # AUC (trapezoidal)
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_adipose = sum(
        0.5 * (c_adipose_list[i] + c_adipose_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Accumulation ratio
    if n_doses > 1 and first_dose_peak_adipose > 0:
        accumulation_ratio = last_dose_peak_adipose / first_dose_peak_adipose
    else:
        accumulation_ratio = 1.0

    # Steady state adipose
    steady_state_adipose_mg_g = c_adipose_list[-1]

    return AdiposeSequestrationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        n_doses=n_doses,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_adipose_mg_g=c_adipose_list,
        kp_adipose=kp,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_adipose_mg_g=cmax_adipose,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_adipose_mg_h_per_g=auc_adipose,
        accumulation_ratio=accumulation_ratio,
        washout_t50_h=washout_t50_h,
        steady_state_adipose_mg_g=steady_state_adipose_mg_g,
        notes="Adipose tissue sequestration model with lipophilic drug accumulation and washout",
    )
