"""
Phase 957 — Antibiotic PK/PD dosing optimization.

Links PK to pharmacodynamic targets:
- fT>MIC (time-dependent, beta-lactams)
- AUC24/MIC (concentration-dependent, fluoroquinolones)
- Cmax/MIC (concentration-dependent, aminoglycosides)

Pure Python, stdlib only. Forward Euler ODE.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "AntibioticDosingResult",
    "optimize_antibiotic_dosing",
    "screen_dose_mic_combinations",
]

_VALID_CLASSES = {"beta_lactam", "fluoroquinolone", "aminoglycoside", "generic"}

_PKPD_INDEX = {
    "beta_lactam": "fT>MIC",
    "fluoroquinolone": "AUC/MIC (AUIC)",
    "aminoglycoside": "Cmax/MIC",
    "generic": "AUC/MIC (AUIC)",
}


@dataclass
class AntibioticDosingResult:
    drug_name: str
    antibiotic_class: str
    dose_mg: float
    dosing_interval_h: float
    mic_mg_L: float
    times_h: list
    c_total_mg_L: list
    cmax_free_mg_L: float
    tmax_h: float
    auc_free_24h_mg_h_per_L: float
    ft_above_mic_pct: float
    auc_mic_ratio: float
    cmax_mic_ratio: float
    target_attained: bool
    pkpd_index: str
    notes: str


def _validate_params(
    dose_mg: float,
    mic_mg_L: float,
    cl_L_per_h: float,
    vd_L: float,
    fu: float,
    antibiotic_class: str,
) -> None:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mic_mg_L <= 0:
        raise ValueError("mic_mg_L must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0 < fu <= 1):
        raise ValueError("fu must be in (0, 1]")
    if antibiotic_class not in _VALID_CLASSES:
        raise ValueError(
            f"antibiotic_class must be one of {_VALID_CLASSES}, got '{antibiotic_class}'"
        )


def _simulate_multidose(
    dose_mg: float,
    dosing_interval_h: float,
    n_doses: int,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list, list]:
    """Forward Euler multi-dose 1-cpt simulation.

    Returns (times_h, c_total_mg_L) for the full simulation period.
    """
    ke = cl_L_per_h / vd_L

    n_steps = int(round(t_end_h / dt_h)) + 1
    times_h = [i * dt_h for i in range(n_steps)]

    # Dose times
    dose_times = [i * dosing_interval_h for i in range(n_doses)]
    dose_set = set(dose_times)

    if ka_per_h == 0.0:
        # IV bolus: single compartment
        c = [0.0] * n_steps
        for step in range(n_steps):
            t = times_h[step]
            # Apply bolus doses at dose times
            if step > 0:
                c[step] = c[step - 1]
                # Check if a dose occurs in (t-dt, t]
                for dt_time in dose_times:
                    prev_t = times_h[step - 1]
                    if prev_t < dt_time <= t:
                        c[step] += dose_mg / vd_L
                # Forward Euler elimination
                c[step] -= ke * c[step] * dt_h
                c[step] = max(c[step], 0.0)
            else:
                # step 0: check if dose at t=0
                if 0.0 in dose_set:
                    c[0] = dose_mg / vd_L
        # Fix step 0 elimination
        # Already handled above — step 0 just gets the bolus
    else:
        # Oral absorption: gut + central compartment
        c = [0.0] * n_steps
        gut = [0.0] * n_steps
        for step in range(1, n_steps):
            t = times_h[step]
            prev_t = times_h[step - 1]
            gut_cur = gut[step - 1]
            c_cur = c[step - 1]
            # Add oral dose to gut at dose times
            for dt_time in dose_times:
                if prev_t < dt_time <= t:
                    gut_cur += dose_mg
            # Forward Euler
            d_gut = -ka_per_h * gut_cur * dt_h
            d_c = (ka_per_h * gut_cur / vd_L - ke * c_cur) * dt_h
            gut[step] = max(gut_cur + d_gut, 0.0)
            c[step] = max(c_cur + d_c, 0.0)

    return times_h, c


def optimize_antibiotic_dosing(
    drug_name: str,
    antibiotic_class: str = "beta_lactam",
    dose_mg: float = 500.0,
    dosing_interval_h: float = 8.0,
    mic_mg_L: float = 1.0,
    n_doses: int = 3,
    cl_L_per_h: float = 10.0,
    vd_L: float = 20.0,
    ka_per_h: float = 0.0,
    fu: float = 0.7,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> AntibioticDosingResult:
    """Simulate antibiotic PK and compute PK/PD indices."""
    _validate_params(dose_mg, mic_mg_L, cl_L_per_h, vd_L, fu, antibiotic_class)

    times_h, c_total = _simulate_multidose(
        dose_mg=dose_mg,
        dosing_interval_h=dosing_interval_h,
        n_doses=n_doses,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        ka_per_h=ka_per_h,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    # Free concentrations
    c_free = [c * fu for c in c_total]

    # Cmax_free and tmax
    cmax_free = max(c_free) if c_free else 0.0
    tmax_h = times_h[c_free.index(cmax_free)] if c_free else 0.0

    # AUC_free over 24h using trapezoidal rule
    n_24 = int(round(24.0 / dt_h))
    n_24 = min(n_24, len(times_h) - 1)
    auc_free_24h = 0.0
    for i in range(n_24):
        auc_free_24h += 0.5 * (c_free[i] + c_free[i + 1]) * (times_h[i + 1] - times_h[i])

    # fT>MIC: fraction of 24h with C_free > MIC
    n_above = sum(1 for cf in c_free[: n_24 + 1] if cf > mic_mg_L)
    ft_above_mic_pct = 100.0 * n_above / (n_24 + 1) if (n_24 + 1) > 0 else 0.0

    # PK/PD ratios
    auc_mic_ratio = auc_free_24h / mic_mg_L
    cmax_mic_ratio = cmax_free / mic_mg_L

    # Target attainment based on class
    pkpd_index = _PKPD_INDEX[antibiotic_class]
    if antibiotic_class == "beta_lactam":
        target_attained = ft_above_mic_pct > 50.0
        target_desc = "fT>MIC > 50% (bacteriostatic)"
    elif antibiotic_class == "fluoroquinolone":
        target_attained = auc_mic_ratio > 25.0
        target_desc = "AUC/MIC > 25 (Gram-positive target)"
    elif antibiotic_class == "aminoglycoside":
        target_attained = cmax_mic_ratio > 10.0
        target_desc = "Cmax/MIC > 10"
    else:  # generic
        target_attained = auc_mic_ratio > 25.0
        target_desc = "AUC/MIC > 25 (generic target)"

    attained_str = "ATTAINED" if target_attained else "NOT attained"
    notes = (
        f"Class: {antibiotic_class}. PK/PD index: {pkpd_index}. "
        f"Target ({target_desc}): {attained_str}. "
        f"fT>MIC={ft_above_mic_pct:.1f}%, AUC/MIC={auc_mic_ratio:.1f}, Cmax/MIC={cmax_mic_ratio:.1f}."
    )

    return AntibioticDosingResult(
        drug_name=drug_name,
        antibiotic_class=antibiotic_class,
        dose_mg=dose_mg,
        dosing_interval_h=dosing_interval_h,
        mic_mg_L=mic_mg_L,
        times_h=times_h,
        c_total_mg_L=c_total,
        cmax_free_mg_L=cmax_free,
        tmax_h=tmax_h,
        auc_free_24h_mg_h_per_L=auc_free_24h,
        ft_above_mic_pct=ft_above_mic_pct,
        auc_mic_ratio=auc_mic_ratio,
        cmax_mic_ratio=cmax_mic_ratio,
        target_attained=target_attained,
        pkpd_index=pkpd_index,
        notes=notes,
    )


def screen_dose_mic_combinations(
    drug_name: str,
    antibiotic_class: str,
    doses_mg: list,
    mics_mg_L: list,
    dosing_interval_h: float = 8.0,
    n_doses: int = 3,
    cl_L_per_h: float = 10.0,
    vd_L: float = 20.0,
    ka_per_h: float = 0.0,
    fu: float = 0.7,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> list:
    """Screen all combinations of doses and MIC values.

    Returns list of AntibioticDosingResult sorted by:
    - target_attained = True first
    - then auc_mic_ratio descending
    """
    results = []
    for dose in doses_mg:
        for mic in mics_mg_L:
            result = optimize_antibiotic_dosing(
                drug_name=drug_name,
                antibiotic_class=antibiotic_class,
                dose_mg=dose,
                dosing_interval_h=dosing_interval_h,
                mic_mg_L=mic,
                n_doses=n_doses,
                cl_L_per_h=cl_L_per_h,
                vd_L=vd_L,
                ka_per_h=ka_per_h,
                fu=fu,
                t_end_h=t_end_h,
                dt_h=dt_h,
            )
            results.append(result)

    # Sort: target_attained=True first, then auc_mic_ratio descending
    results.sort(key=lambda r: (not r.target_attained, -r.auc_mic_ratio))
    return results
