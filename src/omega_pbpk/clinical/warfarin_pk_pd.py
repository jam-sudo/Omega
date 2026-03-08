"""
Phase 965 — Warfarin PK/PD model.

Stereoselective metabolism, INR response (indirect response model),
narrow therapeutic index.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["WarfarinPKPDResult", "simulate_warfarin_pkpd", "compare_cyp2c9_phenotypes"]


@dataclass
class WarfarinPKPDResult:
    """Result of a warfarin PK/PD simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_s_warfarin_mg_L: list
    c_r_warfarin_mg_L: list
    inr: list
    cmax_s_mg_L: float
    cmax_r_mg_L: float
    inr_peak: float
    time_to_inr_peak_h: float
    time_in_therapeutic_range_h: float
    time_above_inr_3: float
    notes: str


def simulate_warfarin_pkpd(
    dose_mg: float = 5.0,
    cl_S_L_per_h: float = 0.18,
    cl_R_L_per_h: float = 0.12,
    vd_L: float = 10.0,
    ka_per_h: float = 0.8,
    t_end_h: float = 168.0,
    dt_h: float = 0.5,
    n_doses: int = 7,
    dosing_interval_h: float = 24.0,
    drug_name: str = "warfarin",
) -> WarfarinPKPDResult:
    """Simulate warfarin PK/PD with stereoselective metabolism and INR response."""
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_S_L_per_h <= 0:
        raise ValueError("cl_S_L_per_h must be > 0")
    if cl_R_L_per_h <= 0:
        raise ValueError("cl_R_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")

    # Bioavailability
    F = 0.93

    # Dose absorbed for each enantiomer (racemic: 50/50 split)
    dose_S_absorbed = 0.5 * dose_mg * F
    dose_R_absorbed = 0.5 * dose_mg * F

    # Elimination rate constants
    ke_S = cl_S_L_per_h / vd_L
    ke_R = cl_R_L_per_h / vd_L

    # PD parameters
    Imax = 1.0
    IC50 = 0.5  # mg/L
    kout = 0.02  # /h
    CF0 = 100.0
    kin = kout * CF0

    # Potency weights
    w_S = 0.75
    w_R = 0.25

    # Dosing times
    dose_times = [i * dosing_interval_h for i in range(n_doses)]

    # State: gut_S, gut_R, C_S, C_R, CF
    # Initialize
    gut_S = 0.0
    gut_R = 0.0
    C_S = 0.0
    C_R = 0.0
    CF = CF0

    n_steps = int(t_end_h / dt_h) + 1
    times_h = []
    c_s_list = []
    c_r_list = []
    inr_list = []

    t = 0.0
    for step in range(n_steps):
        t = step * dt_h

        # Apply doses at dose times (within half a dt_h)
        for dose_t in dose_times:
            if abs(t - dose_t) < dt_h * 0.5:
                gut_S += dose_S_absorbed
                gut_R += dose_R_absorbed

        # Record state
        times_h.append(t)
        c_s_list.append(C_S)
        c_r_list.append(C_R)
        inr_val = (CF0 / CF) if CF > 0 else float("inf")
        inr_list.append(inr_val)

        if step >= n_steps - 1:
            break

        # Forward Euler update
        # PK
        dGut_S = -ka_per_h * gut_S
        dGut_R = -ka_per_h * gut_R
        dC_S = (ka_per_h * gut_S) / vd_L - ke_S * C_S
        dC_R = (ka_per_h * gut_R) / vd_L - ke_R * C_R

        # PD
        Ce = w_S * C_S + w_R * C_R
        inhibition = Imax * Ce / (IC50 + Ce)
        dCF = kin * (1.0 - inhibition) - kout * CF

        gut_S += dGut_S * dt_h
        gut_R += dGut_R * dt_h
        C_S += dC_S * dt_h
        C_R += dC_R * dt_h
        CF += dCF * dt_h

        # Keep non-negative
        if gut_S < 0:
            gut_S = 0.0
        if gut_R < 0:
            gut_R = 0.0
        if C_S < 0:
            C_S = 0.0
        if C_R < 0:
            C_R = 0.0
        if CF < 0.01:
            CF = 0.01

    # Compute summary metrics
    cmax_s = max(c_s_list)
    cmax_r = max(c_r_list)
    inr_peak = max(inr_list)
    time_to_inr_peak = times_h[inr_list.index(inr_peak)]

    # Time in therapeutic range (INR 2-3) using trapezoidal
    time_in_therapeutic = 0.0
    time_above_3 = 0.0
    for i in range(len(times_h) - 1):
        dt = times_h[i + 1] - times_h[i]
        inr_avg = (inr_list[i] + inr_list[i + 1]) / 2.0
        if 2.0 <= inr_avg <= 3.0:
            time_in_therapeutic += dt
        if inr_avg > 3.0:
            time_above_3 += dt

    notes = (
        f"CYP2C9 CL_S={cl_S_L_per_h:.3f} L/h, CL_R={cl_R_L_per_h:.3f} L/h. "
        f"INR peak={inr_peak:.2f} at t={time_to_inr_peak:.1f}h. "
        f"Therapeutic range (INR 2-3): {time_in_therapeutic:.1f}h. "
        f"Supratherapeutic (INR>3): {time_above_3:.1f}h."
    )

    return WarfarinPKPDResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_s_warfarin_mg_L=c_s_list,
        c_r_warfarin_mg_L=c_r_list,
        inr=inr_list,
        cmax_s_mg_L=cmax_s,
        cmax_r_mg_L=cmax_r,
        inr_peak=inr_peak,
        time_to_inr_peak_h=time_to_inr_peak,
        time_in_therapeutic_range_h=time_in_therapeutic,
        time_above_inr_3=time_above_3,
        notes=notes,
    )


def compare_cyp2c9_phenotypes(
    dose_mg: float = 5.0,
    cl_R_L_per_h: float = 0.12,
    **kwargs,
) -> list:
    """Compare warfarin PK/PD across CYP2C9 phenotypes.

    Returns list of WarfarinPKPDResult for [normal, intermediate, poor] metabolizer,
    sorted by time_in_therapeutic_range_h descending.
    """
    phenotypes = [
        ("normal", 0.18),
        ("intermediate_CYP2C9*1/*3", 0.09),
        ("poor_CYP2C9*3/*3", 0.04),
    ]

    results = []
    for phenotype_name, cl_S in phenotypes:
        result = simulate_warfarin_pkpd(
            dose_mg=dose_mg,
            cl_S_L_per_h=cl_S,
            cl_R_L_per_h=cl_R_L_per_h,
            drug_name=f"warfarin_{phenotype_name}",
            **kwargs,
        )
        results.append(result)

    results.sort(key=lambda r: r.time_in_therapeutic_range_h, reverse=True)
    return results
