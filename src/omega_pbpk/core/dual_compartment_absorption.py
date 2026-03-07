"""
Phase 579 — Dual-compartment absorption model with fast and slow absorbing fractions.
"""



def simulate_dual_absorption(
    drug_name: str,
    dose_mg: float,
    f_fast: float,
    ka_fast_per_h: float,
    ka_slow_per_h: float,
    cl_L_per_h: float,
    vd_L: float,
    f_oral: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> dict:
    """Simulate dual-compartment absorption (fast + slow fractions).

    Two gut compartments:
        dA_fast/dt = -ka_fast * A_fast
        dA_slow/dt = -ka_slow * A_slow
        dCp/dt     = (ka_fast * A_fast + ka_slow * A_slow) / Vd - ke * Cp

    Returns dict with: drug_name, times_h, c_plasma_mg_L, cmax, tmax_h, auc, t_half_h, notes
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if not (0 < f_fast < 1):
        raise ValueError("f_fast must be strictly between 0 and 1")
    if ka_fast_per_h <= 0:
        raise ValueError("ka_fast_per_h must be positive")
    if ka_slow_per_h <= 0:
        raise ValueError("ka_slow_per_h must be positive")
    if ka_fast_per_h <= ka_slow_per_h:
        raise ValueError("ka_fast_per_h must be greater than ka_slow_per_h")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be positive")
    if vd_L <= 0:
        raise ValueError("vd_L must be positive")

    ke = cl_L_per_h / vd_L

    # Initial conditions (amounts in mg)
    A_fast = f_oral * dose_mg * f_fast
    A_slow = f_oral * dose_mg * (1.0 - f_fast)
    Cp = 0.0

    times_h = []
    c_plasma = []

    t = 0.0
    steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(steps):
        times_h.append(t)
        c_plasma.append(Cp)

        # Forward Euler
        dA_fast = -ka_fast_per_h * A_fast
        dA_slow = -ka_slow_per_h * A_slow
        dCp = (ka_fast_per_h * A_fast + ka_slow_per_h * A_slow) / vd_L - ke * Cp

        A_fast += dA_fast * dt_h
        A_slow += dA_slow * dt_h
        Cp += dCp * dt_h

        # Clamp negatives
        A_fast = max(A_fast, 0.0)
        A_slow = max(A_slow, 0.0)
        Cp = max(Cp, 0.0)

        t += dt_h

    # Compute metrics
    cmax = max(c_plasma)
    tmax_h = times_h[c_plasma.index(cmax)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma[i - 1] + c_plasma[i]) * (times_h[i] - times_h[i - 1])

    import math

    t_half_h = math.log(2.0) / ke

    notes = (
        f"Dual absorption: f_fast={f_fast:.2f}, "
        f"ka_fast={ka_fast_per_h:.2f}/h, ka_slow={ka_slow_per_h:.2f}/h"
    )

    return {
        "drug_name": drug_name,
        "times_h": times_h,
        "c_plasma_mg_L": c_plasma,
        "cmax": cmax,
        "tmax_h": tmax_h,
        "auc": auc,
        "t_half_h": t_half_h,
        "notes": notes,
    }


def compare_absorption_profiles(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    f_fast_values: list,
) -> list:
    """Simulate dual absorption for each f_fast value.

    Fixed ka_fast=2.0 h^-1, ka_slow=0.2 h^-1.
    Returns list of dicts: f_fast, cmax, tmax_h, auc.
    """
    ka_fast = 2.0
    ka_slow = 0.2
    results = []
    for f_fast in f_fast_values:
        res = simulate_dual_absorption(
            drug_name="compound",
            dose_mg=dose_mg,
            f_fast=f_fast,
            ka_fast_per_h=ka_fast,
            ka_slow_per_h=ka_slow,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
        )
        results.append(
            {
                "f_fast": f_fast,
                "cmax": res["cmax"],
                "tmax_h": res["tmax_h"],
                "auc": res["auc"],
            }
        )
    return results


def biphasic_absorption_signature(times_h: list, c_plasma: list) -> dict:
    """Detect biphasic absorption pattern in a concentration-time profile.

    A biphasic pattern is identified by finding local peaks in the profile.
    Returns: is_biphasic (bool), n_peaks, peak_times, peak_concentrations.
    """
    if len(times_h) != len(c_plasma):
        raise ValueError("times_h and c_plasma must have equal length")
    if len(c_plasma) < 3:
        return {
            "is_biphasic": False,
            "n_peaks": 0,
            "peak_times": [],
            "peak_concentrations": [],
        }

    peak_times = []
    peak_concentrations = []

    for i in range(1, len(c_plasma) - 1):
        if c_plasma[i] > c_plasma[i - 1] and c_plasma[i] > c_plasma[i + 1]:
            peak_times.append(times_h[i])
            peak_concentrations.append(c_plasma[i])

    n_peaks = len(peak_times)
    is_biphasic = n_peaks >= 2

    return {
        "is_biphasic": is_biphasic,
        "n_peaks": n_peaks,
        "peak_times": peak_times,
        "peak_concentrations": peak_concentrations,
    }
