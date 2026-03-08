"""Phase 639 — Drug Thyroid Gland Accumulation.

Models drug accumulation in thyroid gland, relevant for radioiodine,
amiodarone, and thyroid toxicology studies.
"""

import math
from dataclasses import dataclass


@dataclass
class ThyroidPKResult:
    """Result of thyroid PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_thyroid_mg_g: list
    kp_thyroid: float
    cmax_plasma_mg_L: float
    cmax_thyroid_mg_g: float
    auc_plasma_mg_h_per_L: float
    auc_thyroid_mg_h_per_g: float
    thyroid_to_plasma_ratio: float
    thyroid_toxicity_index: float
    iodine_uptake_competition: bool
    t_half_thyroid_h: float
    notes: str


def _trapezoidal_auc(times, values):
    """Manual trapezoidal AUC."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def simulate_thyroid_pk(
    drug_name,
    dose_mg,
    logP,
    mw_Da,
    cl_sys_L_per_h,
    vd_sys_L,
    iodine_content_pct=0.0,
    thyroid_weight_g=20.0,
    t_end_h=48.0,
    dt_h=0.01,
):
    """Simulate drug accumulation in thyroid gland.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Oral dose in mg.
    logP : float
        Octanol-water partition coefficient (log scale).
    mw_Da : float
        Molecular weight in Daltons.
    cl_sys_L_per_h : float
        Systemic clearance in L/h.
    vd_sys_L : float
        Volume of distribution (systemic) in L.
    iodine_content_pct : float
        Iodine content as percentage of molecular weight (e.g., amiodarone ~37%).
    thyroid_weight_g : float
        Thyroid gland weight in grams (default 20 g).
    t_end_h : float
        Simulation end time in hours.
    dt_h : float
        Time step in hours.

    Returns
    -------
    ThyroidPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if thyroid_weight_g <= 0:
        raise ValueError("thyroid_weight_g must be > 0")
    if iodine_content_pct < 0:
        raise ValueError("iodine_content_pct must be >= 0")

    # Thyroid partition coefficient
    # Iodine-rich drugs (e.g., amiodarone) accumulate massively in thyroid
    raw_kp = (logP + 1.0) * 0.5 + iodine_content_pct * 0.5
    kp_thyroid = max(0.5, raw_kp / max(1.0, mw_Da / 300.0))
    kp_thyroid = max(0.2, min(50.0, kp_thyroid))

    # Thyroid volume of distribution (min 0.05 L to keep k_out numerically stable)
    vd_thyroid_L = max(0.05, thyroid_weight_g * kp_thyroid / 1000.0)

    # Thyroid blood flow: ~5 mL/min/g for 20g gland = 6 L/h
    qt_L_per_h = 6.0

    # Rate constants
    ka = 1.2  # absorption rate constant /h
    k_elim = cl_sys_L_per_h / vd_sys_L  # elimination rate constant /h
    k_in = qt_L_per_h / vd_sys_L  # plasma → thyroid /h
    k_out = qt_L_per_h / vd_thyroid_L  # thyroid → plasma /h

    # Half-life in thyroid
    t_half_thyroid_h = math.log(2.0) / k_out

    # Choose internal integration step to guarantee numerical stability.
    # Explicit Euler requires max_rate * dt_int <= 0.5.
    max_rate = max(ka, k_elim, k_in, k_out)
    dt_int = min(dt_h, 0.5 / max_rate)

    # Initial conditions
    # A_gut in mg, C_plasma in mg/L, C_thyroid in mg/L
    A_gut = dose_mg * 0.85
    C_plasma = 0.0
    C_thyroid = 0.0

    # Simulation — sub-step internally, record at output cadence dt_h
    n_steps = int(t_end_h / dt_h) + 1
    times_h = []
    c_plasma_list = []
    c_thyroid_list = []

    t_current = 0.0
    for step in range(n_steps):
        t_target = step * dt_h
        times_h.append(t_target)
        c_plasma_list.append(max(0.0, C_plasma))
        c_thyroid_list.append(max(0.0, C_thyroid))

        if step == n_steps - 1:
            break

        t_next = t_target + dt_h
        while t_current < t_next - 1e-12:
            h = min(dt_int, t_next - t_current)
            dA_gut = -ka * A_gut
            dC_plasma = (
                ka * A_gut / vd_sys_L
                - k_elim * C_plasma
                - k_in * C_plasma
                + k_out * C_thyroid * vd_thyroid_L / vd_sys_L
            )
            dC_thyroid = k_in * C_plasma * vd_sys_L / vd_thyroid_L - k_out * C_thyroid
            A_gut = max(0.0, A_gut + dA_gut * h)
            C_plasma = max(0.0, C_plasma + dC_plasma * h)
            C_thyroid = max(0.0, C_thyroid + dC_thyroid * h)
            t_current += h

    # Convert thyroid concentration to mg/g
    # c_thyroid_mg_g = C_thyroid [mg/L] * Vd_thyroid_L [L] / thyroid_weight_g [g]
    #                = C_thyroid * kp_thyroid / 1000.0
    c_thyroid_mg_g_list = [c * kp_thyroid / 1000.0 for c in c_thyroid_list]

    # Summary metrics
    cmax_plasma = max(c_plasma_list)
    cmax_thyroid = max(c_thyroid_mg_g_list)

    auc_plasma = _trapezoidal_auc(times_h, c_plasma_list)
    auc_thyroid = _trapezoidal_auc(times_h, c_thyroid_mg_g_list)

    thyroid_to_plasma_ratio = auc_thyroid / auc_plasma if auc_plasma > 0.0 else 0.0

    # Toxicity index: cmax_thyroid relative to 1 mg/g threshold
    thyroid_toxicity_index = cmax_thyroid / 1.0

    # Iodine competition: True if drug has significant iodine content (> 0.5%)
    iodine_uptake_competition = iodine_content_pct > 0.5

    notes = (
        f"kp_thyroid={kp_thyroid:.3f}; "
        f"Vd_thyroid={vd_thyroid_L * 1000:.2f} mL; "
        f"t_half_thyroid={t_half_thyroid_h:.2f} h; "
        f"Qt={qt_L_per_h:.1f} L/h"
    )

    return ThyroidPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_thyroid_mg_g=c_thyroid_mg_g_list,
        kp_thyroid=kp_thyroid,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_thyroid_mg_g=cmax_thyroid,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_thyroid_mg_h_per_g=auc_thyroid,
        thyroid_to_plasma_ratio=thyroid_to_plasma_ratio,
        thyroid_toxicity_index=thyroid_toxicity_index,
        iodine_uptake_competition=iodine_uptake_competition,
        t_half_thyroid_h=t_half_thyroid_h,
        notes=notes,
    )
