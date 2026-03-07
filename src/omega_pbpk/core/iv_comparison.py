"""Compare IV bolus vs IV infusion pharmacokinetics for the same total dose.

Provides analytical 1-compartment PK for both routes, safety/efficacy analysis,
and an optimizer for minimum infusion duration to stay below a toxicity threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class IVComparisonResult:
    drug_name: str
    total_dose_mg: float

    # Bolus PK
    cmax_bolus_mg_L: float
    auc_bolus_mg_h_per_L: float
    t_half_h: float
    c_1h_bolus_mg_L: float

    # Infusion PK (variable duration)
    infusion_duration_h: float
    cmax_infusion_mg_L: float
    auc_infusion_mg_h_per_L: float
    c_1h_infusion_mg_L: float
    css_infusion_mg_L: float

    # Comparison
    cmax_ratio: float
    auc_ratio: float

    # Safety/efficacy analysis
    bolus_toxicity_risk: str
    infusion_preferred: bool

    # Therapeutic window
    toxic_threshold_mg_L: float
    below_toxic_bolus: bool
    below_toxic_infusion: bool

    times_h: list[float]
    c_bolus_mg_L: list[float]
    c_infusion_mg_L: list[float]

    notes: str


def _simulate_iv(
    total_dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    infusion_duration_h: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list[float], list[float], list[float]]:
    """Simulate bolus and infusion concentrations using forward Euler.

    Returns (times, c_bolus, c_infusion) arrays.
    """
    ke = cl_L_per_h / vd_L
    rate_mg_per_h = total_dose_mg / infusion_duration_h

    n_steps = max(2, int(round(t_end_h / dt_h)) + 1)
    times: list[float] = []
    c_bolus: list[float] = []
    c_infusion: list[float] = []

    c_b = total_dose_mg / vd_L  # bolus starts at Cmax
    c_inf = 0.0  # infusion starts at 0

    for i in range(n_steps):
        t = i * dt_h
        times.append(t)
        c_bolus.append(max(0.0, c_b))
        c_infusion.append(max(0.0, c_inf))

        # Forward Euler step
        # Bolus: dC/dt = -ke * C
        c_b = c_b - ke * c_b * dt_h

        # Infusion: dC/dt = R/Vd - ke*C  (during), -ke*C (after)
        if t < infusion_duration_h:
            input_rate = rate_mg_per_h / vd_L  # mg/h / L = mg/(L*h)
        else:
            input_rate = 0.0
        c_inf = c_inf + (input_rate - ke * c_inf) * dt_h

        c_b = max(0.0, c_b)
        c_inf = max(0.0, c_inf)

    return times, c_bolus, c_infusion


def compare_iv_bolus_infusion(
    drug_name: str,
    total_dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    infusion_duration_h: float = 0.5,
    toxic_threshold_mg_L: float = 0.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> IVComparisonResult:
    """Compare IV bolus and infusion PK for the same total dose.

    Parameters
    ----------
    drug_name:
        Identifier for the compound.
    total_dose_mg:
        Total dose administered (mg).
    cl_L_per_h:
        Total clearance (L/h).
    vd_L:
        Volume of distribution (L).
    infusion_duration_h:
        Duration of IV infusion (h). Default 0.5 h (30 min).
    toxic_threshold_mg_L:
        Upper safety threshold (mg/L). 0 = not specified.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Euler integration step size (h).

    Returns
    -------
    IVComparisonResult
    """
    if total_dose_mg <= 0:
        raise ValueError(f"total_dose_mg must be > 0, got {total_dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if infusion_duration_h <= 0:
        raise ValueError(f"infusion_duration_h must be > 0, got {infusion_duration_h}")

    ke = cl_L_per_h / vd_L
    t_half_h = math.log(2.0) / ke

    times, c_bolus_list, c_infusion_list = _simulate_iv(
        total_dose_mg, cl_L_per_h, vd_L, infusion_duration_h, t_end_h, dt_h
    )

    # Analytical values (more precise than Euler)
    cmax_bolus = total_dose_mg / vd_L
    auc_bolus = total_dose_mg / cl_L_per_h

    # Analytical Cmax for infusion: at end of infusion
    rate = total_dose_mg / infusion_duration_h
    css_infusion = rate / cl_L_per_h  # true steady state
    cmax_infusion = css_infusion * (1.0 - math.exp(-ke * infusion_duration_h))
    auc_infusion = total_dose_mg / cl_L_per_h  # same total AUC

    # Concentration at 1h post start
    c_1h_bolus = cmax_bolus * math.exp(-ke * 1.0)

    if 1.0 <= infusion_duration_h:
        # Still infusing at 1h
        c_1h_infusion = css_infusion * (1.0 - math.exp(-ke * 1.0))
    else:
        # Post-infusion at 1h
        c_at_end = css_infusion * (1.0 - math.exp(-ke * infusion_duration_h))
        c_1h_infusion = c_at_end * math.exp(-ke * (1.0 - infusion_duration_h))

    cmax_ratio = cmax_bolus / cmax_infusion if cmax_infusion > 0 else float("inf")
    auc_ratio = auc_bolus / auc_infusion if auc_infusion > 0 else 1.0

    # Toxicity risk assessment
    if toxic_threshold_mg_L > 0:
        ratio_to_threshold = cmax_bolus / toxic_threshold_mg_L
        if ratio_to_threshold < 1.0:
            bolus_toxicity_risk = "low"
        elif ratio_to_threshold <= 2.0:
            bolus_toxicity_risk = "moderate"
        else:
            bolus_toxicity_risk = "high"
    else:
        # Use Cmax_bolus / Css_infusion ratio as proxy
        ratio = cmax_bolus / css_infusion if css_infusion > 0 else 1.0
        if ratio < 2.0:
            bolus_toxicity_risk = "low"
        elif ratio < 5.0:
            bolus_toxicity_risk = "moderate"
        else:
            bolus_toxicity_risk = "high"

    infusion_preferred = bolus_toxicity_risk in {"moderate", "high"}

    below_toxic_bolus = True
    below_toxic_infusion = True
    if toxic_threshold_mg_L > 0:
        below_toxic_bolus = cmax_bolus <= toxic_threshold_mg_L
        below_toxic_infusion = cmax_infusion <= toxic_threshold_mg_L

    notes_parts = []
    notes_parts.append(
        f"t1/2={t_half_h:.2f}h; Bolus Cmax={cmax_bolus:.3f} mg/L; "
        f"Infusion Cmax={cmax_infusion:.3f} mg/L ({infusion_duration_h}h infusion)."
    )
    if infusion_preferred:
        notes_parts.append("Infusion preferred to reduce peak concentration risk.")

    return IVComparisonResult(
        drug_name=drug_name,
        total_dose_mg=total_dose_mg,
        cmax_bolus_mg_L=cmax_bolus,
        auc_bolus_mg_h_per_L=auc_bolus,
        t_half_h=t_half_h,
        c_1h_bolus_mg_L=c_1h_bolus,
        infusion_duration_h=infusion_duration_h,
        cmax_infusion_mg_L=cmax_infusion,
        auc_infusion_mg_h_per_L=auc_infusion,
        c_1h_infusion_mg_L=c_1h_infusion,
        css_infusion_mg_L=css_infusion,
        cmax_ratio=cmax_ratio,
        auc_ratio=auc_ratio,
        bolus_toxicity_risk=bolus_toxicity_risk,
        infusion_preferred=infusion_preferred,
        toxic_threshold_mg_L=toxic_threshold_mg_L,
        below_toxic_bolus=below_toxic_bolus,
        below_toxic_infusion=below_toxic_infusion,
        times_h=times,
        c_bolus_mg_L=c_bolus_list,
        c_infusion_mg_L=c_infusion_list,
        notes=" ".join(notes_parts),
    )


def optimize_infusion_duration(
    drug_name: str,
    total_dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    toxic_threshold_mg_L: float,
    target_cmax_mg_L: float = None,
) -> dict:
    """Find minimum infusion duration to keep Cmax below toxic threshold.

    Uses analytical inversion of the infusion Cmax formula:
        Cmax_inf = (R/CL) * (1 - exp(-ke * t_inf))
        where R = dose / t_inf

    Solving for t_inf given a target Cmax uses binary search because the
    relationship is not trivially invertible analytically.

    Parameters
    ----------
    drug_name:
        Compound identifier.
    total_dose_mg:
        Total dose (mg).
    cl_L_per_h:
        Clearance (L/h).
    vd_L:
        Volume of distribution (L).
    toxic_threshold_mg_L:
        Maximum acceptable Cmax (mg/L).
    target_cmax_mg_L:
        If provided, override toxic_threshold for target.

    Returns
    -------
    dict with keys: "optimal_duration_h", "achievable_cmax_mg_L", "recommendation"
    """
    if total_dose_mg <= 0:
        raise ValueError(f"total_dose_mg must be > 0, got {total_dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")

    cmax_target = target_cmax_mg_L if target_cmax_mg_L is not None else toxic_threshold_mg_L
    if cmax_target <= 0:
        raise ValueError("toxic_threshold_mg_L or target_cmax_mg_L must be > 0")

    ke = cl_L_per_h / vd_L

    def cmax_for_duration(t_inf: float) -> float:
        rate = total_dose_mg / t_inf
        css = rate / cl_L_per_h
        return css * (1.0 - math.exp(-ke * t_inf))

    # Bolus Cmax (upper bound)
    bolus_cmax = total_dose_mg / vd_L

    # If bolus is already below target, no infusion needed
    if bolus_cmax <= cmax_target:
        return {
            "optimal_duration_h": 0.0,
            "achievable_cmax_mg_L": bolus_cmax,
            "recommendation": (
                f"Bolus Cmax ({bolus_cmax:.3f} mg/L) is already below target "
                f"({cmax_target:.3f} mg/L). No extended infusion required."
            ),
        }

    # Binary search for minimum t_inf such that cmax_for_duration(t_inf) <= target
    # Increasing t_inf generally decreases Cmax
    lo = 1e-6
    hi = 1000.0  # 1000 hours max

    # Verify that a long enough infusion can achieve target
    if cmax_for_duration(hi) > cmax_target:
        return {
            "optimal_duration_h": hi,
            "achievable_cmax_mg_L": cmax_for_duration(hi),
            "recommendation": (
                f"Cannot achieve Cmax <= {cmax_target:.3f} mg/L even with very "
                f"long infusion. Consider dose reduction."
            ),
        }

    for _ in range(100):  # binary search iterations
        mid = 0.5 * (lo + hi)
        if cmax_for_duration(mid) > cmax_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break

    optimal_duration = hi
    achievable_cmax = cmax_for_duration(optimal_duration)

    return {
        "optimal_duration_h": round(optimal_duration, 4),
        "achievable_cmax_mg_L": round(achievable_cmax, 6),
        "recommendation": (
            f"Infuse {drug_name} over >= {optimal_duration:.2f}h to keep "
            f"Cmax <= {cmax_target:.3f} mg/L (predicted: {achievable_cmax:.3f} mg/L)."
        ),
    }
