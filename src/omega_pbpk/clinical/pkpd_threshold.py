"""PK/PD threshold analysis: determine plasma concentration needed for target effect."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["PKPDThresholdResult", "calculate_pkpd_threshold", "time_above_threshold"]

_VALID_PD_MODELS = ("emax", "linear", "log_linear")


@dataclass(frozen=True)
class PKPDThresholdResult:
    drug_name: str
    pd_model: str  # "emax", "linear", "log_linear"
    ec50_mg_L: float
    emax: float
    target_effect_pct: float  # desired % of Emax
    threshold_conc_mg_L: float  # C needed for target effect
    min_dose_mg: float  # minimum dose to achieve threshold (1-cpt IV)
    cl_L_per_h: float
    vd_L: float
    t_above_threshold_h: float  # time above threshold for given dose
    dose_mg: float
    notes: str


def _compute_threshold(
    pd_model: str,
    ec50_mg_L: float,
    emax: float,
    target_effect_pct: float,
    hill_n: float,
) -> float:
    """Compute threshold concentration for a given PD model and target effect."""
    target_frac = target_effect_pct / 100.0

    if pd_model == "emax":
        # E = Emax * C^n / (EC50^n + C^n)
        # Solve: target_frac = C^n / (EC50^n + C^n)
        # C^n * (1 - target_frac) = EC50^n * target_frac
        # C = EC50 * (target_frac / (1 - target_frac))^(1/n)
        if target_frac >= 1.0:
            return float("inf")
        ratio = target_frac / (1.0 - target_frac)
        return ec50_mg_L * (ratio ** (1.0 / hill_n))

    elif pd_model == "linear":
        # E = emax * C, scale by EC50: threshold = EC50 * target_pct/100
        return ec50_mg_L * target_frac

    elif pd_model == "log_linear":
        # E = emax * log(1 + C/EC50)
        # target_frac (as fraction of emax, so units: emax * target_frac)
        # emax * log(1 + C/EC50) = emax * target_frac
        # log(1 + C/EC50) = target_frac
        # C = EC50 * (exp(target_frac) - 1)
        return ec50_mg_L * (math.exp(target_frac) - 1.0)

    else:
        raise ValueError(f"pd_model must be one of {_VALID_PD_MODELS}, got {pd_model!r}")


def time_above_threshold(
    times_h: list[float],
    concentrations_mg_L: list[float],
    threshold_mg_L: float,
) -> float:
    """Return hours above threshold concentration using trapezoidal approximation."""
    if len(times_h) != len(concentrations_mg_L):
        raise ValueError("times_h and concentrations_mg_L must have the same length")

    total_time = 0.0
    for i in range(1, len(times_h)):
        dt = times_h[i] - times_h[i - 1]
        c_prev = concentrations_mg_L[i - 1]
        c_curr = concentrations_mg_L[i]
        # Both above threshold: full interval
        if c_prev >= threshold_mg_L and c_curr >= threshold_mg_L:
            total_time += dt
        # Crossing from above to below
        elif c_prev >= threshold_mg_L and c_curr < threshold_mg_L:
            if c_prev > c_curr:
                frac = (c_prev - threshold_mg_L) / (c_prev - c_curr)
                total_time += dt * frac
        # Crossing from below to above
        elif c_prev < threshold_mg_L and c_curr >= threshold_mg_L:
            if c_curr > c_prev:
                frac = (c_curr - threshold_mg_L) / (c_curr - c_prev)
                total_time += dt * frac

    return total_time


def calculate_pkpd_threshold(
    drug_name: str,
    pd_model: str,
    ec50_mg_L: float,
    emax: float,
    target_effect_pct: float,
    cl_L_per_h: float,
    vd_L: float,
    dose_mg: float = 100.0,
    hill_n: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PKPDThresholdResult:
    """Calculate PK/PD threshold concentration and time above threshold.

    Uses 1-compartment IV model: C(t) = (dose/Vd) * exp(-ke * t)
    """
    # --- Validation ---
    if pd_model not in _VALID_PD_MODELS:
        raise ValueError(f"pd_model must be one of {_VALID_PD_MODELS}, got {pd_model!r}")
    if ec50_mg_L <= 0:
        raise ValueError(f"ec50_mg_L must be positive, got {ec50_mg_L}")
    if emax <= 0:
        raise ValueError(f"emax must be positive, got {emax}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be positive, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be positive, got {vd_L}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if not (0.0 < target_effect_pct < 100.0):
        raise ValueError(f"target_effect_pct must be in (0, 100), got {target_effect_pct}")
    if hill_n <= 0:
        raise ValueError(f"hill_n must be positive, got {hill_n}")

    # --- Threshold concentration ---
    threshold_conc = _compute_threshold(pd_model, ec50_mg_L, emax, target_effect_pct, hill_n)

    # --- 1-cpt IV simulation ---
    ke = cl_L_per_h / vd_L
    n_steps = int(math.ceil(t_end_h / dt_h))
    times_h = [i * dt_h for i in range(n_steps + 1)]
    c0 = dose_mg / vd_L
    concentrations = [c0 * math.exp(-ke * t) for t in times_h]

    # --- Time above threshold ---
    tat = time_above_threshold(times_h, concentrations, threshold_conc)

    # --- Minimum dose: dose/Vd = threshold → min_dose = threshold * Vd
    min_dose = threshold_conc * vd_L

    notes = (
        f"PK/PD threshold analysis: pd_model={pd_model}, "
        f"EC50={ec50_mg_L:.3f} mg/L, target={target_effect_pct:.1f}%, "
        f"threshold={threshold_conc:.4f} mg/L, min_dose={min_dose:.2f} mg"
    )

    return PKPDThresholdResult(
        drug_name=drug_name,
        pd_model=pd_model,
        ec50_mg_L=ec50_mg_L,
        emax=emax,
        target_effect_pct=target_effect_pct,
        threshold_conc_mg_L=threshold_conc,
        min_dose_mg=min_dose,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        t_above_threshold_h=tat,
        dose_mg=dose_mg,
        notes=notes,
    )
