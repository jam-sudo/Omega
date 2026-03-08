"""Phase 367 — Cmax/AUC Ratio Calculator (Exposure Distribution Metric).

Computes exposure distribution metrics (Cmax/AUC ratio, fluctuation,
accumulation) for interpreting PK profiles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ExposureMetricsResult:
    """Result of exposure distribution metric calculation."""

    drug_name: str
    cmax_mg_L: float
    cmin_mg_L: float
    auc_mg_h_per_L: float
    cmax_auc_ratio_per_h: float
    fluctuation_pct: float
    peak_trough_ratio: float
    css_avg_mg_L: float
    accumulation_ratio: float
    auc_predicted_mg_h_per_L: float
    recovery_pct: float
    fluctuation_class: str
    pk_interpretation: str
    notes: str


def _classify_fluctuation(fluctuation_pct: float) -> str:
    """Classify fluctuation percentage."""
    if fluctuation_pct < 50.0:
        return "low"
    if fluctuation_pct <= 100.0:
        return "moderate"
    return "high"


def _interpret_pk(
    fluctuation_pct: float,
    peak_trough_ratio: float,
    cmax_auc_ratio: float,
) -> str:
    """Derive primary PK interpretation flag."""
    if fluctuation_pct > 100.0:
        return "high_fluctuation"
    if peak_trough_ratio > 10.0:
        return "high_peak_trough"
    if cmax_auc_ratio > 0.5:
        return "rapid_distribution"
    return "normal"


def compute_exposure_metrics(
    drug_name: str,
    cmax_mg_L: float,
    cmin_mg_L: float,
    auc_mg_h_per_L: float,
    tau_h: float,
    t_half_h: float,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    n_doses_simulated: int = 1,
) -> ExposureMetricsResult:
    """Compute exposure distribution metrics for a PK profile.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    cmax_mg_L:
        Maximum observed concentration (mg/L).
    cmin_mg_L:
        Trough (minimum) concentration (mg/L); may be 0.
    auc_mg_h_per_L:
        Area under the curve over one dosing interval (mg·h/L).
    tau_h:
        Dosing interval in hours.
    t_half_h:
        Elimination half-life in hours.
    dose_mg:
        Administered dose in mg.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    n_doses_simulated:
        Number of doses used in the simulation (for accumulation ratio).

    Returns
    -------
    ExposureMetricsResult
    """
    # --- Validation ---
    if cmax_mg_L <= 0:
        raise ValueError("cmax_mg_L must be > 0")
    if auc_mg_h_per_L <= 0:
        raise ValueError("auc_mg_h_per_L must be > 0")
    if tau_h <= 0:
        raise ValueError("tau_h must be > 0")
    if t_half_h <= 0:
        raise ValueError("t_half_h must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    # --- Core metrics ---
    cmax_auc_ratio = cmax_mg_L / auc_mg_h_per_L  # h^-1

    fluctuation_pct = (cmax_mg_L - cmin_mg_L) / auc_mg_h_per_L * tau_h * 100.0

    if cmin_mg_L > 0:
        peak_trough_ratio = cmax_mg_L / cmin_mg_L
    else:
        peak_trough_ratio = math.inf

    css_avg = auc_mg_h_per_L / tau_h

    ke = 0.693 / t_half_h
    accumulation_ratio = 1.0 / (1.0 - math.exp(-ke * tau_h))

    auc_predicted = dose_mg / cl_L_per_h  # mg·h/L (assuming F=1 or IV)

    recovery_pct = auc_mg_h_per_L / auc_predicted * 100.0

    # linearity_index kept internally but not in result per spec
    # linearity_index = cmax_mg_L * t_half_h / (dose_mg * 0.693)

    # --- Classification ---
    fluctuation_class = _classify_fluctuation(fluctuation_pct)
    pk_interpretation = _interpret_pk(fluctuation_pct, peak_trough_ratio, cmax_auc_ratio)

    # --- Notes ---
    note_parts: list[str] = []
    if pk_interpretation == "high_fluctuation":
        note_parts.append("Fluctuation >100%: extended release formulation may be needed.")
    if peak_trough_ratio > 10.0 and pk_interpretation != "high_fluctuation":
        note_parts.append("Peak-trough ratio >10: consider more frequent dosing.")
    if cmax_auc_ratio > 0.5:
        note_parts.append("Cmax/AUC >0.5: rapid distribution/elimination from peaks.")
    note_parts.append(f"Accumulation ratio={accumulation_ratio:.2f} (ke={ke:.4f}/h, tau={tau_h}h).")
    notes = " ".join(note_parts)

    return ExposureMetricsResult(
        drug_name=drug_name,
        cmax_mg_L=cmax_mg_L,
        cmin_mg_L=cmin_mg_L,
        auc_mg_h_per_L=auc_mg_h_per_L,
        cmax_auc_ratio_per_h=cmax_auc_ratio,
        fluctuation_pct=fluctuation_pct,
        peak_trough_ratio=peak_trough_ratio,
        css_avg_mg_L=css_avg,
        accumulation_ratio=accumulation_ratio,
        auc_predicted_mg_h_per_L=auc_predicted,
        recovery_pct=recovery_pct,
        fluctuation_class=fluctuation_class,
        pk_interpretation=pk_interpretation,
        notes=notes,
    )
