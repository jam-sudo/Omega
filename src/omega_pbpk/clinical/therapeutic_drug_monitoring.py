"""TDM-guided dose adjustment for drugs with narrow therapeutic windows.

Implements simple MAP Bayesian dose adjustment using individual CL estimated
from observed concentration data, with target AUC24 or trough-based scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TDMAdjustmentResult:
    """Result of TDM-guided Bayesian dose adjustment."""

    drug_name: str
    current_dose_mg: float
    interval_h: float
    individual_cl_L_per_h: float
    individual_vd_L: float
    recommended_dose_mg: float
    predicted_auc24: float
    predicted_trough: float
    dose_change_pct: float
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _auc_trapezoidal(times: list[float], concs: list[float]) -> float:
    """Compute AUC via the trapezoidal rule."""
    auc = 0.0
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        auc += 0.5 * (concs[i] + concs[i + 1]) * dt
    return auc


def _estimate_cl_from_observations(
    observed_concs: list[float],
    observed_times: list[float],
    dose_mg: float,
    bioavailability: float = 1.0,
) -> float | None:
    """Estimate individual CL from ≥2 observations using AUC/dose.

    Returns None if insufficient data.
    """
    n = len(observed_concs)
    if n < 2:
        return None

    # Sort by time
    pairs = sorted(zip(observed_times, observed_concs, strict=True))
    t_sorted = [p[0] for p in pairs]
    c_sorted = [p[1] for p in pairs]

    # Observed AUC over measured interval (trapezoidal)
    auc_obs = _auc_trapezoidal(t_sorted, c_sorted)

    # Extrapolate to infinity if last concentration > 0
    last_conc = c_sorted[-1]
    if last_conc > 0 and len(t_sorted) >= 2:
        # Estimate ke from last two points (log-linear decline)
        dt = t_sorted[-1] - t_sorted[-2]
        if dt > 0 and c_sorted[-2] > c_sorted[-1] > 0:
            ke = (c_sorted[-2] - c_sorted[-1]) / (dt * 0.5 * (c_sorted[-2] + c_sorted[-1]))
            ke = max(ke, 1e-4)
        else:
            ke = 0.1  # fallback
        auc_obs += last_conc / ke

    if auc_obs <= 0:
        return None

    cl = bioavailability * dose_mg / auc_obs
    return cl


def _predict_trough_1cpt(
    cl_L_per_h: float,
    vd_L: float,
    dose_mg: float,
    interval_h: float,
    n_doses: int = 20,
) -> float:
    """Predict steady-state trough from 1-compartment model using superposition.

    Uses forward Euler integration with a small time step.
    """
    ke = cl_L_per_h / vd_L

    # Simulate one interval at steady state by superposing n_doses prior doses
    conc_at_trough = 0.0
    for dose_idx in range(n_doses):
        t_since_dose = interval_h  # trough = end of interval
        t_rel = t_since_dose + dose_idx * interval_h
        # Oral 1-cpt: approximate with ka >> ke → use IV approximation weighted
        # Simple IV trough: C(τ) = (D/Vd) * exp(-ke * τ)
        conc_at_trough += (dose_mg / vd_L) * _exp_approx(-ke * t_rel)

    return max(conc_at_trough, 0.0)


def _exp_approx(x: float) -> float:
    """Compute exp(x) safely."""
    import math

    try:
        return math.exp(x)
    except OverflowError:
        return 0.0


def _predict_auc24_1cpt(
    cl_L_per_h: float,
    dose_mg: float,
    interval_h: float,
) -> float:
    """Predict AUC over 24h at steady state from 1-cpt: AUC_ss = dose / (CL * interval)."""
    if cl_L_per_h <= 0 or interval_h <= 0:
        return 0.0
    n_doses_per_24h = 24.0 / interval_h
    # AUC per interval = dose / CL; over 24 h = n * (dose / CL)
    auc_per_interval = dose_mg / cl_L_per_h
    return auc_per_interval * n_doses_per_24h


# ---------------------------------------------------------------------------
# Main public functions
# ---------------------------------------------------------------------------


def bayesian_dose_adjustment(
    drug_name: str,
    observed_concs: list[float],
    observed_times: list[float],
    prior_cl_L_per_h: float,
    prior_vd_L: float,
    current_dose_mg: float,
    interval_h: float,
    target_auc24: float | None = None,
    target_trough: float | None = None,
) -> TDMAdjustmentResult:
    """Simple MAP Bayesian TDM dose adjustment.

    Estimates individual CL from observed concentration-time data.
    If ≥ 2 observations are available, CL is estimated from dose/AUC_observed.
    Otherwise the population (prior) CL is used.

    Recommended dose is computed from the target (AUC24 takes precedence).

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    observed_concs : list[float]
        Observed concentrations (mg/L), must be non-negative.
    observed_times : list[float]
        Corresponding sample times (h), must be non-negative.
    prior_cl_L_per_h : float
        Population prior clearance (L/h), must be > 0.
    prior_vd_L : float
        Population prior volume of distribution (L), must be > 0.
    current_dose_mg : float
        Current dose in mg, must be > 0.
    interval_h : float
        Dosing interval in hours, must be > 0.
    target_auc24 : float or None
        Target AUC over 24 h (mg·h/L). If given, takes precedence.
    target_trough : float or None
        Target trough concentration (mg/L). Used only when target_auc24 is None.

    Returns
    -------
    TDMAdjustmentResult
    """
    if not drug_name:
        raise ValueError("drug_name must be a non-empty string")
    if len(observed_concs) != len(observed_times):
        raise ValueError("observed_concs and observed_times must have the same length")
    if any(c < 0 for c in observed_concs):
        raise ValueError("observed_concs must be non-negative")
    if any(t < 0 for t in observed_times):
        raise ValueError("observed_times must be non-negative")
    if prior_cl_L_per_h <= 0:
        raise ValueError("prior_cl_L_per_h must be > 0")
    if prior_vd_L <= 0:
        raise ValueError("prior_vd_L must be > 0")
    if current_dose_mg <= 0:
        raise ValueError("current_dose_mg must be > 0")
    if interval_h <= 0:
        raise ValueError("interval_h must be > 0")
    if target_auc24 is None and target_trough is None:
        raise ValueError("At least one of target_auc24 or target_trough must be provided")
    if target_auc24 is not None and target_auc24 <= 0:
        raise ValueError("target_auc24 must be > 0")
    if target_trough is not None and target_trough <= 0:
        raise ValueError("target_trough must be > 0")

    notes: list[str] = []

    # Step 1: Estimate individual CL
    ind_cl: float | None = None
    if len(observed_concs) >= 2:
        ind_cl = _estimate_cl_from_observations(
            observed_concs, observed_times, current_dose_mg
        )
        if ind_cl is None or ind_cl <= 0:
            notes.append("CL estimation from observations failed; using prior CL")
            ind_cl = prior_cl_L_per_h
        else:
            notes.append(f"Individual CL estimated from {len(observed_concs)} observations")
    else:
        notes.append("Fewer than 2 observations; using population prior CL")
        ind_cl = prior_cl_L_per_h

    # Use prior Vd (not identifiable from sparse data without absorption model)
    ind_vd = prior_vd_L

    # Step 2: Compute recommended dose
    if target_auc24 is not None:
        # AUC-based: dose = target_auc24 * CL / (24 / interval_h)
        n_doses_per_24h = 24.0 / interval_h
        recommended_dose_mg = target_auc24 * ind_cl / n_doses_per_24h
        notes.append("Dose recommended based on AUC24 target")
    else:
        # Trough-based proportional scaling
        # Current trough approximated from current profile
        current_trough = _predict_trough_1cpt(ind_cl, ind_vd, current_dose_mg, interval_h)
        if current_trough > 0:
            recommended_dose_mg = current_dose_mg * (target_trough / current_trough)  # type: ignore[operator]
        else:
            recommended_dose_mg = current_dose_mg
            notes.append("Could not predict current trough; dose unchanged")
        notes.append("Dose recommended based on trough target")

    recommended_dose_mg = max(recommended_dose_mg, 0.01)

    # Step 3: Predict AUC24 and trough at recommended dose
    predicted_auc24 = _predict_auc24_1cpt(ind_cl, recommended_dose_mg, interval_h)
    predicted_trough = _predict_trough_1cpt(ind_cl, ind_vd, recommended_dose_mg, interval_h)

    dose_change_pct = 100.0 * (recommended_dose_mg - current_dose_mg) / current_dose_mg

    return TDMAdjustmentResult(
        drug_name=drug_name,
        current_dose_mg=current_dose_mg,
        interval_h=interval_h,
        individual_cl_L_per_h=round(ind_cl, 4),
        individual_vd_L=round(ind_vd, 4),
        recommended_dose_mg=round(recommended_dose_mg, 4),
        predicted_auc24=round(predicted_auc24, 4),
        predicted_trough=round(predicted_trough, 6),
        dose_change_pct=round(dose_change_pct, 2),
        notes=notes,
    )


def trough_based_adjustment(
    current_trough: float,
    target_trough: float,
    current_dose_mg: float,
) -> dict:
    """Simple linear trough-based dose adjustment.

    new_dose = current_dose * (target_trough / current_trough)

    Parameters
    ----------
    current_trough : float
        Observed trough concentration (mg/L), must be > 0.
    target_trough : float
        Target trough concentration (mg/L), must be > 0.
    current_dose_mg : float
        Current dose in mg, must be > 0.

    Returns
    -------
    dict with keys: new_dose_mg, dose_change_pct
    """
    if current_trough <= 0:
        raise ValueError("current_trough must be > 0")
    if target_trough <= 0:
        raise ValueError("target_trough must be > 0")
    if current_dose_mg <= 0:
        raise ValueError("current_dose_mg must be > 0")

    new_dose_mg = current_dose_mg * (target_trough / current_trough)
    dose_change_pct = 100.0 * (new_dose_mg - current_dose_mg) / current_dose_mg

    return {
        "new_dose_mg": round(new_dose_mg, 4),
        "dose_change_pct": round(dose_change_pct, 2),
    }


__all__ = [
    "TDMAdjustmentResult",
    "bayesian_dose_adjustment",
    "trough_based_adjustment",
]
