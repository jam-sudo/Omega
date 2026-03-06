"""Dose-proportionality statistical analysis using power model.

References
----------
- Smith BP et al., J Biopharm Stat. 2000;10(4):571-96 (power model)
- Gough K et al., J Pharmacokinet Biopharm. 1995;23(3):291-309
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DoseProportionalityResult",
    "PowerModelResult",
    "assess_dose_proportionality",
    "power_model_fit",
]


@dataclass(frozen=True)
class PowerModelResult:
    """Power model fit result: PK = alpha * Dose^beta."""

    doses: list[float]
    pk_values: list[float]
    pk_name: str
    alpha: float
    beta: float
    r2: float
    assessment: str
    predicted_at_doses: list[float]


@dataclass(frozen=True)
class DoseProportionalityResult:
    """Full dose-proportionality assessment result."""

    doses_mg: list[float]
    aucs: list[float]
    beta_auc: float
    beta_cmax: float | None
    alpha_auc: float
    r2_auc: float
    proportionality_assessment: str
    dose_normalized_auc: list[float]
    cv_dose_normalized_auc_pct: float
    notes: str


def _linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, r2) for simple linear regression."""
    n = len(x)
    if n < 2:
        return (0.0, float(y[0]) if n == 1 else 0.0, 0.0)
    sx = np.sum(x)
    sy = np.sum(y)
    sxx = np.sum(x * x)
    sxy = np.sum(x * y)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-30:
        return (0.0, float(np.mean(y)), 0.0)
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0
    return float(slope), float(intercept), float(max(0.0, min(1.0, r2)))


def _assess(beta: float) -> str:
    if abs(beta - 1.0) < 0.1:
        return "proportional"
    elif beta > 1.1:
        return "supra_proportional"
    else:
        return "sub_proportional"


def power_model_fit(
    doses_mg: list[float],
    pk_values: list[float],
    pk_name: str = "AUC",
) -> PowerModelResult:
    """Fit power model PK = alpha * Dose^beta via log-log linear regression.

    Parameters
    ----------
    doses_mg : list of doses.
    pk_values : corresponding PK metric values.
    pk_name : name of the PK metric.

    Returns
    -------
    PowerModelResult
    """
    if len(doses_mg) < 3:
        raise ValueError("At least 3 dose-PK pairs required for power model fit")
    if len(doses_mg) != len(pk_values):
        raise ValueError("doses_mg and pk_values must have same length")
    if any(d <= 0 for d in doses_mg):
        raise ValueError("All doses must be > 0")
    if any(v <= 0 for v in pk_values):
        raise ValueError("All pk_values must be > 0")

    log_d = np.log(np.array(doses_mg, dtype=float))
    log_pk = np.log(np.array(pk_values, dtype=float))

    beta, log_alpha, r2 = _linear_regression(log_d, log_pk)
    alpha = float(math.exp(log_alpha))
    predicted = [float(alpha * (d ** beta)) for d in doses_mg]
    assessment = _assess(beta)

    return PowerModelResult(
        doses=list(doses_mg),
        pk_values=list(pk_values),
        pk_name=pk_name,
        alpha=alpha,
        beta=beta,
        r2=r2,
        assessment=assessment,
        predicted_at_doses=predicted,
    )


def assess_dose_proportionality(
    doses_mg: list[float],
    aucs: list[float],
    cmaxes: list[float] | None = None,
    pk_name: str = "AUC",
) -> DoseProportionalityResult:
    """Assess dose-proportionality for AUC (and optionally Cmax).

    Parameters
    ----------
    doses_mg : List of doses.
    aucs : List of AUC values.
    cmaxes : Optional list of Cmax values.
    pk_name : Name label.

    Returns
    -------
    DoseProportionalityResult
    """
    if len(doses_mg) < 3:
        raise ValueError("At least 3 doses required for dose-proportionality assessment")
    if len(doses_mg) != len(aucs):
        raise ValueError("doses_mg and aucs must have same length")

    auc_pm = power_model_fit(doses_mg, aucs, pk_name)

    beta_cmax: float | None = None
    if cmaxes is not None:
        if len(cmaxes) != len(doses_mg):
            raise ValueError("cmaxes must have same length as doses_mg")
        cmax_pm = power_model_fit(doses_mg, cmaxes, "Cmax")
        beta_cmax = cmax_pm.beta

    # Dose-normalized AUC
    dn_auc = [a / d for a, d in zip(aucs, doses_mg)]
    mean_dn = float(np.mean(dn_auc))
    std_dn = float(np.std(dn_auc, ddof=1)) if len(dn_auc) > 1 else 0.0
    cv_pct = (std_dn / mean_dn * 100.0) if mean_dn > 1e-30 else 0.0

    assessment = auc_pm.assessment
    note_parts = [f"Power model β(AUC)={auc_pm.beta:.3f}: {assessment}"]
    if cv_pct > 30:
        note_parts.append(f"High CV% ({cv_pct:.1f}%) in dose-normalized AUC")
    if beta_cmax is not None:
        note_parts.append(f"β(Cmax)={beta_cmax:.3f}: {_assess(beta_cmax)}")
    notes = "; ".join(note_parts)

    return DoseProportionalityResult(
        doses_mg=list(doses_mg),
        aucs=list(aucs),
        beta_auc=auc_pm.beta,
        beta_cmax=beta_cmax,
        alpha_auc=auc_pm.alpha,
        r2_auc=auc_pm.r2,
        proportionality_assessment=assessment,
        dose_normalized_auc=dn_auc,
        cv_dose_normalized_auc_pct=cv_pct,
        notes=notes,
    )
