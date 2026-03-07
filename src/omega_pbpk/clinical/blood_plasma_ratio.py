"""Blood-to-plasma ratio calculation — Phase 481.

Computes the blood-to-plasma drug concentration ratio (Rb) and its impact
on PK parameters (CL, Vd).  Rb depends on RBC partitioning driven by
lipophilicity (logP) and hematocrit.

References
----------
- Rowland M & Tozer TN. Clinical Pharmacokinetics and Pharmacodynamics, 2011.
- Hinderling PH. Blood cell partitioning of drugs. Pharmacol Rev, 1997.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "RbSensitivityResult",
    "predict_rb",
    "correct_cl_for_rb",
    "correct_vd_for_rb",
    "rb_sensitivity_analysis",
    "classify_rb",
]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class RbSensitivityResult:
    """Result of Rb sensitivity analysis across a range of logP values."""

    logP_values: list[float]
    rb_values: list[float]
    cl_correction_factors: list[float]
    notes: str


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _kp_rbc_from_logP(logP: float) -> float:
    """Estimate RBC partition coefficient from logP.

    Lipophilic drugs (logP >= 0) partition into RBC membranes.
    Hydrophilic drugs (logP < 0) have low partitioning.
    """
    if logP < 0:
        return 0.3
    return 1.0 + logP * 0.3


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def predict_rb(
    logP: float,
    fu_plasma: float,
    hematocrit: float = 0.45,
    rbc_partition: float | None = None,
) -> float:
    """Predict blood-to-plasma ratio (Rb).

    Rb = 1 - Hct + Hct * Kp_rbc

    Parameters
    ----------
    logP:
        Octanol-water partition coefficient (log scale).
    fu_plasma:
        Unbound fraction in plasma (0, 1]. Used for validation context but
        Rb in this model is driven primarily by lipophilicity and hematocrit.
    hematocrit:
        Fractional red blood cell volume (0–1). Default 0.45.
    rbc_partition:
        If provided, use this Kp_rbc directly instead of estimating from logP.

    Returns
    -------
    float
        Blood-to-plasma concentration ratio Rb.
    """
    if not (0.0 <= hematocrit <= 1.0):
        raise ValueError(f"hematocrit must be in [0, 1], got {hematocrit}")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")

    kp = rbc_partition if rbc_partition is not None else _kp_rbc_from_logP(logP)
    if kp < 0.0:
        raise ValueError(f"rbc_partition must be >= 0, got {kp}")

    rb = (1.0 - hematocrit) + hematocrit * kp
    return rb


def correct_cl_for_rb(cl_plasma_L_per_h: float, rb: float) -> float:
    """Convert plasma clearance to blood clearance.

    CL_blood = CL_plasma / Rb

    Parameters
    ----------
    cl_plasma_L_per_h:
        Plasma clearance (L/h). Must be > 0.
    rb:
        Blood-to-plasma ratio. Must be > 0.

    Returns
    -------
    float
        Blood clearance (L/h).
    """
    if cl_plasma_L_per_h <= 0:
        raise ValueError(f"cl_plasma_L_per_h must be > 0, got {cl_plasma_L_per_h}")
    if rb <= 0:
        raise ValueError(f"rb must be > 0, got {rb}")
    return cl_plasma_L_per_h / rb


def correct_vd_for_rb(vd_plasma_L: float, rb: float) -> float:
    """Convert plasma volume of distribution to blood volume of distribution.

    Vd_blood = Vd_plasma / Rb

    Parameters
    ----------
    vd_plasma_L:
        Plasma Vd (L). Must be > 0.
    rb:
        Blood-to-plasma ratio. Must be > 0.

    Returns
    -------
    float
        Blood-based Vd (L).
    """
    if vd_plasma_L <= 0:
        raise ValueError(f"vd_plasma_L must be > 0, got {vd_plasma_L}")
    if rb <= 0:
        raise ValueError(f"rb must be > 0, got {rb}")
    return vd_plasma_L / rb


def rb_sensitivity_analysis(
    logP_values: list[float],
    fu_plasma: float = 0.1,
    hematocrit: float = 0.45,
) -> RbSensitivityResult:
    """Compute Rb and CL correction factors across a range of logP values.

    Parameters
    ----------
    logP_values:
        List of logP values to evaluate.
    fu_plasma:
        Plasma unbound fraction used for each prediction.
    hematocrit:
        Hematocrit fraction (0–1).

    Returns
    -------
    RbSensitivityResult
    """
    if not logP_values:
        raise ValueError("logP_values must not be empty")
    if not (0.0 <= hematocrit <= 1.0):
        raise ValueError(f"hematocrit must be in [0, 1], got {hematocrit}")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")

    rb_values: list[float] = []
    cl_correction_factors: list[float] = []

    for lp in logP_values:
        rb = predict_rb(lp, fu_plasma=fu_plasma, hematocrit=hematocrit)
        rb_values.append(rb)
        # CL_blood = CL_plasma / Rb  →  correction factor = 1/Rb
        cl_correction_factors.append(1.0 / rb)

    notes = (
        f"Rb sensitivity over {len(logP_values)} logP values; "
        f"Hct={hematocrit:.2f}, fu_plasma={fu_plasma:.3f}. "
        f"Higher logP → higher Rb (RBC partitioning). "
        f"CL_blood = CL_plasma / Rb."
    )

    return RbSensitivityResult(
        logP_values=list(logP_values),
        rb_values=rb_values,
        cl_correction_factors=cl_correction_factors,
        notes=notes,
    )


def classify_rb(rb: float) -> str:
    """Classify Rb value into pharmacokinetic categories.

    Parameters
    ----------
    rb:
        Blood-to-plasma ratio. Must be > 0.

    Returns
    -------
    str
        'low' (Rb < 0.55), 'near_unity' (0.55 ≤ Rb ≤ 1.3), or 'high' (Rb > 1.3).
    """
    if rb <= 0:
        raise ValueError(f"rb must be > 0, got {rb}")
    if rb < 0.55:
        return "low"
    if rb <= 1.3:
        return "near_unity"
    return "high"
