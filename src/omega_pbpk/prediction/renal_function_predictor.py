"""Renal function predictor and dose adjustment calculator (Phase 436).

Implements the CKD-EPI 2021 eGFR equation (race-free) and uses GFR to
predict renally adjusted drug clearance and dose adjustment factors.

Reference:
    Inker LA et al. NEJM 2021; 385:1737-1749. CKD-EPI Creatinine 2021.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "RenalFunctionResult",
    "ckd_epi_egfr",
    "predict_renal_drug_clearance",
]

# ---------------------------------------------------------------------------
# GFR category boundaries (KDIGO 2012)
# ---------------------------------------------------------------------------

_GFR_CATEGORIES: list[tuple[float, float, str]] = [
    (90.0, float("inf"), "G1"),
    (60.0, 90.0, "G2"),
    (45.0, 60.0, "G3a"),
    (30.0, 45.0, "G3b"),
    (15.0, 30.0, "G4"),
    (0.0, 15.0, "G5"),
]

# Normal GFR reference value (mL/min)
_NORMAL_GFR = 120.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenalFunctionResult:
    """Output of renal drug clearance prediction."""

    drug_name: str
    gfr_mL_per_min: float
    gfr_category: str                    # G1 through G5
    fu_plasma: float                     # fraction unbound in plasma
    f_renal_excretion: float             # fraction of total CL via renal route (0–1)

    cl_renal_L_per_h: float              # renally adjusted renal CL (L/h)
    cl_nonrenal_L_per_h: float           # non-renal CL (assumed unchanged)
    cl_total_adjusted_L_per_h: float     # sum of the two

    dose_adjustment_factor: float        # adjusted_CL / normal_CL
    notes: str


# ---------------------------------------------------------------------------
# CKD-EPI 2021 eGFR equation
# ---------------------------------------------------------------------------


def ckd_epi_egfr(
    scr_mg_dL: float,
    age: float,
    sex: str,
    race: str = "other",
) -> float:
    """Compute eGFR using the CKD-EPI 2021 creatinine equation (race-free).

    The 2021 revision removes the race coefficient that was present in the
    2009 version.  The ``race`` parameter is accepted for API compatibility
    but has no effect on the calculation.

    Parameters
    ----------
    scr_mg_dL:
        Serum creatinine concentration (mg/dL); must be > 0.
    age:
        Patient age in years; must be in [18, 120].
    sex:
        ``"female"`` or ``"male"`` (case-insensitive).
    race:
        Accepted but ignored (2021 update removed race adjustment).

    Returns
    -------
    float
        Estimated GFR in mL/min/1.73 m².
    """
    if scr_mg_dL <= 0:
        raise ValueError(f"scr_mg_dL must be positive, got {scr_mg_dL}")
    if not (18 <= age <= 120):
        raise ValueError(f"age must be in [18, 120], got {age}")
    sex_lower = sex.strip().lower()
    if sex_lower not in {"female", "male"}:
        raise ValueError(f"sex must be 'female' or 'male', got '{sex}'")

    # CKD-EPI 2021 parameters
    if sex_lower == "female":
        k = 0.7
        if scr_mg_dL <= k:
            alpha = -0.241
        else:
            alpha = -1.200
        sex_multiplier = 1.012
    else:  # male
        k = 0.9
        if scr_mg_dL <= k:
            alpha = -0.302
        else:
            alpha = -1.200
        sex_multiplier = 1.0

    ratio = scr_mg_dL / k
    egfr = (
        142.0
        * (min(ratio, 1.0) ** alpha)
        * (max(ratio, 1.0) ** (-1.200))
        * (0.9938 ** age)
        * sex_multiplier
    )
    return float(egfr)


# ---------------------------------------------------------------------------
# Renal drug clearance prediction
# ---------------------------------------------------------------------------


def predict_renal_drug_clearance(
    drug_name: str,
    gfr_mL_per_min: float,
    fu_plasma: float,
    logP: float,
    mw: float,
    f_renal_excretion: float,
    normal_cl_L_per_h: float,
) -> RenalFunctionResult:
    """Predict renally adjusted drug clearance and dose adjustment factor.

    Algorithm
    ---------
    1. Compute CLr_normal = GFR_normal * fu_plasma (filtration-based renal CL
       in mL/min, then converted to L/h).
    2. Scale to patient GFR: CLr_adjusted = CLr_normal * GFR_patient / GFR_normal
    3. Non-renal CL is unchanged: CL_nr = normal_CL * (1 - f_renal)
    4. Total adjusted CL = CL_nr + CLr_adjusted
    5. Dose adjustment factor = CL_total_adjusted / normal_CL

    Parameters
    ----------
    drug_name:
        Compound identifier.
    gfr_mL_per_min:
        Patient eGFR in mL/min/1.73 m².  Must be >= 0.
    fu_plasma:
        Fraction unbound in plasma (0 < fu <= 1).
    logP:
        Octanol-water partition coefficient (not currently used in the core
        calculation but retained for future tubular reabsorption modelling).
    mw:
        Molecular weight (g/mol); must be > 0.
    f_renal_excretion:
        Fraction of total normal clearance attributable to renal excretion
        (0 <= f <= 1).
    normal_cl_L_per_h:
        Total clearance in a healthy adult with normal renal function (L/h).
        Must be > 0.

    Returns
    -------
    RenalFunctionResult
    """
    # --- Input validation ---
    if not drug_name:
        raise ValueError("drug_name must be a non-empty string")
    if gfr_mL_per_min < 0:
        raise ValueError(f"gfr_mL_per_min must be >= 0, got {gfr_mL_per_min}")
    if not (0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if mw <= 0:
        raise ValueError(f"mw must be positive, got {mw}")
    if not (0.0 <= f_renal_excretion <= 1.0):
        raise ValueError(f"f_renal_excretion must be in [0, 1], got {f_renal_excretion}")
    if normal_cl_L_per_h <= 0:
        raise ValueError(f"normal_cl_L_per_h must be positive, got {normal_cl_L_per_h}")

    # Clamp GFR to sane ceiling
    gfr_patient = min(gfr_mL_per_min, 200.0)

    # GFR category
    gfr_category = "G5"
    for lo, hi, cat in _GFR_CATEGORIES:
        if lo <= gfr_patient < hi:
            gfr_category = cat
            break
    # Edge case: exactly >= 90
    if gfr_patient >= 90.0:
        gfr_category = "G1"

    # Normal renal CL from the declared renal fraction of total CL.
    # This encompasses both glomerular filtration (CLr_filt = GFR * fu_plasma)
    # and active tubular secretion. Using f_renal * normal_CL ensures that at
    # GFR=120 (normal), the computed total CL equals normal_CL exactly.
    cl_renal_normal_L_per_h = normal_cl_L_per_h * f_renal_excretion

    # Patient renal CL: scale CLr_normal linearly with GFR
    # (GFR is the primary driver; active secretion is assumed to scale similarly)
    cl_renal_adj = cl_renal_normal_L_per_h * (gfr_patient / _NORMAL_GFR)

    # Non-renal CL (hepatic, biliary) — assumed preserved
    cl_nonrenal = normal_cl_L_per_h * (1.0 - f_renal_excretion)

    cl_total_adj = cl_nonrenal + cl_renal_adj

    # Dose adjustment factor
    dose_adj_factor = cl_total_adj / normal_cl_L_per_h

    # Notes
    notes_parts: list[str] = [
        f"GFR category: {gfr_category} ({gfr_patient:.1f} mL/min)",
        f"CLr_adj: {cl_renal_adj:.3f} L/h, CL_nr: {cl_nonrenal:.3f} L/h",
        f"Dose adjustment factor: {dose_adj_factor:.3f}",
    ]
    if gfr_category in {"G4", "G5"}:
        notes_parts.append("Severe renal impairment — consider dose reduction or extended interval")
    elif gfr_category == "G3b":
        notes_parts.append("Moderate-severe renal impairment — monitor closely")
    elif gfr_category in {"G2", "G3a"}:
        notes_parts.append("Mild-moderate renal impairment")

    return RenalFunctionResult(
        drug_name=drug_name,
        gfr_mL_per_min=float(gfr_patient),
        gfr_category=gfr_category,
        fu_plasma=float(fu_plasma),
        f_renal_excretion=float(f_renal_excretion),
        cl_renal_L_per_h=float(cl_renal_adj),
        cl_nonrenal_L_per_h=float(cl_nonrenal),
        cl_total_adjusted_L_per_h=float(cl_total_adj),
        dose_adjustment_factor=float(dose_adj_factor),
        notes="; ".join(notes_parts),
    )
