"""Phase 345 — Renal Drug Clearance Prediction.

Predicts renal drug clearance from physicochemical properties using
filtration, secretion, and reabsorption mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass

# Renal plasma flow
_Q_RENAL_ML_PER_MIN: float = 700.0
_DEFAULT_TUBULAR_PH: float = 5.0


@dataclass(frozen=True)
class RenalClearancePredictionResult:
    """Result of renal clearance prediction."""

    drug_name: str
    fu_plasma: float
    gfr_mL_per_min: float
    cl_filtration_mL_per_min: float
    cl_secretion_mL_per_min: float
    cl_reabsorption_factor: float
    cl_renal_mL_per_min: float
    cl_renal_L_per_h: float
    fe_renal: float
    renal_elimination_class: str
    ckd_sensitivity: str
    notes: str


def _validate_inputs(
    gfr_mL_per_min: float,
    fu_plasma: float,
    mw_Da: float,
    reabsorption_fraction: float,
    protein_binding_pct: float,
) -> None:
    """Validate input parameters."""
    if gfr_mL_per_min <= 0:
        raise ValueError(f"gfr_mL_per_min must be > 0, got {gfr_mL_per_min}")
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if not (0.0 <= reabsorption_fraction <= 1.0):
        raise ValueError(f"reabsorption_fraction must be in [0, 1], got {reabsorption_fraction}")
    if not (0.0 <= protein_binding_pct <= 100.0):
        raise ValueError(f"protein_binding_pct must be in [0, 100], got {protein_binding_pct}")


def _compute_ionization_factor(
    drug_type: str,
    pKa: float,
    tubular_pH: float,
) -> float:
    """Compute ionization factor for pH-dependent reabsorption."""
    if drug_type == "neutral":
        return 1.0
    if drug_type == "acid":
        # Fraction ionized for acid: [A-] / ([HA] + [A-])
        pct_ionized = 100.0 / (1.0 + 10.0 ** (pKa - tubular_pH))
        return 1.0 - pct_ionized / 100.0
    if drug_type == "base":
        # For base: ionized at low pH means more protonated (BH+)
        # pct_ionized = 100 / (1 + 10^(pH - pKa))
        pct_ionized = 100.0 / (1.0 + 10.0 ** (tubular_pH - pKa))
        return pct_ionized / 100.0
    raise ValueError(f"drug_type must be 'acid', 'base', or 'neutral', got {drug_type!r}")


def predict_renal_clearance(
    drug_name: str,
    mw_Da: float,
    logP: float,
    pKa: float,
    drug_type: str,
    fu_plasma: float,
    gfr_mL_per_min: float = 120.0,
    tubular_secretion_clint_mL_per_min: float = 0.0,
    reabsorption_fraction: float = 0.0,
    protein_binding_pct: float = 0.0,
    nonrenal_cl_L_per_h: float | None = None,
    tubular_pH: float = _DEFAULT_TUBULAR_PH,
) -> RenalClearancePredictionResult:
    """Predict renal clearance from physicochemical properties.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    mw_Da:
        Molecular weight in Daltons.
    logP:
        Octanol-water partition coefficient.
    pKa:
        Acid/base dissociation constant.
    drug_type:
        'acid', 'base', or 'neutral'.
    fu_plasma:
        Unbound fraction in plasma (0 < fu <= 1).
    gfr_mL_per_min:
        Glomerular filtration rate in mL/min (default 120).
    tubular_secretion_clint_mL_per_min:
        Intrinsic secretion clearance via OAT/OCT transporters (mL/min).
    reabsorption_fraction:
        Fraction of filtered/secreted drug reabsorbed from tubular fluid (0–1).
    protein_binding_pct:
        Protein binding percentage (0–100); used for notes only.
    nonrenal_cl_L_per_h:
        Non-renal clearance in L/h. If None, defaults to 3 * CL_renal.
    tubular_pH:
        Tubular fluid pH for pH-dependent reabsorption calculation.

    Returns
    -------
    RenalClearancePredictionResult
    """
    _validate_inputs(gfr_mL_per_min, fu_plasma, mw_Da, reabsorption_fraction, protein_binding_pct)

    if drug_type not in ("acid", "base", "neutral"):
        raise ValueError(f"drug_type must be 'acid', 'base', or 'neutral', got {drug_type!r}")

    # Filtration clearance
    cl_filtration = gfr_mL_per_min * fu_plasma

    # Tubular secretion clearance (well-stirred model)
    if tubular_secretion_clint_mL_per_min > 0.0:
        q = _Q_RENAL_ML_PER_MIN
        cl_int_eff = tubular_secretion_clint_mL_per_min * fu_plasma
        cl_sec = (q * cl_int_eff) / (q + cl_int_eff)
    else:
        cl_sec = 0.0

    # pH-dependent reabsorption adjustment
    ionization_factor = _compute_ionization_factor(drug_type, pKa, tubular_pH)
    adjusted_reabsorption = reabsorption_fraction * ionization_factor
    # Clamp to [0, 1]
    adjusted_reabsorption = max(0.0, min(1.0, adjusted_reabsorption))
    cl_reabsorption_factor = 1.0 - adjusted_reabsorption

    # Total renal clearance
    cl_renal_mL = (cl_filtration + cl_sec) * cl_reabsorption_factor
    cl_renal_L_h = cl_renal_mL * 60.0 / 1000.0

    # Non-renal clearance
    if nonrenal_cl_L_per_h is None:
        nonrenal_cl_L_per_h = 3.0 * cl_renal_L_h

    # Fraction excreted unchanged
    total_cl = cl_renal_L_h + nonrenal_cl_L_per_h
    if total_cl > 0.0:
        fe = cl_renal_L_h / total_cl
    else:
        fe = 0.0
    fe = max(0.0, min(1.0, fe))

    # Classify renal elimination
    if fe > 0.5:
        renal_class = "primary"
        ckd_sens = "high"
    elif fe >= 0.25:
        renal_class = "significant"
        ckd_sens = "moderate"
    else:
        renal_class = "minor"
        ckd_sens = "low"

    # Build notes
    notes_parts = [
        f"Drug type: {drug_type}",
        f"Tubular pH: {tubular_pH}",
        f"Ionization factor: {ionization_factor:.3f}",
        f"Adjusted reabsorption: {adjusted_reabsorption:.3f}",
        f"Protein binding: {protein_binding_pct:.1f}%",
        f"logP: {logP}",
        f"MW: {mw_Da} Da",
    ]
    if tubular_secretion_clint_mL_per_min > 0:
        notes_parts.append(f"Active secretion CLint: {tubular_secretion_clint_mL_per_min} mL/min")

    return RenalClearancePredictionResult(
        drug_name=drug_name,
        fu_plasma=fu_plasma,
        gfr_mL_per_min=gfr_mL_per_min,
        cl_filtration_mL_per_min=cl_filtration,
        cl_secretion_mL_per_min=cl_sec,
        cl_reabsorption_factor=cl_reabsorption_factor,
        cl_renal_mL_per_min=cl_renal_mL,
        cl_renal_L_per_h=cl_renal_L_h,
        fe_renal=fe,
        renal_elimination_class=renal_class,
        ckd_sensitivity=ckd_sens,
        notes="; ".join(notes_parts),
    )
