"""Phase 554 — Renal Clearance Mechanistic Model.

Models renal clearance as: filtration + active secretion - reabsorption.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RenalClearanceResult:
    drug_name: str
    gfr_mL_per_min: float
    fup: float
    cl_filtration_mL_per_min: float
    cl_secretion_mL_per_min: float
    cl_reabsorption_mL_per_min: float
    cl_renal_total_mL_per_min: float
    fe_urine: float
    renal_elimination_class: str
    notes: str


def predict_renal_clearance(
    drug_name: str,
    gfr_mL_per_min: float,
    fup: float,
    logP: float,
    vmax_secretion_mL_per_min: float = 0.0,
    km_secretion_mg_L: float = 1.0,
    cl_hepatic_mL_per_min: float = 100.0,
) -> RenalClearanceResult:
    """Predict renal clearance using mechanistic filtration/secretion/reabsorption model."""
    if not drug_name or not isinstance(drug_name, str):
        raise ValueError("drug_name must be a non-empty string")
    if gfr_mL_per_min <= 0:
        raise ValueError("gfr_mL_per_min must be > 0")
    if not (0 < fup <= 1):
        raise ValueError("fup must be in (0, 1]")

    # Filtration
    cl_filtration = gfr_mL_per_min * fup

    # Active secretion (simplified Michaelis-Menten at typical tubular conc)
    if vmax_secretion_mL_per_min > 0:
        cl_secretion = vmax_secretion_mL_per_min * 0.5
    else:
        cl_secretion = 0.0

    # Passive reabsorption from logP
    if logP > 1:
        reab_frac = max(0.0, min(0.95, 0.15 * (logP - 1)))
    else:
        reab_frac = 0.0
    cl_reabsorption = (cl_filtration + cl_secretion) * reab_frac

    # Total renal clearance
    cl_renal_total = max(0.0, cl_filtration + cl_secretion - cl_reabsorption)

    # Fraction excreted in urine
    denom = cl_renal_total + cl_hepatic_mL_per_min
    fe_urine = cl_renal_total / denom if denom > 0 else 0.0

    # Classification
    if cl_secretion > 1.5 * cl_filtration:
        renal_class = "secretion_dominant"
    elif cl_reabsorption > 0.5 * (cl_filtration + cl_secretion):
        renal_class = "reabsorption_dominant"
    elif cl_filtration > 0.5 * cl_renal_total and cl_renal_total > 0:
        renal_class = "filtration_only"
    else:
        renal_class = "mixed"

    return RenalClearanceResult(
        drug_name=drug_name,
        gfr_mL_per_min=gfr_mL_per_min,
        fup=fup,
        cl_filtration_mL_per_min=cl_filtration,
        cl_secretion_mL_per_min=cl_secretion,
        cl_reabsorption_mL_per_min=cl_reabsorption,
        cl_renal_total_mL_per_min=cl_renal_total,
        fe_urine=fe_urine,
        renal_elimination_class=renal_class,
        notes=f"Mechanistic renal model; logP={logP}, reab_frac={reab_frac:.3f}",
    )
