"""
Phase 590 — Plasma Protein Binding Displacement
Model drug-drug interaction via plasma protein binding displacement.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PPBDisplacementResult:
    """Result of plasma protein binding displacement prediction."""

    drug_name: str
    displacer_name: str
    fu_baseline: float
    fu_displaced: float
    fu_ratio: float
    delta_auc_pct: float
    clinical_significance: str
    narrow_therapeutic_index: bool
    notes: str


def predict_ppb_displacement(
    drug_name: str,
    fu_drug: float,
    vd_L: float,
    cl_L_per_h: float,
    displacer_name: str,
    fu_displacer: float,
    protein_binding_site: str,
) -> PPBDisplacementResult:
    """
    Predict the effect of plasma protein binding displacement on drug PK.

    Parameters
    ----------
    drug_name : str
        Name of the displaced drug.
    fu_drug : float
        Baseline unbound fraction of the drug (0 < fu_drug <= 1).
    vd_L : float
        Volume of distribution in liters (must be > 0).
    cl_L_per_h : float
        Clearance in L/h (must be > 0).
    displacer_name : str
        Name of the displacing drug.
    fu_displacer : float
        Unbound fraction of the displacer (0 < fu_displacer <= 1).
    protein_binding_site : str
        Binding site: "albumin", "alpha1_agp", or "both".

    Returns
    -------
    PPBDisplacementResult
    """
    # Validation
    if not (0 < fu_drug <= 1.0):
        raise ValueError(f"fu_drug must be in (0, 1], got {fu_drug}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if not (0 < fu_displacer <= 1.0):
        raise ValueError(f"fu_displacer must be in (0, 1], got {fu_displacer}")

    # Determine displacement factor based on protein binding site
    # If same site: 30% displacement scaled by displacer protein binding
    # If different site: 5% displacement
    if protein_binding_site == "both":
        # Both share the same site (albumin and alpha1-AGP both considered shared)
        displacement = 0.3 * (1.0 - fu_displacer)
    elif protein_binding_site in ("albumin", "alpha1_agp"):
        # Single shared site
        displacement = 0.3 * (1.0 - fu_displacer)
    else:
        # Different or unknown site — minimal displacement
        displacement = 0.05 * (1.0 - fu_displacer)

    # Displaced unbound fraction
    fu_displaced = min(1.0, fu_drug + displacement * (1.0 - fu_drug))
    fu_ratio = fu_displaced / fu_drug

    # Hepatic extraction ratio
    hepatic_blood_flow = 90.0  # L/h
    extraction = cl_L_per_h / (cl_L_per_h + hepatic_blood_flow)

    # AUC change: restricted for high extraction or high Vd drugs
    if extraction > 0.7 or vd_L > 50.0:
        delta_auc_pct = (fu_ratio - 1.0) * 5.0
    else:
        delta_auc_pct = (fu_ratio - 1.0) * 100.0 * (1.0 - extraction)

    # Clinical significance based on absolute AUC change
    abs_delta = abs(delta_auc_pct)
    if abs_delta < 10.0:
        clinical_significance = "negligible"
    elif abs_delta < 25.0:
        clinical_significance = "minor"
    elif abs_delta < 50.0:
        clinical_significance = "moderate"
    else:
        clinical_significance = "major"

    # Narrow therapeutic index flag
    narrow_therapeutic_index = delta_auc_pct > 20.0

    notes = (
        f"Extraction ratio={extraction:.3f}, Vd={vd_L} L. "
        f"fu: {fu_drug:.4f} -> {fu_displaced:.4f} (ratio={fu_ratio:.3f}). "
        f"delta_AUC={delta_auc_pct:.1f}%."
    )

    return PPBDisplacementResult(
        drug_name=drug_name,
        displacer_name=displacer_name,
        fu_baseline=fu_drug,
        fu_displaced=fu_displaced,
        fu_ratio=fu_ratio,
        delta_auc_pct=delta_auc_pct,
        clinical_significance=clinical_significance,
        narrow_therapeutic_index=narrow_therapeutic_index,
        notes=notes,
    )
