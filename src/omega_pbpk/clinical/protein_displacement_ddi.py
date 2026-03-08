"""Phase 487 — Drug Interaction via Plasma Protein Displacement.

Models pharmacokinetic drug-drug interactions via plasma protein binding
displacement (competitive binding for albumin / AAG sites).

FDA guidance: protein displacement alone is rarely clinically significant
unless the victim drug also has a narrow therapeutic index and low clearance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProteinDisplacementDDIResult:
    """Result of a protein-displacement DDI prediction."""

    victim_drug: str
    displacer_drug: str
    fu_baseline: float
    fu_displaced: float
    displacement_ratio: float  # fu_displaced / fu_baseline
    extraction_ratio: float
    auc_ratio: float  # expected AUC_displaced / AUC_baseline
    vd_ratio: float  # Vd_displaced / Vd_baseline
    clinical_significance: str  # none | minor | moderate | major
    mechanism: str
    recommendation: str  # monitor | no_action
    notes: str


def _classify_significance(auc_ratio: float) -> str:
    """Classify clinical significance based on AUC change."""
    pct_change = abs(auc_ratio - 1.0) * 100.0
    if pct_change < 10.0:
        return "none"
    if pct_change < 25.0:
        return "minor"
    if pct_change < 50.0:
        return "moderate"
    return "major"


def predict_protein_displacement_ddi(
    victim_drug: str,
    displacer_drug: str,
    fu_baseline: float,
    cl_sys_L_per_h: float,
    conc_displacer_mg_L: float,
    kd_displacer_mg_L: float = 1.0,
    Q_hepatic_L_per_h: float = 90.0,
) -> ProteinDisplacementDDIResult:
    """Predict pharmacokinetic DDI caused by plasma protein displacement.

    Parameters
    ----------
    victim_drug:
        Name of the victim (displaced) drug.
    displacer_drug:
        Name of the displacing drug.
    fu_baseline:
        Unbound fraction of the victim drug at baseline (0 < fu <= 1).
    cl_sys_L_per_h:
        Systemic clearance of the victim drug (L/h). Used to derive the
        hepatic extraction ratio.
    conc_displacer_mg_L:
        Steady-state plasma concentration of the displacer (mg/L).
    kd_displacer_mg_L:
        Apparent dissociation constant for the displacer–albumin interaction
        (mg/L). Default 1.0 mg/L.
    Q_hepatic_L_per_h:
        Hepatic blood flow (L/h). Default 90 L/h.

    Returns
    -------
    ProteinDisplacementDDIResult
    """
    # ----- Validation -----
    if not (0.0 < fu_baseline <= 1.0):
        raise ValueError(f"fu_baseline must be in (0, 1], got {fu_baseline}")
    if cl_sys_L_per_h <= 0.0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if conc_displacer_mg_L < 0.0:
        raise ValueError(f"conc_displacer_mg_L must be >= 0, got {conc_displacer_mg_L}")
    if kd_displacer_mg_L <= 0.0:
        raise ValueError(f"kd_displacer_mg_L must be > 0, got {kd_displacer_mg_L}")

    # ----- Displaced fu -----
    # Simplified competitive binding model:
    #   fu_displaced = fu_baseline * (1 + conc / (Kd + conc))
    # Maximum effect: fu_displaced -> 2 * fu_baseline when conc >> Kd.
    # When conc = 0: fu_displaced = fu_baseline (no displacement).
    displacement_factor = conc_displacer_mg_L / (kd_displacer_mg_L + conc_displacer_mg_L)
    fu_displaced_raw = fu_baseline * (1.0 + displacement_factor)
    # fu cannot exceed 1.0
    fu_displaced = min(fu_displaced_raw, 1.0)

    displacement_ratio = fu_displaced / fu_baseline

    # ----- Extraction ratio -----
    extraction_ratio = cl_sys_L_per_h / Q_hepatic_L_per_h
    # Clamp to [0, 1] for safety
    extraction_ratio = max(0.0, min(1.0, extraction_ratio))

    # ----- AUC ratio (using well-stirred model) -----
    # High-ER (E > 0.7): CL ≈ Q (flow-limited); fu change has negligible effect
    #   → auc_ratio ≈ 1.0
    # Low-ER (E < 0.3): CL ≈ fu * CLint (restriction model)
    #   → CL_displaced/CL_baseline = fu_displaced/fu_baseline
    #   → auc_ratio = fu_baseline/fu_displaced = 1/displacement_ratio
    # Moderate (0.3 <= E <= 0.7): linear interpolation between both extremes.
    if extraction_ratio > 0.7:
        # Flow-limited: AUC unchanged
        auc_ratio = 1.0
    elif extraction_ratio < 0.3:
        # Restriction model: AUC inversely proportional to fu increase
        auc_ratio = 1.0 / displacement_ratio
    else:
        # Interpolate linearly between 0.3 and 0.7
        weight = (extraction_ratio - 0.3) / 0.4  # 0 at E=0.3, 1 at E=0.7
        low_er_auc = 1.0 / displacement_ratio
        high_er_auc = 1.0
        auc_ratio = low_er_auc + weight * (high_er_auc - low_er_auc)

    # ----- Vd ratio -----
    # For drugs with restricted tissue distribution:
    #   Vd = Vp + (fu_p / fu_t) * Vt
    # Approximation for Vd ratio: Vd_displaced / Vd_baseline ≈ fu_displaced / fu_baseline
    vd_ratio = displacement_ratio

    # ----- Classification -----
    clinical_significance = _classify_significance(auc_ratio)

    if clinical_significance in ("none", "minor"):
        recommendation = "no_action"
    else:
        recommendation = "monitor"

    # ----- Mechanism & notes -----
    mechanism = (
        f"{displacer_drug} competitively displaces {victim_drug} from plasma "
        "protein binding sites (albumin/AAG), increasing unbound fraction."
    )

    er_desc = (
        "high" if extraction_ratio > 0.7 else ("low" if extraction_ratio < 0.3 else "moderate")
    )
    notes = (
        f"Extraction ratio {extraction_ratio:.2f} ({er_desc}-clearance drug). "
        f"fu increased from {fu_baseline:.3f} to {fu_displaced:.3f} "
        f"(displacement_ratio={displacement_ratio:.3f}). "
        f"AUC change {(auc_ratio - 1.0) * 100:.1f}%. "
        "FDA guidance: protein displacement is rarely clinically significant "
        "unless the drug has a narrow therapeutic index."
    )

    return ProteinDisplacementDDIResult(
        victim_drug=victim_drug,
        displacer_drug=displacer_drug,
        fu_baseline=fu_baseline,
        fu_displaced=fu_displaced,
        displacement_ratio=displacement_ratio,
        extraction_ratio=extraction_ratio,
        auc_ratio=auc_ratio,
        vd_ratio=vd_ratio,
        clinical_significance=clinical_significance,
        mechanism=mechanism,
        recommendation=recommendation,
        notes=notes,
    )
