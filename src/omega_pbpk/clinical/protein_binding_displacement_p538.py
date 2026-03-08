"""
Phase 538 — Protein Binding Displacement Drug Interaction

Models the pharmacokinetic consequences of protein-binding displacement
DDI, where a displacing drug increases the free fraction of a displaced drug.
"""

from __future__ import annotations

from dataclasses import dataclass

# Extraction ratio thresholds
_EH_HIGH = 0.7
_EH_LOW = 0.3

# Hepatic blood flow default (L/h)
_Q_DEFAULT = 90.0


def _extraction_ratio(q: float, fup: float, cl_int: float) -> float:
    """Well-stirred hepatic extraction ratio."""
    return fup * cl_int / (q + fup * cl_int)


def _hepatic_cl(q: float, fup: float, cl_int: float) -> float:
    """Well-stirred hepatic clearance (L/h)."""
    return q * fup * cl_int / (q + fup * cl_int)


def _displacement_significance(fu_ratio: float) -> str:
    if fu_ratio > 2.0:
        return "significant"
    if fu_ratio > 1.5:
        return "moderate"
    if fu_ratio > 1.1:
        return "mild"
    return "negligible"


@dataclass(frozen=True)
class ProteinBindingDisplacementResult:
    """Result of a protein-binding displacement interaction assessment."""

    displaced_drug: str
    displacing_drug: str
    c_displacing_mg_L: float
    ki_displacement_mg_L: float

    fu_displaced_baseline: float
    fu_displaced_new: float
    fu_ratio: float  # fu_displaced_new / fu_displaced_baseline

    eh_displaced: float  # extraction ratio at baseline fu
    cl_int_displaced_L_per_h: float

    cl_hepatic_baseline_L_per_h: float
    cl_hepatic_new_L_per_h: float

    # AUC inversely proportional to CL; for protein binding DDI AUC ratio ≈ 1.0
    auc_ratio: float  # CLh_baseline / CLh_new  (ratio of AUCs)
    free_cmax_ratio: float  # fu_new / fu_old — transient free drug increase

    displacement_significance: str  # "negligible"/"mild"/"moderate"/"significant"
    clinical_concern: bool  # True if sig != negligible AND EH < 0.3
    notes: str


def assess_protein_binding_displacement(
    displaced_drug: str,
    displacing_drug: str,
    c_displacing_mg_L: float,
    ki_displacement_mg_L: float,
    fu_displaced_baseline: float,
    cl_int_displaced_L_per_h: float,
    fu_displacing: float = 0.05,
    q_hepatic_L_per_h: float = _Q_DEFAULT,
) -> ProteinBindingDisplacementResult:
    """Assess the PK impact of protein-binding displacement.

    Parameters
    ----------
    displaced_drug:
        Name of the drug that is displaced from protein binding.
    displacing_drug:
        Name of the drug causing displacement.
    c_displacing_mg_L:
        Total plasma concentration of the displacing drug (mg/L).
    ki_displacement_mg_L:
        Inhibition constant for binding displacement (mg/L).
    fu_displaced_baseline:
        Free fraction of displaced drug without displacing drug (0 < fu <= 1).
    cl_int_displaced_L_per_h:
        Intrinsic clearance of displaced drug (L/h).
    fu_displacing:
        Free fraction of displacing drug; used to compute unbound [I].
    q_hepatic_L_per_h:
        Hepatic blood flow (L/h); default 90.

    Returns
    -------
    ProteinBindingDisplacementResult
    """
    if c_displacing_mg_L < 0:
        raise ValueError(f"c_displacing_mg_L must be >= 0; got {c_displacing_mg_L}")
    if ki_displacement_mg_L <= 0:
        raise ValueError(f"ki_displacement_mg_L must be > 0; got {ki_displacement_mg_L}")
    if not (0 < fu_displaced_baseline <= 1.0):
        raise ValueError(f"fu_displaced_baseline must be in (0, 1]; got {fu_displaced_baseline}")
    if cl_int_displaced_L_per_h <= 0:
        raise ValueError(f"cl_int_displaced_L_per_h must be > 0; got {cl_int_displaced_L_per_h}")

    # Unbound concentration of displacing drug
    c_unbound_displacing = c_displacing_mg_L * fu_displacing

    # Competitive displacement: fu increases with [I]/Ki
    fu_new_raw = fu_displaced_baseline * (1.0 + c_unbound_displacing / ki_displacement_mg_L)
    # Clamp between baseline and 1.0
    fu_new = min(max(fu_new_raw, fu_displaced_baseline), 1.0)

    fu_ratio = fu_new / fu_displaced_baseline

    # Hepatic clearances
    clh_baseline = _hepatic_cl(q_hepatic_L_per_h, fu_displaced_baseline, cl_int_displaced_L_per_h)
    clh_new = _hepatic_cl(q_hepatic_L_per_h, fu_new, cl_int_displaced_L_per_h)

    # Extraction ratio (baseline)
    eh = _extraction_ratio(q_hepatic_L_per_h, fu_displaced_baseline, cl_int_displaced_L_per_h)

    # AUC ratio: AUC ∝ 1/CLh; ratio = CLh_baseline / CLh_new
    # For high-extraction: CLh ≈ Q (unchanged) → AUC ratio ≈ 1.0
    # For low-extraction: CLh increases proportionally → AUC ratio ≈ 1.0
    # Either way the long-term total AUC does not change appreciably
    auc_ratio = clh_baseline / clh_new if clh_new > 0 else 1.0

    free_cmax_ratio = fu_ratio  # transient free-drug peak increase

    sig = _displacement_significance(fu_ratio)
    clinical_concern = sig != "negligible" and eh < _EH_LOW

    notes = (
        f"Unbound [I]={c_unbound_displacing:.4f} mg/L, Ki={ki_displacement_mg_L:.4f} mg/L. "
        f"EH={eh:.3f} ({('flow_limited' if eh > _EH_HIGH else ('enzyme_limited' if eh < _EH_LOW else 'intermediate'))}). "
        "Long-term AUC typically unchanged (CLh and Vd increase proportionally); "
        "transient free Cmax may rise for low-extraction drugs."
    )

    return ProteinBindingDisplacementResult(
        displaced_drug=displaced_drug,
        displacing_drug=displacing_drug,
        c_displacing_mg_L=c_displacing_mg_L,
        ki_displacement_mg_L=ki_displacement_mg_L,
        fu_displaced_baseline=fu_displaced_baseline,
        fu_displaced_new=fu_new,
        fu_ratio=fu_ratio,
        eh_displaced=eh,
        cl_int_displaced_L_per_h=cl_int_displaced_L_per_h,
        cl_hepatic_baseline_L_per_h=clh_baseline,
        cl_hepatic_new_L_per_h=clh_new,
        auc_ratio=auc_ratio,
        free_cmax_ratio=free_cmax_ratio,
        displacement_significance=sig,
        clinical_concern=clinical_concern,
        notes=notes,
    )


def screen_displacement_interactions(
    displaced_drug: str,
    cl_int_L_per_h: float,
    fu_baseline: float,
    perpetrators: list,
) -> list:
    """Screen multiple perpetrators for protein-binding displacement potential.

    Parameters
    ----------
    displaced_drug:
        Name of the drug being displaced.
    cl_int_L_per_h:
        Intrinsic clearance of displaced drug (L/h).
    fu_baseline:
        Free fraction of displaced drug at baseline.
    perpetrators:
        List of dicts, each with keys:
        ``displacing_drug``, ``c_displacing_mg_L``, ``ki_displacement_mg_L``,
        ``fu_displacing``.

    Returns
    -------
    List of ProteinBindingDisplacementResult sorted by fu_ratio descending.
    """
    results = []
    for p in perpetrators:
        r = assess_protein_binding_displacement(
            displaced_drug=displaced_drug,
            displacing_drug=p["displacing_drug"],
            c_displacing_mg_L=p["c_displacing_mg_L"],
            ki_displacement_mg_L=p["ki_displacement_mg_L"],
            fu_displaced_baseline=fu_baseline,
            cl_int_displaced_L_per_h=cl_int_L_per_h,
            fu_displacing=p.get("fu_displacing", 0.05),
        )
        results.append(r)
    results.sort(key=lambda r: r.fu_ratio, reverse=True)
    return results
