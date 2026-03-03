"""MIST (Metabolites in Safety Testing) assessment.

Implements FDA MIST guidance (2016 revision) for evaluating whether
circulating metabolites require separate non-clinical safety coverage.

Core rule: metabolites exceeding 10% of parent drug AUC at steady state
are considered "disproportionate" and may require dedicated toxicology
studies, especially if pharmacologically active or unique to humans.

Approach:
  Parent PK: 1-compartment analytical model
    AUC_parent = F × Dose / CL_parent
    Cmax_parent = F × Dose / Vd  (IV: Dose/Vd)

  Metabolite AUC:
    AUC_met = fm × AUC_parent × (CL_parent / CL_met) × (MW_met / MW_parent)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omega_pbpk.drugs.drug import Drug

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIST_THRESHOLD = 0.10  # FDA 10% parent AUC threshold
_CL_MET_FLOOR = 1e-6  # L/h — clamp floor to avoid division by zero


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetaboliteSpec:
    """Specification for a single metabolite."""

    name: str
    mw: float  # g/mol
    fm_parent: float  # fraction of parent metabolised to this metabolite (0-1)
    clint_met_L_per_h: float  # metabolite intrinsic clearance (L/h)
    fup_met: float = 0.5
    pharmacologically_active: bool = False
    unique_human: bool = False


@dataclass(frozen=True)
class MetaboliteMISTResult:
    """MIST classification result for one metabolite."""

    name: str
    auc_met_mg_h_L: float
    auc_parent_mg_h_L: float
    met_to_parent_ratio: float
    exceeds_10pct: bool
    classification: str  # "disproportionate", "covered", "unique_human"
    risk_level: str  # "low", "moderate", "high"
    recommendation: str
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MISTReport:
    """Aggregated MIST assessment for a drug and its metabolites."""

    drug_name: str
    parent_auc_mg_h_L: float
    parent_cmax_mg_L: float
    metabolite_results: list[MetaboliteMISTResult]
    n_flagged: int
    overall_mist_risk: str  # "low", "moderate", "high"
    summary: str
    recommendations: list[str]
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def estimate_metabolite_auc(
    parent_auc: float,
    parent_mw: float,
    met_spec: MetaboliteSpec,
    parent_cl: float,
) -> tuple[float, list[str]]:
    """Estimate metabolite AUC using formation-clearance model.

    AUC_met = fm × AUC_parent × (CL_parent / CL_met) × (MW_met / MW_parent)

    Returns:
        Tuple of (auc_met, warnings).
    """
    warnings: list[str] = []

    cl_met = met_spec.clint_met_L_per_h
    if cl_met < _CL_MET_FLOOR:
        warnings.append(
            f"Metabolite '{met_spec.name}' CL_met ({cl_met:.2e} L/h) near zero; "
            f"clamped to {_CL_MET_FLOOR:.1e} L/h — AUC may be overestimated"
        )
        cl_met = _CL_MET_FLOOR

    mw_ratio = met_spec.mw / max(parent_mw, 1e-12)
    cl_ratio = parent_cl / cl_met

    auc_met = met_spec.fm_parent * parent_auc * cl_ratio * mw_ratio
    return auc_met, warnings


def classify_metabolite(
    ratio: float,
    active: bool,
    unique_human: bool,
) -> tuple[str, str, str]:
    """Classify a metabolite under MIST guidance.

    Args:
        ratio: metabolite AUC / parent AUC.
        active: pharmacologically active metabolite.
        unique_human: metabolite unique to humans (not in tox species).

    Returns:
        (classification, risk_level, recommendation)
    """
    if unique_human:
        return (
            "unique_human",
            "high",
            "Unique human metabolite — dedicated safety studies required "
            "regardless of exposure ratio (FDA MIST guidance §IV.B)",
        )

    if ratio > MIST_THRESHOLD:
        if active:
            return (
                "disproportionate",
                "high",
                f"Disproportionate active metabolite (>{MIST_THRESHOLD * 100:.0f}% parent AUC) — "
                "dedicated safety and pharmacology studies recommended",
            )
        return (
            "disproportionate",
            "moderate",
            f"Disproportionate metabolite (>{MIST_THRESHOLD * 100:.0f}% parent AUC) — "
            "ensure adequate toxicology coverage in non-clinical species",
        )

    return (
        "covered",
        "low",
        f"Metabolite ≤{MIST_THRESHOLD * 100:.0f}% parent AUC — adequately covered "
        "under standard non-clinical studies",
    )


def run_mist_assessment(
    drug: Drug,
    metabolites: list[MetaboliteSpec],
    dose_mg: float,
    route: str = "oral",
    body_weight: float = 70.0,
    t_end_h: float = 24.0,
    vdss_L_per_kg: float = 0.6,
) -> MISTReport:
    """Run MIST assessment for a drug and its metabolites.

    Uses 1-compartment analytical model for parent PK:
      AUC_parent = F × Dose / CL
      Cmax_parent = F × Dose / Vd  (simplified)

    For IV: F=1, AUC = Dose / CL, Cmax = Dose / Vd.

    Args:
        drug: Drug with PK parameters.
        metabolites: List of MetaboliteSpec objects.
        dose_mg: Administered dose (mg).
        route: 'oral' or 'iv'.
        body_weight: Body weight in kg (for Vd scaling).
        t_end_h: Not used directly; kept for interface consistency.
        vdss_L_per_kg: Volume of distribution at steady state (L/kg).

    Returns:
        MISTReport with per-metabolite classification and overall risk.
    """
    all_warnings: list[str] = []

    # --- Parent PK (1-compartment analytical) ---
    vdss_L = vdss_L_per_kg * body_weight
    cl_parent = drug.clint_scaled_L_per_h + drug.clr_L_per_h
    if cl_parent < 1e-12:
        cl_parent = 0.1  # fallback minimal clearance
        all_warnings.append("Parent CL near zero; using fallback 0.1 L/h")

    if route == "iv":
        f_bioavail = 1.0
    else:
        # Oral bioavailability: F = Fa × Fg × Fh
        # Simplified: Fh = 1 - E_h, E_h = CL_hepatic / Q_h
        q_liver = 90.0  # L/h hepatic blood flow
        cl_hep = drug.clint_scaled_L_per_h
        e_h = cl_hep / (q_liver + cl_hep) if (q_liver + cl_hep) > 0 else 0.0
        f_bioavail = max(1.0 - e_h, 0.01)  # floor at 1%

    parent_auc = f_bioavail * dose_mg / cl_parent
    parent_cmax = f_bioavail * dose_mg / max(vdss_L, 1e-12)

    # --- Metabolite assessment ---
    met_results: list[MetaboliteMISTResult] = []
    recommendations: list[str] = []

    for met in metabolites:
        auc_met, met_warnings = estimate_metabolite_auc(
            parent_auc=parent_auc,
            parent_mw=drug.mw,
            met_spec=met,
            parent_cl=cl_parent,
        )
        all_warnings.extend(met_warnings)

        ratio = auc_met / max(parent_auc, 1e-12)
        exceeds = ratio > MIST_THRESHOLD

        classification, risk_level, recommendation = classify_metabolite(
            ratio=ratio,
            active=met.pharmacologically_active,
            unique_human=met.unique_human,
        )

        met_results.append(
            MetaboliteMISTResult(
                name=met.name,
                auc_met_mg_h_L=round(auc_met, 6),
                auc_parent_mg_h_L=round(parent_auc, 6),
                met_to_parent_ratio=round(ratio, 6),
                exceeds_10pct=exceeds,
                classification=classification,
                risk_level=risk_level,
                recommendation=recommendation,
                warnings=met_warnings,
            )
        )

        if exceeds or met.unique_human:
            recommendations.append(f"{met.name}: {recommendation}")

    # --- Overall risk ---
    n_flagged = sum(1 for r in met_results if r.exceeds_10pct or r.classification == "unique_human")
    risk_levels = [r.risk_level for r in met_results]
    n_high = risk_levels.count("high")
    n_moderate = risk_levels.count("moderate")

    if n_high >= 1:
        overall = "high"
    elif n_moderate >= 1:
        overall = "moderate"
    else:
        overall = "low"

    summary = (
        f"MIST assessment for {drug.name}: {len(metabolites)} metabolite(s) evaluated, "
        f"{n_flagged} flagged (>{MIST_THRESHOLD * 100:.0f}% parent AUC or unique human). "
        f"Overall MIST risk: {overall}."
    )

    return MISTReport(
        drug_name=drug.name,
        parent_auc_mg_h_L=round(parent_auc, 6),
        parent_cmax_mg_L=round(parent_cmax, 6),
        metabolite_results=met_results,
        n_flagged=n_flagged,
        overall_mist_risk=overall,
        summary=summary,
        recommendations=recommendations,
        warnings=all_warnings,
    )


def format_mist_table(report: MISTReport) -> str:
    """Format MIST report as a readable text table.

    Returns:
        Multi-line string with MIST summary and per-metabolite results.
    """
    lines: list[str] = []
    lines.append(f"MIST Assessment — {report.drug_name}")
    lines.append("=" * 70)
    lines.append(f"Parent AUC: {report.parent_auc_mg_h_L:.4f} mg·h/L")
    lines.append(f"Parent Cmax: {report.parent_cmax_mg_L:.4f} mg/L")
    lines.append("")

    # Header
    header = (
        f"{'Metabolite':<20} {'AUC (mg·h/L)':>14} {'Ratio':>8} "
        f"{'Flag':>6} {'Risk':>10} {'Classification':<20}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in report.metabolite_results:
        flag = "YES" if r.exceeds_10pct else "no"
        lines.append(
            f"{r.name:<20} {r.auc_met_mg_h_L:>14.4f} {r.met_to_parent_ratio:>8.4f} "
            f"{flag:>6} {r.risk_level:>10} {r.classification:<20}"
        )

    lines.append("")
    lines.append(f"Flagged: {report.n_flagged}/{len(report.metabolite_results)}")
    lines.append(f"Overall MIST risk: {report.overall_mist_risk}")

    if report.recommendations:
        lines.append("")
        lines.append("Recommendations:")
        for rec in report.recommendations:
            lines.append(f"  • {rec}")

    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in report.warnings:
            lines.append(f"  ⚠ {w}")

    return "\n".join(lines)


__all__ = [
    "MIST_THRESHOLD",
    "MetaboliteSpec",
    "MetaboliteMISTResult",
    "MISTReport",
    "estimate_metabolite_auc",
    "classify_metabolite",
    "run_mist_assessment",
    "format_mist_table",
]
