"""Metabolic pathway reaction phenotyping — CYP contribution analysis.

Determines the fractional contribution (fm) of individual CYP enzymes
to overall drug metabolism using selective chemical inhibition and/or
recombinant CYP (rCYP) data with ISEF/RAF scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SELECTIVE_INHIBITORS: dict[str, dict[str, float]] = {
    "CYP1A2": {"inhibitor": "furafylline", "conc_uM": 10.0, "ki_uM": 0.6},
    "CYP2B6": {"inhibitor": "ticlopidine", "conc_uM": 1.0, "ki_uM": 0.2},
    "CYP2C8": {"inhibitor": "montelukast", "conc_uM": 1.0, "ki_uM": 0.019},
    "CYP2C9": {"inhibitor": "sulfaphenazole", "conc_uM": 10.0, "ki_uM": 0.3},
    "CYP2C19": {"inhibitor": "N-3-benzylnirvanol", "conc_uM": 2.0, "ki_uM": 0.22},
    "CYP2D6": {"inhibitor": "quinidine", "conc_uM": 1.0, "ki_uM": 0.06},
    "CYP3A4": {"inhibitor": "ketoconazole", "conc_uM": 1.0, "ki_uM": 0.015},
}

DEFAULT_ISEF: dict[str, float] = {
    "CYP1A2": 8.1,
    "CYP2B6": 22.0,
    "CYP2C8": 5.7,
    "CYP2C9": 4.3,
    "CYP2C19": 6.5,
    "CYP2D6": 4.8,
    "CYP3A4": 1.0,
}

DEFAULT_ABUNDANCE: dict[str, float] = {
    "CYP1A2": 45.0,
    "CYP2B6": 11.0,
    "CYP2C8": 24.0,
    "CYP2C9": 73.0,
    "CYP2C19": 14.0,
    "CYP2D6": 8.0,
    "CYP3A4": 137.0,
}

POLYMORPHIC_ENZYMES: set[str] = {"CYP2D6", "CYP2C19", "CYP2C9"}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InhibitionExperiment:
    """Single selective inhibition experiment result."""

    enzyme: str
    inhibitor_name: str
    control_clint: float
    inhibited_clint: float
    inhibitor_conc_uM: float | None = None
    ki_uM: float | None = None


@dataclass(frozen=True)
class RecombinantCYPData:
    """Recombinant CYP velocity data for a single enzyme."""

    enzyme: str
    velocity_pmol_min_pmol: float
    isef: float | None = None
    raf: float | None = None
    abundance_pmol_mg: float | None = None


@dataclass(frozen=True)
class CYPContribution:
    """Fractional contribution of one CYP enzyme."""

    enzyme: str
    fm: float
    clint_enzyme_uL_min_mg: float
    method: str
    confidence: str


@dataclass(frozen=True)
class PhenotypingResult:
    """Complete reaction phenotyping result."""

    drug_name: str
    total_clint_uL_min_mg: float
    contributions: list[CYPContribution]
    fm_dict: dict[str, float]
    primary_enzyme: str
    fm_primary: float
    ddi_vulnerability: str
    pgx_sensitivity: str
    method: str
    warnings: list[str] = field(default_factory=list)
    recommendation: str = ""


# ---------------------------------------------------------------------------
# 1. Inhibition-based phenotyping
# ---------------------------------------------------------------------------


def phenotype_by_inhibition(
    experiments: list[InhibitionExperiment],
) -> list[CYPContribution]:
    """Determine CYP contributions from selective chemical inhibition data.

    fm = (control_clint - inhibited_clint) / control_clint
    With optional Ki correction when inhibitor_conc_uM and ki_uM are provided.
    """
    contributions: list[CYPContribution] = []
    warnings_list: list[str] = []

    for exp in experiments:
        control = max(exp.control_clint, 1e-12)
        fm_apparent = (control - exp.inhibited_clint) / control
        fm_apparent = max(fm_apparent, 0.0)

        # Ki correction: adjust for incomplete inhibition
        if exp.inhibitor_conc_uM is not None and exp.ki_uM is not None:
            inh_conc = exp.inhibitor_conc_uM
            ki = exp.ki_uM
            fractional_inhibition = inh_conc / (inh_conc + ki)
            fractional_inhibition = max(fractional_inhibition, 1e-12)
            fm_corrected = fm_apparent / fractional_inhibition
        else:
            fm_corrected = fm_apparent

        fm_corrected = min(fm_corrected, 1.0)
        clint_enzyme = fm_corrected * control

        contributions.append(
            CYPContribution(
                enzyme=exp.enzyme,
                fm=round(fm_corrected, 4),
                clint_enzyme_uL_min_mg=round(clint_enzyme, 4),
                method="inhibition",
                confidence="high" if fm_corrected >= 0.25 else "moderate",
            )
        )

    # Normalize if sum > 1.0
    fm_sum = sum(c.fm for c in contributions)
    if fm_sum > 1.0:
        warnings_list.append(f"fm sum ({fm_sum:.3f}) exceeded 1.0; normalized")
        contributions = [
            CYPContribution(
                enzyme=c.enzyme,
                fm=round(c.fm / fm_sum, 4),
                clint_enzyme_uL_min_mg=c.clint_enzyme_uL_min_mg,
                method=c.method,
                confidence=c.confidence,
            )
            for c in contributions
        ]

    return contributions


# ---------------------------------------------------------------------------
# 2. Recombinant CYP (rCYP) phenotyping
# ---------------------------------------------------------------------------


def phenotype_by_recombinant(
    data: list[RecombinantCYPData],
) -> list[CYPContribution]:
    """Determine CYP contributions from recombinant enzyme velocity data.

    ISEF method: CLint_enzyme = velocity * abundance * ISEF
    RAF method:  CLint_enzyme = velocity * RAF
    """
    clint_values: list[tuple[str, float, str]] = []

    for d in data:
        if d.raf is not None and d.raf > 0:
            clint_enzyme = d.velocity_pmol_min_pmol * d.raf
            method = "raf"
        else:
            isef = d.isef if d.isef is not None else DEFAULT_ISEF.get(d.enzyme, 1.0)
            abundance = (
                d.abundance_pmol_mg
                if d.abundance_pmol_mg is not None
                else DEFAULT_ABUNDANCE.get(d.enzyme, 50.0)
            )
            clint_enzyme = d.velocity_pmol_min_pmol * abundance * isef
            method = "isef"

        clint_values.append((d.enzyme, clint_enzyme, method))

    total = sum(v for _, v, _ in clint_values)
    total = max(total, 1e-12)

    contributions: list[CYPContribution] = []
    for enzyme, clint_enzyme, method in clint_values:
        fm = clint_enzyme / total
        contributions.append(
            CYPContribution(
                enzyme=enzyme,
                fm=round(fm, 4),
                clint_enzyme_uL_min_mg=round(clint_enzyme, 4),
                method=f"recombinant_{method}",
                confidence="moderate",
            )
        )

    return contributions


# ---------------------------------------------------------------------------
# 3. Combined consensus phenotyping
# ---------------------------------------------------------------------------


def phenotype_combined(
    inhibition: list[InhibitionExperiment],
    recombinant: list[RecombinantCYPData],
    inhibition_weight: float = 0.7,
) -> list[CYPContribution]:
    """Weighted consensus of inhibition and recombinant methods.

    Inhibition data takes priority (default 70% weight).
    """
    inh_results = phenotype_by_inhibition(inhibition)
    rec_results = phenotype_by_recombinant(recombinant)

    inh_fm: dict[str, float] = {c.enzyme: c.fm for c in inh_results}
    rec_fm: dict[str, float] = {c.enzyme: c.fm for c in rec_results}
    inh_clint: dict[str, float] = {c.enzyme: c.clint_enzyme_uL_min_mg for c in inh_results}
    rec_clint: dict[str, float] = {c.enzyme: c.clint_enzyme_uL_min_mg for c in rec_results}

    all_enzymes = set(inh_fm.keys()) | set(rec_fm.keys())
    rec_weight = 1.0 - inhibition_weight

    contributions: list[CYPContribution] = []
    for enzyme in sorted(all_enzymes):
        fm_i = inh_fm.get(enzyme, 0.0)
        fm_r = rec_fm.get(enzyme, 0.0)
        fm_combined = inhibition_weight * fm_i + rec_weight * fm_r

        clint_i = inh_clint.get(enzyme, 0.0)
        clint_r = rec_clint.get(enzyme, 0.0)
        clint_combined = inhibition_weight * clint_i + rec_weight * clint_r

        contributions.append(
            CYPContribution(
                enzyme=enzyme,
                fm=round(fm_combined, 4),
                clint_enzyme_uL_min_mg=round(clint_combined, 4),
                method="combined",
                confidence="high" if fm_combined >= 0.25 else "moderate",
            )
        )

    # Normalize
    fm_sum = sum(c.fm for c in contributions)
    if fm_sum > 0 and abs(fm_sum - 1.0) > 0.001:
        contributions = [
            CYPContribution(
                enzyme=c.enzyme,
                fm=round(c.fm / max(fm_sum, 1e-12), 4),
                clint_enzyme_uL_min_mg=c.clint_enzyme_uL_min_mg,
                method=c.method,
                confidence=c.confidence,
            )
            for c in contributions
        ]

    return contributions


# ---------------------------------------------------------------------------
# 4. DDI vulnerability assessment
# ---------------------------------------------------------------------------


def assess_ddi_vulnerability(fm_dict: dict[str, float]) -> str:
    """Assess DDI vulnerability based on fm values.

    Returns 'high' if any fm >= 0.8, 'moderate' if >= 0.5, 'low' otherwise.
    """
    if not fm_dict:
        return "low"
    max_fm = max(fm_dict.values())
    if max_fm >= 0.8:
        return "high"
    if max_fm >= 0.5:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# 5. PGx sensitivity assessment
# ---------------------------------------------------------------------------


def assess_pgx_sensitivity(fm_dict: dict[str, float]) -> str:
    """Assess pharmacogenomic sensitivity.

    Checks polymorphic enzymes (CYP2D6, CYP2C19, CYP2C9).
    Returns 'high' if any polymorphic fm >= 0.5, 'moderate' if >= 0.25,
    'low' otherwise.
    """
    polymorphic_fm = {e: fm for e, fm in fm_dict.items() if e in POLYMORPHIC_ENZYMES}
    if not polymorphic_fm:
        return "low"
    max_poly_fm = max(polymorphic_fm.values())
    if max_poly_fm >= 0.5:
        return "high"
    if max_poly_fm >= 0.25:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# 6. Main entry point
# ---------------------------------------------------------------------------


def run_reaction_phenotyping(
    drug_name: str,
    inhibition_experiments: list[InhibitionExperiment] | None = None,
    recombinant_data: list[RecombinantCYPData] | None = None,
    method: str = "auto",
) -> PhenotypingResult:
    """Run complete reaction phenotyping analysis.

    Args:
        drug_name: Name of the drug compound.
        inhibition_experiments: Selective inhibition data.
        recombinant_data: Recombinant CYP velocity data.
        method: 'inhibition', 'recombinant', 'combined', or 'auto'.

    Returns:
        PhenotypingResult with CYP contributions and risk assessments.
    """
    warnings: list[str] = []
    has_inhibition = inhibition_experiments is not None and len(inhibition_experiments) > 0
    has_recombinant = recombinant_data is not None and len(recombinant_data) > 0

    # Determine method
    if method == "auto":
        if has_inhibition and has_recombinant:
            method = "combined"
        elif has_inhibition:
            method = "inhibition"
        elif has_recombinant:
            method = "recombinant"
        else:
            warnings.append("No experimental data provided")
            return PhenotypingResult(
                drug_name=drug_name,
                total_clint_uL_min_mg=0.0,
                contributions=[],
                fm_dict={},
                primary_enzyme="unknown",
                fm_primary=0.0,
                ddi_vulnerability="unknown",
                pgx_sensitivity="unknown",
                method="none",
                warnings=warnings,
                recommendation="Provide inhibition or recombinant CYP data",
            )

    # Run phenotyping
    if method == "combined" and has_inhibition and has_recombinant:
        contributions = phenotype_combined(
            inhibition_experiments,
            recombinant_data,  # type: ignore[arg-type]
        )
    elif method == "inhibition" and has_inhibition:
        contributions = phenotype_by_inhibition(inhibition_experiments)  # type: ignore[arg-type]
    elif method == "recombinant" and has_recombinant:
        contributions = phenotype_by_recombinant(recombinant_data)  # type: ignore[arg-type]
    else:
        warnings.append(f"Requested method '{method}' but data not available; using available data")
        if has_inhibition:
            contributions = phenotype_by_inhibition(inhibition_experiments)  # type: ignore[arg-type]
            method = "inhibition"
        elif has_recombinant:
            contributions = phenotype_by_recombinant(recombinant_data)  # type: ignore[arg-type]
            method = "recombinant"
        else:
            contributions = []

    # Build fm_dict
    fm_dict = {c.enzyme: c.fm for c in contributions}
    total_clint = sum(c.clint_enzyme_uL_min_mg for c in contributions)

    # Flag unaccounted fraction
    fm_sum = sum(fm_dict.values())
    if fm_sum < 0.7 and contributions:
        warnings.append(
            f"Only {fm_sum:.1%} of metabolism accounted for; "
            f"consider additional enzymes (UGT, FMO, etc.)"
        )

    # Primary enzyme
    if fm_dict:
        primary_enzyme = max(fm_dict, key=fm_dict.get)  # type: ignore[arg-type]
        fm_primary = fm_dict[primary_enzyme]
    else:
        primary_enzyme = "unknown"
        fm_primary = 0.0

    # Assessments
    ddi_vuln = assess_ddi_vulnerability(fm_dict)
    pgx_sens = assess_pgx_sensitivity(fm_dict)

    # Recommendation
    recommendations: list[str] = []
    if ddi_vuln == "high":
        recommendations.append(
            f"High DDI risk via {primary_enzyme} (fm={fm_primary:.2f}); "
            f"conduct clinical DDI studies with strong inhibitors/inducers"
        )
    if pgx_sens == "high":
        poly_enzymes = [e for e in fm_dict if e in POLYMORPHIC_ENZYMES and fm_dict[e] >= 0.5]
        recommendations.append(
            f"High PGx sensitivity via {', '.join(poly_enzymes)}; consider genotype-guided dosing"
        )
    if ddi_vuln == "low" and pgx_sens == "low":
        recommendations.append("Low DDI and PGx risk; standard development pathway")

    return PhenotypingResult(
        drug_name=drug_name,
        total_clint_uL_min_mg=round(total_clint, 4),
        contributions=contributions,
        fm_dict=fm_dict,
        primary_enzyme=primary_enzyme,
        fm_primary=round(fm_primary, 4),
        ddi_vulnerability=ddi_vuln,
        pgx_sensitivity=pgx_sens,
        method=method,
        warnings=warnings,
        recommendation="; ".join(recommendations),
    )


# ---------------------------------------------------------------------------
# 7. Utility: convert to Drug.fm format
# ---------------------------------------------------------------------------


def phenotyping_to_drug_fm(result: PhenotypingResult) -> dict[str, float]:
    """Convert phenotyping result to Drug.fm format."""
    return {enzyme: fm for enzyme, fm in result.fm_dict.items() if fm > 0.0}
