"""Pharmacogenomics (PGx) — CYP genotype-based dose individualization."""

from __future__ import annotations

from dataclasses import dataclass


# Metabolizer phenotype activity scores
_PHENOTYPE_ACTIVITY = {
    "UM": 2.0,   # Ultra-rapid metabolizer
    "EM": 1.0,   # Extensive (normal) metabolizer
    "IM": 0.5,   # Intermediate metabolizer
    "PM": 0.1,   # Poor metabolizer
}

# CYP allele → activity score mapping (simplified)
_ALLELE_ACTIVITY: dict[str, dict[str, float]] = {
    "CYP2D6": {
        "*1": 1.0, "*2": 1.0, "*3": 0.0, "*4": 0.0,
        "*5": 0.0, "*6": 0.0, "*9": 0.5, "*10": 0.25,
        "*17": 0.5, "*41": 0.5, "*1xN": 2.0, "*2xN": 2.0,
    },
    "CYP2C19": {
        "*1": 1.0, "*2": 0.0, "*3": 0.0, "*17": 1.5,
        "*1xN": 2.0,
    },
    "CYP2C9": {
        "*1": 1.0, "*2": 0.5, "*3": 0.25, "*5": 0.0,
        "*6": 0.0,
    },
    "CYP3A4": {
        "*1": 1.0, "*22": 0.5, "*1G": 1.0,
    },
    "CYP3A5": {
        "*1": 1.0, "*3": 0.0, "*6": 0.0, "*7": 0.0,
    },
}


@dataclass(frozen=True)
class PGxResult:
    drug_name: str
    cyp_enzyme: str
    allele1: str
    allele2: str
    activity_score: float
    phenotype: str           # UM / EM / IM / PM
    cl_scaling_factor: float # relative to EM (1.0)
    recommended_dose_mg: float
    standard_dose_mg: float
    dose_adjustment_pct: float  # relative to standard dose
    fm: float                   # fraction metabolized by this CYP
    clinical_note: str


def _allele_to_activity(enzyme: str, allele: str) -> float:
    """Look up activity score for a given enzyme allele."""
    enzyme_map = _ALLELE_ACTIVITY.get(enzyme, {})
    # Check exact match, then prefix match (*1 → *1xN fallback)
    if allele in enzyme_map:
        return enzyme_map[allele]
    # Unknown allele → assume normal (1.0)
    return 1.0


def _activity_to_phenotype(activity_score: float) -> str:
    """Convert total activity score (sum of two alleles) to phenotype."""
    if activity_score == 0.0:
        return "PM"
    elif activity_score < 1.0:
        return "IM"
    elif activity_score <= 2.0:
        return "EM"
    else:
        return "UM"


def predict_pgx_dose(
    drug_name: str,
    cyp_enzyme: str,
    allele1: str,
    allele2: str,
    standard_dose_mg: float,
    fm: float = 0.8,
) -> PGxResult:
    """Predict dosing recommendation based on CYP genotype.

    Parameters
    ----------
    cyp_enzyme : str
        Enzyme (e.g. 'CYP2D6', 'CYP2C19').
    allele1, allele2 : str
        Two diploid alleles (e.g. '*1', '*4').
    standard_dose_mg : float
        Standard dose for EM patients.
    fm : float
        Fraction of clearance mediated by this enzyme.
    """
    if standard_dose_mg <= 0:
        raise ValueError("standard_dose_mg must be > 0")
    if not 0 < fm <= 1.0:
        raise ValueError("fm must be in (0, 1]")

    a1 = _allele_to_activity(cyp_enzyme, allele1)
    a2 = _allele_to_activity(cyp_enzyme, allele2)
    activity_score = a1 + a2

    phenotype = _activity_to_phenotype(activity_score)

    # CL scaling: CL_patient / CL_EM
    # CL = fm * CL_enzymatic + (1-fm) * CL_other
    # CL_enzymatic proportional to activity score / 2 (EM has score 2)
    em_activity = 2.0  # reference (EM: *1/*1)
    activity_ratio = activity_score / em_activity
    cl_scale = fm * activity_ratio + (1.0 - fm) * 1.0
    cl_scale = max(cl_scale, 0.05)  # floor: even PM has some basal CL

    # Dose adjustment: to maintain same AUC, dose ∝ CL
    recommended_dose = standard_dose_mg * cl_scale
    adjustment_pct = 100.0 * (recommended_dose / standard_dose_mg - 1.0)

    # Clinical note
    notes = {
        "PM": f"Poor metabolizer: reduce dose by ~{abs(adjustment_pct):.0f}% or consider alternative drug",
        "IM": f"Intermediate metabolizer: consider {abs(adjustment_pct):.0f}% dose reduction",
        "EM": "Normal metabolizer: standard dose appropriate",
        "UM": f"Ultra-rapid metabolizer: consider {adjustment_pct:.0f}% dose increase or alternative",
    }
    note = notes.get(phenotype, "Monitor drug levels closely")

    return PGxResult(
        drug_name=drug_name,
        cyp_enzyme=cyp_enzyme,
        allele1=allele1,
        allele2=allele2,
        activity_score=activity_score,
        phenotype=phenotype,
        cl_scaling_factor=cl_scale,
        recommended_dose_mg=recommended_dose,
        standard_dose_mg=standard_dose_mg,
        dose_adjustment_pct=adjustment_pct,
        fm=fm,
        clinical_note=note,
    )


def batch_pgx_screen(patients: list[dict]) -> list[PGxResult]:
    """Screen multiple patients with genotype data.

    Each patient dict: {drug_name, cyp_enzyme, allele1, allele2, standard_dose_mg, fm}
    """
    results = []
    for p in patients:
        r = predict_pgx_dose(
            drug_name=p.get("drug_name", "Drug"),
            cyp_enzyme=p.get("cyp_enzyme", "CYP2D6"),
            allele1=p.get("allele1", "*1"),
            allele2=p.get("allele2", "*1"),
            standard_dose_mg=p.get("standard_dose_mg", 100.0),
            fm=p.get("fm", 0.8),
        )
        results.append(r)
    return results


def cyp_phenotype_distribution(
    enzyme: str = "CYP2D6",
    population: str = "caucasian",
) -> dict[str, float]:
    """Return approximate phenotype frequencies for a population."""
    _distributions = {
        "CYP2D6": {
            "caucasian": {"UM": 0.05, "EM": 0.77, "IM": 0.12, "PM": 0.06},
            "asian": {"UM": 0.01, "EM": 0.51, "IM": 0.45, "PM": 0.03},
            "african": {"UM": 0.03, "EM": 0.73, "IM": 0.20, "PM": 0.04},
        },
        "CYP2C19": {
            "caucasian": {"UM": 0.02, "EM": 0.65, "IM": 0.25, "PM": 0.08},
            "asian": {"UM": 0.01, "EM": 0.31, "IM": 0.35, "PM": 0.33},
            "african": {"UM": 0.03, "EM": 0.54, "IM": 0.32, "PM": 0.11},
        },
    }
    default = {"UM": 0.05, "EM": 0.75, "IM": 0.15, "PM": 0.05}
    return _distributions.get(enzyme, {}).get(population, default)


__all__ = [
    "PGxResult",
    "predict_pgx_dose",
    "batch_pgx_screen",
    "cyp_phenotype_distribution",
]
