"""Phase 1089 — Pharmacogenomics-Driven Dose Personalization.

Integrates PGx genotype + phenotype data to personalize drug dosing
for 6 drugs with drug-specific dosing tables and CPIC recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Supported drugs and genes
# ---------------------------------------------------------------------------

SUPPORTED_DRUGS = frozenset(
    {"clopidogrel", "warfarin", "codeine", "tamoxifen", "atomoxetine", "mercaptopurine"}
)

VALID_PHENOTYPES: dict[str, tuple[str, ...]] = {
    "CYP2C19": ("PM", "IM", "NM", "RM", "UM"),
    "CYP2D6": ("PM", "IM", "NM", "UM"),
    "CYP2C9": ("PM", "IM", "NM"),
    "TPMT": ("PM", "IM", "NM", "UM"),
}

VALID_GENES = frozenset(VALID_PHENOTYPES.keys())

# Drug → (gene, standard dose mg, cpic_level)
DRUG_GENE_MAP: dict[str, tuple[str, float, str]] = {
    "codeine": ("CYP2D6", 30.0, "A"),
    "clopidogrel": ("CYP2C19", 75.0, "A"),
    "atomoxetine": ("CYP2D6", 40.0, "A"),
    "warfarin": ("CYP2C9", 5.0, "A"),
    "mercaptopurine": ("TPMT", 75.0, "A"),
    "tamoxifen": ("CYP2D6", 20.0, "A"),
}


# ---------------------------------------------------------------------------
# Dose adjustment tables (drug → phenotype → config dict)
# ---------------------------------------------------------------------------

_DOSE_TABLES: dict[str, dict[str, dict]] = {
    "codeine": {
        "PM": {
            "factor": 0.0,
            "efficacy": "absent",
            "toxicity": "low",
            "alternative": "morphine",
            "rec": "Avoid codeine. PM patients cannot convert codeine to morphine; no analgesia.",
            "notes": "CYP2D6 PM: codeine is a prodrug requiring conversion to morphine. Use alternative opioid.",
        },
        "IM": {
            "factor": 0.75,
            "efficacy": "reduced",
            "toxicity": "low",
            "alternative": "",
            "rec": "Use with caution; reduced analgesic effect expected.",
            "notes": "CYP2D6 IM: partial metabolizer, reduced morphine formation.",
        },
        "NM": {
            "factor": 1.0,
            "efficacy": "normal",
            "toxicity": "low",
            "alternative": "",
            "rec": "Standard codeine dose appropriate.",
            "notes": "CYP2D6 NM: normal metabolism.",
        },
        "UM": {
            "factor": 0.0,
            "efficacy": "absent",
            "toxicity": "high",
            "alternative": "morphine",
            "rec": "Contraindicated. UM patients rapidly convert codeine to morphine causing toxicity.",
            "notes": "CYP2D6 UM: ultra-rapid conversion to morphine may cause respiratory depression.",
        },
    },
    "clopidogrel": {
        "PM": {
            "factor": 0.0,
            "efficacy": "absent",
            "toxicity": "low",
            "alternative": "ticagrelor",
            "rec": "Avoid clopidogrel. Switch to ticagrelor or prasugrel (non-CYP2C19 dependent).",
            "notes": "CYP2C19 PM: inadequate conversion to active metabolite; high risk of cardiovascular events.",
        },
        "IM": {
            "factor": 0.75,
            "efficacy": "reduced",
            "toxicity": "low",
            "alternative": "",
            "rec": "Consider alternative antiplatelet therapy. Reduced platelet inhibition expected.",
            "notes": "CYP2C19 IM: reduced activation of clopidogrel.",
        },
        "NM": {
            "factor": 1.0,
            "efficacy": "normal",
            "toxicity": "low",
            "alternative": "",
            "rec": "Standard clopidogrel dose appropriate.",
            "notes": "CYP2C19 NM: normal activation.",
        },
        "RM": {
            "factor": 1.0,
            "efficacy": "normal",
            "toxicity": "low",
            "alternative": "",
            "rec": "Standard clopidogrel dose appropriate.",
            "notes": "CYP2C19 RM: slightly increased activation; standard dose adequate.",
        },
        "UM": {
            "factor": 1.0,
            "efficacy": "normal",
            "toxicity": "low",
            "alternative": "",
            "rec": "Standard clopidogrel dose appropriate. Clopidogrel is an active drug.",
            "notes": "CYP2C19 UM: rapid metabolism; standard dose sufficient for active drug.",
        },
    },
    "atomoxetine": {
        "PM": {
            "factor": 0.5,
            "efficacy": "normal",
            "toxicity": "high",
            "alternative": "",
            "rec": "Reduce dose by 50%. PM patients have ~10x higher AUC; monitor for adverse effects.",
            "notes": "CYP2D6 PM: markedly elevated plasma levels; titrate carefully.",
        },
        "IM": {
            "factor": 0.75,
            "efficacy": "normal",
            "toxicity": "moderate",
            "alternative": "",
            "rec": "Consider dose reduction. Monitor for cardiovascular adverse effects.",
            "notes": "CYP2D6 IM: elevated exposure relative to NM.",
        },
        "NM": {
            "factor": 1.0,
            "efficacy": "normal",
            "toxicity": "low",
            "alternative": "",
            "rec": "Standard atomoxetine dose appropriate.",
            "notes": "CYP2D6 NM: normal metabolism.",
        },
        "UM": {
            "factor": 1.25,
            "efficacy": "reduced",
            "toxicity": "low",
            "alternative": "",
            "rec": "May require higher dose. Monitor for reduced efficacy.",
            "notes": "CYP2D6 UM: rapid elimination may reduce efficacy at standard doses.",
        },
    },
    "warfarin": {
        "PM": {
            "factor": 0.3,
            "efficacy": "normal",
            "toxicity": "high",
            "alternative": "",
            "rec": "Reduce dose by ~70%. CYP2C9 PM has significantly impaired warfarin clearance.",
            "notes": "CYP2C9 PM (*3/*3 or similar): greatly prolonged t1/2; high bleeding risk.",
        },
        "IM": {
            "factor": 0.6,
            "efficacy": "normal",
            "toxicity": "moderate",
            "alternative": "",
            "rec": "Reduce dose by ~40%. Monitor INR closely during initiation.",
            "notes": "CYP2C9 IM (*1/*3 or *2/*2): moderately impaired clearance.",
        },
        "NM": {
            "factor": 1.0,
            "efficacy": "normal",
            "toxicity": "low",
            "alternative": "",
            "rec": "Standard warfarin dosing algorithm. Monitor INR.",
            "notes": "CYP2C9 NM (*1/*1): normal warfarin metabolism.",
        },
    },
    "mercaptopurine": {
        "PM": {
            "factor": 0.1,
            "efficacy": "normal",
            "toxicity": "very_high",
            "alternative": "",
            "rec": "Reduce dose to 10% of standard. TPMT PM patients are at very high risk of myelosuppression.",
            "notes": "TPMT PM: thiopurine accumulation causes life-threatening bone marrow suppression.",
        },
        "IM": {
            "factor": 0.5,
            "efficacy": "normal",
            "toxicity": "moderate",
            "alternative": "",
            "rec": "Reduce dose by ~50%. Monitor CBC weekly.",
            "notes": "TPMT IM: intermediate metabolizer; elevated thiopurine levels.",
        },
        "NM": {
            "factor": 1.0,
            "efficacy": "normal",
            "toxicity": "low",
            "alternative": "",
            "rec": "Standard mercaptopurine dose appropriate.",
            "notes": "TPMT NM: normal thiopurine metabolism.",
        },
        "UM": {
            "factor": 1.5,
            "efficacy": "reduced",
            "toxicity": "low",
            "alternative": "",
            "rec": "May need higher dose due to rapid metabolism. Monitor drug levels.",
            "notes": "TPMT UM: rapid metabolism may reduce efficacy.",
        },
    },
    "tamoxifen": {
        "PM": {
            "factor": 0.0,
            "efficacy": "absent",
            "toxicity": "low",
            "alternative": "aromatase_inhibitor",
            "rec": "Avoid tamoxifen. Consider aromatase inhibitor for post-menopausal patients.",
            "notes": "CYP2D6 PM: cannot form endoxifen (active metabolite); no anticancer benefit.",
        },
        "IM": {
            "factor": 0.75,
            "efficacy": "reduced",
            "toxicity": "low",
            "alternative": "",
            "rec": "Use with caution; reduced endoxifen levels expected. Consider higher dose (40 mg).",
            "notes": "CYP2D6 IM: lower endoxifen concentrations; potentially reduced efficacy.",
        },
        "NM": {
            "factor": 1.0,
            "efficacy": "normal",
            "toxicity": "low",
            "alternative": "",
            "rec": "Standard tamoxifen dose appropriate.",
            "notes": "CYP2D6 NM: normal endoxifen formation.",
        },
        "UM": {
            "factor": 1.0,
            "efficacy": "normal",
            "toxicity": "low",
            "alternative": "",
            "rec": "Standard tamoxifen dose appropriate.",
            "notes": "CYP2D6 UM: rapid metabolism but adequate endoxifen levels at standard dose.",
        },
    },
}

VALID_EFFICACY = frozenset({"normal", "reduced", "absent", "enhanced"})
VALID_TOXICITY = frozenset({"low", "moderate", "high", "very_high"})


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PGxDosePersonalizationResult:
    """Result of pharmacogenomics-driven dose personalization."""

    drug_name: str
    gene: str
    phenotype: str
    standard_dose_mg: float
    recommended_dose_mg: float
    dose_adjustment_factor: float
    efficacy_prediction: str
    toxicity_risk: str
    alternative_drug: str
    cpic_level: str
    recommendation: str
    notes: str


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def personalize_dose(
    drug_name: str,
    gene: str,
    phenotype: str,
    standard_dose_mg: float = 100.0,
) -> PGxDosePersonalizationResult:
    """Personalize dose based on PGx genotype/phenotype.

    Parameters
    ----------
    drug_name:
        Drug name (one of the 6 supported drugs).
    gene:
        Pharmacogene (e.g. "CYP2D6", "CYP2C19", "CYP2C9", "TPMT").
    phenotype:
        Metabolizer phenotype for the given gene.
    standard_dose_mg:
        Standard (population) dose in mg.

    Returns
    -------
    PGxDosePersonalizationResult with personalized dose and CPIC guidance.

    Raises
    ------
    ValueError
        If drug_name, gene, or phenotype is invalid.
    """
    if drug_name not in SUPPORTED_DRUGS:
        raise ValueError(
            f"drug_name '{drug_name}' not supported. Supported drugs: {sorted(SUPPORTED_DRUGS)}"
        )
    if gene not in VALID_GENES:
        raise ValueError(f"gene '{gene}' not recognized. Valid genes: {sorted(VALID_GENES)}")
    valid_phenos = VALID_PHENOTYPES[gene]
    if phenotype not in valid_phenos:
        raise ValueError(
            f"phenotype '{phenotype}' not valid for gene '{gene}'. Valid phenotypes: {valid_phenos}"
        )

    # Look up the dose table for this drug
    drug_table = _DOSE_TABLES[drug_name]
    # Get known gene for the drug
    expected_gene, default_std_dose, cpic_level = DRUG_GENE_MAP[drug_name]

    # Phenotype entry (may not exist for all combinations)
    entry = drug_table.get(phenotype)
    if entry is None:
        # Fallback: gene mismatch means phenotype not applicable
        factor = 1.0
        efficacy = "normal"
        toxicity = "low"
        alternative = ""
        rec = f"No specific guidance for {drug_name}/{gene}/{phenotype} combination."
        notes = "Use standard dose. Consult pharmacist."
    else:
        factor = entry["factor"]
        efficacy = entry["efficacy"]
        toxicity = entry["toxicity"]
        alternative = entry["alternative"]
        rec = entry["rec"]
        notes = entry["notes"]

    recommended_dose = round(standard_dose_mg * factor, 4)

    return PGxDosePersonalizationResult(
        drug_name=drug_name,
        gene=gene,
        phenotype=phenotype,
        standard_dose_mg=standard_dose_mg,
        recommended_dose_mg=recommended_dose,
        dose_adjustment_factor=factor,
        efficacy_prediction=efficacy,
        toxicity_risk=toxicity,
        alternative_drug=alternative,
        cpic_level=cpic_level,
        recommendation=rec,
        notes=notes,
    )


def screen_drug_pgx(
    drug_name: str,
    standard_dose_mg: float,
    phenotypes: list[str],
    gene: str,
) -> list[PGxDosePersonalizationResult]:
    """Screen a drug across multiple phenotypes.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    standard_dose_mg:
        Standard dose in mg.
    phenotypes:
        List of phenotype strings to evaluate.
    gene:
        Gene to evaluate against.

    Returns
    -------
    List of PGxDosePersonalizationResult sorted by dose_adjustment_factor ascending
    (lowest recommended dose first).
    """
    results = [personalize_dose(drug_name, gene, ph, standard_dose_mg) for ph in phenotypes]
    results.sort(key=lambda r: r.dose_adjustment_factor)
    return results
