"""Phase 601 — Reference therapeutic drug levels for common drugs.

Provides lookup, comparison, and safety margin calculation against
established therapeutic ranges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DrugLevelReference:
    drug_name: str
    drug_class: str
    therapeutic_low_mg_L: float
    therapeutic_high_mg_L: float
    toxic_threshold_mg_L: float
    sampling_time: str  # e.g. "trough", "peak", "2h post-dose"
    unit_note: str


@dataclass(frozen=True)
class TherapeuticLevelResult:
    drug_name: str
    drug_class: str
    measured_conc_mg_L: float
    therapeutic_low_mg_L: float
    therapeutic_high_mg_L: float
    toxic_threshold_mg_L: float
    status: str  # "subtherapeutic","therapeutic","supratherapeutic","toxic"
    safety_margin: float  # (toxic_threshold - measured) / measured
    therapeutic_index: float  # toxic_threshold / therapeutic_high
    recommendation: str
    notes: str


# Built-in reference database — drug_name_lower → DrugLevelReference
DRUG_LEVEL_DATABASE: dict[str, DrugLevelReference] = {
    "vancomycin": DrugLevelReference(
        drug_name="vancomycin",
        drug_class="antibiotic",
        therapeutic_low_mg_L=10.0,
        therapeutic_high_mg_L=20.0,
        toxic_threshold_mg_L=40.0,
        sampling_time="trough",
        unit_note="mg/L; AUC/MIC-guided dosing increasingly preferred",
    ),
    "digoxin": DrugLevelReference(
        drug_name="digoxin",
        drug_class="cardiac_glycoside",
        therapeutic_low_mg_L=0.0008,
        therapeutic_high_mg_L=0.002,
        toxic_threshold_mg_L=0.003,
        sampling_time="6-8h post-dose",
        unit_note="mg/L (0.8–2 ng/mL); narrow therapeutic index",
    ),
    "phenytoin": DrugLevelReference(
        drug_name="phenytoin",
        drug_class="anticonvulsant",
        therapeutic_low_mg_L=10.0,
        therapeutic_high_mg_L=20.0,
        toxic_threshold_mg_L=30.0,
        sampling_time="trough",
        unit_note="mg/L (10–20 mcg/mL); free phenytoin 1–2 mcg/mL",
    ),
    "carbamazepine": DrugLevelReference(
        drug_name="carbamazepine",
        drug_class="anticonvulsant",
        therapeutic_low_mg_L=4.0,
        therapeutic_high_mg_L=12.0,
        toxic_threshold_mg_L=15.0,
        sampling_time="trough",
        unit_note="mg/L (4–12 mcg/mL)",
    ),
    "valproic_acid": DrugLevelReference(
        drug_name="valproic_acid",
        drug_class="anticonvulsant",
        therapeutic_low_mg_L=50.0,
        therapeutic_high_mg_L=100.0,
        toxic_threshold_mg_L=150.0,
        sampling_time="trough",
        unit_note="mg/L (50–100 mcg/mL)",
    ),
    "lithium": DrugLevelReference(
        drug_name="lithium",
        drug_class="mood_stabilizer",
        therapeutic_low_mg_L=4.16,
        therapeutic_high_mg_L=8.33,
        toxic_threshold_mg_L=13.88,
        sampling_time="12h post-dose",
        unit_note="mg/L proxy (1 mmol/L = 6.94 mg/L); clinical range 0.6–1.2 mmol/L",
    ),
    "cyclosporine": DrugLevelReference(
        drug_name="cyclosporine",
        drug_class="immunosuppressant",
        therapeutic_low_mg_L=0.1,
        therapeutic_high_mg_L=0.4,
        toxic_threshold_mg_L=0.8,
        sampling_time="trough",
        unit_note="mg/L (100–400 ng/mL); target varies by indication",
    ),
    "tacrolimus": DrugLevelReference(
        drug_name="tacrolimus",
        drug_class="immunosuppressant",
        therapeutic_low_mg_L=0.005,
        therapeutic_high_mg_L=0.020,
        toxic_threshold_mg_L=0.030,
        sampling_time="trough",
        unit_note="mg/L (5–20 ng/mL); target varies by transplant type",
    ),
    "theophylline": DrugLevelReference(
        drug_name="theophylline",
        drug_class="bronchodilator",
        therapeutic_low_mg_L=10.0,
        therapeutic_high_mg_L=20.0,
        toxic_threshold_mg_L=25.0,
        sampling_time="2h post-dose",
        unit_note="mg/L (10–20 mcg/mL)",
    ),
    "amikacin": DrugLevelReference(
        drug_name="amikacin",
        drug_class="antibiotic",
        therapeutic_low_mg_L=20.0,
        therapeutic_high_mg_L=30.0,
        toxic_threshold_mg_L=40.0,
        sampling_time="peak",
        unit_note="mg/L peak; trough <8 mg/L to avoid nephrotoxicity",
    ),
    "gentamicin": DrugLevelReference(
        drug_name="gentamicin",
        drug_class="antibiotic",
        therapeutic_low_mg_L=5.0,
        therapeutic_high_mg_L=10.0,
        toxic_threshold_mg_L=12.0,
        sampling_time="peak",
        unit_note="mg/L peak; trough <2 mg/L to avoid nephrotoxicity",
    ),
    "methotrexate": DrugLevelReference(
        drug_name="methotrexate",
        drug_class="antimetabolite",
        therapeutic_low_mg_L=0.0,
        therapeutic_high_mg_L=0.0001,
        toxic_threshold_mg_L=0.001,
        sampling_time="24h post-dose",
        unit_note="mg/L; high-dose protocols require leucovorin rescue",
    ),
    "sirolimus": DrugLevelReference(
        drug_name="sirolimus",
        drug_class="immunosuppressant",
        therapeutic_low_mg_L=0.004,
        therapeutic_high_mg_L=0.012,
        toxic_threshold_mg_L=0.020,
        sampling_time="trough",
        unit_note="mg/L (4–12 ng/mL)",
    ),
    "phenobarbital": DrugLevelReference(
        drug_name="phenobarbital",
        drug_class="anticonvulsant",
        therapeutic_low_mg_L=15.0,
        therapeutic_high_mg_L=40.0,
        toxic_threshold_mg_L=55.0,
        sampling_time="trough",
        unit_note="mg/L (15–40 mcg/mL)",
    ),
    "ibuprofen": DrugLevelReference(
        drug_name="ibuprofen",
        drug_class="nsaid",
        therapeutic_low_mg_L=10.0,
        therapeutic_high_mg_L=50.0,
        toxic_threshold_mg_L=200.0,
        sampling_time="1-2h post-dose",
        unit_note="mg/L; routine TDM not standard; used in overdose assessment",
    ),
}


def lookup_drug_level(drug_name: str) -> DrugLevelReference:
    """Look up a drug in the reference database (case-insensitive)."""
    key = drug_name.strip().lower().replace(" ", "_")
    if key not in DRUG_LEVEL_DATABASE:
        available = ", ".join(sorted(DRUG_LEVEL_DATABASE.keys()))
        raise ValueError(
            f"Drug '{drug_name}' not found in reference database. Available drugs: {available}"
        )
    return DRUG_LEVEL_DATABASE[key]


def evaluate_drug_level(
    drug_name: str,
    measured_conc_mg_L: float,
    drug_class: str | None = None,
    therapeutic_low: float | None = None,
    therapeutic_high: float | None = None,
    toxic_threshold: float | None = None,
) -> TherapeuticLevelResult:
    """Evaluate measured concentration against therapeutic range."""
    ref = lookup_drug_level(drug_name)

    # Apply overrides
    effective_class = drug_class if drug_class is not None else ref.drug_class
    effective_low = therapeutic_low if therapeutic_low is not None else ref.therapeutic_low_mg_L
    effective_high = therapeutic_high if therapeutic_high is not None else ref.therapeutic_high_mg_L
    effective_toxic = toxic_threshold if toxic_threshold is not None else ref.toxic_threshold_mg_L

    # Determine status
    if measured_conc_mg_L < effective_low:
        status = "subtherapeutic"
    elif measured_conc_mg_L <= effective_high:
        status = "therapeutic"
    elif measured_conc_mg_L < effective_toxic:
        status = "supratherapeutic"
    else:
        status = "toxic"

    # Safety margin: (toxic - measured) / measured
    if measured_conc_mg_L > 0:
        safety_margin = (effective_toxic - measured_conc_mg_L) / measured_conc_mg_L
    else:
        safety_margin = math.inf

    # Therapeutic index: toxic / high
    if effective_high > 0:
        therapeutic_index = effective_toxic / effective_high
    else:
        therapeutic_index = math.inf

    # Recommendation
    if status == "subtherapeutic":
        recommendation = (
            f"Concentration below therapeutic range ({effective_low}–{effective_high} mg/L). "
            "Consider increasing dose or checking adherence."
        )
    elif status == "therapeutic":
        recommendation = (
            f"Concentration within therapeutic range ({effective_low}–{effective_high} mg/L). "
            "Continue current regimen and monitor."
        )
    elif status == "supratherapeutic":
        recommendation = (
            f"Concentration above therapeutic range but below toxic threshold ({effective_toxic} mg/L). "
            "Consider dose reduction and increase monitoring frequency."
        )
    else:
        recommendation = (
            f"Concentration at or above toxic threshold ({effective_toxic} mg/L). "
            "Hold dose, manage toxicity, and reassess regimen."
        )

    notes = (
        f"Sampling time: {ref.sampling_time}. {ref.unit_note}. "
        f"TI={therapeutic_index:.2f}; safety margin={safety_margin:.2f}."
    )

    return TherapeuticLevelResult(
        drug_name=ref.drug_name,
        drug_class=effective_class,
        measured_conc_mg_L=measured_conc_mg_L,
        therapeutic_low_mg_L=effective_low,
        therapeutic_high_mg_L=effective_high,
        toxic_threshold_mg_L=effective_toxic,
        status=status,
        safety_margin=safety_margin,
        therapeutic_index=therapeutic_index,
        recommendation=recommendation,
        notes=notes,
    )


def list_available_drugs() -> list:
    """Return sorted list of drug names in the database."""
    return sorted(DRUG_LEVEL_DATABASE.keys())
