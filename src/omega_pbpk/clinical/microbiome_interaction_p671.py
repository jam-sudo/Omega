"""Drug–microbiome interaction prediction.

Models drug interactions with the gut microbiome including bacterial drug
metabolism, microbiome-mediated activation/inactivation, and bidirectional
effects of drugs on microbiome diversity.

References
----------
- Zimmermann M et al. Nature (2019) — mapping human microbiome drug metabolism
- Maier L et al. Nature (2018) — extensive impact of non-antibiotic drugs on gut bacteria
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_DRUG_CLASSES = {"antibiotic", "NSAID", "PPI", "beta_blocker", "statin", "other"}


@dataclass(frozen=True)
class MicrobiomeInteractionResult:
    """Result of drug–microbiome interaction prediction."""

    drug_name: str
    gut_bacteria_class: str
    bacterial_metabolism_fraction: float
    drug_bioavailability_with_microbiome: float
    drug_bioavailability_without_microbiome: float
    bioavailability_change_pct: float
    active_metabolite_formed_pct: float
    microbiome_disruption_score: float
    antibiotic_class: str
    recovery_time_weeks: float
    clinical_impact: str
    mechanistic_notes: str
    notes: str


def predict_microbiome_interaction(
    drug_name: str,
    drug_class: str,
    logP: float,
    mw_Da: float,
    pka: float = 7.0,
    is_antibiotic: bool = False,
    antibiotic_spectrum: str = "none",
) -> MicrobiomeInteractionResult:
    """Predict drug–microbiome interactions.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    drug_class : str
        One of "antibiotic", "NSAID", "PPI", "beta_blocker", "statin", "other".
    logP : float
        Octanol–water partition coefficient.
    mw_Da : float
        Molecular weight in Daltons (must be > 0).
    pka : float
        Acid dissociation constant (default 7.0).
    is_antibiotic : bool
        Whether the drug is an antibiotic.
    antibiotic_spectrum : str
        "broad", "narrow", or "none".
    """
    # --- validation ---
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if drug_class not in VALID_DRUG_CLASSES:
        raise ValueError(
            f"drug_class must be one of {sorted(VALID_DRUG_CLASSES)}, got '{drug_class}'"
        )

    # --- bacterial metabolism fraction ---
    if logP < 0:
        bacterial_met_fraction = min(0.5, 0.3 + abs(pka - 7.0) * 0.02)
    elif logP < 2:
        bacterial_met_fraction = 0.15
    elif logP < 4:
        bacterial_met_fraction = 0.05
    else:
        bacterial_met_fraction = 0.02

    # --- microbiome disruption ---
    if drug_class == "antibiotic":
        bacterial_met_fraction = 0.02
        microbiome_disruption = min(10.0, 5.0 + (1 - logP / 5.0) * 3.0)
        if antibiotic_spectrum == "broad":
            microbiome_disruption = min(10.0, microbiome_disruption * 1.5)
        elif antibiotic_spectrum == "narrow":
            microbiome_disruption = max(1.0, microbiome_disruption * 0.5)
    elif drug_class == "PPI":
        microbiome_disruption = 4.0
    elif drug_class == "NSAID":
        microbiome_disruption = 3.0
    else:
        microbiome_disruption = max(0.0, logP * 0.5 - 1.0)

    microbiome_disruption_score = max(0.0, min(10.0, microbiome_disruption))

    # --- bioavailability ---
    f_without = min(1.0, 0.5 + logP * 0.1)
    f_with = f_without * (1 - bacterial_met_fraction)
    drug_bioavailability_with_microbiome = max(0.0, min(1.0, f_with))
    drug_bioavailability_without_microbiome = max(0.0, min(1.0, f_without))
    if f_without > 0:
        bioavailability_change_pct = (f_with - f_without) / f_without * 100
    else:
        bioavailability_change_pct = 0.0

    # --- active metabolite ---
    if drug_class in ("other", "NSAID") and bacterial_met_fraction > 0.1:
        active_metabolite_formed_pct = bacterial_met_fraction * 30
    else:
        active_metabolite_formed_pct = 0.0

    # --- recovery time ---
    recovery_time_weeks = max(0, microbiome_disruption_score * 1.5)
    recovery_time_weeks = round(recovery_time_weeks, 1)

    # --- gut bacteria class ---
    gut_bacteria_class = "gram_negative_predominant" if logP > 2 else "gram_positive_predominant"

    # --- antibiotic class ---
    antibiotic_class = antibiotic_spectrum if is_antibiotic else "N/A"

    # --- clinical impact ---
    if microbiome_disruption_score > 6:
        clinical_impact = "significant"
    elif microbiome_disruption_score > 3:
        clinical_impact = "moderate"
    else:
        clinical_impact = "minimal"

    # --- mechanistic notes ---
    notes_map = {
        "antibiotic": "Direct bactericidal/bacteriostatic action disrupts commensal flora.",
        "NSAID": "NSAIDs alter gut permeability and affect bacterial composition.",
        "PPI": "Proton pump inhibition raises gastric pH, shifting microbial populations.",
        "beta_blocker": "Beta-blockers have limited direct microbiome effects.",
        "statin": "Statins may modestly influence bile acid metabolism and gut flora.",
        "other": "Drug–microbiome interaction driven by physicochemical properties.",
    }
    mechanistic_notes = notes_map[drug_class]

    return MicrobiomeInteractionResult(
        drug_name=drug_name,
        gut_bacteria_class=gut_bacteria_class,
        bacterial_metabolism_fraction=bacterial_met_fraction,
        drug_bioavailability_with_microbiome=drug_bioavailability_with_microbiome,
        drug_bioavailability_without_microbiome=drug_bioavailability_without_microbiome,
        bioavailability_change_pct=bioavailability_change_pct,
        active_metabolite_formed_pct=active_metabolite_formed_pct,
        microbiome_disruption_score=microbiome_disruption_score,
        antibiotic_class=antibiotic_class,
        recovery_time_weeks=recovery_time_weeks,
        clinical_impact=clinical_impact,
        mechanistic_notes=mechanistic_notes,
        notes=f"Predicted microbiome interaction for {drug_name} (class={drug_class}).",
    )
