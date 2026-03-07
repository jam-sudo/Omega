"""Genotoxicity/mutagenicity structural alert screening — Phase 259.

Screens SMILES for structural alerts associated with genotoxicity,
Ames test positivity, and DNA damage. Based on Derek/Sarah alert databases
and ICH M7 guideline for mutagenic impurities.

References
----------
- ICH M7(R1): Assessment and Control of DNA Reactive (Mutagenic) Impurities. 2017.
- Benigni R & Bossa C, Chem Rev. 2011;111(4):2507-36
- Kazius J et al., J Med Chem. 2005;48(1):312-20
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["GenotoxAlert", "GenotoxResult", "assess_genotoxicity", "screen_genotoxicity"]


@dataclass(frozen=True)
class GenotoxAlert:
    """A single genotoxicity structural alert."""

    name: str
    triggered: bool
    weight: float
    mechanism: str
    description: str


@dataclass(frozen=True)
class GenotoxResult:
    """Result from genotoxicity structural alert screening."""

    compound_name: str
    smiles: str
    alerts: list[GenotoxAlert]
    n_alerts: int
    total_weight: float
    ames_prediction: str  # "positive", "negative", "equivocal"
    icm_m7_class: str  # "class 1" through "class 5" (simplified)
    daily_acceptable_intake_ng: float  # TTC for mutagenic impurities
    mitigation_strategy: list[str]
    notes: str


# Structural alert definitions: (description, weight, mechanism)
_GENO_ALERTS: dict[str, tuple[str, float, str]] = {
    "nitro_aromatic": (
        "Nitroaromatic compound: nitroreduction to reactive nitroso/hydroxylamine",
        2.0,
        "nitroreduction",
    ),
    "primary_aromatic_amine": (
        "Primary aromatic amine: N-hydroxylation to reactive nitrenium ion",
        2.0,
        "CYP-oxidation",
    ),
    "alkyl_halide": (
        "Alkyl halide: direct alkylation of DNA bases",
        2.0,
        "direct_alkylation",
    ),
    "epoxide": (
        "Epoxide: direct alkylation of nucleophilic DNA sites",
        2.0,
        "direct_alkylation",
    ),
    "hydrazine": (
        "Hydrazine: forms reactive diazonium ions under oxidation",
        2.0,
        "diazonium",
    ),
    "aziridine": (
        "Aziridine ring: high-strain electrophile alkylating DNA",
        2.5,
        "direct_alkylation",
    ),
    "beta_lactam_mutagenic": (
        "Reactive beta-lactam with additional electron-withdrawing group",
        1.0,
        "direct_alkylation",
    ),
    "quinone": (
        "Quinone/hydroquinone: redox cycling, DNA strand breaks",
        1.5,
        "redox_cycling",
    ),
    "michael_acceptor_dna": (
        "Alpha,beta-unsaturated carbonyl: Michael addition to DNA bases",
        1.5,
        "michael_addition",
    ),
    "n_nitroso": (
        "N-nitroso compound: N-nitrosamine, forms diazonium via alpha-hydroxylation",
        2.5,
        "nitrosamine",
    ),
    "acylating_agent": (
        "Acyl halide or anhydride: protein/DNA acylation",
        1.5,
        "direct_acylation",
    ),
}


def _detect_genotox_alerts(smiles: str) -> list[GenotoxAlert]:
    """Detect genotoxicity structural alerts from SMILES string."""
    alerts = []
    s = smiles  # Case-sensitive for some patterns

    for name, (desc, weight, mech) in _GENO_ALERTS.items():
        triggered = False

        if name == "nitro_aromatic":
            triggered = "[N+](=O)[O-]" in s and "c" in s.lower()
        elif name == "primary_aromatic_amine":
            triggered = "Nc1" in s or "N c1" in s or ("N" in s and "c1ccc" in s)
        elif name == "alkyl_halide":
            triggered = (
                "CBr" in s
                or "CI" in s
                or ("CCl" in s and "c" not in s.lower()[: s.lower().find("CCl") + 3])
            )
        elif name == "epoxide":
            triggered = "C1OC1" in s or "c1oc1" in s.lower()
        elif name == "hydrazine":
            triggered = "NN" in s and "NNC" not in s  # exclude simple hydrazides partially
        elif name == "aziridine":
            triggered = "C1CN1" in s or "C1NC1" in s
        elif name == "beta_lactam_mutagenic":
            triggered = "C1(=O)NC1" in s and ("[N+]" in s or "F" in s)
        elif name == "quinone":
            triggered = "C(=O)c1ccc(=O)" in s or "O=C1C=CC(=O)" in s
        elif name == "michael_acceptor_dna":
            triggered = "C=CC=O" in s or "C=CC(=O)" in s
        elif name == "n_nitroso":
            triggered = "N-N=O" in s or "NN=O" in s or "N(N=O)" in s
        elif name == "acylating_agent":
            triggered = "C(=O)Cl" in s or "C(=O)Br" in s or "C(=O)OC(=O)" in s

        alerts.append(
            GenotoxAlert(
                name=name,
                triggered=triggered,
                weight=weight,
                mechanism=mech,
                description=desc,
            )
        )

    return alerts


def assess_genotoxicity(
    compound_name: str,
    smiles: str,
    daily_dose_mg: float = 1.0,
) -> GenotoxResult:
    """Assess genotoxicity risk from structural alerts.

    Parameters
    ----------
    compound_name : Name of the compound.
    smiles : SMILES string. Must not be empty.
    daily_dose_mg : Daily dose (mg). Used for risk context.

    Returns
    -------
    GenotoxResult
    """
    if not smiles or not smiles.strip():
        raise ValueError("smiles must not be empty")
    if daily_dose_mg <= 0:
        raise ValueError("daily_dose_mg must be > 0")

    alerts = _detect_genotox_alerts(smiles)
    triggered = [a for a in alerts if a.triggered]
    n_alerts = len(triggered)
    total_weight = sum(a.weight for a in triggered)

    # Ames prediction
    if total_weight >= 2.0:
        ames = "positive"
    elif total_weight >= 1.0:
        ames = "equivocal"
    else:
        ames = "negative"

    # ICH M7 class (simplified):
    # Class 1: known mutagen+carcinogen, Class 2: known mutagen
    # Class 3: unknown, alerting, Class 4: non-alerting, Class 5: no concern
    triggered_names = {a.name for a in triggered}
    if "n_nitroso" in triggered_names or "aziridine" in triggered_names:
        icm_class = "class 1"
    elif n_alerts >= 2:
        icm_class = "class 2"
    elif n_alerts == 1:
        icm_class = "class 3"
    else:
        icm_class = "class 5"

    # TTC (Threshold of Toxicological Concern) for mutagenic impurities: 1.5 μg/day = 1500 ng
    # For APIs: 120 μg/day permitted for clinical development; 1.5 μg/day for lifetime
    daily_ai_ng = 1500.0 if ames == "positive" else 120000.0

    # Mitigation
    mitigations = []
    if "nitro_aromatic" in triggered_names:
        mitigations.append("Replace nitro group with bioisostere (e.g., CN, CF3)")
    if "primary_aromatic_amine" in triggered_names:
        mitigations.append("Block para-position; use secondary or tertiary amine")
    if "alkyl_halide" in triggered_names:
        mitigations.append("Replace alkyl halide with less reactive moiety")
    if "n_nitroso" in triggered_names:
        mitigations.append("Avoid N-nitroso formation; control nitrite impurities")
    if not mitigations:
        mitigations = (
            ["No specific structural modifications required"]
            if n_alerts == 0
            else ["Perform confirmatory Ames test; consider scaffold redesign"]
        )

    note_parts = []
    if n_alerts == 0:
        note_parts.append("No genotoxicity structural alerts detected.")
    else:
        names = ", ".join(sorted(triggered_names))
        note_parts.append(f"{n_alerts} alert(s): {names}.")
    note_parts.append(f"Ames prediction: {ames}. ICH M7: {icm_class}.")

    return GenotoxResult(
        compound_name=compound_name,
        smiles=smiles,
        alerts=alerts,
        n_alerts=n_alerts,
        total_weight=total_weight,
        ames_prediction=ames,
        icm_m7_class=icm_class,
        daily_acceptable_intake_ng=daily_ai_ng,
        mitigation_strategy=mitigations,
        notes=" ".join(note_parts),
    )


def screen_genotoxicity(compounds: list[dict]) -> list[GenotoxResult]:
    """Screen a list of compounds for genotoxicity.

    Parameters
    ----------
    compounds : List of dicts with keys: 'name', 'smiles', optional 'daily_dose_mg'.

    Returns sorted by total_weight descending.
    """
    results = [
        assess_genotoxicity(c["name"], c["smiles"], c.get("daily_dose_mg", 1.0)) for c in compounds
    ]
    return sorted(results, key=lambda r: r.total_weight, reverse=True)
