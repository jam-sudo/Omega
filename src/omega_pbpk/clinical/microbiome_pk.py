"""Phase 193 — Gut microbiome influence on drug metabolism and absorption."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MicrobiomeEffect",
    "MicrobiomePKResult",
    "assess_microbiome_impact",
    "compare_microbiome_states",
]

VALID_STATES = {"normal", "dysbiotic", "antibiotic_treated", "germ_free"}

STATE_ACTIVITY: dict[str, float] = {
    "normal": 1.0,
    "dysbiotic": 0.6,
    "antibiotic_treated": 0.1,
    "germ_free": 0.0,
}


@dataclass(frozen=True)
class MicrobiomeEffect:
    """A single microbiome-mediated effect on drug PK."""

    pathway: str
    triggered: bool
    magnitude: float
    direction: str
    description: str


@dataclass(frozen=True)
class MicrobiomePKResult:
    """Result of microbiome PK impact assessment."""

    drug_name: str
    microbiome_state: str
    effects: list[MicrobiomeEffect]
    f_oral_baseline: float
    f_oral_adjusted: float
    bioavailability_change_pct: float
    auc_ratio: float
    risk_category: str
    recommendation: str
    notes: str


def _build_effects(
    smiles: str,
    microbiome_state: str,
    has_azo_bond: bool,
    has_glucuronide: bool,
    has_sulfate: bool,
    logP: float,
) -> list[MicrobiomeEffect]:
    effects: list[MicrobiomeEffect] = []
    activity = STATE_ACTIVITY[microbiome_state]

    if has_azo_bond:
        mag = 1.0 + 0.5 * activity
        effects.append(
            MicrobiomeEffect(
                "azo_reduction",
                True,
                mag,
                "increase",
                "Gut bacteria activate azo-containing prodrug",
            )
        )

    if has_glucuronide:
        mag = 1.0 + 0.3 * activity
        effects.append(
            MicrobiomeEffect(
                "glucuronide_hydrolysis",
                True,
                mag,
                "increase",
                "Beta-glucuronidase hydrolyzes conjugate, enabling enterohepatic recycling",
            )
        )

    if "NN" in smiles:
        effects.append(
            MicrobiomeEffect(
                "hydrazine_reduction",
                True,
                0.8,
                "decrease",
                "Gut bacteria may reduce hydrazine compounds",
            )
        )

    if logP > 4 and microbiome_state == "dysbiotic":
        effects.append(
            MicrobiomeEffect(
                "lipophilic_trapping",
                True,
                0.85,
                "decrease",
                "Dysbiotic microbiome may trap lipophilic drugs",
            )
        )

    return effects


def assess_microbiome_impact(
    drug_name: str,
    smiles: str,
    f_oral_baseline: float,
    microbiome_state: str = "normal",
    has_azo_bond: bool = False,
    has_glucuronide: bool = False,
    has_sulfate: bool = False,
    logP: float = 2.0,
) -> MicrobiomePKResult:
    """Assess the impact of gut microbiome state on oral drug PK.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    smiles:
        SMILES string (used for structural feature detection).
    f_oral_baseline:
        Baseline oral bioavailability fraction (0 < f <= 1).
    microbiome_state:
        One of "normal", "dysbiotic", "antibiotic_treated", "germ_free".
    has_azo_bond:
        True if drug contains an azo (–N=N–) bond (prodrug activation).
    has_glucuronide:
        True if drug undergoes glucuronidation/enterohepatic recycling.
    has_sulfate:
        True if drug is a sulfate-reduction substrate (reserved for future use).
    logP:
        Octanol-water partition coefficient.

    Returns
    -------
    MicrobiomePKResult
    """
    if not (0 < f_oral_baseline <= 1.0):
        raise ValueError(f"f_oral_baseline must be in (0, 1], got {f_oral_baseline}")
    if microbiome_state not in VALID_STATES:
        raise ValueError(
            f"microbiome_state must be one of {sorted(VALID_STATES)}, got '{microbiome_state}'"
        )

    effects = _build_effects(
        smiles, microbiome_state, has_azo_bond, has_glucuronide, has_sulfate, logP
    )

    combined_mag = 1.0
    for e in effects:
        combined_mag *= e.magnitude

    f_adjusted = min(1.0, f_oral_baseline * combined_mag)
    ba_change_pct = (f_adjusted - f_oral_baseline) / f_oral_baseline * 100.0
    auc_ratio = f_adjusted / f_oral_baseline if f_oral_baseline > 0 else 1.0

    if abs(ba_change_pct) < 10:
        risk = "low"
    elif abs(ba_change_pct) < 30:
        risk = "moderate"
    else:
        risk = "high"

    if ba_change_pct > 20:
        rec = "Consider dose reduction in patients with normal microbiome"
    elif ba_change_pct < -20:
        rec = "Monitor for reduced efficacy in antibiotic-treated patients"
    else:
        rec = "Microbiome impact is likely clinically insignificant"

    n_effects = len(effects)
    notes = (
        f"{n_effects} microbiome pathway(s) active; "
        f"combined magnitude {combined_mag:.3f}x; "
        f"state activity={STATE_ACTIVITY[microbiome_state]:.1f}"
    )

    return MicrobiomePKResult(
        drug_name=drug_name,
        microbiome_state=microbiome_state,
        effects=effects,
        f_oral_baseline=f_oral_baseline,
        f_oral_adjusted=f_adjusted,
        bioavailability_change_pct=ba_change_pct,
        auc_ratio=auc_ratio,
        risk_category=risk,
        recommendation=rec,
        notes=notes,
    )


def compare_microbiome_states(
    drug_name: str,
    smiles: str,
    f_oral_baseline: float,
    **kwargs,
) -> list[MicrobiomePKResult]:
    """Compare microbiome impact across all four microbiome states.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    smiles:
        SMILES string.
    f_oral_baseline:
        Baseline oral bioavailability fraction (0 < f <= 1).
    **kwargs:
        Additional keyword arguments forwarded to :func:`assess_microbiome_impact`
        (e.g. ``has_azo_bond``, ``has_glucuronide``, ``logP``).

    Returns
    -------
    list[MicrobiomePKResult]
        One result per microbiome state, in the order:
        ``normal``, ``dysbiotic``, ``antibiotic_treated``, ``germ_free``.
    """
    states = ["normal", "dysbiotic", "antibiotic_treated", "germ_free"]
    return [
        assess_microbiome_impact(
            drug_name=drug_name,
            smiles=smiles,
            f_oral_baseline=f_oral_baseline,
            microbiome_state=state,
            **kwargs,
        )
        for state in states
    ]
