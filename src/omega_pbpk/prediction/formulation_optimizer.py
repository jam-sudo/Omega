"""Formulation optimizer — recommend optimal drug delivery strategy.

Recommends formulation strategies based on drug physicochemical properties
(MW, logP, PSA, solubility, permeability) and BCS classification.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormulationRecommendation:
    """Recommended formulation strategy for a drug candidate."""

    drug_name: str
    bcs_class: str
    target_tmax_h: float
    target_duration_h: float
    primary_strategy: str
    secondary_strategy: str
    tertiary_strategy: str
    risk_flags: tuple[str, ...]
    rationale: str
    notes: str


# ---------------------------------------------------------------------------
# Strategy tables
# ---------------------------------------------------------------------------

# BCS class → (duration_normal, duration_extended)
_STRATEGY_MAP: dict[str, dict[str, list[str]]] = {
    "I": {
        "normal": ["IR tablet", "ER/MR tablet", "oral solution"],
        "extended": ["ER/MR tablet", "IR tablet", "oral solution"],
    },
    "II": {
        "normal": ["ASD/nanoparticle", "lipid formulation", "cyclodextrin complex"],
        "extended": ["ASD/nanoparticle ER", "lipid ER formulation", "cyclodextrin ER"],
    },
    "III": {
        "normal": ["absorption enhancer", "prodrug strategy", "nanoparticle"],
        "extended": ["absorption enhancer ER", "prodrug ER", "nanoparticle ER"],
    },
    "IV": {
        "normal": ["ASD + enhancer", "lipid formulation", "nanoparticle"],
        "extended": ["ASD + enhancer ER", "lipid ER formulation", "nanoparticle ER"],
    },
}

_VALID_BCS_CLASSES = {"I", "II", "III", "IV"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _needs_solubility_enhancement(dose_mg: float, solubility_mg_mL: float) -> bool:
    """Return True if dose/solubility > GI fluid volume (250 mL)."""
    if solubility_mg_mL <= 0:
        return True
    gut_volume_mL = 250.0
    return (dose_mg / solubility_mg_mL) > gut_volume_mL


def _collect_risk_flags(
    logP: float,
    solubility_mg_mL: float,
    mw: float,
    psa: float,
) -> list[str]:
    """Return list of physicochemical risk flags."""
    flags: list[str] = []
    if solubility_mg_mL < 0.01:
        flags.append("very_low_solubility_flag")
    if logP > 5:
        flags.append("high_lipophilicity_flag")
    if mw > 500:
        flags.append("high_mw_flag")
    if psa > 140:
        flags.append("high_psa_flag")
    return flags


def _build_rationale(
    bcs_class: str,
    needs_enhancement: bool,
    risk_flags: list[str],
    target_duration_h: float,
    logP: float,
    solubility_mg_mL: float,
) -> str:
    """Construct a human-readable rationale string."""
    parts: list[str] = [f"BCS Class {bcs_class}:"]

    if bcs_class == "I":
        parts.append("high solubility and permeability; IR preferred for fast onset.")
    elif bcs_class == "II":
        parts.append("low solubility limits absorption; solubilization technology required.")
    elif bcs_class == "III":
        parts.append("high solubility but low permeability; absorption enhancement needed.")
    else:
        parts.append(
            "low solubility and permeability; combined solubilization + enhancement required."
        )

    if needs_enhancement:
        parts.append(
            f"Dose/solubility ratio exceeds 250 mL GI volume "
            f"(solubility={solubility_mg_mL:.4g} mg/mL)."
        )

    if target_duration_h > 12:
        parts.append("Extended-release preferred to achieve target duration.")

    if "very_low_solubility_flag" in risk_flags:
        parts.append(f"Very low solubility ({solubility_mg_mL:.4g} mg/mL): special care needed.")
    if "high_lipophilicity_flag" in risk_flags:
        parts.append(f"High lipophilicity (logP={logP:.1f}): lipid-based systems advantageous.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recommend_formulation(
    drug_name: str,
    mw: float,
    logP: float,
    psa: float,
    solubility_mg_mL: float,
    permeability_cm_s: float,
    dose_mg: float,
    bcs_class: str,
    target_tmax_h: float,
    target_duration_h: float,
) -> FormulationRecommendation:
    """Recommend optimal formulation strategy based on drug properties.

    Parameters
    ----------
    drug_name : str
        Drug or compound name.
    mw : float
        Molecular weight (g/mol), must be > 0.
    logP : float
        Octanol-water partition coefficient (log scale).
    psa : float
        Polar surface area (Å²), must be ≥ 0.
    solubility_mg_mL : float
        Aqueous solubility (mg/mL), must be ≥ 0.
    permeability_cm_s : float
        Caco-2 effective permeability (cm/s), must be ≥ 0.
    dose_mg : float
        Target dose (mg), must be > 0.
    bcs_class : str
        BCS classification: "I", "II", "III", or "IV".
    target_tmax_h : float
        Desired time to peak concentration (h), must be > 0.
    target_duration_h : float
        Desired duration of action (h), must be > 0.

    Returns
    -------
    FormulationRecommendation
    """
    if not drug_name:
        raise ValueError("drug_name must be a non-empty string")
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")
    if solubility_mg_mL < 0:
        raise ValueError("solubility_mg_mL must be >= 0")
    if permeability_cm_s < 0:
        raise ValueError("permeability_cm_s must be >= 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if bcs_class not in _VALID_BCS_CLASSES:
        raise ValueError(f"bcs_class must be one of {sorted(_VALID_BCS_CLASSES)}")
    if target_tmax_h <= 0:
        raise ValueError("target_tmax_h must be > 0")
    if target_duration_h <= 0:
        raise ValueError("target_duration_h must be > 0")

    # Select strategy tier based on target duration
    duration_key = "extended" if target_duration_h > 12 else "normal"
    strategies = _STRATEGY_MAP[bcs_class][duration_key]
    primary = strategies[0]
    secondary = strategies[1]
    tertiary = strategies[2]

    # Risk flags
    risk_flags = _collect_risk_flags(logP, solubility_mg_mL, mw, psa)

    # Solubility enhancement check
    needs_enhancement = _needs_solubility_enhancement(dose_mg, solubility_mg_mL)
    if needs_enhancement and bcs_class == "I":
        # Even BCS I may need help if dose >> solubility
        risk_flags = risk_flags + ["solubility_enhancement_needed_flag"]

    # Rationale
    rationale = _build_rationale(
        bcs_class, needs_enhancement, risk_flags, target_duration_h, logP, solubility_mg_mL
    )

    # Notes
    note_parts: list[str] = []
    if target_tmax_h < 1.0:
        note_parts.append("Very short Tmax target; consider ODT or liquid formulation.")
    if target_tmax_h > 6.0:
        note_parts.append("Long Tmax target; extended-release matrix suitable.")
    if not note_parts:
        note_parts.append("Standard formulation development recommended.")
    notes = " ".join(note_parts)

    return FormulationRecommendation(
        drug_name=drug_name,
        bcs_class=bcs_class,
        target_tmax_h=target_tmax_h,
        target_duration_h=target_duration_h,
        primary_strategy=primary,
        secondary_strategy=secondary,
        tertiary_strategy=tertiary,
        risk_flags=tuple(risk_flags),
        rationale=rationale,
        notes=notes,
    )


def formulation_risk_assessment(
    drug_name: str,
    mw: float,
    logP: float,
    psa: float,
    solubility_mg_mL: float,
) -> dict:
    """Assess formulation risk based on physicochemical properties.

    Parameters
    ----------
    drug_name : str
        Drug or compound name.
    mw : float
        Molecular weight (g/mol), must be > 0.
    logP : float
        Octanol-water partition coefficient.
    psa : float
        Polar surface area (Å²), must be ≥ 0.
    solubility_mg_mL : float
        Aqueous solubility (mg/mL), must be ≥ 0.

    Returns
    -------
    dict with keys: risk_flags (list[str]), overall_risk (str), rationale (str)
    """
    if not drug_name:
        raise ValueError("drug_name must be a non-empty string")
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")
    if solubility_mg_mL < 0:
        raise ValueError("solubility_mg_mL must be >= 0")

    risk_flags = _collect_risk_flags(logP, solubility_mg_mL, mw, psa)

    n_flags = len(risk_flags)
    if n_flags == 0:
        overall_risk = "low"
    elif n_flags == 1:
        overall_risk = "moderate"
    else:
        overall_risk = "high"

    # Build rationale
    rationale_parts: list[str] = [f"Drug: {drug_name}."]
    if not risk_flags:
        rationale_parts.append("No physicochemical risk flags identified.")
    else:
        rationale_parts.append(f"Risk flags: {', '.join(risk_flags)}.")
    if "very_low_solubility_flag" in risk_flags:
        rationale_parts.append(
            "Solubility < 0.01 mg/mL: dissolution likely rate-limiting; solubilization needed."
        )
    if "high_lipophilicity_flag" in risk_flags:
        rationale_parts.append(
            "logP > 5: poor aqueous wettability; self-emulsifying or lipid system recommended."
        )
    if "high_mw_flag" in risk_flags:
        rationale_parts.append("MW > 500: potential permeability and dissolution challenges.")
    if "high_psa_flag" in risk_flags:
        rationale_parts.append("PSA > 140 Å²: poor membrane permeability expected.")

    rationale = " ".join(rationale_parts)

    return {
        "risk_flags": risk_flags,
        "overall_risk": overall_risk,
        "rationale": rationale,
    }


__all__ = [
    "FormulationRecommendation",
    "recommend_formulation",
    "formulation_risk_assessment",
]
