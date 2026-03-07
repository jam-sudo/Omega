"""Excipient-drug interaction prediction — Phase 451.

Predicts how common pharmaceutical excipients affect drug solubility,
permeability, and stability based on physicochemical drug properties.

Models are empirical correlations for early formulation screening.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ExcipientEffectResult",
    "predict_excipient_effect",
    "screen_excipients",
    "SUPPORTED_EXCIPIENTS",
]

# ---------------------------------------------------------------------------
# Excipient database
# ---------------------------------------------------------------------------

# Each entry: name -> dict with:
#   type: 'polymer' | 'surfactant' | 'filler' | 'lubricant' | 'disintegrant'
#   solubility_base: baseline fold-change for a neutral logP=0 drug
#   solubility_logp_slope: additional enhancement per unit of logP (clipped to >=1)
#   permeability_effect: 'enhance' | 'neutral' | 'reduce'
#   stability_effect: 'positive' | 'neutral' | 'negative'
#   mechanism: description
#   notes: additional info

_EXCIPIENT_DB: dict[str, dict] = {
    "HPMC": {
        "type": "polymer",
        "solubility_base": 1.5,
        "solubility_logp_slope": 0.15,
        "permeability_effect": "reduce",
        "stability_effect": "positive",
        "compatibility_base": 78.0,
        "mechanism": "Polymer matrix retards crystallization; viscosity increase"
        " reduces dissolution rate for BCS I compounds",
        "notes": "Widely used in matrix tablets and hot-melt extrusion; improves"
        " chemical stability via moisture barrier",
    },
    "PVP": {
        "type": "polymer",
        "solubility_base": 1.8,
        "solubility_logp_slope": 0.18,
        "permeability_effect": "neutral",
        "stability_effect": "positive",
        "compatibility_base": 82.0,
        "mechanism": "Complexation and co-precipitation inhibit crystallization;"
        " amorphous solid dispersion formation",
        "notes": "Effective amorphous stabilizer; hygroscopic — moisture may"
        " induce crystallization at high RH",
    },
    "SDS": {
        "type": "surfactant",
        "solubility_base": 3.0,
        "solubility_logp_slope": 0.55,
        "permeability_effect": "enhance",
        "stability_effect": "neutral",
        "compatibility_base": 68.0,
        "mechanism": "Anionic surfactant forms mixed micelles; critical micelle"
        " concentration ~8 mM in GI fluid analogues",
        "notes": "Strong solubilizer for BCS II/IV drugs; may irritate GI mucosa"
        " at high concentrations",
    },
    "Poloxamer 407": {
        "type": "surfactant",
        "solubility_base": 2.5,
        "solubility_logp_slope": 0.40,
        "permeability_effect": "enhance",
        "stability_effect": "neutral",
        "compatibility_base": 75.0,
        "mechanism": "Non-ionic triblock copolymer (PEO-PPO-PEO) micellization;"
        " P-gp inhibition enhances absorption",
        "notes": "Mild P-gp inhibitor; thermoresponsive gelation useful for"
        " sustained-release applications",
    },
    "Lactose": {
        "type": "filler",
        "solubility_base": 1.0,
        "solubility_logp_slope": 0.0,
        "permeability_effect": "neutral",
        "stability_effect": "negative",
        "compatibility_base": 60.0,
        "mechanism": "Inert diluent; reducing sugar — Maillard reaction with"
        " primary amines compromises stability",
        "notes": "Avoid with amine-containing drugs; not suitable for lactose"
        "-intolerant patients",
    },
    "MCC": {
        "type": "filler",
        "solubility_base": 1.0,
        "solubility_logp_slope": 0.0,
        "permeability_effect": "neutral",
        "stability_effect": "positive",
        "compatibility_base": 85.0,
        "mechanism": "Microcrystalline cellulose — chemically inert diluent/binder;"
        " promotes tablet disintegration",
        "notes": "Highly compressible; excellent chemical compatibility; low"
        " moisture may limit disintegration",
    },
    "Stearic acid": {
        "type": "lubricant",
        "solubility_base": 0.85,
        "solubility_logp_slope": -0.05,
        "permeability_effect": "reduce",
        "stability_effect": "neutral",
        "compatibility_base": 55.0,
        "mechanism": "Hydrophobic lubricant forms film on tablet surface;"
        " retards wetting and dissolution",
        "notes": "Use at minimum effective concentration (<1%) to limit"
        " solubility reduction; avoid with basic drugs that may form soaps",
    },
    "Tween 80": {
        "type": "surfactant",
        "solubility_base": 2.8,
        "solubility_logp_slope": 0.48,
        "permeability_effect": "enhance",
        "stability_effect": "neutral",
        "compatibility_base": 72.0,
        "mechanism": "Non-ionic surfactant; micellar solubilization; P-gp"
        " efflux pump inhibition at GI epithelium",
        "notes": "Mild P-gp inhibitor; oxidative degradation of polyoxyethylene"
        " chains may affect susceptible drugs",
    },
}

SUPPORTED_EXCIPIENTS: list[str] = list(_EXCIPIENT_DB.keys())


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExcipientEffectResult:
    """Predicted effect of a single excipient on a drug.

    Attributes
    ----------
    excipient_name : str
        Name of the excipient.
    solubility_fold_change : float
        Predicted fold-change in apparent solubility (>1 = enhancement).
    permeability_effect : str
        Direction of permeability change: 'enhance', 'neutral', or 'reduce'.
    stability_effect : str
        Effect on chemical/physical stability: 'positive', 'neutral', or 'negative'.
    compatibility_score : float
        Overall compatibility score 0–100 (higher = better).
    mechanism : str
        Primary interaction mechanism.
    notes : str
        Formulation guidance notes.
    """

    excipient_name: str
    solubility_fold_change: float
    permeability_effect: str
    stability_effect: str
    compatibility_score: float
    mechanism: str
    notes: str


# ---------------------------------------------------------------------------
# Core logic helpers
# ---------------------------------------------------------------------------


def _solubility_fold_change(entry: dict, drug_logp: float) -> float:
    """Compute solubility fold-change from excipient entry and drug logP."""
    base = entry["solubility_base"]
    slope = entry["solubility_logp_slope"]
    # For hydrophobic drugs (high logP) surfactants are more effective
    # Clip logP to [0, 8] to avoid extreme extrapolation
    effective_logp = max(0.0, min(drug_logp, 8.0))
    fold = base + slope * effective_logp
    # Minimum fold-change is 0.5 (some reduction at most)
    return max(0.5, fold)


def _pka_stability_modifier(entry: dict, drug_pka: float | None) -> float:
    """Adjust compatibility score based on pKa — Maillard risk with lactose."""
    if drug_pka is None:
        return 0.0
    excipient = entry
    # Lactose Maillard: amine drugs (pKa 8-11) are most reactive
    if excipient.get("stability_effect") == "negative" and 7.0 <= drug_pka <= 11.0:
        return -10.0  # extra penalty for amine + reducing sugar
    return 0.0


def _compatibility_score(
    entry: dict,
    drug_logp: float,
    drug_mw: float,
    drug_pka: float | None,
    fold_change: float,
) -> float:
    """Calculate overall compatibility score 0–100."""
    score = entry["compatibility_base"]

    # Reward strong solubility enhancement
    if fold_change >= 3.0:
        score += 8.0
    elif fold_change >= 2.0:
        score += 4.0
    elif fold_change < 0.9:
        score -= 10.0

    # Permeability effect
    perm = entry["permeability_effect"]
    if perm == "enhance":
        score += 5.0
    elif perm == "reduce":
        score -= 5.0

    # Stability effect
    stab = entry["stability_effect"]
    if stab == "positive":
        score += 5.0
    elif stab == "negative":
        score -= 8.0

    # pKa-based modifier (e.g., Maillard risk)
    score += _pka_stability_modifier(entry, drug_pka)

    # High-MW drugs may have slower dissolution — polymers are less beneficial
    if drug_mw > 500 and entry["type"] == "polymer":
        score -= 3.0

    return max(0.0, min(100.0, score))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_excipient_effect(
    drug_logp: float,
    drug_mw: float,
    excipient_name: str,
    drug_pka: float | None = None,
) -> ExcipientEffectResult:
    """Predict how a single excipient affects drug solubility, permeability, and stability.

    Parameters
    ----------
    drug_logp : float
        Drug octanol-water partition coefficient.
    drug_mw : float
        Drug molecular weight (Da).
    excipient_name : str
        Excipient name (must be in SUPPORTED_EXCIPIENTS).
    drug_pka : float or None
        Drug pKa (used for Maillard risk assessment with reducing sugars).

    Returns
    -------
    ExcipientEffectResult

    Raises
    ------
    ValueError
        If excipient_name is not in the database.
    """
    key = excipient_name.strip()
    if key not in _EXCIPIENT_DB:
        supported = ", ".join(sorted(_EXCIPIENT_DB.keys()))
        raise ValueError(
            f"Unknown excipient '{excipient_name}'. Supported: {supported}"
        )

    entry = _EXCIPIENT_DB[key]
    fold = _solubility_fold_change(entry, drug_logp)
    score = _compatibility_score(entry, drug_logp, drug_mw, drug_pka, fold)

    return ExcipientEffectResult(
        excipient_name=key,
        solubility_fold_change=round(fold, 4),
        permeability_effect=entry["permeability_effect"],
        stability_effect=entry["stability_effect"],
        compatibility_score=round(score, 2),
        mechanism=entry["mechanism"],
        notes=entry["notes"],
    )


def screen_excipients(
    drug_logp: float,
    drug_mw: float,
    excipients: list[str] | None = None,
    drug_pka: float | None = None,
) -> list[ExcipientEffectResult]:
    """Screen multiple excipients for a drug and rank by compatibility score.

    Parameters
    ----------
    drug_logp : float
        Drug octanol-water partition coefficient.
    drug_mw : float
        Drug molecular weight (Da).
    excipients : list[str] or None
        Excipient names to screen. Defaults to all supported excipients.
    drug_pka : float or None
        Drug pKa for Maillard risk assessment.

    Returns
    -------
    list[ExcipientEffectResult]
        Results sorted by compatibility_score descending.

    Raises
    ------
    ValueError
        If any excipient name is not in the database.
    """
    if excipients is None:
        excipients = SUPPORTED_EXCIPIENTS

    results: list[ExcipientEffectResult] = []
    for name in excipients:
        results.append(predict_excipient_effect(drug_logp, drug_mw, name, drug_pka))

    results.sort(key=lambda r: r.compatibility_score, reverse=True)
    return results
