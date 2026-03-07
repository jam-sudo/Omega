"""Excipient-drug interaction prediction — Phase 451 + Phase 757.

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
        "notes": "Avoid with amine-containing drugs; not suitable for lactose-intolerant patients",
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
        raise ValueError(f"Unknown excipient '{excipient_name}'. Supported: {supported}")

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


# ---------------------------------------------------------------------------
# Phase 757 — Drug-Excipient Interaction Predictor (F-based model)
# ---------------------------------------------------------------------------

# Bioavailability-focused excipient effects table
EXCIPIENT_EFFECTS: dict[str, dict[str, float]] = {
    "cremophor_el": {
        "pgp_inhibition": 0.8,
        "cyp3a4_inhibition": 0.3,
        "solubility_enhancement": 2.0,
    },
    "tween_80": {
        "pgp_inhibition": 0.5,
        "cyp3a4_inhibition": 0.1,
        "solubility_enhancement": 1.5,
    },
    "tpgs": {
        "pgp_inhibition": 0.7,
        "cyp3a4_inhibition": 0.05,
        "solubility_enhancement": 1.8,
    },
    "pvp_k30": {
        "pgp_inhibition": 0.0,
        "cyp3a4_inhibition": 0.0,
        "solubility_enhancement": 1.3,
    },
    "sds": {
        "pgp_inhibition": 0.1,
        "cyp3a4_inhibition": 0.0,
        "solubility_enhancement": 3.0,
    },
    "hpmc": {
        "pgp_inhibition": 0.0,
        "cyp3a4_inhibition": 0.0,
        "solubility_enhancement": 1.1,
    },
    "citric_acid": {
        "pgp_inhibition": 0.0,
        "cyp3a4_inhibition": 0.0,
        "solubility_enhancement": 0.8,
        "ph_modifier": -1.5,
    },
}


@dataclass(frozen=True)
class ExcipientInteractionResult:
    """Result of excipient-drug interaction prediction (Phase 757).

    Attributes
    ----------
    drug_name : str
        Name of the drug.
    excipients_used : str
        Comma-joined list of excipients evaluated.
    pgp_inhibition_combined : float
        Combined P-gp inhibition fraction [0, 1].
    cyp3a4_inhibition_combined : float
        Combined CYP3A4 inhibition fraction [0, 1].
    solubility_enhancement_fold : float
        Multiplicative solubility enhancement fold.
    ph_change : float
        Net pH change from buffer excipients.
    fa_corrected : float
        Corrected fraction absorbed after excipient effects.
    fg_corrected : float
        Corrected gut-wall availability after CYP3A4 inhibition.
    f_total : float
        Total bioavailability with excipients (fa × fg × fh).
    f_baseline : float
        Baseline bioavailability without excipients (fa_base × fh).
    f_change_pct : float
        Percent change in F relative to baseline.
    recommendation : str
        Clinical recommendation based on F change.
    notes : str
        Summary notes on dominant excipient effects.
    """

    drug_name: str
    excipients_used: str
    pgp_inhibition_combined: float
    cyp3a4_inhibition_combined: float
    solubility_enhancement_fold: float
    ph_change: float
    fa_corrected: float
    fg_corrected: float
    f_total: float
    f_baseline: float
    f_change_pct: float
    recommendation: str
    notes: str


def predict_excipient_interactions(
    drug_name: str,
    excipients: list[str],
    fa_base: float = 0.7,
    fm_cyp3a4: float = 0.3,
    fh: float = 0.7,
    is_pgp_substrate: bool = False,
) -> ExcipientInteractionResult:
    """Predict how a combination of excipients affects oral bioavailability.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    excipients : list[str]
        Excipient names (from EXCIPIENT_EFFECTS; unknown excipients have no effect).
    fa_base : float
        Base fraction absorbed (0–1).
    fm_cyp3a4 : float
        Fraction metabolised by CYP3A4 (0–1).
    fh : float
        Hepatic first-pass availability (0–1].
    is_pgp_substrate : bool
        If True, P-gp inhibition boosts fa further.

    Returns
    -------
    ExcipientInteractionResult

    Raises
    ------
    ValueError
        If fa_base, fm_cyp3a4, or fh are out of valid range.
    """
    if not (0.0 <= fa_base <= 1.0):
        raise ValueError(f"fa_base must be between 0 and 1, got {fa_base}")
    if not (0.0 <= fm_cyp3a4 <= 1.0):
        raise ValueError(f"fm_cyp3a4 must be between 0 and 1, got {fm_cyp3a4}")
    if not (0.0 < fh <= 1.0):
        raise ValueError(f"fh must be in (0, 1], got {fh}")

    # Collect effects from known excipients only
    pgp_inh_list: list[float] = []
    cyp_inh_list: list[float] = []
    sol_enh_list: list[float] = []
    ph_delta: float = 0.0

    for exc in excipients:
        key = exc.lower().replace(" ", "_").replace("-", "_")
        effects = EXCIPIENT_EFFECTS.get(key, {})
        pgp_inh_list.append(effects.get("pgp_inhibition", 0.0))
        cyp_inh_list.append(effects.get("cyp3a4_inhibition", 0.0))
        sol_enh_list.append(effects.get("solubility_enhancement", 1.0))
        ph_delta += effects.get("ph_modifier", 0.0)

    # Combined effects
    pgp_inh_combined = 1.0
    for p in pgp_inh_list:
        pgp_inh_combined *= 1.0 - p
    pgp_inh_combined = 1.0 - pgp_inh_combined

    cyp_inh_combined = 1.0
    for c in cyp_inh_list:
        cyp_inh_combined *= 1.0 - c
    cyp_inh_combined = 1.0 - cyp_inh_combined

    sol_enh_combined = 1.0
    for s in sol_enh_list:
        sol_enh_combined *= s

    # Impact on F components
    fa_corrected = min(1.0, fa_base * sol_enh_combined)
    if is_pgp_substrate:
        fa_corrected = min(1.0, fa_corrected + pgp_inh_combined * 0.2)

    fg_corrected = 1.0 / (1.0 + fm_cyp3a4 * cyp_inh_combined * 5.0)

    f_total = fa_corrected * fg_corrected * fh
    f_baseline = fa_base * 1.0 * fh  # fg=1 (no inhibition), fh unchanged

    if f_baseline > 0.0:
        f_change_pct = (f_total - f_baseline) / f_baseline * 100.0
    else:
        f_change_pct = 0.0

    # Recommendation
    if f_change_pct > 50.0:
        recommendation = "Significant F enhancement expected; consider dose reduction"
    elif f_change_pct > 20.0:
        recommendation = "Moderate F enhancement; monitor plasma levels"
    elif f_change_pct < -20.0:
        recommendation = "F reduction; consider dose increase"
    else:
        recommendation = "Minimal excipient impact on bioavailability"

    # Build notes
    note_parts: list[str] = []
    if pgp_inh_combined > 0.3:
        note_parts.append(f"Strong P-gp inhibition ({pgp_inh_combined:.2f})")
    if cyp_inh_combined > 0.1:
        note_parts.append(f"CYP3A4 inhibition ({cyp_inh_combined:.2f})")
    if sol_enh_combined > 1.5:
        note_parts.append(f"High solubility enhancement ({sol_enh_combined:.2f}x)")
    elif sol_enh_combined < 1.0:
        note_parts.append(f"Solubility reduction ({sol_enh_combined:.2f}x)")
    if abs(ph_delta) > 0.1:
        note_parts.append(f"pH change {ph_delta:+.1f} from buffer excipients")
    if not note_parts:
        note_parts.append("No dominant excipient interactions identified")

    notes = "; ".join(note_parts)

    return ExcipientInteractionResult(
        drug_name=drug_name,
        excipients_used=", ".join(excipients) if excipients else "none",
        pgp_inhibition_combined=round(pgp_inh_combined, 6),
        cyp3a4_inhibition_combined=round(cyp_inh_combined, 6),
        solubility_enhancement_fold=round(sol_enh_combined, 6),
        ph_change=round(ph_delta, 6),
        fa_corrected=round(fa_corrected, 6),
        fg_corrected=round(fg_corrected, 6),
        f_total=round(f_total, 6),
        f_baseline=round(f_baseline, 6),
        f_change_pct=round(f_change_pct, 4),
        recommendation=recommendation,
        notes=notes,
    )


def screen_excipient_combinations(
    drug_name: str,
    excipient_combos: list[list[str]],
    fa_base: float = 0.7,
    fm_cyp3a4: float = 0.3,
    fh: float = 0.7,
    is_pgp_substrate: bool = False,
) -> list[ExcipientInteractionResult]:
    """Screen multiple excipient combinations and rank by total bioavailability.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    excipient_combos : list[list[str]]
        Each inner list is a combination of excipient names to evaluate.
    fa_base, fm_cyp3a4, fh, is_pgp_substrate :
        Shared drug PK parameters (see predict_excipient_interactions).

    Returns
    -------
    list[ExcipientInteractionResult]
        Sorted by f_total descending.
    """
    results: list[ExcipientInteractionResult] = []
    for combo in excipient_combos:
        result = predict_excipient_interactions(
            drug_name=drug_name,
            excipients=combo,
            fa_base=fa_base,
            fm_cyp3a4=fm_cyp3a4,
            fh=fh,
            is_pgp_substrate=is_pgp_substrate,
        )
        results.append(result)

    results.sort(key=lambda r: r.f_total, reverse=True)
    return results
