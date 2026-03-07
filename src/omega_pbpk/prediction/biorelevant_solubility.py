"""Phase 1090 — Drug Solubility in Biorelevant Media.

Predicts drug solubility in FaSSIF, FeSSIF, FaSSGF, water, and PBS_pH6_8,
accounting for pH ionization, bile salt micellar solubilization, lecithin
vesicle effects, and lipid digestion products.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Media definitions: (pH, bile_salt_mM, lecithin_mM, lipid_mM)
# ---------------------------------------------------------------------------

MEDIA_COMPOSITIONS: dict[str, tuple[float, float, float, float]] = {
    "FaSSIF": (6.5, 3.0, 0.75, 0.0),
    "FeSSIF": (5.0, 15.0, 3.75, 28.4),
    "FaSSGF": (1.6, 0.08, 0.02, 0.0),
    "water": (7.0, 0.0, 0.0, 0.0),
    "PBS_pH6_8": (6.8, 0.0, 0.0, 0.0),
}

VALID_MEDIA = frozenset(MEDIA_COMPOSITIONS.keys())
VALID_IONIZATION_TYPES = frozenset({"acid", "base", "neutral"})


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BiorelevantSolubilityResult:
    """Result of biorelevant solubility prediction."""

    drug_name: str
    medium: str
    pka: float
    logP: float
    ionization_type: str
    s_intrinsic_mg_mL: float
    s_biorelevant_mg_mL: float
    ionization_factor: float
    micellar_enhancement: float
    food_effect_ratio: float
    solubility_class: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _ionization_factor(ionization_type: str, pka: float, ph: float) -> float:
    """Henderson-Hasselbalch ionization factor."""
    if ionization_type == "acid":
        return 1.0 + math.pow(10.0, ph - pka)
    if ionization_type == "base":
        return 1.0 + math.pow(10.0, pka - ph)
    # neutral
    return 1.0


def _k_bile(logP: float) -> float:
    """Micellar partition coefficient K_bile (mM^-1)."""
    raw = math.pow(10.0, 0.3 * logP - 0.5)
    return _clamp(raw, 0.01, 100.0)


def _k_lec(logP: float) -> float:
    """Lecithin partition coefficient K_lec (mM^-1)."""
    raw = math.pow(10.0, 0.2 * logP)
    return _clamp(raw, 0.1, 50.0)


def _k_lipid(logP: float) -> float:
    """Lipid partition coefficient K_lipid (mM^-1)."""
    raw = math.pow(10.0, 0.4 * logP - 1.0)
    return _clamp(raw, 0.001, 200.0)


def _solubility_class(s_mg_mL: float) -> str:
    """BCS-inspired solubility classification."""
    if s_mg_mL >= 1.0:
        return "high"
    if s_mg_mL >= 0.1:
        return "moderate"
    return "low"


def _build_notes(
    ionization_type: str,
    ionization_factor: float,
    micellar_enhancement: float,
    medium: str,
    s_biorelevant: float,
) -> str:
    """Build human-readable notes."""
    parts: list[str] = []
    parts.append(f"Ionization factor: {ionization_factor:.2f} ({ionization_type}).")
    if micellar_enhancement > 1.0:
        parts.append(f"Micellar/lipid enhancement: {micellar_enhancement:.2f}x.")
    else:
        parts.append("No micellar enhancement (aqueous medium).")
    parts.append(f"Predicted solubility in {medium}: {s_biorelevant:.4f} mg/mL.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------


def _compute_solubility(
    pka: float,
    logP: float,
    ionization_type: str,
    intrinsic_solubility_mg_mL: float,
    medium: str,
) -> tuple[float, float, float, float]:
    """Return (s_biorelevant, ionization_factor, micellar_enhancement, s_intrinsic_used).

    Enhancement formula:
        S_total = S_intrinsic * f_ion * (1 + K_bile*bile + K_lec*lec + K_lipid*lipid)
    """
    ph, bile_salt, lecithin, lipid = MEDIA_COMPOSITIONS[medium]

    f_ion = _ionization_factor(ionization_type, pka, ph)
    kb = _k_bile(logP)
    kl = _k_lec(logP)
    klp = _k_lipid(logP)

    # Micellar/lipid factor
    micelle_factor = 1.0 + kb * bile_salt + kl * lecithin + klp * lipid

    s_biorelevant = intrinsic_solubility_mg_mL * f_ion * micelle_factor

    # micellar_enhancement = contribution purely from bile/lec/lipid
    # = total / (S_intrinsic * f_ion) → which equals micelle_factor
    micellar_enhancement = micelle_factor

    return s_biorelevant, f_ion, micellar_enhancement, intrinsic_solubility_mg_mL


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_biorelevant_solubility(
    drug_name: str,
    pka: float,
    logP: float,
    ionization_type: str,
    intrinsic_solubility_mg_mL: float = 0.05,
    medium: str = "FaSSIF",
) -> BiorelevantSolubilityResult:
    """Predict drug solubility in a biorelevant medium.

    Parameters
    ----------
    drug_name:
        Name of the drug (for labeling).
    pka:
        Ionization constant (0–14).
    logP:
        Octanol-water partition coefficient (-5 to 10).
    ionization_type:
        One of "acid", "base", "neutral".
    intrinsic_solubility_mg_mL:
        Thermodynamic (unionized) solubility in mg/mL (must be > 0).
    medium:
        Biorelevant medium: "FaSSIF", "FeSSIF", "FaSSGF", "water", "PBS_pH6_8".

    Returns
    -------
    BiorelevantSolubilityResult.

    Raises
    ------
    ValueError
        If any input parameter is out of valid range.
    """
    if medium not in VALID_MEDIA:
        raise ValueError(f"medium '{medium}' not recognized. Valid media: {sorted(VALID_MEDIA)}")
    if ionization_type not in VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type '{ionization_type}' not valid. "
            f"Must be one of: {sorted(VALID_IONIZATION_TYPES)}"
        )
    if not (0.0 <= pka <= 14.0):
        raise ValueError(f"pka must be in [0, 14], got {pka}")
    if not (-5.0 <= logP <= 10.0):
        raise ValueError(f"logP must be in [-5, 10], got {logP}")
    if intrinsic_solubility_mg_mL <= 0.0:
        raise ValueError(
            f"intrinsic_solubility_mg_mL must be > 0, got {intrinsic_solubility_mg_mL}"
        )

    s_biorelevant, f_ion, micellar_enhancement, s_intrinsic = _compute_solubility(
        pka, logP, ionization_type, intrinsic_solubility_mg_mL, medium
    )

    sol_class = _solubility_class(s_biorelevant)
    notes = _build_notes(ionization_type, f_ion, micellar_enhancement, medium, s_biorelevant)

    return BiorelevantSolubilityResult(
        drug_name=drug_name,
        medium=medium,
        pka=pka,
        logP=logP,
        ionization_type=ionization_type,
        s_intrinsic_mg_mL=s_intrinsic,
        s_biorelevant_mg_mL=s_biorelevant,
        ionization_factor=f_ion,
        micellar_enhancement=micellar_enhancement,
        food_effect_ratio=0.0,  # Not computable from single medium
        solubility_class=sol_class,
        notes=notes,
    )


def screen_media(
    drug_name: str,
    pka: float,
    logP: float,
    ionization_type: str,
    intrinsic_solubility_mg_mL: float = 0.05,
) -> list[BiorelevantSolubilityResult]:
    """Run solubility prediction across all 5 biorelevant media.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    pka:
        Ionization constant.
    logP:
        Partition coefficient.
    ionization_type:
        "acid", "base", or "neutral".
    intrinsic_solubility_mg_mL:
        Thermodynamic solubility (> 0).

    Returns
    -------
    List of BiorelevantSolubilityResult for all 5 media, sorted by
    s_biorelevant_mg_mL descending (highest solubility first).
    The food_effect_ratio field is populated as FeSSIF/FaSSIF.
    """
    # First compute all base results
    base_results: dict[str, BiorelevantSolubilityResult] = {}
    for med in VALID_MEDIA:
        base_results[med] = predict_biorelevant_solubility(
            drug_name, pka, logP, ionization_type, intrinsic_solubility_mg_mL, med
        )

    # Compute food effect ratio: FeSSIF / FaSSIF
    fessif_sol = base_results["FeSSIF"].s_biorelevant_mg_mL
    fassif_sol = base_results["FaSSIF"].s_biorelevant_mg_mL
    if fassif_sol > 0.0:
        food_ratio = fessif_sol / fassif_sol
    else:
        food_ratio = 0.0

    # Rebuild results with food_effect_ratio populated
    final: list[BiorelevantSolubilityResult] = []
    for _med, res in base_results.items():
        updated = BiorelevantSolubilityResult(
            drug_name=res.drug_name,
            medium=res.medium,
            pka=res.pka,
            logP=res.logP,
            ionization_type=res.ionization_type,
            s_intrinsic_mg_mL=res.s_intrinsic_mg_mL,
            s_biorelevant_mg_mL=res.s_biorelevant_mg_mL,
            ionization_factor=res.ionization_factor,
            micellar_enhancement=res.micellar_enhancement,
            food_effect_ratio=food_ratio,
            solubility_class=res.solubility_class,
            notes=res.notes,
        )
        final.append(updated)

    final.sort(key=lambda r: r.s_biorelevant_mg_mL, reverse=True)
    return final
