"""Phase 1106 — Drug Plasma Half-Life Prediction from Structure.

Empirical rule-based half-life prediction from molecular properties:
MW, logP, ionization type, pKa, PSA, rotatable bonds, ester flag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ── Constants ────────────────────────────────────────────────────────────────
_BASE_T_HALF_H = 2.0  # reference t½ (MW=300, logP=2, neutral)
_REF_MW = 300.0  # reference MW for scaling
_MW_EXPONENT = 0.3
_VD_ASSUMED_L = 60.0  # assumed Vd for CL back-calculation (1-cpt)
_REF_PSA_A2 = 80.0
_PSA_DENOMINATOR = 200.0
_ROT_BOND_PENALTY = 0.05  # per rotatable bond

_VALID_IONIZATION = {"acid", "base", "neutral"}

# ── Half-life classification thresholds ──────────────────────────────────────
_ULTRA_SHORT_H = 1.0
_SHORT_H = 6.0
_INTERMEDIATE_H = 24.0


def _classify_half_life(t_half_h: float) -> str:
    if t_half_h < _ULTRA_SHORT_H:
        return "ultra_short"
    if t_half_h < _SHORT_H:
        return "short"
    if t_half_h < _INTERMEDIATE_H:
        return "intermediate"
    return "long"


def _predict_t_half_raw(
    mw_Da: float,
    logP: float,
    psa_A2: float,
    ionization_type: str,
    pka: float,
    n_rotatable_bonds: int,
    has_ester: bool,
) -> float:
    """Core half-life prediction logic. Returns t½ in hours."""
    t_half = _BASE_T_HALF_H

    # MW factor: larger molecules cleared more slowly
    t_half *= (mw_Da / _REF_MW) ** _MW_EXPONENT

    # logP factor
    if logP < 0:
        t_half *= 0.5
    elif logP <= 3:
        t_half *= 1.0
    elif logP <= 5:
        t_half *= 1.5
    else:
        t_half *= 2.5

    # Ionization factor at physiological pH=7.4
    if ionization_type == "acid":
        t_half *= 1.3
    elif ionization_type == "base":
        t_half *= 1.2
    # neutral → 1.0

    # PSA factor (high PSA → rapid renal excretion → shorter t½)
    psa_factor = max(0.5, min(2.0, 1.0 - (psa_A2 - _REF_PSA_A2) / _PSA_DENOMINATOR))
    t_half *= psa_factor

    # Rotatable bonds factor (flexibility → faster metabolism)
    rot_factor = max(0.5, 1.0 - n_rotatable_bonds * _ROT_BOND_PENALTY)
    t_half *= rot_factor

    # Ester hydrolysis boost
    if has_ester:
        t_half *= 0.7

    return t_half


# ── Result dataclass ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HalfLifePredictionResult:
    """Frozen result of empirical half-life prediction."""

    drug_name: str
    mw_Da: float
    logP: float
    psa_A2: float
    ionization_type: str
    pka: float
    n_rotatable_bonds: int
    has_ester: bool
    t_half_predicted_h: float
    cl_predicted_L_per_h: float  # CL = 0.693 * Vd / t½
    vd_assumed_L: float  # = 60 L (fixed)
    half_life_class: str  # ultra_short / short / intermediate / long
    notes: str


# ── Public API ───────────────────────────────────────────────────────────────
def predict_half_life(
    drug_name: str,
    mw_Da: float,
    logP: float,
    psa_A2: float,
    ionization_type: str = "neutral",
    pka: float = 7.0,
    n_rotatable_bonds: int = 5,
    has_ester: bool = False,
) -> HalfLifePredictionResult:
    """Predict plasma half-life from molecular descriptors.

    Parameters
    ----------
    drug_name:          Human-readable identifier.
    mw_Da:              Molecular weight (Da). Must be > 0.
    logP:               Octanol-water partition coefficient. Must be in [-5, 10].
    psa_A2:             Polar surface area (Å²). Must be >= 0.
    ionization_type:    One of {"acid", "base", "neutral"}.
    pka:                pKa of the ionizable group (informational; used for notes).
    n_rotatable_bonds:  Number of rotatable bonds. Must be >= 0.
    has_ester:          True if the molecule contains an ester group.
    """
    # ── Validation ──────────────────────────────────────────────────────────
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if not (-5.0 <= logP <= 10.0):
        raise ValueError(f"logP must be in [-5, 10], got {logP}")
    if psa_A2 < 0:
        raise ValueError(f"psa_A2 must be >= 0, got {psa_A2}")
    if n_rotatable_bonds < 0:
        raise ValueError(f"n_rotatable_bonds must be >= 0, got {n_rotatable_bonds}")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION}, got {ionization_type!r}"
        )

    t_half = _predict_t_half_raw(
        mw_Da=mw_Da,
        logP=logP,
        psa_A2=psa_A2,
        ionization_type=ionization_type,
        pka=pka,
        n_rotatable_bonds=n_rotatable_bonds,
        has_ester=has_ester,
    )

    cl = (math.log(2) / t_half) * _VD_ASSUMED_L  # L/h
    half_life_class = _classify_half_life(t_half)

    note_parts = [
        f"MW_factor=({mw_Da}/{_REF_MW})^{_MW_EXPONENT:.1f}",
        f"logP={logP}",
        f"ion={ionization_type}",
    ]
    if has_ester:
        note_parts.append("ester_hydrolysis_boost")
    notes = "; ".join(note_parts)

    return HalfLifePredictionResult(
        drug_name=drug_name,
        mw_Da=mw_Da,
        logP=logP,
        psa_A2=psa_A2,
        ionization_type=ionization_type,
        pka=pka,
        n_rotatable_bonds=n_rotatable_bonds,
        has_ester=has_ester,
        t_half_predicted_h=t_half,
        cl_predicted_L_per_h=cl,
        vd_assumed_L=_VD_ASSUMED_L,
        half_life_class=half_life_class,
        notes=notes,
    )


def screen_compounds(
    compounds: list[dict],
) -> list[HalfLifePredictionResult]:
    """Screen a list of compound dicts through half-life prediction.

    Each dict must contain ``drug_name``, ``mw_Da``, ``logP``, ``psa_A2``
    and optionally any keyword arguments accepted by :func:`predict_half_life`.

    Returns results sorted by ``t_half_predicted_h`` ascending (shortest first).
    """
    if not compounds:
        raise ValueError("compounds list must not be empty")

    results: list[HalfLifePredictionResult] = []
    for cmp in compounds:
        cmp_copy = dict(cmp)
        name = cmp_copy.pop("drug_name")
        mw = cmp_copy.pop("mw_Da")
        logp = cmp_copy.pop("logP")
        psa = cmp_copy.pop("psa_A2")
        results.append(predict_half_life(name, mw, logp, psa, **cmp_copy))

    results.sort(key=lambda r: r.t_half_predicted_h)
    return results
