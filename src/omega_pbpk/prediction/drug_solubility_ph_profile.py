"""Phase 506 — Drug Solubility pH Profile Prediction.

Predict pH-dependent solubility across GI tract compartments using
Henderson-Hasselbalch equations for ionizable drugs.

Models
------
- Monoprotic acid:   S(pH) = S0 * (1 + 10^(pH - pKa))
- Monoprotic base:   S(pH) = S0 * (1 + 10^(pKa - pH))
- Neutral:           S(pH) = S0 (constant)

GI pH compartments used:
    stomach=1.5, duodenum=6.0, jejunum=6.5, ileum=7.2, colon=7.0
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "SolubilityPhProfileResult",
    "predict_solubility_ph_profile",
    "find_optimal_gi_segment",
]

_VALID_DRUG_TYPES = {"acid", "base", "neutral"}

# GI compartment pH values
_GI_PH = {
    "stomach": 1.5,
    "duodenum": 6.0,
    "jejunum": 6.5,
    "ileum": 7.2,
    "colon": 7.0,
}

_S_MIN = 1e-6  # mg/mL absolute floor
_S_MAX = 1e6  # mg/mL absolute ceiling

# pH profile range: 1.0 to 10.0 in 0.5 steps → 19 points
_PH_VALUES = [1.0 + 0.5 * i for i in range(19)]  # 1.0, 1.5, ..., 10.0


@dataclass
class SolubilityPhProfileResult:
    """Result of pH-solubility profile prediction."""

    drug_name: str
    intrinsic_solubility_mg_mL: float
    pKa: float
    drug_type: str
    ph_values: list
    solubility_values_mg_mL: list
    stomach_solubility_mg_mL: float
    duodenum_solubility_mg_mL: float
    jejunum_solubility_mg_mL: float
    ileum_solubility_mg_mL: float
    colon_solubility_mg_mL: float
    max_solubility_mg_mL: float
    min_solubility_mg_mL: float
    max_solubility_ph: float
    dissolution_limited_segments: list
    notes: str


def _solubility_at_ph(
    s0: float,
    ph: float,
    pKa: float,
    drug_type: str,
) -> float:
    """Compute solubility at a given pH using Henderson-Hasselbalch."""
    if drug_type == "neutral":
        s = s0
    elif drug_type == "acid":
        exponent = ph - pKa
        exponent = max(-300.0, min(300.0, exponent))
        s = s0 * (1.0 + 10.0**exponent)
    else:  # base
        exponent = pKa - ph
        exponent = max(-300.0, min(300.0, exponent))
        s = s0 * (1.0 + 10.0**exponent)

    # Clamp to absolute limits
    return max(_S_MIN, min(_S_MAX, s))


def predict_solubility_ph_profile(
    drug_name: str,
    intrinsic_solubility_mg_mL: float,
    pKa: float,
    drug_type: str,
) -> SolubilityPhProfileResult:
    """Predict pH-solubility profile across the GI tract.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    intrinsic_solubility_mg_mL : float
        Intrinsic (unionized) solubility in mg/mL. Must be > 0.
    pKa : float
        Acid/base dissociation constant.
    drug_type : str
        One of "acid", "base", "neutral".

    Returns
    -------
    SolubilityPhProfileResult
    """
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError(
            f"intrinsic_solubility_mg_mL must be > 0, got {intrinsic_solubility_mg_mL}"
        )
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(f"drug_type must be one of {sorted(_VALID_DRUG_TYPES)}, got '{drug_type}'")

    s0 = intrinsic_solubility_mg_mL

    # Build pH-solubility profile (1.0 to 10.0 in 0.5 steps)
    ph_values = list(_PH_VALUES)
    solubility_values = [_solubility_at_ph(s0, ph, pKa, drug_type) for ph in ph_values]

    # GI compartment solubilities
    stomach_sol = _solubility_at_ph(s0, _GI_PH["stomach"], pKa, drug_type)
    duodenum_sol = _solubility_at_ph(s0, _GI_PH["duodenum"], pKa, drug_type)
    jejunum_sol = _solubility_at_ph(s0, _GI_PH["jejunum"], pKa, drug_type)
    ileum_sol = _solubility_at_ph(s0, _GI_PH["ileum"], pKa, drug_type)
    colon_sol = _solubility_at_ph(s0, _GI_PH["colon"], pKa, drug_type)

    # Max/min across pH profile
    max_sol = max(solubility_values)
    min_sol = min(solubility_values)
    max_sol_ph = ph_values[solubility_values.index(max_sol)]

    # Dissolution-limited segments: solubility < 0.1 mg/mL
    dissolution_limited: list[str] = []
    gi_sols = {
        "stomach": stomach_sol,
        "duodenum": duodenum_sol,
        "jejunum": jejunum_sol,
        "ileum": ileum_sol,
        "colon": colon_sol,
    }
    for seg_name, sol in gi_sols.items():
        if sol < 0.1:
            dissolution_limited.append(seg_name)

    # Notes
    notes_parts = [
        f"Drug type: {drug_type}; pKa={pKa:.2f}; S0={s0:.4g} mg/mL.",
        f"Max solubility {max_sol:.4g} mg/mL at pH {max_sol_ph:.1f}.",
    ]
    if dissolution_limited:
        notes_parts.append(f"Dissolution-limited segments: {', '.join(dissolution_limited)}.")
    notes = " ".join(notes_parts)

    return SolubilityPhProfileResult(
        drug_name=drug_name,
        intrinsic_solubility_mg_mL=s0,
        pKa=pKa,
        drug_type=drug_type,
        ph_values=ph_values,
        solubility_values_mg_mL=solubility_values,
        stomach_solubility_mg_mL=stomach_sol,
        duodenum_solubility_mg_mL=duodenum_sol,
        jejunum_solubility_mg_mL=jejunum_sol,
        ileum_solubility_mg_mL=ileum_sol,
        colon_solubility_mg_mL=colon_sol,
        max_solubility_mg_mL=max_sol,
        min_solubility_mg_mL=min_sol,
        max_solubility_ph=max_sol_ph,
        dissolution_limited_segments=dissolution_limited,
        notes=notes,
    )


def find_optimal_gi_segment(
    drug_name: str,
    intrinsic_solubility_mg_mL: float,
    pKa: float,
    drug_type: str,
) -> str:
    """Return the GI segment with highest solubility for the drug.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    intrinsic_solubility_mg_mL : float
        Intrinsic solubility in mg/mL.
    pKa : float
        Dissociation constant.
    drug_type : str
        One of "acid", "base", "neutral".

    Returns
    -------
    str
        One of "stomach", "duodenum", "jejunum", "ileum", "colon".
    """
    result = predict_solubility_ph_profile(
        drug_name=drug_name,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        pKa=pKa,
        drug_type=drug_type,
    )

    seg_sols = {
        "stomach": result.stomach_solubility_mg_mL,
        "duodenum": result.duodenum_solubility_mg_mL,
        "jejunum": result.jejunum_solubility_mg_mL,
        "ileum": result.ileum_solubility_mg_mL,
        "colon": result.colon_solubility_mg_mL,
    }
    return max(seg_sols, key=lambda seg: seg_sols[seg])
