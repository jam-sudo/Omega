"""Phase 643 — Drug tissue accumulation risk assessment.

Assesses accumulation risk based on physicochemical properties and PK parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AccumulationRiskResult:
    compound_name: str
    vd_L_per_kg: float
    logP: float
    pka_base: float | None
    adipose_kp: float
    muscle_kp: float
    liver_kp: float
    brain_kp: float
    accumulation_index: float
    risk_level: str
    tissue_of_concern: str
    wash_out_t_half_h: float
    steady_state_ratio: float
    notes: str


def _compute_kps(
    logP: float,
    psa: float,
    pka_base: float | None,
) -> tuple[float, float, float, float]:
    """Return (adipose_kp, muscle_kp, liver_kp, brain_kp)."""
    # Adipose
    if logP > 1.0:
        adipose_kp = max(1.0, logP * 3.0 + 0.5)
    else:
        adipose_kp = 0.5

    # Muscle
    pka_base_bonus = 2.0 if (pka_base is not None and pka_base > 7.0) else 0.0
    muscle_kp = max(0.5, logP * 0.8 + pka_base_bonus)

    # Liver
    liver_base_bonus = 1.0 if (pka_base is not None and pka_base > 7.0) else 0.0
    liver_kp = max(1.5, logP * 1.5 + 1.0 + liver_base_bonus)

    # Brain
    if logP > 2.0 and psa < 90.0:
        brain_kp = logP * 0.5 - psa * 0.01
    else:
        brain_kp = 0.1

    return adipose_kp, muscle_kp, liver_kp, brain_kp


def _compute_accumulation_index(
    adipose_kp: float,
    muscle_kp: float,
    liver_kp: float,
    brain_kp: float,
) -> float:
    """Weighted log10 accumulation index, scaled 0-100."""
    w_adipose = 0.30
    w_muscle = 0.30
    w_liver = 0.25
    w_brain = 0.15

    raw = (
        w_adipose * math.log10(max(1.0, adipose_kp))
        + w_muscle * math.log10(max(1.0, muscle_kp))
        + w_liver * math.log10(max(1.0, liver_kp))
        + w_brain * math.log10(max(1.0, brain_kp))
    )

    # Scale: raw=0 → 0, raw=1.5 → 100
    index = (raw / 1.5) * 100.0
    return min(100.0, max(0.0, index))


def _risk_level(index: float) -> str:
    if index < 25.0:
        return "low"
    elif index < 50.0:
        return "moderate"
    elif index < 75.0:
        return "high"
    else:
        return "very_high"


def assess_accumulation_risk(
    compound_name: str,
    logP: float,
    mw: float,
    psa: float,
    vd_L_per_kg: float = 1.0,
    cl_L_per_h_per_kg: float = 0.1,
    t_half_h: float | None = None,
    pka_base: float | None = None,
    pka_acid: float | None = None,
    fup: float = 0.1,
) -> AccumulationRiskResult:
    """Assess tissue accumulation risk for a drug compound."""
    # Validation
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if psa < 0:
        raise ValueError(f"psa must be >= 0, got {psa}")
    if vd_L_per_kg <= 0:
        raise ValueError(f"vd_L_per_kg must be > 0, got {vd_L_per_kg}")
    if cl_L_per_h_per_kg <= 0:
        raise ValueError(f"cl_L_per_h_per_kg must be > 0, got {cl_L_per_h_per_kg}")
    if not (0.0 < fup <= 1.0):
        raise ValueError(f"fup must be in (0, 1], got {fup}")

    # Estimate t_half if not provided
    if t_half_h is None:
        t_half_h = (vd_L_per_kg / cl_L_per_h_per_kg) * math.log(2.0)

    # Compute tissue Kp values
    adipose_kp, muscle_kp, liver_kp, brain_kp = _compute_kps(logP, psa, pka_base)

    # Accumulation index
    index = _compute_accumulation_index(adipose_kp, muscle_kp, liver_kp, brain_kp)

    # Risk level
    level = _risk_level(index)

    # Tissue of concern (highest Kp)
    tissue_kps = {
        "adipose": adipose_kp,
        "muscle": muscle_kp,
        "liver": liver_kp,
        "brain": brain_kp,
    }
    tissue_of_concern = max(tissue_kps, key=lambda k: tissue_kps[k])
    max_kp = tissue_kps[tissue_of_concern]

    # Wash-out t½: effective tissue t½ scales with Kp
    wash_out_t_half_h = t_half_h * max_kp

    # Steady-state ratio (worst-case tissue:plasma)
    steady_state_ratio = max_kp / fup if fup > 0 else max_kp

    # Build notes
    note_parts = []
    if pka_base is not None and pka_base > 7.0:
        note_parts.append(f"Basic drug (pKa={pka_base:.1f}) enhances muscle/liver accumulation.")
    if logP > 3.0:
        note_parts.append(f"High lipophilicity (logP={logP:.1f}) drives adipose accumulation.")
    if brain_kp > 1.0:
        note_parts.append(f"Brain penetration predicted (brain_kp={brain_kp:.2f}).")
    if level in ("high", "very_high"):
        note_parts.append(f"Accumulation index {index:.1f}: {level} risk; monitor tissue levels.")
    else:
        note_parts.append(f"Accumulation index {index:.1f}: {level} risk.")
    notes = " ".join(note_parts) if note_parts else f"Accumulation index {index:.1f}: {level} risk."

    return AccumulationRiskResult(
        compound_name=compound_name,
        vd_L_per_kg=vd_L_per_kg,
        logP=logP,
        pka_base=pka_base,
        adipose_kp=adipose_kp,
        muscle_kp=muscle_kp,
        liver_kp=liver_kp,
        brain_kp=brain_kp,
        accumulation_index=index,
        risk_level=level,
        tissue_of_concern=tissue_of_concern,
        wash_out_t_half_h=wash_out_t_half_h,
        steady_state_ratio=steady_state_ratio,
        notes=notes,
    )


def screen_accumulation_risk(compounds: list[dict]) -> list[AccumulationRiskResult]:
    """Screen multiple compounds, sorted by accumulation_index descending."""
    if not compounds:
        return []

    results = []
    for c in compounds:
        result = assess_accumulation_risk(**c)
        results.append(result)

    results.sort(key=lambda r: r.accumulation_index, reverse=True)
    return results
