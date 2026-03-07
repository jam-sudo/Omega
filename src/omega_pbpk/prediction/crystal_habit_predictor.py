"""Phase 207 — Crystal habit predictor.

Predict crystal habit (morphology) and its impact on dissolution and
flowability based on molecular properties.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CrystalHabitResult:
    """Result of crystal habit prediction (immutable)."""

    drug_name: str
    logP: float
    mw_Da: float
    hbd: int
    hba: int
    habit_class: str  # "needle", "plate", "prism", "cube"
    dissolution_modifier: float
    flowability: str  # "poor", "moderate", "good", "excellent"
    compressibility: str  # "poor", "moderate", "good"
    processing_risk: str  # "low", "moderate", "high"
    symmetry_index: float
    notes: str


_VALID_HABITS = {"needle", "plate", "prism", "cube"}
_VALID_FLOWABILITY = {"poor", "moderate", "good", "excellent"}
_VALID_COMPRESSIBILITY = {"poor", "moderate", "good"}
_VALID_PROCESSING_RISK = {"low", "moderate", "high"}

# dissolution modifier per habit
_DISSOLUTION_MODIFIER = {
    "needle": 1.2,
    "plate": 0.8,
    "prism": 1.0,
    "cube": 0.9,
}

_FLOWABILITY = {
    "needle": "poor",
    "plate": "moderate",
    "prism": "good",
    "cube": "excellent",
}

_COMPRESSIBILITY = {
    "needle": "poor",
    "plate": "good",
    "prism": "moderate",
    "cube": "good",
}

_PROCESSING_RISK = {
    "needle": "high",
    "plate": "moderate",
    "prism": "low",
    "cube": "low",
}


def _predict_habit(logP: float, hbd: int, hba: int, symmetry_index: float) -> str:
    """Determine crystal habit from molecular properties.

    Decision rules (in priority order):
    1. High symmetry (>= 0.7) → cube
    2. High H-bonding capacity (hbd + hba > 8) → needle
    3. Hydrophobic with low H-bond donor count (logP > 3 and hbd < 2) → plate
    4. Otherwise → prism
    """
    if symmetry_index >= 0.7:
        return "cube"
    if hbd + hba > 8:
        return "needle"
    if logP > 3 and hbd < 2:
        return "plate"
    return "prism"


def predict_crystal_habit(
    drug_name: str,
    logP: float,
    mw_Da: float,
    hbd: int,
    hba: int,
    symmetry_index: float,
    notes: str = "",
) -> CrystalHabitResult:
    """Predict crystal habit and its pharmaceutical impact.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Octanol-water partition coefficient.
    mw_Da:
        Molecular weight in Daltons.
    hbd:
        Hydrogen bond donor count.
    hba:
        Hydrogen bond acceptor count.
    symmetry_index:
        Molecular symmetry index in range [0, 1]; higher values indicate
        more equidimensional (cube/prism) crystal habits.
    notes:
        Optional extra notes to append.

    Returns
    -------
    CrystalHabitResult
    """
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if not 0.0 <= symmetry_index <= 1.0:
        raise ValueError(f"symmetry_index must be in [0, 1], got {symmetry_index}")

    habit = _predict_habit(logP, hbd, hba, symmetry_index)
    dissolution_modifier = _DISSOLUTION_MODIFIER[habit]
    flowability = _FLOWABILITY[habit]
    compressibility = _COMPRESSIBILITY[habit]
    processing_risk = _PROCESSING_RISK[habit]

    auto_notes = (
        f"habit={habit}; logP={logP:.2f}, MW={mw_Da:.1f} Da, "
        f"HBD={hbd}, HBA={hba}, symm={symmetry_index:.2f}"
    )
    if notes:
        auto_notes = auto_notes + "; " + notes

    return CrystalHabitResult(
        drug_name=drug_name,
        logP=logP,
        mw_Da=mw_Da,
        hbd=hbd,
        hba=hba,
        habit_class=habit,
        dissolution_modifier=dissolution_modifier,
        flowability=flowability,
        compressibility=compressibility,
        processing_risk=processing_risk,
        symmetry_index=symmetry_index,
        notes=auto_notes,
    )


def compare_polymorphs(
    drug_name: str,
    polymorph_conditions: list[dict],
) -> list[CrystalHabitResult]:
    """Compare multiple polymorph conditions for a drug.

    Each entry in *polymorph_conditions* must contain keys matching the
    keyword arguments of :func:`predict_crystal_habit` (excluding
    ``drug_name``): ``logP``, ``mw_Da``, ``hbd``, ``hba``,
    ``symmetry_index``, and optionally ``notes``.

    Returns
    -------
    list[CrystalHabitResult]
        Results sorted by *dissolution_modifier* descending (best
        dissolution first).
    """
    results: list[CrystalHabitResult] = []
    for cond in polymorph_conditions:
        result = predict_crystal_habit(
            drug_name=drug_name,
            logP=cond["logP"],
            mw_Da=cond["mw_Da"],
            hbd=cond["hbd"],
            hba=cond["hba"],
            symmetry_index=cond["symmetry_index"],
            notes=cond.get("notes", ""),
        )
        results.append(result)

    results.sort(key=lambda r: r.dissolution_modifier, reverse=True)
    return results
