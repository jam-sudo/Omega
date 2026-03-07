"""Molecular weight prediction from atom counts and fragment analysis.

Provides MW estimation, atom count estimation from MW, classification,
and fragment-based MW calculation.

Note: mw_predictor.py (Phase 113) already exists. This module uses different
function names to avoid conflicts.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Atomic weights (IUPAC standard)
# ---------------------------------------------------------------------------

ATOMIC_WEIGHTS: dict[str, float] = {
    "C": 12.011,
    "H": 1.008,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
    "F": 19.00,
    "Cl": 35.45,
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MolecularWeightResult:
    compound_name: str
    estimated_mw: float
    mw_class: str  # fragment / small / medium / large / macromolecule
    bcs_size_category: str  # bcs_small / bcs_medium / bcs_large
    mw_range_likely: tuple  # (low_estimate, high_estimate)
    n_heavy_atoms_estimate: int
    formula_guess: str
    lipinski_mw_ok: bool  # MW <= 500
    veber_mw_ok: bool  # MW <= 480
    notes: str


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def classify_mw(mw: float) -> str:
    """Classify MW into drug-size categories.

    fragment < 200; small 200-400; medium 400-600; large 600-1000; macromolecule > 1000.
    """
    if mw < 200:
        return "fragment"
    if mw < 400:
        return "small"
    if mw < 600:
        return "medium"
    if mw <= 1000:
        return "large"
    return "macromolecule"


def _bcs_size_category(mw: float) -> str:
    if mw < 300:
        return "bcs_small"
    if mw < 500:
        return "bcs_medium"
    return "bcs_large"


def _build_formula(
    n_carbon: int,
    n_hydrogen: int,
    n_nitrogen: int,
    n_oxygen: int,
    n_sulfur: int,
    n_fluorine: int,
    n_chlorine: int,
) -> str:
    """Build a simplified molecular formula string (only non-zero atoms)."""
    parts = []
    if n_carbon > 0:
        parts.append(f"C{n_carbon}")
    if n_hydrogen > 0:
        parts.append(f"H{n_hydrogen}")
    if n_nitrogen > 0:
        parts.append(f"N{n_nitrogen}")
    if n_oxygen > 0:
        parts.append(f"O{n_oxygen}")
    if n_sulfur > 0:
        parts.append(f"S{n_sulfur}")
    if n_fluorine > 0:
        parts.append(f"F{n_fluorine}")
    if n_chlorine > 0:
        parts.append(f"Cl{n_chlorine}")
    return "".join(parts) if parts else "unknown"


# ---------------------------------------------------------------------------
# Main functions
# ---------------------------------------------------------------------------


def estimate_mw_from_formula(
    n_carbon: int,
    n_hydrogen: int,
    n_nitrogen: int = 0,
    n_oxygen: int = 0,
    n_sulfur: int = 0,
    n_fluorine: int = 0,
    n_chlorine: int = 0,
    compound_name: str = "unknown",
) -> MolecularWeightResult:
    """Calculate MW from atom counts using standard atomic weights."""
    # Validation
    if n_carbon < 0 or n_hydrogen < 0 or n_nitrogen < 0 or n_oxygen < 0:
        raise ValueError("Atom counts must be >= 0")
    if n_sulfur < 0 or n_fluorine < 0 or n_chlorine < 0:
        raise ValueError("Atom counts must be >= 0")
    if n_carbon + n_nitrogen + n_oxygen + n_sulfur == 0:
        raise ValueError("At least one heavy non-H atom required (C, N, O, or S)")

    mw = (
        n_carbon * ATOMIC_WEIGHTS["C"]
        + n_hydrogen * ATOMIC_WEIGHTS["H"]
        + n_nitrogen * ATOMIC_WEIGHTS["N"]
        + n_oxygen * ATOMIC_WEIGHTS["O"]
        + n_sulfur * ATOMIC_WEIGHTS["S"]
        + n_fluorine * ATOMIC_WEIGHTS["F"]
        + n_chlorine * ATOMIC_WEIGHTS["Cl"]
    )

    n_heavy = n_carbon + n_nitrogen + n_oxygen + n_sulfur + n_fluorine + n_chlorine
    formula = _build_formula(
        n_carbon, n_hydrogen, n_nitrogen, n_oxygen, n_sulfur, n_fluorine, n_chlorine
    )

    notes = f"MW calculated from atom counts; formula={formula}; heavy_atoms={n_heavy}"

    return MolecularWeightResult(
        compound_name=compound_name,
        estimated_mw=round(mw, 3),
        mw_class=classify_mw(mw),
        bcs_size_category=_bcs_size_category(mw),
        mw_range_likely=(round(mw * 0.95, 3), round(mw * 1.05, 3)),
        n_heavy_atoms_estimate=n_heavy,
        formula_guess=formula,
        lipinski_mw_ok=mw <= 500,
        veber_mw_ok=mw <= 480,
        notes=notes,
    )


def estimate_atoms_from_mw(mw: float) -> dict:
    """Estimate atom counts from MW using typical organic drug-like ratios.

    Returns dict: {n_heavy, n_carbon, n_hydrogen, n_heteroatoms}
    Empirical: n_heavy ≈ MW / 7.5
    """
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")

    n_heavy = int(round(mw / 7.5))
    if n_heavy < 1:
        n_heavy = 1
    n_carbon = int(round(0.65 * n_heavy))
    n_hydrogen = int(round(n_heavy * 1.1))
    n_heteroatoms = int(round(0.35 * n_heavy))

    return {
        "n_heavy": n_heavy,
        "n_carbon": n_carbon,
        "n_hydrogen": n_hydrogen,
        "n_heteroatoms": n_heteroatoms,
    }


def mw_from_fragments(fragments: list) -> float:
    """Compute MW from a list of fragment dicts.

    Each fragment: {"smarts": str, "count": int, "fragment_mw": float}
    Returns sum of fragment_mw * count.
    """
    total = 0.0
    for frag in fragments:
        count = frag.get("count", 1)
        frag_mw = frag.get("fragment_mw", 0.0)
        total += frag_mw * count
    return total
