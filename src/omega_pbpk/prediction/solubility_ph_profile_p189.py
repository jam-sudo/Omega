"""Phase 189 — pH-solubility profile for ionizable drugs using Henderson-Hasselbalch.

Generates a full pH-solubility profile across a given pH range for acid, base,
or neutral ionization types.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "SolubilityPhProfileResult",
    "generate_ph_profile",
    "compare_ionization_types",
]

_VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}


def _default_ph_range() -> list[float]:
    """Return default pH range: 1.0 to 9.0 in 0.5 steps."""
    result: list[float] = []
    ph = 1.0
    while ph <= 9.0 + 1e-9:
        result.append(round(ph, 10))
        ph += 0.5
    return result


def _solubility_at_ph(
    intrinsic_solubility_mg_mL: float,
    pka: float,
    ionization_type: str,
    ph: float,
) -> float:
    """Compute solubility at a single pH using Henderson-Hasselbalch."""
    if ionization_type == "acid":
        return intrinsic_solubility_mg_mL * (1.0 + 10.0 ** (ph - pka))
    if ionization_type == "base":
        return intrinsic_solubility_mg_mL * (1.0 + 10.0 ** (pka - ph))
    # neutral
    return intrinsic_solubility_mg_mL


@dataclass
class SolubilityPhProfileResult:
    """Full pH-solubility profile result."""

    drug_name: str
    ionization_type: str
    pka: float
    intrinsic_solubility_mg_mL: float
    ph_values: list = field(default_factory=list)
    solubility_mg_mL: list = field(default_factory=list)
    ph_at_max_solubility: float = 0.0
    ph_at_min_solubility: float = 0.0
    max_solubility_mg_mL: float = 0.0
    min_solubility_mg_mL: float = 0.0
    fold_increase: float = 1.0
    notes: str = ""


def generate_ph_profile(
    drug_name: str,
    intrinsic_solubility_mg_mL: float,
    pka: float,
    ionization_type: str,
    ph_range: list[float] | None = None,
) -> SolubilityPhProfileResult:
    """Generate a full pH-solubility profile using Henderson-Hasselbalch.

    Args:
        drug_name: Name of the drug.
        intrinsic_solubility_mg_mL: Intrinsic solubility of the neutral form (mg/mL).
            Must be > 0.
        pka: pKa of the ionizable group. Must be in [0, 14].
        ionization_type: One of "acid", "base", or "neutral".
        ph_range: List of pH values to evaluate. Defaults to 1.0 to 9.0 in 0.5 steps.

    Returns:
        SolubilityPhProfileResult with profile data.

    Raises:
        ValueError: On invalid inputs.
    """
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError(
            f"intrinsic_solubility_mg_mL must be > 0, got {intrinsic_solubility_mg_mL}"
        )
    if not (0.0 <= pka <= 14.0):
        raise ValueError(f"pka must be in [0, 14], got {pka}")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION_TYPES}, got {ionization_type!r}"
        )

    if ph_range is None:
        ph_range = _default_ph_range()

    ph_values = list(ph_range)
    solubility_values = [
        _solubility_at_ph(intrinsic_solubility_mg_mL, pka, ionization_type, ph) for ph in ph_values
    ]

    max_sol = max(solubility_values)
    min_sol = min(solubility_values)
    max_idx = solubility_values.index(max_sol)
    min_idx = solubility_values.index(min_sol)
    ph_at_max = ph_values[max_idx]
    ph_at_min = ph_values[min_idx]

    fold_increase = max_sol / min_sol if min_sol > 0 else 1.0

    # Determine pH at which S = 10 * S0 (if exists in range), if applicable
    target_10x = intrinsic_solubility_mg_mL * 10.0
    ph_10x_note = ""
    if ionization_type in ("acid", "base"):
        # For acid: pH_10x = pKa + log10(9) ~ pKa + 0.954
        # For base: pH_10x = pKa - log10(9) ~ pKa - 0.954
        log9 = 0.9542425094393248  # math.log10(9)
        if ionization_type == "acid":
            ph_10x = pka + log9
        else:
            ph_10x = pka - log9

        ph_min_range = min(ph_values)
        ph_max_range = max(ph_values)
        if ph_min_range <= ph_10x <= ph_max_range:
            ph_10x_note = f"; pH at S=10xS0: {ph_10x:.2f}"
        # Verify via direct calculation
        check_sol = _solubility_at_ph(intrinsic_solubility_mg_mL, pka, ionization_type, ph_10x)
        if abs(check_sol - target_10x) > target_10x * 0.01:
            ph_10x_note = ""

    notes_parts = [
        f"ionization_type={ionization_type}",
        f"pKa={pka}",
        f"S0={intrinsic_solubility_mg_mL:.4f} mg/mL",
        f"pH range {min(ph_values):.1f}-{max(ph_values):.1f}",
        f"fold_increase={fold_increase:.2f}",
    ]
    if ph_10x_note:
        notes_parts.append(ph_10x_note.lstrip("; "))
    notes = "; ".join(notes_parts)

    return SolubilityPhProfileResult(
        drug_name=drug_name,
        ionization_type=ionization_type,
        pka=pka,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        ph_values=ph_values,
        solubility_mg_mL=solubility_values,
        ph_at_max_solubility=ph_at_max,
        ph_at_min_solubility=ph_at_min,
        max_solubility_mg_mL=max_sol,
        min_solubility_mg_mL=min_sol,
        fold_increase=fold_increase,
        notes=notes,
    )


def compare_ionization_types(
    drug_name: str,
    s0: float,
    pka: float,
    ph_range: list[float] | None = None,
) -> list[SolubilityPhProfileResult]:
    """Generate profiles for acid, base, and neutral ionization types.

    Args:
        drug_name: Name of the drug.
        s0: Intrinsic solubility of the neutral form (mg/mL).
        pka: pKa value.
        ph_range: Optional list of pH values. Defaults to 1.0-9.0 in 0.5 steps.

    Returns:
        List of 3 SolubilityPhProfileResult (acid, base, neutral).
    """
    results: list[SolubilityPhProfileResult] = []
    for ion_type in ("acid", "base", "neutral"):
        result = generate_ph_profile(
            drug_name=drug_name,
            intrinsic_solubility_mg_mL=s0,
            pka=pka,
            ionization_type=ion_type,
            ph_range=ph_range,
        )
        results.append(result)
    return results
