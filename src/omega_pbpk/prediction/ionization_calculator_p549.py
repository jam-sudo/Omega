"""Drug ionization state calculator — Phase 549.

Computes ionization state of drugs across pH range using Henderson-Hasselbalch.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_DRUG_TYPES = {"acid", "base", "amphoteric"}


@dataclass(frozen=True)
class IonizationResult:
    """Result of ionization state calculation across a pH range."""

    drug_name: str
    pKa: float
    drug_type: str
    ph_values: tuple  # type: ignore[type-arg]
    ionized_fractions: tuple  # type: ignore[type-arg]
    unionized_fractions: tuple  # type: ignore[type-arg]
    ph_50_pct_ionized: float
    dominant_form_at_physiological_ph: str
    notes: str


def ionization_at_ph(pKa: float, drug_type: str, ph: float) -> float:
    """Return ionized fraction at a single pH value.

    Parameters
    ----------
    pKa : float
        Acid dissociation constant.
    drug_type : str
        One of "acid", "base", "amphoteric".
    ph : float
        pH value.

    Returns
    -------
    float
        Ionized fraction in [0, 1].
    """
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(
            f"Invalid drug_type '{drug_type}'. Must be one of {sorted(_VALID_DRUG_TYPES)}."
        )

    if drug_type == "acid":
        return 1.0 / (1.0 + 10.0 ** (pKa - ph))
    else:
        # base and amphoteric use base formula
        return 1.0 / (1.0 + 10.0 ** (ph - pKa))


def calculate_ionization(
    drug_name: str,
    pKa: float,
    drug_type: str,
    ph_range: tuple = (1.0, 10.0),  # type: ignore[type-arg]
    n_points: int = 19,
) -> IonizationResult:
    """Calculate ionization profile across a pH range.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    pKa : float
        Acid dissociation constant.
    drug_type : str
        One of "acid", "base", "amphoteric".
    ph_range : tuple
        (min_ph, max_ph) range to evaluate.
    n_points : int
        Number of evenly spaced pH points (inclusive endpoints).

    Returns
    -------
    IonizationResult
    """
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(
            f"Invalid drug_type '{drug_type}'. Must be one of {sorted(_VALID_DRUG_TYPES)}."
        )

    ph_min, ph_max = ph_range
    if n_points < 2:
        raise ValueError("n_points must be >= 2")

    step = (ph_max - ph_min) / (n_points - 1)
    ph_values = tuple(ph_min + i * step for i in range(n_points))

    ionized_fractions = tuple(ionization_at_ph(pKa, drug_type, ph) for ph in ph_values)
    unionized_fractions = tuple(1.0 - f for f in ionized_fractions)

    ph_50_pct_ionized = pKa

    ionized_at_7_4 = ionization_at_ph(pKa, drug_type, 7.4)
    dominant_form = "ionized" if ionized_at_7_4 > 0.5 else "unionized"

    return IonizationResult(
        drug_name=drug_name,
        pKa=pKa,
        drug_type=drug_type,
        ph_values=ph_values,
        ionized_fractions=ionized_fractions,
        unionized_fractions=unionized_fractions,
        ph_50_pct_ionized=ph_50_pct_ionized,
        dominant_form_at_physiological_ph=dominant_form,
        notes=f"Henderson-Hasselbalch ionization profile for {drug_type} drug",
    )


def compare_ionization_profiles(
    compounds: list,  # type: ignore[type-arg]
) -> list:  # type: ignore[type-arg]
    """Compare ionization profiles of multiple compounds.

    Parameters
    ----------
    compounds : list[dict]
        Each dict must have keys: drug_name, pKa, drug_type.

    Returns
    -------
    list[IonizationResult]
        Sorted by ph_50_pct_ionized ascending.
    """
    results = [
        calculate_ionization(
            drug_name=c["drug_name"],
            pKa=c["pKa"],
            drug_type=c["drug_type"],
        )
        for c in compounds
    ]
    results.sort(key=lambda r: r.ph_50_pct_ionized)
    return results
