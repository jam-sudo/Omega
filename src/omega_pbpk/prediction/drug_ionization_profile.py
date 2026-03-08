"""
Phase 988 — Drug ionization profile generator.

Computes Henderson-Hasselbalch ionization fractions across pH 0.5-14.0
for acids, bases, and neutral compounds.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IonizationProfileResult:
    """Comprehensive ionization profile for a drug across the pH range."""

    drug_name: str
    pka: float
    ionization_type: str
    ph_values: tuple
    fraction_ionized: tuple
    fraction_neutral: tuple
    ph_at_50pct_ionized: float
    predominantly_ionized_ph_range: tuple
    gi_ionization_fasted: float
    gi_ionization_fed: float
    intestinal_ionization: float
    plasma_ionization: float
    notes: str


def _ionized_fraction(pka: float, ph: float, ionization_type: str) -> float:
    """Compute fraction ionized at a given pH using Henderson-Hasselbalch."""
    if ionization_type == "neutral":
        return 0.0
    elif ionization_type == "acid":
        # HA <-> H+ + A-; fraction ionized = [A-]/([HA]+[A-]) = 1/(1+10^(pKa-pH))
        exponent = pka - ph
    else:  # base
        # BH+ <-> B + H+; fraction ionized = [BH+]/([B]+[BH+]) = 1/(1+10^(pH-pKa))
        exponent = ph - pka

    # Clamp exponent to avoid overflow
    exponent = max(-300.0, min(300.0, exponent))
    return 1.0 / (1.0 + 10.0**exponent)


def compute_ionization_profile(
    drug_name: str,
    pka: float,
    ionization_type: str = "acid",
) -> IonizationProfileResult:
    """
    Generate a comprehensive ionization profile vs pH for a drug.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    pka : float
        pKa value (must be in [0, 14]).
    ionization_type : str
        One of "acid", "base", or "neutral".

    Returns
    -------
    IonizationProfileResult
    """
    if not (0.0 <= pka <= 14.0):
        raise ValueError(f"pka must be between 0 and 14, got {pka}")
    valid_types = {"acid", "base", "neutral"}
    if ionization_type not in valid_types:
        raise ValueError(f"ionization_type must be one of {valid_types}, got '{ionization_type}'")

    # pH grid: 0.5, 1.0, ..., 14.0 (28 values)
    ph_list: list[float] = []
    step = 0.5
    ph = 0.5
    while ph <= 14.0 + 1e-9:
        ph_list.append(round(ph, 1))
        ph += step

    ph_values = tuple(ph_list)

    # Compute fractions
    fi_list: list[float] = []
    fn_list: list[float] = []
    for ph_val in ph_values:
        fi = _ionized_fraction(pka, ph_val, ionization_type)
        fi_list.append(fi)
        fn_list.append(1.0 - fi)

    fraction_ionized = tuple(fi_list)
    fraction_neutral = tuple(fn_list)

    # 50% ionized at pH = pKa by definition
    ph_at_50pct_ionized = pka

    # Predominantly ionized pH range (where fraction_ionized > 0.5)
    if ionization_type == "acid":
        # Ionized when pH > pKa
        predominantly_ionized_ph_range = (round(pka + 0.01, 2), 14.0)
    elif ionization_type == "base":
        # Ionized when pH < pKa
        predominantly_ionized_ph_range = (0.0, round(pka - 0.01, 2))
    else:
        # Neutral: never predominantly ionized; use sentinel
        predominantly_ionized_ph_range = (14.1, 14.1)

    # GI ionization
    gi_ionization_fasted = _ionized_fraction(pka, 2.0, ionization_type)
    gi_ionization_fed = _ionized_fraction(pka, 4.0, ionization_type)

    # Intestinal (pH 6.8) and plasma (pH 7.4)
    intestinal_ionization = _ionized_fraction(pka, 6.8, ionization_type)
    plasma_ionization = _ionized_fraction(pka, 7.4, ionization_type)

    # Build notes
    if ionization_type == "neutral":
        notes = (
            f"{drug_name} is a neutral compound (pKa concept does not apply). "
            "No significant pH-dependent ionization."
        )
    elif ionization_type == "acid":
        notes = (
            f"{drug_name} is a weak acid (pKa={pka:.2f}). "
            f"Predominantly neutral at gastric pH; {plasma_ionization * 100:.1f}% "
            f"ionized at plasma pH 7.4."
        )
    else:
        notes = (
            f"{drug_name} is a weak base (pKa={pka:.2f}). "
            f"Predominantly ionized at gastric pH; {plasma_ionization * 100:.1f}% "
            f"ionized at plasma pH 7.4."
        )

    return IonizationProfileResult(
        drug_name=drug_name,
        pka=pka,
        ionization_type=ionization_type,
        ph_values=ph_values,
        fraction_ionized=fraction_ionized,
        fraction_neutral=fraction_neutral,
        ph_at_50pct_ionized=ph_at_50pct_ionized,
        predominantly_ionized_ph_range=predominantly_ionized_ph_range,
        gi_ionization_fasted=gi_ionization_fasted,
        gi_ionization_fed=gi_ionization_fed,
        intestinal_ionization=intestinal_ionization,
        plasma_ionization=plasma_ionization,
        notes=notes,
    )


def compare_ionization_profiles(drugs: list) -> list:
    """
    Generate and compare ionization profiles for multiple drugs.

    Parameters
    ----------
    drugs : list of dict
        Each dict must have keys: drug_name, pka, ionization_type.

    Returns
    -------
    list of IonizationProfileResult
        Sorted by plasma_ionization ascending.
    """
    results = []
    for drug in drugs:
        result = compute_ionization_profile(
            drug_name=drug["drug_name"],
            pka=drug["pka"],
            ionization_type=drug.get("ionization_type", "acid"),
        )
        results.append(result)

    results.sort(key=lambda r: r.plasma_ionization)
    return results
