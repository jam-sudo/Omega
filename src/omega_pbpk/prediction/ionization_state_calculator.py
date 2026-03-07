"""
Phase 374 — Drug Ionization State Calculator

Calculates ionization state and charge distribution of drugs across
a range of pH values using Henderson-Hasselbalch equations.
"""

import math
from dataclasses import dataclass


@dataclass
class IonizationStateResult:
    """Result of ionization state calculation across pH range."""

    drug_name: str
    drug_type: str  # "acid", "base", "zwitterion", "neutral"
    pka_acid: float  # pKa of acidic group (math.nan if none)
    pka_base: float  # pKa of basic group (math.nan if none)
    ph_values: list
    fraction_neutral: list
    fraction_ionized: list
    fraction_cationic: list
    fraction_anionic: list
    ph_50_ionized: float  # pH at which 50% ionized
    optimal_absorption_ph: float
    gi_ionization_fraction: float  # fraction ionized at GI pH=6.5
    notes: str


_VALID_DRUG_TYPES = frozenset({"acid", "base", "zwitterion", "neutral"})
_GI_PH = 6.5


def _linspace(start: float, stop: float, n: int) -> list:
    """Generate n evenly spaced points from start to stop inclusive."""
    if n <= 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def _ionization_at_ph(
    drug_type: str,
    pka_acid: float | None,
    pka_base: float | None,
    ph: float,
) -> tuple[float, float, float, float]:
    """
    Compute (fraction_neutral, fraction_ionized, fraction_cationic, fraction_anionic)
    at a given pH for the specified drug type.
    """
    if drug_type == "neutral":
        return 1.0, 0.0, 0.0, 0.0

    if drug_type == "acid":
        # Henderson-Hasselbalch: HA <-> H+ + A-
        # fraction ionized (anionic) = 1 / (1 + 10^(pKa - pH))
        f_anionic = 1.0 / (1.0 + 10.0 ** (pka_acid - ph))
        f_neutral = 1.0 - f_anionic
        f_cationic = 0.0
        f_ionized = f_anionic
        return f_neutral, f_ionized, f_cationic, f_anionic

    if drug_type == "base":
        # BH+ <-> B + H+
        # fraction ionized (cationic) = 1 / (1 + 10^(pH - pKa))
        f_cationic = 1.0 / (1.0 + 10.0 ** (ph - pka_base))
        f_neutral = 1.0 - f_cationic
        f_anionic = 0.0
        f_ionized = f_cationic
        return f_neutral, f_ionized, f_cationic, f_anionic

    if drug_type == "zwitterion":
        # Approximate: combine acid and base ionizations
        # f_anionic from acid group: 1/(1+10^(pKa_acid - pH))
        f_anionic = 1.0 / (1.0 + 10.0 ** (pka_acid - ph))
        # f_cationic from base group: 1/(1+10^(pH - pKa_base))
        f_cationic = 1.0 / (1.0 + 10.0 ** (ph - pka_base))
        # neutral = max(0, 1 - f_anionic - f_cationic)
        f_neutral = max(0.0, 1.0 - f_anionic - f_cationic)
        f_ionized = f_anionic + f_cationic
        # Clamp ionized to [0, 1]
        if f_ionized > 1.0:
            f_ionized = 1.0
        return f_neutral, f_ionized, f_cationic, f_anionic

    raise ValueError(f"Unknown drug_type: {drug_type}")


def calculate_ionization(
    drug_name: str,
    drug_type: str,
    pka_acid: float | None = None,
    pka_base: float | None = None,
    ph_range: tuple[float, float] = (1.0, 10.0),
    n_points: int = 50,
) -> IonizationStateResult:
    """
    Calculate ionization state distribution across a pH range.

    Parameters
    ----------
    drug_name : str
    drug_type : str — "acid", "base", "zwitterion", or "neutral"
    pka_acid : float or None — pKa of acidic group (required for acid/zwitterion)
    pka_base : float or None — pKa of basic group (required for base/zwitterion)
    ph_range : (float, float) — (pH_min, pH_max) for calculation
    n_points : int — number of pH points (default 50)

    Returns
    -------
    IonizationStateResult
    """
    # Validation
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(f"drug_type must be one of {sorted(_VALID_DRUG_TYPES)}, got '{drug_type}'")

    if drug_type == "acid" and pka_acid is None:
        raise ValueError("pka_acid is required for drug_type='acid'")
    if drug_type == "base" and pka_base is None:
        raise ValueError("pka_base is required for drug_type='base'")
    if drug_type == "zwitterion" and (pka_acid is None or pka_base is None):
        raise ValueError("Both pka_acid and pka_base are required for drug_type='zwitterion'")

    pka_acid_val = pka_acid if pka_acid is not None else math.nan
    pka_base_val = pka_base if pka_base is not None else math.nan

    ph_min, ph_max = ph_range
    ph_values = _linspace(ph_min, ph_max, n_points)

    fraction_neutral = []
    fraction_ionized = []
    fraction_cationic = []
    fraction_anionic = []

    for ph in ph_values:
        fn, fi, fc, fa = _ionization_at_ph(drug_type, pka_acid, pka_base, ph)
        fraction_neutral.append(fn)
        fraction_ionized.append(fi)
        fraction_cationic.append(fc)
        fraction_anionic.append(fa)

    # pH at which 50% is ionized
    if drug_type == "acid":
        ph_50_ionized = pka_acid_val
    elif drug_type == "base":
        ph_50_ionized = pka_base_val
    elif drug_type == "zwitterion":
        ph_50_ionized = min(pka_acid_val, pka_base_val)
    else:
        ph_50_ionized = math.nan

    # Optimal absorption pH = pH where fraction_neutral is maximum
    max_fn = -1.0
    optimal_ph = ph_values[0]
    for ph, fn in zip(ph_values, fraction_neutral):
        if fn > max_fn:
            max_fn = fn
            optimal_ph = ph
    optimal_absorption_ph = optimal_ph

    # GI ionization fraction at pH=6.5
    _, gi_fi, _, _ = _ionization_at_ph(drug_type, pka_acid, pka_base, _GI_PH)
    gi_ionization_fraction = gi_fi

    # Notes
    notes_parts = [f"drug_type={drug_type}"]
    if not math.isnan(pka_acid_val):
        notes_parts.append(f"pKa_acid={pka_acid_val:.2f}")
    if not math.isnan(pka_base_val):
        notes_parts.append(f"pKa_base={pka_base_val:.2f}")
    notes_parts.append(f"optimal_pH={optimal_absorption_ph:.2f}")
    notes_parts.append(f"gi_ionized={gi_ionization_fraction:.3f}")
    notes = ", ".join(notes_parts)

    return IonizationStateResult(
        drug_name=drug_name,
        drug_type=drug_type,
        pka_acid=pka_acid_val,
        pka_base=pka_base_val,
        ph_values=ph_values,
        fraction_neutral=fraction_neutral,
        fraction_ionized=fraction_ionized,
        fraction_cationic=fraction_cationic,
        fraction_anionic=fraction_anionic,
        ph_50_ionized=ph_50_ionized,
        optimal_absorption_ph=optimal_absorption_ph,
        gi_ionization_fraction=gi_ionization_fraction,
        notes=notes,
    )


def screen_ionization_window(
    drug_name: str,
    drug_type: str,
    pka_acid: float | None = None,
    pka_base: float | None = None,
    target_ph_range: tuple[float, float] = (6.0, 8.0),
) -> dict:
    """
    Screen the neutral fraction within a target pH window.

    Parameters
    ----------
    drug_name : str
    drug_type : str
    pka_acid : float or None
    pka_base : float or None
    target_ph_range : (float, float) — pH window of interest (default 6.0–8.0)

    Returns
    -------
    dict with keys:
        "mean_neutral_fraction", "min_neutral_fraction", "max_neutral_fraction",
        "ph_values", "neutral_fractions"
    """
    # Validate (reuse calculate_ionization logic)
    result = calculate_ionization(
        drug_name=drug_name,
        drug_type=drug_type,
        pka_acid=pka_acid,
        pka_base=pka_base,
        ph_range=target_ph_range,
        n_points=50,
    )

    neutral_fractions = result.fraction_neutral
    ph_values = result.ph_values

    mean_nf = sum(neutral_fractions) / len(neutral_fractions)
    min_nf = min(neutral_fractions)
    max_nf = max(neutral_fractions)

    return {
        "mean_neutral_fraction": mean_nf,
        "min_neutral_fraction": min_nf,
        "max_neutral_fraction": max_nf,
        "ph_values": ph_values,
        "neutral_fractions": neutral_fractions,
    }
