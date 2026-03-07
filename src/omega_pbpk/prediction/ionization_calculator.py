"""Drug ionization and charge state prediction — Phase 485."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GIChargeResult:
    """Charge state across GI compartments."""

    drug_type: str
    pka: float
    stomach_fraction_neutral: float  # pH 1.5
    duodenum_fraction_neutral: float  # pH 6.0
    jejunum_fraction_neutral: float  # pH 6.5
    ileum_fraction_neutral: float  # pH 7.2
    colon_fraction_neutral: float  # pH 7.4
    absorption_favorable_segment: str
    notes: str


# GI segment pH values
_GI_PH = {
    "stomach": 1.5,
    "duodenum": 6.0,
    "jejunum": 6.5,
    "ileum": 7.2,
    "colon": 7.4,
}

_VALID_DRUG_TYPES = ("acid", "base", "neutral", "zwitterion")


def _validate_drug_type(drug_type: str) -> None:
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(
            f"drug_type must be one of {_VALID_DRUG_TYPES}, got '{drug_type}'"
        )


def _validate_ph(ph: float) -> None:
    if not (0.0 <= ph <= 14.0):
        raise ValueError(f"pH must be between 0 and 14, got {ph}")


def _validate_pka(pka: float) -> None:
    if not (-5.0 <= pka <= 20.0):
        raise ValueError(f"pKa must be between -5 and 20, got {pka}")


def fraction_ionized(ph: float, pka: float, drug_type: str = "acid") -> float:
    """Return the fraction of drug in ionized form at given pH.

    Henderson-Hasselbalch:
    - Acid:    fi = 1 / (1 + 10^(pKa - pH))
    - Base:    fi = 1 / (1 + 10^(pH - pKa))
    - Neutral: fi = 0.0  (never ionized)
    """
    _validate_drug_type(drug_type)
    _validate_ph(ph)
    _validate_pka(pka)

    if drug_type == "neutral":
        return 0.0

    # Henderson-Hasselbalch:
    #   Acid: fi = 1/(1 + 10^(pKa-pH))  => compute 10^(pKa-pH) directly
    #   Base: fi = 1/(1 + 10^(pH-pKa))  => compute 10^(pH-pKa) directly
    if drug_type == "acid":
        exp_val = pka - ph
    elif drug_type == "base":
        exp_val = ph - pka
    else:
        # zwitterion: use base-like ionization
        exp_val = ph - pka

    # Clamp exp_val to avoid overflow
    exp_val = max(-100.0, min(100.0, exp_val))
    return 1.0 / (1.0 + math.pow(10.0, exp_val))


def fraction_neutral(ph: float, pka: float, drug_type: str = "acid") -> float:
    """Return the fraction of drug in neutral (un-ionized) form at given pH."""
    return 1.0 - fraction_ionized(ph, pka, drug_type)


def effective_charge(
    ph: float,
    pka_values: list[float],
    drug_types: list[str],
) -> float:
    """Return net charge of a multi-protic drug at given pH.

    Each ionized acid contributes -1; each ionized base contributes +1.
    """
    _validate_ph(ph)
    if len(pka_values) != len(drug_types):
        raise ValueError("pka_values and drug_types must have the same length")
    if not pka_values:
        raise ValueError("pka_values must not be empty")

    charge = 0.0
    for pka, dtype in zip(pka_values, drug_types, strict=True):
        _validate_pka(pka)
        _validate_drug_type(dtype)
        fi = fraction_ionized(ph, pka, dtype)
        if dtype == "acid":
            charge += -1.0 * fi
        elif dtype == "base":
            charge += 1.0 * fi
        elif dtype == "zwitterion":
            charge += 1.0 * fi
        # neutral contributes 0
    return charge


def ionization_profile(
    pka: float,
    drug_type: str = "acid",
    ph_range: tuple[float, float] = (1, 14),
    n_points: int = 50,
) -> dict:
    """Return ionization profile across a pH range.

    Returns dict with:
    - 'ph_values': list of pH values
    - 'fraction_ionized': list of fi values
    - 'fraction_neutral': list of fn values
    - 'pka': pka
    - 'drug_type': drug_type
    """
    _validate_pka(pka)
    _validate_drug_type(drug_type)
    ph_min, ph_max = ph_range
    if ph_min < 0 or ph_max > 14 or ph_min >= ph_max:
        raise ValueError(
            f"ph_range must be within [0, 14] and min < max, got {ph_range}"
        )
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")

    step = (ph_max - ph_min) / (n_points - 1)
    ph_values = [ph_min + i * step for i in range(n_points)]

    fi_values = [fraction_ionized(ph, pka, drug_type) for ph in ph_values]
    fn_values = [1.0 - fi for fi in fi_values]

    return {
        "ph_values": ph_values,
        "fraction_ionized": fi_values,
        "fraction_neutral": fn_values,
        "pka": pka,
        "drug_type": drug_type,
    }


def predict_gi_charge_states(pka: float, drug_type: str = "acid") -> GIChargeResult:
    """Predict charge state at each GI segment and identify most favorable for absorption."""
    _validate_pka(pka)
    _validate_drug_type(drug_type)

    fn = {
        seg: fraction_neutral(_GI_PH[seg], pka, drug_type)
        for seg in _GI_PH
    }

    # Best absorption where fraction neutral is highest (neutral form permeates best)
    best_seg = max(fn, key=lambda s: fn[s])

    # Build notes
    if drug_type == "acid":
        if pka < 3.0:
            notes = "Strong acid; largely ionized across all GI segments."
        elif pka < 6.0:
            notes = "Weak acid; stomach absorption favored due to low pH."
        else:
            notes = "Very weak acid; predominantly neutral across GI tract."
    elif drug_type == "base":
        if pka > 10.0:
            notes = "Strong base; largely ionized across all GI segments."
        elif pka > 7.0:
            notes = "Moderate base; significant ionization in upper GI."
        else:
            notes = "Weak base; predominantly neutral in intestine; intestinal absorption favored."
    elif drug_type == "neutral":
        notes = "Neutral compound; absorption not pH-dependent."
        best_seg = "jejunum"  # default for neutral
    else:
        notes = "Zwitterion; charge state complex; intestinal absorption likely."

    return GIChargeResult(
        drug_type=drug_type,
        pka=pka,
        stomach_fraction_neutral=fn["stomach"],
        duodenum_fraction_neutral=fn["duodenum"],
        jejunum_fraction_neutral=fn["jejunum"],
        ileum_fraction_neutral=fn["ileum"],
        colon_fraction_neutral=fn["colon"],
        absorption_favorable_segment=best_seg,
        notes=notes,
    )


def solubility_from_ionization(
    intrinsic_sol_mg_mL: float,
    ph: float,
    pka: float,
    drug_type: str = "acid",
) -> float:
    """Calculate pH-dependent solubility using Henderson-Hasselbalch.

    S(pH) = S0 * (1 + 10^(pH - pKa))  for acids
    S(pH) = S0 * (1 + 10^(pKa - pH))  for bases
    Neutral: S(pH) = S0 (no pH dependence)

    Assumes ionized species is freely soluble (common approximation).
    """
    _validate_ph(ph)
    _validate_pka(pka)
    _validate_drug_type(drug_type)

    if intrinsic_sol_mg_mL < 0:
        raise ValueError(
            f"intrinsic_sol_mg_mL must be >= 0, got {intrinsic_sol_mg_mL}"
        )

    if drug_type == "neutral":
        return intrinsic_sol_mg_mL

    if drug_type == "acid":
        exponent = ph - pka
    elif drug_type == "base":
        exponent = pka - ph
    else:
        # zwitterion: use base-like behavior
        exponent = pka - ph

    # Clamp to avoid overflow
    exponent = max(-100.0, min(50.0, exponent))
    factor = 1.0 + math.pow(10.0, exponent)
    return intrinsic_sol_mg_mL * factor
