"""Predict renal excretion parameters from physicochemical properties.

Uses logP-based sigmoid reabsorption and ionization at urine pH to estimate
CLr, fraction reabsorbed, and pH-sensitivity of renal clearance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

MoleculeType = Literal["acid", "base", "neutral", "zwitterion"]

_VALID_MOLECULE_TYPES = {"acid", "base", "neutral", "zwitterion"}


@dataclass(frozen=True)
class RenalExcretionResult:
    """Result of renal excretion prediction."""

    mw: float
    logP: float
    pka: float
    molecule_type: str
    fu_plasma: float
    gfr_mL_per_min: float
    cl_filtration_mL_per_min: float
    f_reabsorption: float
    cl_renal_mL_per_min: float
    cl_renal_L_per_h: float
    ionization_fraction_at_urine_pH: float
    notes: str


def _ionization_fraction(pka: float, molecule_type: str, urine_pH: float) -> float:
    """Compute ionized fraction at given urine pH.

    For acids:  fi = 1 / (1 + 10^(pKa - pH))
    For bases:  fi = 1 / (1 + 10^(pH - pKa))
    For neutral/zwitterion: fi = 0
    """
    if molecule_type == "acid":
        try:
            return 1.0 / (1.0 + math.pow(10.0, pka - urine_pH))
        except OverflowError:
            return 0.0
    elif molecule_type == "base":
        try:
            return 1.0 / (1.0 + math.pow(10.0, urine_pH - pka))
        except OverflowError:
            return 0.0
    else:
        # neutral or zwitterion: treated as non-ionizable
        return 0.0


def _compute_renal_cl(
    fu_plasma: float,
    gfr_mL_per_min: float,
    logP: float,
    pka: float,
    molecule_type: str,
    urine_pH: float,
) -> tuple[float, float, float, float]:
    """Shared calculation core.

    Returns
    -------
    tuple of (cl_filtration, f_reabs, cl_renal_mL_per_min, ionization_fraction)
    """
    cl_filtration = gfr_mL_per_min * fu_plasma

    # Base reabsorption from logP sigmoid: f = 1/(1+exp(-(logP-1)))
    f_reabs_base = 1.0 / (1.0 + math.exp(-(logP - 1.0)))

    # Ionization reduces reabsorption
    fi = _ionization_fraction(pka, molecule_type, urine_pH)
    f_reabs = f_reabs_base * (1.0 - fi * 0.8)
    f_reabs = max(0.0, min(f_reabs, 1.0))

    cl_renal = cl_filtration * (1.0 - f_reabs)
    return cl_filtration, f_reabs, cl_renal, fi


def predict_renal_excretion(
    mw: float,
    logP: float,
    pka: float,
    molecule_type: str,
    fu_plasma: float,
    gfr_mL_per_min: float = 120.0,
) -> RenalExcretionResult:
    """Predict renal excretion parameters from physicochemical properties.

    Parameters
    ----------
    mw : float
        Molecular weight (Da), must be > 0.
    logP : float
        Lipophilicity (log octanol/water partition coefficient).
    pka : float
        Most relevant pKa value.
    molecule_type : str
        One of 'acid', 'base', 'neutral', 'zwitterion'.
    fu_plasma : float
        Fraction unbound in plasma (0, 1].
    gfr_mL_per_min : float
        Glomerular filtration rate (mL/min), default 120. Must be > 0.

    Returns
    -------
    RenalExcretionResult
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if not (-10 <= logP <= 15):
        raise ValueError("logP must be between -10 and 15")
    if not math.isfinite(pka):
        raise ValueError("pka must be a finite number")
    if molecule_type not in _VALID_MOLECULE_TYPES:
        raise ValueError(
            f"molecule_type must be one of {_VALID_MOLECULE_TYPES}, got '{molecule_type}'"
        )
    if not (0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be in (0, 1]")
    if gfr_mL_per_min <= 0:
        raise ValueError("gfr_mL_per_min must be > 0")

    urine_pH = 6.0
    cl_filtration, f_reabs, cl_renal, fi = _compute_renal_cl(
        fu_plasma, gfr_mL_per_min, logP, pka, molecule_type, urine_pH
    )

    cl_renal_L_per_h = cl_renal * 60.0 / 1000.0

    notes_parts = [
        f"CLf={cl_filtration:.2f} mL/min",
        f"f_reabs={f_reabs:.3f}",
        f"CLr={cl_renal:.2f} mL/min at urine pH {urine_pH}",
    ]
    if molecule_type in ("acid", "base"):
        notes_parts.append(f"ionized fraction={fi:.3f}")

    return RenalExcretionResult(
        mw=mw,
        logP=logP,
        pka=pka,
        molecule_type=molecule_type,
        fu_plasma=fu_plasma,
        gfr_mL_per_min=gfr_mL_per_min,
        cl_filtration_mL_per_min=cl_filtration,
        f_reabsorption=f_reabs,
        cl_renal_mL_per_min=cl_renal,
        cl_renal_L_per_h=cl_renal_L_per_h,
        ionization_fraction_at_urine_pH=fi,
        notes="; ".join(notes_parts),
    )


def urine_pH_sensitivity(
    mw: float,
    logP: float,
    pka: float,
    molecule_type: str,
    fu_plasma: float,
    gfr_mL_per_min: float,
    urine_pH_values: list[float],
) -> list[dict]:
    """Sweep urine pH values and compute CLr at each pH.

    Parameters
    ----------
    mw : float
        Molecular weight (Da), must be > 0.
    logP : float
        Lipophilicity.
    pka : float
        Most relevant pKa.
    molecule_type : str
        One of 'acid', 'base', 'neutral', 'zwitterion'.
    fu_plasma : float
        Fraction unbound in plasma (0, 1].
    gfr_mL_per_min : float
        GFR (mL/min), must be > 0.
    urine_pH_values : list[float]
        pH values to evaluate (typically 4.5–8.5).

    Returns
    -------
    list[dict]
        One dict per pH with keys: urine_pH, cl_filtration_mL_per_min,
        f_reabsorption, cl_renal_mL_per_min, cl_renal_L_per_h,
        ionization_fraction.
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if not (-10 <= logP <= 15):
        raise ValueError("logP must be between -10 and 15")
    if not math.isfinite(pka):
        raise ValueError("pka must be a finite number")
    if molecule_type not in _VALID_MOLECULE_TYPES:
        raise ValueError(
            f"molecule_type must be one of {_VALID_MOLECULE_TYPES}, got '{molecule_type}'"
        )
    if not (0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be in (0, 1]")
    if gfr_mL_per_min <= 0:
        raise ValueError("gfr_mL_per_min must be > 0")
    if not urine_pH_values:
        raise ValueError("urine_pH_values must be a non-empty list")

    results: list[dict] = []
    for ph in urine_pH_values:
        cl_filtration, f_reabs, cl_renal, fi = _compute_renal_cl(
            fu_plasma, gfr_mL_per_min, logP, pka, molecule_type, ph
        )
        cl_renal_L_per_h = cl_renal * 60.0 / 1000.0
        results.append(
            {
                "urine_pH": ph,
                "cl_filtration_mL_per_min": cl_filtration,
                "f_reabsorption": f_reabs,
                "cl_renal_mL_per_min": cl_renal,
                "cl_renal_L_per_h": cl_renal_L_per_h,
                "ionization_fraction": fi,
            }
        )

    return results


__all__ = [
    "RenalExcretionResult",
    "predict_renal_excretion",
    "urine_pH_sensitivity",
]
