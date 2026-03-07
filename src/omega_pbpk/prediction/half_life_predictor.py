"""Half-life prediction from physicochemical properties (Phase 378)."""

from __future__ import annotations

import math
from dataclasses import dataclass

_BODY_WEIGHT_KG = 70.0

_VALID_ROUTES = frozenset({"hepatic", "renal", "mixed"})


@dataclass(frozen=True)
class HalfLifeResult:
    """Result of half-life prediction."""

    mw: float
    logP: float
    psa: float
    n_hbd: int
    fu_plasma: float
    route_of_elimination: str
    t_half_h: float
    cl_L_per_h_per_kg: float
    vd_L_per_kg: float
    vd_L: float            # for 70 kg
    cl_L_per_h: float      # for 70 kg
    half_life_class: str
    confidence: str
    notes: str


def _cl_hepatic(fu_plasma: float, logP: float) -> float:
    """Empirical hepatic CL (L/h/kg)."""
    return 15.0 * fu_plasma * math.exp(-0.3 * logP)


def _cl_renal(fu_plasma: float, logP: float) -> float:
    """Empirical renal CL (L/h/kg)."""
    return 6.0 * fu_plasma * math.exp(-0.1 * logP)


def _vd_per_kg(mw: float, logP: float, psa: float, n_hbd: int) -> float:
    """Empirical Vd (L/kg), clamped to [0.3, 200]."""
    vd = 0.5 + 0.3 * logP + 0.01 * mw - 0.01 * psa - 0.05 * n_hbd
    return max(0.3, min(200.0, vd))


def classify_half_life(t_half_h: float) -> str:
    """Classify half-life into a descriptive category.

    Parameters
    ----------
    t_half_h:
        Half-life in hours.

    Returns
    -------
    str
        One of "ultra-short", "short", "medium", "long", "very-long".
    """
    if t_half_h < 0:
        raise ValueError(f"t_half_h must be non-negative; got {t_half_h}.")
    if t_half_h < 1.0:
        return "ultra-short"
    if t_half_h < 6.0:
        return "short"
    if t_half_h < 24.0:
        return "medium"
    if t_half_h < 72.0:
        return "long"
    return "very-long"


def predict_half_life(
    mw: float,
    logP: float,
    psa: float,
    n_hbd: int,
    fu_plasma: float,
    route_of_elimination: str,
) -> HalfLifeResult:
    """Predict drug half-life from physicochemical descriptors.

    Parameters
    ----------
    mw:
        Molecular weight (g/mol). Must be > 0.
    logP:
        Octanol-water partition coefficient (log scale).
    psa:
        Polar surface area (Å²). Must be ≥ 0.
    n_hbd:
        Number of hydrogen bond donors. Must be ≥ 0.
    fu_plasma:
        Fraction unbound in plasma (0 < fu_plasma ≤ 1).
    route_of_elimination:
        Primary route: "hepatic", "renal", or "mixed".

    Returns
    -------
    HalfLifeResult
    """
    # --- Input validation ---
    if mw <= 0:
        raise ValueError(f"mw must be positive; got {mw}.")
    if psa < 0:
        raise ValueError(f"psa must be non-negative; got {psa}.")
    if n_hbd < 0:
        raise ValueError(f"n_hbd must be non-negative; got {n_hbd}.")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1]; got {fu_plasma}.")
    if route_of_elimination not in _VALID_ROUTES:
        raise ValueError(
            f"route_of_elimination must be one of {sorted(_VALID_ROUTES)}; "
            f"got '{route_of_elimination}'."
        )

    # --- CL prediction ---
    if route_of_elimination == "hepatic":
        cl_per_kg = _cl_hepatic(fu_plasma, logP)
    elif route_of_elimination == "renal":
        cl_per_kg = _cl_renal(fu_plasma, logP)
    else:  # mixed
        cl_per_kg = 0.5 * (_cl_hepatic(fu_plasma, logP) + _cl_renal(fu_plasma, logP))

    # Ensure CL > 0 (cannot be 0 due to positive inputs, but guard anyway)
    cl_per_kg = max(1e-9, cl_per_kg)

    # --- Vd prediction ---
    vd_per_kg = _vd_per_kg(mw, logP, psa, n_hbd)

    # --- Half-life ---
    t_half_h = 0.693147 * vd_per_kg / cl_per_kg

    # --- Scaled to 70 kg ---
    vd_L = vd_per_kg * _BODY_WEIGHT_KG
    cl_L_per_h = cl_per_kg * _BODY_WEIGHT_KG

    # --- Classification ---
    half_life_class = classify_half_life(t_half_h)

    # --- Confidence (based on logP range) ---
    if -1.0 <= logP <= 5.0:
        confidence = "high"
    elif -3.0 <= logP <= 7.0:
        confidence = "moderate"
    else:
        confidence = "low"

    notes = (
        f"Empirical model: CL={cl_per_kg:.4f} L/h/kg, Vd={vd_per_kg:.2f} L/kg, "
        f"t½={t_half_h:.2f} h ({half_life_class}). "
        f"Confidence based on logP={logP:.2f} range: {confidence}."
    )

    return HalfLifeResult(
        mw=mw,
        logP=logP,
        psa=psa,
        n_hbd=n_hbd,
        fu_plasma=fu_plasma,
        route_of_elimination=route_of_elimination,
        t_half_h=t_half_h,
        cl_L_per_h_per_kg=cl_per_kg,
        vd_L_per_kg=vd_per_kg,
        vd_L=vd_L,
        cl_L_per_h=cl_L_per_h,
        half_life_class=half_life_class,
        confidence=confidence,
        notes=notes,
    )


__all__ = [
    "HalfLifeResult",
    "predict_half_life",
    "classify_half_life",
]
