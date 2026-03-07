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
    vd_L: float  # for 70 kg
    cl_L_per_h: float  # for 70 kg
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
    "HalfLifePrediction",
    "predict_half_life_from_props",
    "screen_half_lives",
]


# ---------------------------------------------------------------------------
# Phase 641: QSPR-based half-life prediction from physicochemical properties
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HalfLifePrediction:
    """QSPR-predicted half-life from physicochemical properties."""

    compound_name: str
    predicted_t_half_h: float
    ci_lower_h: float
    ci_upper_h: float
    predicted_ke_per_h: float
    vd_class: str
    cl_class: str
    elimination_half_lives_in_24h: float
    bcs_class: str
    notes: str


def _bcs_class(logP: float, mw: float, psa: float, solubility_mg_mL: float) -> str:
    """Classify BCS class based on permeability and solubility proxies."""
    # Permeability proxy: high if logP in 0-3 and PSA < 140 and MW < 500
    high_perm = logP >= 0.0 and psa < 140.0 and mw < 500.0
    # Solubility proxy: high if solubility >= 1 mg/mL
    high_sol = solubility_mg_mL >= 1.0
    if high_perm and high_sol:
        return "I"
    if not high_perm and high_sol:
        return "III"
    if high_perm and not high_sol:
        return "II"
    return "IV"


def _vd_class(logP: float) -> tuple[str, float]:
    """Return (vd_class, vd_factor) based on logP."""
    vd_factor = max(0.2, logP * 0.5 + 1.0)
    # Estimate Vd per kg from vd_factor (rough: vd_factor as L/kg)
    vd_per_kg = vd_factor
    if vd_per_kg < 1.0:
        return "low", vd_factor
    if vd_per_kg <= 5.0:
        return "moderate", vd_factor
    return "high", vd_factor


def _cl_class_from_clint(clint: float) -> tuple[str, float]:
    """Return (cl_class, cl_factor) from CLint."""
    cl_factor = 1.0 / (clint * 0.001 + 0.5)
    # Approximate CL as inverse of cl_factor in L/h/kg space
    # cl_factor ~ 2 for clint=0, smaller for higher clint
    # low CL: < 5 L/h/kg proxy → cl_factor > 0.1, moderate 5-20, high > 20
    cl_approx = 1.0 / cl_factor if cl_factor > 0 else float("inf")
    if cl_approx < 5.0:
        return "low", cl_factor
    if cl_approx <= 20.0:
        return "moderate", cl_factor
    return "high", cl_factor


def predict_half_life_from_props(
    compound_name: str,
    logP: float,
    mw: float,
    psa: float,
    clint_uL_min_per_mg: float = 10.0,
    solubility_mg_mL: float = 1.0,
    pka_acid: float | None = None,
    pka_base: float | None = None,
    body_weight_kg: float = 70.0,
) -> HalfLifePrediction:
    """Predict drug half-life from physicochemical properties (QSPR model).

    Parameters
    ----------
    compound_name:
        Name of the compound.
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight (g/mol). Must be > 0.
    psa:
        Polar surface area (Å²). Must be >= 0.
    clint_uL_min_per_mg:
        Intrinsic clearance (µL/min/mg protein). Must be >= 0.
    solubility_mg_mL:
        Aqueous solubility (mg/mL).
    pka_acid:
        Acidic pKa (optional).
    pka_base:
        Basic pKa (optional).
    body_weight_kg:
        Body weight (kg). Must be > 0.

    Returns
    -------
    HalfLifePrediction
    """
    if mw <= 0:
        raise ValueError(f"mw must be positive; got {mw}.")
    if psa < 0:
        raise ValueError(f"psa must be non-negative; got {psa}.")
    if clint_uL_min_per_mg < 0:
        raise ValueError(f"clint_uL_min_per_mg must be non-negative; got {clint_uL_min_per_mg}.")
    if body_weight_kg <= 0:
        raise ValueError(f"body_weight_kg must be positive; got {body_weight_kg}.")

    base_t_half = 6.0

    vd_class_str, vd_factor = _vd_class(logP)
    cl_class_str, cl_factor = _cl_class_from_clint(clint_uL_min_per_mg)

    # Ionization factor
    if pka_acid is not None and pka_acid < 4.0:
        ionization_factor = 1.3
        ion_note = f"acidic (pKa_acid={pka_acid:.1f} < 4)"
    elif pka_base is not None and pka_base > 8.0:
        ionization_factor = 0.9
        ion_note = f"basic (pKa_base={pka_base:.1f} > 8)"
    else:
        ionization_factor = 1.0
        ion_note = "neutral/typical ionization"

    predicted_t_half_h = base_t_half * vd_factor * cl_factor * ionization_factor
    predicted_ke_per_h = math.log(2.0) / predicted_t_half_h

    bcs = _bcs_class(logP, mw, psa, solubility_mg_mL)

    # CI based on BCS class
    if bcs == "I":
        ci_pct = 0.20
    elif bcs == "IV":
        ci_pct = 0.60
    elif bcs in ("II", "III"):
        ci_pct = 0.40
    else:
        ci_pct = 0.40

    ci_lower_h = predicted_t_half_h * (1.0 - ci_pct)
    ci_upper_h = predicted_t_half_h * (1.0 + ci_pct)

    elimination_half_lives_in_24h = float(int(24.0 / predicted_t_half_h))

    notes = (
        f"QSPR model: logP={logP:.2f}, MW={mw:.1f}, CLint={clint_uL_min_per_mg:.1f} µL/min/mg; "
        f"vd_factor={vd_factor:.3f}, cl_factor={cl_factor:.3f}, ionization={ion_note}; "
        f"BCS={bcs}, 90% CI ±{ci_pct * 100:.0f}%."
    )

    return HalfLifePrediction(
        compound_name=compound_name,
        predicted_t_half_h=predicted_t_half_h,
        ci_lower_h=ci_lower_h,
        ci_upper_h=ci_upper_h,
        predicted_ke_per_h=predicted_ke_per_h,
        vd_class=vd_class_str,
        cl_class=cl_class_str,
        elimination_half_lives_in_24h=elimination_half_lives_in_24h,
        bcs_class=bcs,
        notes=notes,
    )


def screen_half_lives(compounds: list[dict]) -> list[HalfLifePrediction]:
    """Screen multiple compounds, sorted by predicted_t_half_h ascending.

    Each dict should contain keys matching predict_half_life_from_props parameters.
    Required: compound_name, logP, mw, psa.
    """
    results: list[HalfLifePrediction] = []
    for c in compounds:
        result = predict_half_life_from_props(
            compound_name=c["compound_name"],
            logP=c["logP"],
            mw=c["mw"],
            psa=c["psa"],
            clint_uL_min_per_mg=c.get("clint_uL_min_per_mg", 10.0),
            solubility_mg_mL=c.get("solubility_mg_mL", 1.0),
            pka_acid=c.get("pka_acid", None),
            pka_base=c.get("pka_base", None),
            body_weight_kg=c.get("body_weight_kg", 70.0),
        )
        results.append(result)
    return sorted(results, key=lambda r: r.predicted_t_half_h)
