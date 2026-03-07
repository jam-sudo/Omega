"""Volume of distribution prediction using QSPR and Rodgers-Rowland methods."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "VdPrediction",
    "predict_vd",
    "screen_vd",
]

# ---------------------------------------------------------------------------
# Tissue composition data (simplified Rodgers-Rowland)
# fIW: intracellular water fraction, fEW: extracellular water fraction
# fN: neutral lipid fraction, fP: phospholipid fraction
# ---------------------------------------------------------------------------
_TISSUE_DATA: dict[str, dict[str, float]] = {
    "muscle": {"fIW": 0.63, "fEW": 0.12, "fN": 0.012, "fP": 0.002, "V_L_per_kg": 0.43},
    "adipose": {"fIW": 0.02, "fEW": 0.05, "fN": 0.853, "fP": 0.002, "V_L_per_kg": 0.19},
    "lung": {"fIW": 0.45, "fEW": 0.30, "fN": 0.022, "fP": 0.009, "V_L_per_kg": 0.010},
    "liver": {"fIW": 0.57, "fEW": 0.16, "fN": 0.038, "fP": 0.012, "V_L_per_kg": 0.026},
    "kidney": {"fIW": 0.48, "fEW": 0.20, "fN": 0.012, "fP": 0.005, "V_L_per_kg": 0.004},
    "brain": {"fIW": 0.62, "fEW": 0.04, "fN": 0.039, "fP": 0.015, "V_L_per_kg": 0.020},
    "heart": {"fIW": 0.55, "fEW": 0.25, "fN": 0.014, "fP": 0.007, "V_L_per_kg": 0.003},
    "bone": {"fIW": 0.10, "fEW": 0.10, "fN": 0.017, "fP": 0.003, "V_L_per_kg": 0.071},
    "skin": {"fIW": 0.38, "fEW": 0.12, "fN": 0.060, "fP": 0.004, "V_L_per_kg": 0.111},
    "gi": {"fIW": 0.52, "fEW": 0.14, "fN": 0.022, "fP": 0.005, "V_L_per_kg": 0.017},
    "spleen": {"fIW": 0.54, "fEW": 0.21, "fN": 0.020, "fP": 0.005, "V_L_per_kg": 0.003},
    "thymus": {"fIW": 0.50, "fEW": 0.18, "fN": 0.015, "fP": 0.004, "V_L_per_kg": 0.002},
}

_V_PLASMA_L_PER_KG = 0.043
_V_MUSCLE_L_PER_KG = 0.43
_V_ADIPOSE_L_PER_KG = 0.19
_V_LIVER_L_PER_KG = 0.026
_V_BRAIN_L_PER_KG = 0.020


@dataclass(frozen=True)
class VdPrediction:
    compound_name: str
    fup: float
    logP: float
    pka_base: float | None
    pka_acid: float | None
    predicted_vd_L: float
    predicted_vd_L_per_kg: float
    vd_class: str
    kp_muscle: float
    kp_adipose: float
    kp_liver: float
    kp_brain: float
    predicted_vss_L: float
    notes: str


def _vd_class(vd_L_per_kg: float) -> str:
    if vd_L_per_kg < 0.7:
        return "low"
    if vd_L_per_kg < 5.0:
        return "moderate"
    if vd_L_per_kg < 20.0:
        return "high"
    return "very_high"


def _qspr_vd(
    logP: float,
    psa: float,
    fup: float,
    pka_base: float | None,
    pka_acid: float | None,
) -> float:
    """QSPR estimate of Vd in L/kg."""
    logP_factor = max(0.0, logP) * 0.5
    psa_penalty = max(0.0, psa - 60.0) * 0.005
    base_vd = 0.7 + logP_factor - psa_penalty

    if pka_base is not None and pka_base > 7.0:
        base_vd += (pka_base - 7.0) * 0.5

    if pka_acid is not None and pka_acid < 5.0:
        base_vd *= 0.6

    vd_L_per_kg = max(0.1, base_vd / fup)
    return vd_L_per_kg


def _kp_muscle(logP: float, fup: float) -> float:
    return max(0.5, (0.75 + logP * 0.3) / fup)


def _kp_adipose(logP: float, fup: float) -> float:
    if logP > 0:
        return max(1.0, (1.0 + logP * 2.5) / fup)
    return max(0.01, 1.0 / fup)


def _kp_liver(logP: float, fup: float, pka_base: float | None) -> float:
    base_bonus = 0.5 if (pka_base is not None and pka_base > 7.0) else 0.0
    return max(1.0, (1.2 + logP * 0.5) / fup + base_bonus)


def _kp_brain(logP: float, psa: float, fup: float) -> float:
    return max(0.01, (0.3 + logP * 0.2 - psa * 0.002) / fup)


def _rodgers_rowland_vd(
    logP: float,
    psa: float,
    fup: float,
    pka_base: float | None,
    body_weight_kg: float,
) -> tuple[float, float, float, float, float]:
    """Simplified Rodgers-Rowland Vss computation.

    Returns
    -------
    (vd_L_per_kg, kp_muscle, kp_adipose, kp_liver, kp_brain)
    """
    kpm = _kp_muscle(logP, fup)
    kpa = _kp_adipose(logP, fup)
    kpl = _kp_liver(logP, fup, pka_base)
    kpb = _kp_brain(logP, psa, fup)

    vss_per_kg = (
        _V_PLASMA_L_PER_KG
        + _V_MUSCLE_L_PER_KG * kpm
        + _V_ADIPOSE_L_PER_KG * kpa
        + _V_LIVER_L_PER_KG * kpl
        + _V_BRAIN_L_PER_KG * kpb
    )
    return vss_per_kg, kpm, kpa, kpl, kpb


def predict_vd(
    compound_name: str,
    logP: float,
    mw: float,
    psa: float,
    fup: float,
    pka_base: float | None = None,
    pka_acid: float | None = None,
    body_weight_kg: float = 70.0,
    method: str = "qspr",
) -> VdPrediction:
    """Predict volume of distribution from physicochemical properties.

    Parameters
    ----------
    compound_name:
        Identifier for the compound.
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight (g/mol). Must be > 0.
    psa:
        Polar surface area (Å²). Must be >= 0.
    fup:
        Fraction unbound in plasma (0 < fup <= 1).
    pka_base:
        Basic pKa (optional); enables lysosomal trapping bonus.
    pka_acid:
        Acidic pKa (optional); high acidity reduces Vd.
    body_weight_kg:
        Body weight in kg (default 70.0). Must be > 0.
    method:
        Prediction method: "qspr" or "rodgers_rowland".

    Returns
    -------
    VdPrediction

    Raises
    ------
    ValueError
        On invalid inputs or unknown method.
    """
    if fup <= 0:
        raise ValueError("fup must be > 0")
    if fup > 1:
        raise ValueError("fup must be <= 1")
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")
    if body_weight_kg <= 0:
        raise ValueError("body_weight_kg must be > 0")
    if method not in {"qspr", "rodgers_rowland"}:
        raise ValueError(f"method must be 'qspr' or 'rodgers_rowland', got '{method}'")

    if method == "qspr":
        vd_L_per_kg = _qspr_vd(logP, psa, fup, pka_base, pka_acid)

        kpm = _kp_muscle(logP, fup)
        kpa = _kp_adipose(logP, fup)
        kpl = _kp_liver(logP, fup, pka_base)
        kpb = _kp_brain(logP, psa, fup)

        vss_per_kg = (
            _V_PLASMA_L_PER_KG
            + _V_MUSCLE_L_PER_KG * kpm
            + _V_ADIPOSE_L_PER_KG * kpa
            + _V_LIVER_L_PER_KG * kpl
            + _V_BRAIN_L_PER_KG * kpb
        )
        vss_L = vss_per_kg * body_weight_kg
        method_note = "QSPR"

    else:  # rodgers_rowland
        vd_L_per_kg, kpm, kpa, kpl, kpb = _rodgers_rowland_vd(
            logP, psa, fup, pka_base, body_weight_kg
        )
        vss_L = vd_L_per_kg * body_weight_kg
        method_note = "Rodgers-Rowland simplified"

    vd_L = vd_L_per_kg * body_weight_kg
    cls = _vd_class(vd_L_per_kg)

    notes_parts = [
        f"Method: {method_note}",
        f"Vd={vd_L_per_kg:.2f} L/kg",
        f"class={cls}",
    ]
    if pka_base is not None and pka_base > 7.0:
        notes_parts.append("basic drug: lysosomal trapping expected")
    if pka_acid is not None and pka_acid < 5.0:
        notes_parts.append("acidic drug: reduced tissue distribution")
    notes = "; ".join(notes_parts)

    return VdPrediction(
        compound_name=compound_name,
        fup=fup,
        logP=logP,
        pka_base=pka_base,
        pka_acid=pka_acid,
        predicted_vd_L=vd_L,
        predicted_vd_L_per_kg=vd_L_per_kg,
        vd_class=cls,
        kp_muscle=kpm,
        kp_adipose=kpa,
        kp_liver=kpl,
        kp_brain=kpb,
        predicted_vss_L=vss_L,
        notes=notes,
    )


def screen_vd(compounds: list[dict]) -> list[VdPrediction]:
    """Screen multiple compounds and sort by predicted_vd_L_per_kg descending.

    Each dict must contain keys accepted by predict_vd:
    compound_name, logP, mw, psa, fup, and optionally pka_base, pka_acid,
    body_weight_kg, method.

    Parameters
    ----------
    compounds:
        List of parameter dicts.

    Returns
    -------
    list[VdPrediction] sorted by predicted_vd_L_per_kg descending.
    """
    if not compounds:
        return []

    results = []
    for c in compounds:
        result = predict_vd(
            compound_name=c["compound_name"],
            logP=c["logP"],
            mw=c["mw"],
            psa=c["psa"],
            fup=c["fup"],
            pka_base=c.get("pka_base"),
            pka_acid=c.get("pka_acid"),
            body_weight_kg=c.get("body_weight_kg", 70.0),
            method=c.get("method", "qspr"),
        )
        results.append(result)

    results.sort(key=lambda r: r.predicted_vd_L_per_kg, reverse=True)
    return results
