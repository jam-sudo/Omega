"""Hepatic clearance prediction from in vitro CLint data (Phase 390)."""

from __future__ import annotations

from dataclasses import dataclass

# Microsomal scaling constants
_MICROSOMAL_PROTEIN_PER_G_LIVER: float = 45.0  # mg protein / g liver
_LIVER_WEIGHT_PER_KG_BW: float = 25.7  # g liver / kg body weight

# Hepatocyte scaling constants
_HEPATOCYTES_PER_G_LIVER: float = 120.0e6  # cells / g liver

# Hepatic blood flow allometric: QH = 90.6 * BW^0.75 mL/min
_QH_COEFFICIENT: float = 90.6
_QH_EXPONENT: float = 0.75

_VALID_METHODS: frozenset[str] = frozenset({"microsomal", "hepatocyte"})


@dataclass(frozen=True)
class ClearancePredictionResult:
    """Result of hepatic clearance prediction from in vitro data."""

    clint_uL_per_min_per_mg: float
    fu_mic: float
    method: str
    body_weight_kg: float
    cl_hepatic_L_per_h: float
    extraction_ratio: float
    cl_classification: str  # "high" / "intermediate" / "low"
    qh_mL_per_min: float
    clint_invivo_mL_per_min: float
    notes: str


def predict_hepatic_cl(
    clint_uL_per_min_per_mg_protein: float,
    fu_mic: float,
    hepatocyte_scaling: float = 1.0,
    method: str = "microsomal",
    body_weight_kg: float = 70.0,
) -> ClearancePredictionResult:
    """Predict hepatic clearance using well-stirred liver model.

    Parameters
    ----------
    clint_uL_per_min_per_mg_protein:
        In vitro intrinsic clearance. For "microsomal" method: µL/min/mg
        microsomal protein. For "hepatocyte" method: µL/min/10^6 cells
        (pass as this parameter).
    fu_mic:
        Fraction unbound in microsomal incubation (0 < fu_mic ≤ 1).
    hepatocyte_scaling:
        Additional scaling factor (default 1.0; not used in microsomal path).
    method:
        Scaling method: "microsomal" or "hepatocyte".
    body_weight_kg:
        Body weight for allometric scaling of hepatic blood flow (kg).

    Returns
    -------
    ClearancePredictionResult
    """
    if clint_uL_per_min_per_mg_protein <= 0:
        raise ValueError("clint_uL_per_min_per_mg_protein must be > 0")
    if not (0 < fu_mic <= 1.0):
        raise ValueError("fu_mic must be in (0, 1]")
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {sorted(_VALID_METHODS)}, got '{method}'")
    if body_weight_kg <= 0:
        raise ValueError("body_weight_kg must be > 0")
    if hepatocyte_scaling <= 0:
        raise ValueError("hepatocyte_scaling must be > 0")

    clint_input = clint_uL_per_min_per_mg_protein

    # Scale in vitro CLint to in vivo CLint (mL/min)
    if method == "microsomal":
        # clint [µL/min/mg protein] * [µL→mL /1000] * [mg protein/g liver]
        # * [g liver/kg BW] * BW [kg]
        clint_invivo = (
            clint_input
            / 1000.0  # µL → mL
            * _MICROSOMAL_PROTEIN_PER_G_LIVER
            * _LIVER_WEIGHT_PER_KG_BW
            * body_weight_kg
        )
        notes = (
            f"Microsomal scaling: {_MICROSOMAL_PROTEIN_PER_G_LIVER} mg protein/g liver, "
            f"{_LIVER_WEIGHT_PER_KG_BW} g liver/kg BW"
        )
    else:  # hepatocyte
        # clint [µL/min/10^6 cells] * [µL→mL /1000] * [cells/g liver /1e6] * [g liver/kg] * BW
        clint_invivo = (
            clint_input
            / 1000.0  # µL → mL
            * (_HEPATOCYTES_PER_G_LIVER / 1.0e6)  # cells/g liver expressed as 10^6 cells/g
            * _LIVER_WEIGHT_PER_KG_BW
            * body_weight_kg
            * hepatocyte_scaling
        )
        notes = (
            f"Hepatocyte scaling: {_HEPATOCYTES_PER_G_LIVER / 1e6:.0f}×10^6 cells/g liver, "
            f"{_LIVER_WEIGHT_PER_KG_BW} g liver/kg BW"
        )

    # Hepatic blood flow: QH = 90.6 * BW^0.75 (mL/min)
    qh = _QH_COEFFICIENT * (body_weight_kg**_QH_EXPONENT)

    # Well-stirred model: CLH = QH * fu_mic * CLint / (QH + fu_mic * CLint)
    fu_clint = fu_mic * clint_invivo
    cl_hepatic_mL_min = qh * fu_clint / (qh + fu_clint)

    # Extraction ratio EH = CLH / QH
    extraction_ratio = cl_hepatic_mL_min / qh

    # Convert to L/h
    cl_hepatic_L_h = cl_hepatic_mL_min * 60.0 / 1000.0

    # Classification
    if extraction_ratio > 0.7:
        cl_classification = "high"
    elif extraction_ratio >= 0.3:
        cl_classification = "intermediate"
    else:
        cl_classification = "low"

    notes += (
        f"; fu_mic correction applied; EH={extraction_ratio:.3f} ({cl_classification} extraction)"
    )

    return ClearancePredictionResult(
        clint_uL_per_min_per_mg=clint_input,
        fu_mic=fu_mic,
        method=method,
        body_weight_kg=body_weight_kg,
        cl_hepatic_L_per_h=cl_hepatic_L_h,
        extraction_ratio=extraction_ratio,
        cl_classification=cl_classification,
        qh_mL_per_min=qh,
        clint_invivo_mL_per_min=clint_invivo,
        notes=notes,
    )


def rank_clearance_compounds(
    compounds: list[dict],
    method: str = "microsomal",
    body_weight_kg: float = 70.0,
) -> list[dict]:
    """Rank compounds by predicted hepatic clearance (highest first).

    Parameters
    ----------
    compounds:
        List of dicts, each with keys "name", "clint" (µL/min/mg protein),
        "fu_mic" (fraction unbound in microsomes).
    method:
        Scaling method: "microsomal" or "hepatocyte".
    body_weight_kg:
        Body weight for allometric QH scaling.

    Returns
    -------
    List of dicts sorted by cl_hepatic_L_per_h descending, each containing
    original compound data plus prediction fields.
    """
    if not compounds:
        raise ValueError("compounds list must not be empty")

    results: list[dict] = []
    for i, cpd in enumerate(compounds):
        name = cpd.get("name", f"compound_{i}")
        clint = cpd.get("clint")
        fu_mic = cpd.get("fu_mic")
        if clint is None or fu_mic is None:
            raise ValueError(f"Compound '{name}' must have 'clint' and 'fu_mic' fields")

        pred = predict_hepatic_cl(
            clint_uL_per_min_per_mg_protein=float(clint),
            fu_mic=float(fu_mic),
            method=method,
            body_weight_kg=body_weight_kg,
        )
        results.append(
            {
                "name": name,
                "clint_uL_per_min_per_mg": float(clint),
                "fu_mic": float(fu_mic),
                "cl_hepatic_L_per_h": pred.cl_hepatic_L_per_h,
                "extraction_ratio": pred.extraction_ratio,
                "cl_classification": pred.cl_classification,
                "qh_mL_per_min": pred.qh_mL_per_min,
                "clint_invivo_mL_per_min": pred.clint_invivo_mL_per_min,
            }
        )

    results.sort(key=lambda x: x["cl_hepatic_L_per_h"], reverse=True)
    return results


__all__ = [
    "ClearancePredictionResult",
    "predict_hepatic_cl",
    "rank_clearance_compounds",
]
