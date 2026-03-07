"""Hepatic extraction ratio prediction from physicochemical properties."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["HepExtractionResult", "predict_hepatic_extraction", "screen_hepatic_extraction"]

# Constants
_MPPGL = 45.0  # mg microsomal protein per g liver
_LIVER_WT_G = 1500.0  # g


@dataclass(frozen=True)
class HepExtractionResult:
    """Result of hepatic extraction ratio prediction."""

    compound_name: str
    er: float
    cl_hepatic_L_per_h: float
    fh: float
    fu_mic: float
    extraction_class: str
    first_pass_loss_pct: float
    notes: str


def predict_hepatic_extraction(
    compound_name: str,
    logP: float,
    mw: float,
    psa: float = 80.0,
    fup: float = 0.1,
    n_h_donors: int = 1,
    cl_int_uL_per_min_per_mg: float = 50.0,
    qh_L_per_h: float = 90.0,
) -> HepExtractionResult:
    """Predict hepatic extraction ratio from physicochemical and in vitro data.

    Args:
        compound_name: Name of the compound.
        logP: Lipophilicity (log octanol-water partition coefficient).
        mw: Molecular weight (g/mol).
        psa: Polar surface area (Angstrom^2).
        fup: Fraction unbound in plasma (0 < fup <= 1).
        n_h_donors: Number of hydrogen bond donors.
        cl_int_uL_per_min_per_mg: In vitro intrinsic clearance (uL/min/mg microsomal protein).
        qh_L_per_h: Hepatic blood flow (L/h), default 90 L/h.

    Returns:
        HepExtractionResult with hepatic extraction predictions.
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if not (0 < fup <= 1):
        raise ValueError("fup must be in (0, 1]")
    if cl_int_uL_per_min_per_mg <= 0:
        raise ValueError("cl_int_uL_per_min_per_mg must be > 0")
    if qh_L_per_h <= 0:
        raise ValueError("qh_L_per_h must be > 0")

    # Microsomal unbound fraction (Obach 1997 simplified)
    logP_for_fumic = logP if logP > 0 else 0.1
    fu_mic = max(0.01, min(1.0, 1.0 / (1.0 + 0.072 * logP_for_fumic)))

    # CLint scaling: uL/min/mg -> L/h in vivo
    # CLint_in_vivo (L/h) = cl_int * MPPGL * liver_wt / 1e6 * 60
    cl_int_in_vivo_L_per_h = cl_int_uL_per_min_per_mg * _MPPGL * _LIVER_WT_G / 1.0e6 * 60.0

    # Unbound CLint
    cl_int_u = cl_int_in_vivo_L_per_h / fu_mic

    # Well-stirred hepatic clearance model
    cl_hepatic = qh_L_per_h * (fup * cl_int_u) / (qh_L_per_h + fup * cl_int_u)

    # Extraction ratio
    er = cl_hepatic / qh_L_per_h

    # Physicochemical modulation
    notes_parts = []
    if psa > 120.0:
        er *= 0.8
        notes_parts.append("High PSA reduces hepatic uptake (factor 0.8)")
    if logP < 0.0:
        er *= 0.9
        notes_parts.append("Negative logP reduces hepatic uptake (factor 0.9)")
    if mw > 800.0:
        er *= 0.85
        notes_parts.append("High MW reduces hepatic uptake (factor 0.85)")

    # Clamp ER to valid range
    er = max(0.0, min(1.0, er))

    # Recalculate CLh after modulation
    cl_hepatic_final = er * qh_L_per_h

    # Classification
    if er < 0.3:
        extraction_class = "low"
    elif er < 0.7:
        extraction_class = "intermediate"
    else:
        extraction_class = "high"

    fh = 1.0 - er
    first_pass_loss_pct = er * 100.0

    if not notes_parts:
        notes_parts.append("Standard physicochemical properties; well-stirred model applied")

    notes = "; ".join(notes_parts)

    return HepExtractionResult(
        compound_name=compound_name,
        er=er,
        cl_hepatic_L_per_h=cl_hepatic_final,
        fh=fh,
        fu_mic=fu_mic,
        extraction_class=extraction_class,
        first_pass_loss_pct=first_pass_loss_pct,
        notes=notes,
    )


def screen_hepatic_extraction(compounds: list) -> list:
    """Screen a list of compounds for hepatic extraction ratio.

    Args:
        compounds: List of dicts, each with keys matching predict_hepatic_extraction args.
            Required keys: compound_name, logP, mw.
            Optional keys: psa, fup, n_h_donors, cl_int_uL_per_min_per_mg, qh_L_per_h.

    Returns:
        List of HepExtractionResult sorted by er descending.
    """
    if not compounds:
        return []

    results = []
    for cpd in compounds:
        result = predict_hepatic_extraction(
            compound_name=cpd["compound_name"],
            logP=cpd["logP"],
            mw=cpd["mw"],
            psa=cpd.get("psa", 80.0),
            fup=cpd.get("fup", 0.1),
            n_h_donors=cpd.get("n_h_donors", 1),
            cl_int_uL_per_min_per_mg=cpd.get("cl_int_uL_per_min_per_mg", 50.0),
            qh_L_per_h=cpd.get("qh_L_per_h", 90.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.er, reverse=True)
    return results
