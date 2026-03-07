"""Oral bioavailability classification from physicochemical descriptors."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BioavailabilityResult:
    drug_name: str
    logP: float
    mw: float
    psa: float
    solubility_mg_mL: float
    predicted_f_pct: float
    f_class: str
    fa_fraction: float
    fg_fraction: float
    fh_fraction: float
    predicted_dose_number: float
    lipinski_pass: bool
    bcs_class: str
    confidence: str
    notes: str


def lipinski_violations(mw: float, logP: float, n_hbd: int, n_hba: int) -> int:
    """Count Ro5 violations: MW>500, logP>5, HBD>5, HBA>10."""
    count = 0
    if mw > 500:
        count += 1
    if logP > 5:
        count += 1
    if n_hbd > 5:
        count += 1
    if n_hba > 10:
        count += 1
    return count


def classify_f(f_pct: float) -> str:
    """< 10->very_low; <30->low; <60->moderate; <80->high; else->very_high."""
    if f_pct < 10:
        return "very_low"
    elif f_pct < 30:
        return "low"
    elif f_pct < 60:
        return "moderate"
    elif f_pct < 80:
        return "high"
    else:
        return "very_high"


def predict_bioavailability(
    drug_name: str,
    logP: float,
    mw: float,
    psa: float,
    solubility_mg_mL: float,
    dose_mg: float = 100.0,
    n_hbd: int = 2,
    n_hba: int = 4,
    n_violations: int | None = None,
    pka_acid: float | None = None,
    pka_base: float | None = None,
    cyp3a4_er: float = 0.3,
    fu_plasma: float = 0.5,
) -> BioavailabilityResult:
    """Predict oral bioavailability from physicochemical descriptors."""
    # Validation
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (0 < fu_plasma <= 1):
        raise ValueError("fu_plasma must be in (0, 1]")
    if not (0 <= cyp3a4_er <= 1):
        raise ValueError("cyp3a4_er must be in [0, 1]")

    # Dose number
    do = dose_mg / (250.0 * solubility_mg_mL)

    # Peff proxy
    peff_high = psa < 90 and logP > 0 and mw < 500

    # fa (fraction absorbed)
    fa = 1.0 / math.sqrt(1.0 + do)
    fa = min(fa, 1.0)
    if not peff_high:
        fa *= 0.5

    # fg (gut wall fraction)
    fg = 1.0 - 0.5 * max(0.0, logP - 3.0) / 5.0
    fg = max(0.1, min(1.0, fg))
    if psa > 120:
        fg *= 0.8
    fg = max(0.1, min(1.0, fg))

    # fh (hepatic first-pass)
    fh = 1.0 - cyp3a4_er

    # predicted F
    predicted_f_pct = fa * fg * fh * 100.0
    predicted_f_pct = max(0.0, min(100.0, predicted_f_pct))

    # Lipinski violations
    if n_violations is None:
        n_violations = lipinski_violations(mw, logP, n_hbd, n_hba)
    lipo_pass = n_violations <= 1

    # BCS class
    if do <= 1 and peff_high:
        bcs = "I"
    elif do > 1 and peff_high:
        bcs = "II"
    elif do <= 1 and not peff_high:
        bcs = "III"
    else:
        bcs = "IV"

    # Confidence
    if solubility_mg_mL > 0.1 and lipo_pass:
        confidence = "high"
    elif lipo_pass:
        confidence = "medium"
    else:
        confidence = "low"

    # Notes
    note_parts = []
    if do > 1:
        note_parts.append(f"High dose number (Do={do:.2f}): solubility-limited")
    if not peff_high:
        note_parts.append("Low permeability (high PSA or low logP or high MW)")
    if cyp3a4_er > 0.5:
        note_parts.append(f"High CYP3A4 extraction ratio ({cyp3a4_er:.2f}): first-pass limited")
    if not lipo_pass:
        note_parts.append(f"Lipinski violations ({n_violations}): poor oral drug-likeness")
    if not note_parts:
        note_parts.append("Good drug-like properties predicted")
    notes = "; ".join(note_parts)

    return BioavailabilityResult(
        drug_name=drug_name,
        logP=logP,
        mw=mw,
        psa=psa,
        solubility_mg_mL=solubility_mg_mL,
        predicted_f_pct=predicted_f_pct,
        f_class=classify_f(predicted_f_pct),
        fa_fraction=fa,
        fg_fraction=fg,
        fh_fraction=fh,
        predicted_dose_number=do,
        lipinski_pass=lipo_pass,
        bcs_class=bcs,
        confidence=confidence,
        notes=notes,
    )
