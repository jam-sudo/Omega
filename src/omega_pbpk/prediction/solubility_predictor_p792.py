"""Phase 792 — Solubility Predictor (p792 variant).

Predict aqueous solubility from physicochemical properties using
the General Solubility Equation (GSE, Jain 2001).

Reference:
  Jain N, Yalkowsky SH (2001). Estimation of the aqueous solubility I:
  Application to organic nonelectrolytes. J Pharm Sci 90:234-252.
  logS = 0.5 - logP - 0.01*(MP - 25)  [S in mol/L]
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SolubilityPredictionResult:
    """Result of GSE-based solubility prediction."""

    compound_name: str
    logp: float
    mw_da: float
    mp_celsius: float  # melting point
    logS_intrinsic: float  # log10(intrinsic solubility, mg/L)
    s_intrinsic_mg_L: float
    logS_ph7_mg_L: float  # log10(solubility at pH 7.4 in mg/L)
    s_ph7_mg_L: float
    solubility_class: str  # "high" >100, "moderate" 10-100, "low" 1-10, "very_low" <1
    bcs_solubility: str  # "BCS_high" or "BCS_low"
    dose_number: float  # Do = dose_mg / (S_ph7_mg_L * 0.001 * 250mL)
    formulation_recommendation: str
    notes: str


def predict_solubility(
    compound_name: str,
    logp: float,
    mw_da: float,
    mp_celsius: float = 150.0,
    pka: float | None = None,
    ionization_type: str = "neutral",  # "acid", "base", "neutral"
    dose_mg: float = 100.0,
) -> SolubilityPredictionResult:
    """Predict aqueous solubility using the General Solubility Equation.

    GSE (Jain 2001):
      logS_mol_L = 0.5 - logP - 0.01 * (MP - 25)
      S_mg_L = 10^logS_mol_L * MW * 1000

    pH-dependent solubility at pH 7.4 (Henderson-Hasselbalch):
      Acid: S_ph7 = S0 * (1 + 10^(pH - pKa))
      Base: S_ph7 = S0 * (1 + 10^(pKa - pH))
      Neutral: S_ph7 = S0

    BCS dose number: Do = dose_mg / (S_ph7_mg_L * 0.250 L)
      BCS_high if Do < 1

    Args:
        compound_name: Name of the compound.
        logp: Octanol-water log partition coefficient.
        mw_da: Molecular weight in Daltons (g/mol).
        mp_celsius: Melting point in degrees Celsius (default 150).
        pka: pKa for ionization correction (optional).
        ionization_type: "acid", "base", or "neutral" (default "neutral").
        dose_mg: Dose in mg for BCS dose number (default 100).

    Returns:
        SolubilityPredictionResult dataclass.

    Raises:
        ValueError: If mw_da <= 0.
    """
    if mw_da <= 0.0:
        raise ValueError(f"mw_da must be > 0, got {mw_da}")

    # GSE: logS in mol/L
    logS_mol_L = 0.5 - logp - 0.01 * (mp_celsius - 25.0)
    s_mol_L = 10.0**logS_mol_L

    # Convert to mg/L: mol/L * g/mol * 1000 mg/g = mg/L
    s_intrinsic_mg_L = s_mol_L * mw_da * 1000.0
    logS_intrinsic = math.log10(s_intrinsic_mg_L) if s_intrinsic_mg_L > 0.0 else -99.0

    # pH-dependent solubility at pH 7.4
    ph = 7.4
    if ionization_type == "acid" and pka is not None:
        s_ph7_mg_L = s_intrinsic_mg_L * (1.0 + 10.0 ** (ph - pka))
    elif ionization_type == "base" and pka is not None:
        s_ph7_mg_L = s_intrinsic_mg_L * (1.0 + 10.0 ** (pka - ph))
    else:
        s_ph7_mg_L = s_intrinsic_mg_L

    logS_ph7_mg_L = math.log10(s_ph7_mg_L) if s_ph7_mg_L > 0.0 else -99.0

    # Solubility class based on s_ph7 in mg/L
    if s_ph7_mg_L >= 100.0:
        solubility_class = "high"
    elif s_ph7_mg_L >= 10.0:
        solubility_class = "moderate"
    elif s_ph7_mg_L >= 1.0:
        solubility_class = "low"
    else:
        solubility_class = "very_low"

    # BCS dose number: Do = dose_mg / (S [mg/L] * 0.250 L)
    # Do < 1 → BCS high solubility
    volume_L = 0.250
    dose_number = dose_mg / (s_ph7_mg_L * volume_L) if s_ph7_mg_L > 0.0 else float("inf")
    bcs_solubility = "BCS_high" if dose_number < 1.0 else "BCS_low"

    # Formulation recommendation
    if solubility_class == "high":
        formulation_recommendation = (
            "No special formulation needed; standard tablet/capsule appropriate."
        )
    elif solubility_class == "moderate":
        formulation_recommendation = (
            "Consider salt formation or wet granulation to ensure adequate dissolution."
        )
    elif solubility_class == "low":
        formulation_recommendation = "Amorphous solid dispersion (ASD) or nano-milling recommended."
    else:  # very_low
        formulation_recommendation = (
            "Nanoparticle formulation, self-emulsifying drug delivery system (SEDDS), "
            "or co-solvent required."
        )

    # Notes
    notes_parts = [
        f"logS_intrinsic={logS_intrinsic:.2f} (mg/L)",
        f"S_intrinsic={s_intrinsic_mg_L:.2f} mg/L",
        f"S_pH7.4={s_ph7_mg_L:.2f} mg/L",
        f"class={solubility_class}",
        f"Do={dose_number:.2f}",
        f"BCS={bcs_solubility}",
    ]
    if ionization_type != "neutral" and pka is not None:
        notes_parts.append(f"ion_type={ionization_type}, pKa={pka}")
    notes = "; ".join(notes_parts)

    return SolubilityPredictionResult(
        compound_name=compound_name,
        logp=logp,
        mw_da=mw_da,
        mp_celsius=mp_celsius,
        logS_intrinsic=logS_intrinsic,
        s_intrinsic_mg_L=s_intrinsic_mg_L,
        logS_ph7_mg_L=logS_ph7_mg_L,
        s_ph7_mg_L=s_ph7_mg_L,
        solubility_class=solubility_class,
        bcs_solubility=bcs_solubility,
        dose_number=dose_number,
        formulation_recommendation=formulation_recommendation,
        notes=notes,
    )


def screen_solubility(compounds: list[dict]) -> list[SolubilityPredictionResult]:
    """Screen multiple compounds for solubility, sorted by s_intrinsic_mg_L descending.

    Each dict must have keys: 'compound_name', 'logp', 'mw_da'.
    Optional keys: 'mp_celsius', 'pka', 'ionization_type', 'dose_mg'.

    Args:
        compounds: List of dicts with compound physicochemical properties.

    Returns:
        List of SolubilityPredictionResult sorted by s_intrinsic_mg_L descending.
    """
    if not compounds:
        return []

    results: list[SolubilityPredictionResult] = []
    for c in compounds:
        result = predict_solubility(
            compound_name=c["compound_name"],
            logp=c["logp"],
            mw_da=c["mw_da"],
            mp_celsius=c.get("mp_celsius", 150.0),
            pka=c.get("pka", None),
            ionization_type=c.get("ionization_type", "neutral"),
            dose_mg=c.get("dose_mg", 100.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.s_intrinsic_mg_L, reverse=True)
    return results
