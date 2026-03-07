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
    # Phase 581 additions
    "predict_excretion_route",
    "renal_excretion_mechanism",
    "biliary_excretion_likelihood",
    "urine_ph_effect_on_excretion",
]


# ---------------------------------------------------------------------------
# Phase 581 — excretion route prediction functions
# ---------------------------------------------------------------------------


def predict_excretion_route(
    mw: float,
    logP: float,
    fu_plasma: float,
    cl_renal_L_per_h: float,
    cl_hepatic_L_per_h: float,
) -> dict:
    """Predict primary excretion route and fractional clearances.

    Parameters
    ----------
    mw : float
        Molecular weight (g/mol)
    logP : float
        Lipophilicity
    fu_plasma : float
        Unbound fraction in plasma (0-1)
    cl_renal_L_per_h : float
        Renal clearance (L/h)
    cl_hepatic_L_per_h : float
        Hepatic clearance (L/h)

    Returns
    -------
    dict with fe_renal, fe_biliary, total_cl_L_per_h, primary_route, notes
    """
    if cl_renal_L_per_h < 0:
        raise ValueError("cl_renal_L_per_h must be >= 0")
    if cl_hepatic_L_per_h < 0:
        raise ValueError("cl_hepatic_L_per_h must be >= 0")
    if not (0.0 <= fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be between 0 and 1")

    total_cl = cl_renal_L_per_h + cl_hepatic_L_per_h
    if total_cl == 0.0:
        return {
            "fe_renal": 0.0,
            "fe_biliary": 0.0,
            "total_cl_L_per_h": 0.0,
            "primary_route": "undetermined",
            "notes": "Total clearance is zero; route cannot be determined.",
        }

    fe_renal = cl_renal_L_per_h / total_cl
    fe_biliary = 1.0 - fe_renal

    # Determine primary route
    if fe_renal > 0.6:
        primary_route = "renal"
    elif fe_biliary > 0.6:
        primary_route = "hepatic"
    else:
        primary_route = "mixed"

    notes_parts = []
    if mw > 400 and fe_biliary > 0.4:
        notes_parts.append("High MW (>400) supports biliary excretion.")
    if fu_plasma < 0.1:
        notes_parts.append("Low fu_plasma may limit renal filtration.")
    if logP > 2:
        notes_parts.append("Lipophilic compound may undergo tubular reabsorption.")

    return {
        "fe_renal": round(fe_renal, 4),
        "fe_biliary": round(fe_biliary, 4),
        "total_cl_L_per_h": round(total_cl, 4),
        "primary_route": primary_route,
        "notes": " ".join(notes_parts) if notes_parts else "No special notes.",
    }


def renal_excretion_mechanism(
    fu_plasma: float,
    logP: float,
    mw: float,
    gfr_mL_min: float = 120.0,
) -> dict:
    """Characterise renal excretion mechanism.

    Parameters
    ----------
    fu_plasma : float
        Unbound fraction in plasma (0-1)
    logP : float
        Lipophilicity
    mw : float
        Molecular weight (g/mol)
    gfr_mL_min : float
        Glomerular filtration rate (mL/min), default 120

    Returns
    -------
    dict with cl_filtration, reabsorption_fraction, active_secretion_bonus,
         cl_renal_net_L_per_h, mechanism_notes
    """
    if not (0.0 <= fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be between 0 and 1")
    if gfr_mL_min < 0:
        raise ValueError("gfr_mL_min must be >= 0")

    # Filtration clearance (L/h): fu * GFR_mL/min * 0.06 = L/h
    cl_filtration = fu_plasma * gfr_mL_min * 0.06

    # Tubular reabsorption
    if logP > 0:
        reabsorption_fraction = min(0.9, logP * 0.15)
    else:
        reabsorption_fraction = 0.0

    # Active secretion bonus
    if mw < 400 and logP < 1:
        active_secretion_bonus = 0.5 * cl_filtration
    else:
        active_secretion_bonus = 0.0

    cl_renal_net = cl_filtration * (1.0 - reabsorption_fraction) + active_secretion_bonus

    mechanism_notes_parts = []
    if reabsorption_fraction > 0:
        mechanism_notes_parts.append(
            f"Tubular reabsorption {reabsorption_fraction:.0%} (logP={logP:.2f})."
        )
    else:
        mechanism_notes_parts.append("Minimal tubular reabsorption (logP <= 0).")
    if active_secretion_bonus > 0:
        mechanism_notes_parts.append("Active secretion likely (MW<400, logP<1).")

    return {
        "cl_filtration": round(cl_filtration, 4),
        "reabsorption_fraction": round(reabsorption_fraction, 4),
        "active_secretion_bonus": round(active_secretion_bonus, 4),
        "cl_renal_net_L_per_h": round(cl_renal_net, 4),
        "mechanism_notes": " ".join(mechanism_notes_parts),
    }


def biliary_excretion_likelihood(mw: float, logP: float, n_hba: int) -> dict:
    """Predict biliary excretion likelihood.

    Parameters
    ----------
    mw : float
        Molecular weight (g/mol)
    logP : float
        Lipophilicity
    n_hba : int
        Number of hydrogen-bond acceptors

    Returns
    -------
    dict with likelihood ('high'/'moderate'/'low'), score (0-3), notes
    """
    if mw < 0:
        raise ValueError("mw must be >= 0")
    if n_hba < 0:
        raise ValueError("n_hba must be >= 0")

    score = 0
    notes_parts = []

    # Criterion 1: MW > 500
    if mw > 500:
        score += 1
        notes_parts.append("MW>500 favours biliary transport.")

    # Criterion 2: logP > 2
    if logP > 2:
        score += 1
        notes_parts.append("logP>2 supports biliary uptake.")

    # Criterion 3: HBA > 4
    if n_hba > 4:
        score += 1
        notes_parts.append("HBA>4 consistent with biliary substrates.")

    # Determine likelihood
    if score == 3:
        likelihood = "high"
    elif mw >= 300 and logP >= 1 and logP <= 4 and score >= 1:
        likelihood = "moderate"
    elif score >= 2:
        likelihood = "moderate"
    else:
        likelihood = "low"

    return {
        "likelihood": likelihood,
        "score": score,
        "notes": " ".join(notes_parts)
        if notes_parts
        else "Criteria not met for biliary excretion.",
    }


def urine_ph_effect_on_excretion(
    pka: float,
    drug_type: str,
    urine_ph: float,
    cl_renal_baseline_L_per_h: float,
) -> dict:
    """Ion-trapping effect of urine pH on renal excretion.

    Parameters
    ----------
    pka : float
        Drug pKa
    drug_type : str
        'acid' or 'base'
    urine_ph : float
        Urine pH (typically 4.5-8.0)
    cl_renal_baseline_L_per_h : float
        Baseline renal clearance (L/h)

    Returns
    -------
    dict with fi_urine, cl_renal_adjusted_L_per_h, effect_direction
    """
    if drug_type not in ("acid", "base"):
        raise ValueError("drug_type must be 'acid' or 'base'")
    if cl_renal_baseline_L_per_h < 0:
        raise ValueError("cl_renal_baseline_L_per_h must be >= 0")
    if not (0 < urine_ph < 14):
        raise ValueError("urine_ph must be between 0 and 14 (exclusive)")

    if drug_type == "acid":
        # Henderson-Hasselbalch: fraction ionized at urine pH
        try:
            fi_urine = 1.0 / (1.0 + 10 ** (pka - urine_ph))
        except OverflowError:
            fi_urine = 0.0
        # Basic urine → more ionized acid → less reabsorption → higher CL
        if urine_ph > pka:
            effect_direction = "increased"
        else:
            effect_direction = "decreased"
    else:  # base
        try:
            fi_urine = 1.0 / (1.0 + 10 ** (urine_ph - pka))
        except OverflowError:
            fi_urine = 0.0
        # Acidic urine → more ionized base → less reabsorption → higher CL
        if urine_ph < pka:
            effect_direction = "increased"
        else:
            effect_direction = "decreased"

    cl_adjusted = cl_renal_baseline_L_per_h * (1.0 + fi_urine)

    return {
        "fi_urine": round(fi_urine, 4),
        "cl_renal_adjusted_L_per_h": round(cl_adjusted, 4),
        "effect_direction": effect_direction,
    }
