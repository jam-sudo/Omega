"""Phase 1058 — Drug Rectal Absorption Predictor.

Predicts drug absorption from rectal formulations (suppositories, enemas, gels, foams).
Models ionization, permeability, formulation effects, and portal bypass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RectalAbsorptionResult:
    drug_name: str
    formulation: str
    fa_rectal: float
    f_portal_bypass_pct: float
    f_systemic: float
    tmax_rectal_h: float
    absorption_rate_constant: float
    onset_classification: str
    portal_contribution_pct: float
    notes: str


_VALID_FORMULATIONS = {"suppository", "enema", "gel", "foam"}
_VALID_IONIZATION_TYPES = {"acid", "base", "neutral", "zwitterion"}
_VALID_SUPPOSITORY_BASES = {"fatty", "water_soluble"}

# Portal bypass fraction (lower rectum → IVC, bypassing portal)
_F_PORTAL_BYPASS = 0.5

# Rectal parameters
_RECTAL_PH = 7.8
_RECTAL_ABSORPTION_WINDOW_H = 2.0


def _ionization_fraction(pka: float, ionization_type: str, ph: float) -> float:
    """Fraction ionized at given pH."""
    if ionization_type == "acid":
        return 1.0 / (1.0 + 10 ** (pka - ph))
    elif ionization_type == "base":
        return 1.0 / (1.0 + 10 ** (ph - pka))
    else:
        return 0.0


def _rectal_peff(logp: float, mw_Da: float) -> float:
    """Rectal permeability (cm/s): logP/MW-based, similar to intestinal Peff.

    Returns value in (1e-6, 0.05] cm/s.
    """
    peff = 10 ** (0.3 * logp - 0.002 * mw_Da - 1.0)
    return max(1e-6, min(0.05, peff))


def _peff_to_ka(peff_cm_s: float) -> float:
    """Convert Peff (cm/s) to absorption rate constant (/h).

    Uses a log-linear mapping calibrated so that:
      peff = 1e-6 → ka ≈ 0.05 /h  (very low permeability)
      peff = 0.05  → ka ≈ 5.0 /h  (high permeability, upper physiological limit)

    The mapping: ka = 0.05 * (peff / 1e-6)^exponent, solved for exponent such that
    ka(0.05) = 5.0:
      5.0 / 0.05 = (0.05 / 1e-6)^exponent
      100 = 50000^exponent
      exponent = log(100) / log(50000) ≈ 0.435
    """
    exponent = math.log(100.0) / math.log(50000.0)
    ka = 0.05 * (peff_cm_s / 1e-6) ** exponent
    return max(0.05, min(5.0, ka))


def predict_rectal_absorption(
    drug_name: str,
    mw_Da: float,
    logp: float,
    pka: float = 7.0,
    ionization_type: str = "neutral",
    formulation: str = "suppository",
    suppository_base: str = "fatty",
    dose_mg: float = 50.0,
    fu_rectal: float = 0.3,
    cl_hepatic_L_per_h: float = 10.0,
    q_hepatic_L_per_h: float = 90.0,
) -> RectalAbsorptionResult:
    """Predict drug absorption from rectal formulation.

    Args:
        drug_name: Name of the drug.
        mw_Da: Molecular weight in Da.
        logp: Lipophilicity (log octanol-water partition coefficient).
        pka: Acid/base dissociation constant.
        ionization_type: One of "acid", "base", "neutral", "zwitterion".
        formulation: One of "suppository", "enema", "gel", "foam".
        suppository_base: "fatty" or "water_soluble" (only for suppository).
        dose_mg: Dose in mg.
        fu_rectal: Unbound fraction in rectal mucosa (0-1].
        cl_hepatic_L_per_h: Hepatic clearance (L/h).
        q_hepatic_L_per_h: Hepatic blood flow (L/h).

    Returns:
        RectalAbsorptionResult with predicted PK parameters.

    Raises:
        ValueError: If invalid parameters are provided.
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if fu_rectal <= 0 or fu_rectal > 1:
        raise ValueError(f"fu_rectal must be in (0, 1], got {fu_rectal}")
    if cl_hepatic_L_per_h <= 0:
        raise ValueError(f"cl_hepatic_L_per_h must be > 0, got {cl_hepatic_L_per_h}")
    if q_hepatic_L_per_h <= 0:
        raise ValueError(f"q_hepatic_L_per_h must be > 0, got {q_hepatic_L_per_h}")
    if formulation not in _VALID_FORMULATIONS:
        raise ValueError(f"formulation must be one of {_VALID_FORMULATIONS}, got {formulation!r}")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION_TYPES}, got {ionization_type!r}"
        )

    # Ionization at rectal pH 7.8
    fi = _ionization_fraction(pka, ionization_type, ph=_RECTAL_PH)
    fu_rectal_adj = 1.0 - 0.5 * fi

    # Rectal permeability (cm/s)
    peff = _rectal_peff(logp, mw_Da)

    # Base absorption rate constant from Peff (log-linear mapping, /h)
    ka_base = _peff_to_ka(peff)

    # Formulation modifier
    if formulation == "suppository":
        if suppository_base == "water_soluble":
            form_factor = 1.2
        else:  # fatty
            form_factor = 0.8
    elif formulation == "enema":
        form_factor = 1.5
    elif formulation == "gel":
        form_factor = 0.9
    else:  # foam
        form_factor = 1.1

    # Ionization adjustment; cap final ka at physiological maximum
    ka = min(5.0, ka_base * form_factor * fu_rectal_adj)

    # Fraction absorbed (first-order, 2h absorption window)
    fa_rectal = 1.0 - math.exp(-ka * _RECTAL_ABSORPTION_WINDOW_H)
    fa_rectal = min(0.9, fa_rectal)

    # Portal bypass: 50% bypasses portal (lower rectum → IVC)
    f_portal_bypass = _F_PORTAL_BYPASS

    # Hepatic extraction ratio
    e_h = min(0.99, cl_hepatic_L_per_h / q_hepatic_L_per_h)
    f_hepatic = 1.0 - e_h

    # Systemic bioavailability
    f_systemic = fa_rectal * (f_portal_bypass + (1.0 - f_portal_bypass) * f_hepatic)
    f_systemic = min(1.0, max(0.0, f_systemic))

    # Time to peak (time to 63% absorbed = 1/ka)
    tmax_rectal_h = 1.0 / ka if ka > 0 else 99.0

    # Onset classification
    if tmax_rectal_h < 0.5:
        onset = "rapid"
    elif tmax_rectal_h < 2.0:
        onset = "moderate"
    else:
        onset = "slow"

    portal_contribution_pct = (1.0 - f_portal_bypass) * 100.0

    notes = (
        f"Drug: {drug_name}, formulation: {formulation}. "
        f"Peff (cm/s): {peff:.2e}, ka (/h): {ka:.4f}. "
        f"fa_rectal: {fa_rectal:.3f}, f_systemic: {f_systemic:.3f}. "
        f"50% portal bypass assumed (lower rectal anatomy). "
        f"E_h: {e_h:.3f}, f_hepatic: {f_hepatic:.3f}."
    )

    return RectalAbsorptionResult(
        drug_name=drug_name,
        formulation=formulation,
        fa_rectal=fa_rectal,
        f_portal_bypass_pct=f_portal_bypass * 100.0,
        f_systemic=f_systemic,
        tmax_rectal_h=tmax_rectal_h,
        absorption_rate_constant=ka,
        onset_classification=onset,
        portal_contribution_pct=portal_contribution_pct,
        notes=notes,
    )


def compare_rectal_formulations(
    drug_name: str,
    mw_Da: float,
    logp: float,
    pka: float = 7.0,
    ionization_type: str = "neutral",
) -> list:
    """Compare rectal formulations for a drug, sorted by systemic bioavailability.

    Evaluates all 4 formulation types: suppository (fatty base), enema, gel, foam.

    Args:
        drug_name: Name of the drug.
        mw_Da: Molecular weight in Da.
        logp: Lipophilicity.
        pka: Acid/base dissociation constant.
        ionization_type: One of "acid", "base", "neutral", "zwitterion".

    Returns:
        List of RectalAbsorptionResult sorted by f_systemic descending.
    """
    formulations = [
        ("suppository", "fatty"),
        ("enema", "fatty"),
        ("gel", "fatty"),
        ("foam", "fatty"),
    ]

    results = []
    for form, base in formulations:
        result = predict_rectal_absorption(
            drug_name=drug_name,
            mw_Da=mw_Da,
            logp=logp,
            pka=pka,
            ionization_type=ionization_type,
            formulation=form,
            suppository_base=base,
        )
        results.append(result)

    results.sort(key=lambda r: r.f_systemic, reverse=True)
    return results
