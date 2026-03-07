"""Lipophilicity profiling — comprehensive logP/logD analysis for drug development decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "LipophilicityProfile",
    "profile_lipophilicity",
    "compare_lipophilicity_optimization",
    "screen_lipophilicity",
]


@dataclass(frozen=True)
class LipophilicityProfile:
    """Comprehensive lipophilicity profile for a compound."""

    compound_name: str
    logP: float
    mw: float
    psa: float
    logD_50: float
    logD_68: float
    logD_74: float
    logP_class: str
    cns_penetration_score: float
    adipose_kp_estimate: float
    papp_caco2_estimate: float
    lipinski_pass: bool
    oral_optimal: bool
    notes: str


def _compute_logD(logP: float, ph: float, pka_acid: float | None, pka_base: float | None) -> float:
    """Compute logD at a given pH.

    For acids: logD = logP - log10(1 + 10^(pH - pKa_acid))
    For bases: logD = logP - log10(1 + 10^(pKa_base - pH))
    For amphoteric: use whichever correction is larger (most ionizable).
    """
    correction = 0.0
    if pka_acid is not None and pka_base is not None:
        acid_correction = math.log10(1 + 10 ** (ph - pka_acid))
        base_correction = math.log10(1 + 10 ** (pka_base - ph))
        correction = max(acid_correction, base_correction)
    elif pka_acid is not None:
        correction = math.log10(1 + 10 ** (ph - pka_acid))
    elif pka_base is not None:
        correction = math.log10(1 + 10 ** (pka_base - ph))
    return logP - correction


def _logP_class(logP: float) -> str:
    if logP < 0:
        return "hydrophilic"
    elif logP <= 3:
        return "moderate"
    elif logP <= 5:
        return "lipophilic"
    else:
        return "highly_lipophilic"


def _cns_penetration_score(logP: float, psa: float) -> float:
    if 1 <= logP <= 5 and psa < 90:
        return 1.0
    elif 0 <= logP <= 5 and psa < 120:
        return 0.5
    else:
        return 0.0


def _adipose_kp(logP: float) -> float:
    """Kp_adipose = 10^(0.77 * logP - 0.90), clamped to [0.01, 100]."""
    val = 10 ** (0.77 * logP - 0.90)
    return max(0.01, min(100.0, val))


def _papp_caco2(logD_68: float) -> float:
    """Estimated Caco-2 Papp = 10^(0.5 * logD_68 - 1.0) × 10⁻⁶ cm/s, clamped to [0.001, 100]."""
    val = 10 ** (0.5 * logD_68 - 1.0)
    return max(0.001, min(100.0, val))


def profile_lipophilicity(
    compound_name: str,
    logP: float,
    pka_acid: float | None = None,
    pka_base: float | None = None,
    mw: float = 400.0,
    psa: float = 80.0,
) -> LipophilicityProfile:
    """Generate a comprehensive lipophilicity profile for a compound.

    Parameters
    ----------
    compound_name:
        Name of the compound.
    logP:
        Octanol-water partition coefficient (intrinsic, neutral form).
    pka_acid:
        Acidic pKa, if applicable.
    pka_base:
        Basic pKa, if applicable.
    mw:
        Molecular weight (Da).
    psa:
        Polar surface area (Å²).

    Returns
    -------
    LipophilicityProfile
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")

    logD_50 = _compute_logD(logP, 5.0, pka_acid, pka_base)
    logD_68 = _compute_logD(logP, 6.8, pka_acid, pka_base)
    logD_74 = _compute_logD(logP, 7.4, pka_acid, pka_base)

    lp_class = _logP_class(logP)
    cns_score = _cns_penetration_score(logP, psa)
    kp_adipose = _adipose_kp(logP)
    papp = _papp_caco2(logD_68)
    lipinski = logP <= 5
    oral_opt = 0 <= logP <= 5 and mw < 500

    note_parts = [
        f"logP={logP:.2f} ({lp_class})",
        f"logD(5.0/6.8/7.4)={logD_50:.2f}/{logD_68:.2f}/{logD_74:.2f}",
        f"Lipinski logP pass: {lipinski}",
        f"Oral optimal: {oral_opt}",
        f"CNS penetration score: {cns_score:.1f}",
        f"Estimated adipose Kp: {kp_adipose:.2f}",
        f"Estimated Caco-2 Papp: {papp:.3g} ×10⁻⁶ cm/s",
    ]
    if logP > 5:
        note_parts.append(
            "Highly lipophilic — increased risk of metabolic liability and non-specific binding"
        )
    if logP < 0:
        note_parts.append("Hydrophilic — poor membrane permeability expected")
    if cns_score == 0.0:
        note_parts.append("Poor CNS penetration predicted (high PSA or unfavorable logP)")
    notes = "; ".join(note_parts)

    return LipophilicityProfile(
        compound_name=compound_name,
        logP=logP,
        mw=mw,
        psa=psa,
        logD_50=logD_50,
        logD_68=logD_68,
        logD_74=logD_74,
        logP_class=lp_class,
        cns_penetration_score=cns_score,
        adipose_kp_estimate=kp_adipose,
        papp_caco2_estimate=papp,
        lipinski_pass=lipinski,
        oral_optimal=oral_opt,
        notes=notes,
    )


def compare_lipophilicity_optimization(
    compound_name: str,
    logP_variants: list[float],
    mw: float = 400.0,
    psa: float = 80.0,
) -> list[LipophilicityProfile]:
    """Compare lipophilicity variants for a compound series.

    Returns profiles sorted by proximity to optimal oral logP (closest to 2.5 first).
    """
    profiles = [
        profile_lipophilicity(f"{compound_name}_logP{lp:.1f}", lp, mw=mw, psa=psa)
        for lp in logP_variants
    ]
    profiles.sort(key=lambda p: abs(p.logP - 2.5))
    return profiles


def screen_lipophilicity(compounds: list[dict]) -> list[LipophilicityProfile]:
    """Screen a list of compounds by lipophilicity profile.

    Each dict must contain: compound_name, logP.
    Optional keys: pka_acid, pka_base, mw, psa.

    Returns list sorted by logD_74 descending (most lipophilic at physiological pH first).
    """
    results = []
    for c in compounds:
        profile = profile_lipophilicity(
            compound_name=c["compound_name"],
            logP=c["logP"],
            pka_acid=c.get("pka_acid"),
            pka_base=c.get("pka_base"),
            mw=c.get("mw", 400.0),
            psa=c.get("psa", 80.0),
        )
        results.append(profile)
    results.sort(key=lambda p: p.logD_74, reverse=True)
    return results
