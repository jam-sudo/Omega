"""
Phase 284 — pH-Partition Theory GI Absorption Predictor.

Predict segmental GI absorption using pH-partition theory.
Henderson-Hasselbalch ionization at each GI segment pH, combined with
permeability and solubility constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

__all__ = [
    "GIAbsorptionResult",
    "predict_gi_absorption",
    "compare_ionization_types",
]

# GI segment parameters: (ph, volume_mL, length_fraction, transit_h)
_GI_SEGMENTS: list[tuple[str, float, float, float, float]] = [
    # name, ph, volume_mL, length_fraction, transit_h
    ("stomach", 1.5, 250.0, 0.05, 1.0),
    ("duodenum", 5.5, 150.0, 0.10, 1.0),
    ("jejunum", 6.5, 400.0, 0.35, 1.0),
    ("ileum", 7.2, 400.0, 0.35, 1.0),
    ("colon", 6.8, 300.0, 0.15, 10.0),
]

# Radius used for ka calculation (jejunum radius ~ 0.15 cm)
_RADIUS_CM = 0.15

_VALID_IONIZATION = frozenset({"acid", "base", "neutral"})


@dataclass
class GIAbsorptionResult:
    """Result of GI absorption prediction. Plain dataclass (has list field)."""

    drug_name: str
    pka: float
    ionization_type: str
    logp: float
    mw: float
    dose_mg: float
    f_abs: float
    peff_cm_per_s: float
    bcs_class: str
    absorption_class: str
    rate_limiting_factor: str
    segmental_fa: list[float]
    notes: str


def _validate_inputs(
    pka: float,
    logp: float,
    mw: float,
    dose_mg: float,
    solubility_mg_mL_at_neutral: float,
    ionization_type: str,
) -> None:
    if not (0 <= pka <= 14):
        raise ValueError(f"pka must be in [0, 14], got {pka}")
    if not (-5 <= logp <= 10):
        raise ValueError(f"logp must be in [-5, 10], got {logp}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if solubility_mg_mL_at_neutral <= 0:
        raise ValueError(
            f"solubility_mg_mL_at_neutral must be > 0, got {solubility_mg_mL_at_neutral}"
        )
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {sorted(_VALID_IONIZATION)}, got {ionization_type!r}"
        )


def _neutral_fraction(ionization_type: str, pka: float, ph: float) -> float:
    """Return fraction of drug in neutral (un-ionized) form at given pH."""
    if ionization_type == "neutral":
        return 1.0
    if ionization_type == "acid":
        # fi = 1 / (1 + 10^(pH - pKa)); fu = 1 - fi
        fi = 1.0 / (1.0 + 10.0 ** (ph - pka))
        return 1.0 - fi
    # base
    fu = 1.0 / (1.0 + 10.0 ** (pka - ph))
    return fu


def _base_peff(logp: float) -> float:
    """Simplified QSAR: Peff = 10^(0.5*logP - 3) cm/s, capped [1e-6, 1e-3]."""
    peff = 10.0 ** (0.5 * logp - 3.0)
    return max(1e-6, min(1e-3, peff))


def predict_gi_absorption(
    drug_name: str,
    pka: float,
    ionization_type: str,
    logp: float,
    mw: float,
    dose_mg: float,
    solubility_mg_mL_at_neutral: float,
) -> GIAbsorptionResult:
    """
    Predict segmental GI absorption using pH-partition theory.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    pka : float
        pKa of the drug (must be in [0, 14]).
    ionization_type : str
        Ionization type: "acid", "base", or "neutral".
    logp : float
        Octanol-water partition coefficient (must be in [-5, 10]).
    mw : float
        Molecular weight (Da). Must be > 0.
    dose_mg : float
        Administered dose in mg. Must be > 0.
    solubility_mg_mL_at_neutral : float
        Aqueous solubility at neutral pH in mg/mL. Must be > 0.

    Returns
    -------
    GIAbsorptionResult
    """
    _validate_inputs(pka, logp, mw, dose_mg, solubility_mg_mL_at_neutral, ionization_type)

    peff = _base_peff(logp)

    segmental_fa: list[float] = []
    weighted_fa = 0.0

    for _seg_name, ph, volume_mL, length_fraction, transit_h in _GI_SEGMENTS:
        fu = _neutral_fraction(ionization_type, pka, ph)

        # Effective permeability adjusted by ionization
        peff_seg = peff * fu + peff * 0.01 * (1.0 - fu)

        # Solubility at segment (simplified)
        cs_seg = solubility_mg_mL_at_neutral * (fu + 0.1 * (1.0 - fu))

        # Absorption rate constant (1/h): ka = 2 * Peff_seg [cm/s] * 3600 [s/h] / radius [cm]
        ka_seg = 2.0 * peff_seg * 3600.0 / _RADIUS_CM

        # Solubility-limited fraction
        solubility_factor = min(1.0, cs_seg * volume_mL / dose_mg)

        # Absorption fraction from segment
        fa_seg = (1.0 - exp(-ka_seg * transit_h)) * solubility_factor
        fa_seg = max(0.0, min(1.0, fa_seg))

        segmental_fa.append(fa_seg)
        weighted_fa += fa_seg * length_fraction

    # Total absorption fraction capped at 1.0
    f_abs = min(1.0, weighted_fa)

    # BCS classification
    high_perm = peff > 1e-4
    high_sol = solubility_mg_mL_at_neutral > 0.1

    if high_perm and high_sol:
        bcs_class = "I"
    elif high_perm and not high_sol:
        bcs_class = "II"
    elif not high_perm and high_sol:
        bcs_class = "III"
    else:
        bcs_class = "IV"

    # Absorption class
    if f_abs > 0.8:
        absorption_class = "complete"
    elif f_abs >= 0.2:
        absorption_class = "partial"
    else:
        absorption_class = "poor"

    # Rate limiting factor
    if bcs_class in ("III", "IV"):
        rate_limiting_factor = "permeability"
    elif bcs_class == "II":
        rate_limiting_factor = "solubility"
    else:
        rate_limiting_factor = "neither"

    notes_parts = [
        f"BCS class {bcs_class}",
        f"peff={peff:.2e} cm/s",
    ]
    if ionization_type == "acid" and pka < 4:
        notes_parts.append("highly ionized in stomach; better absorption in intestine")
    elif ionization_type == "base" and pka > 9:
        notes_parts.append("highly ionized in intestine; better absorption in stomach")
    notes = "; ".join(notes_parts)

    return GIAbsorptionResult(
        drug_name=drug_name,
        pka=pka,
        ionization_type=ionization_type,
        logp=logp,
        mw=mw,
        dose_mg=dose_mg,
        f_abs=f_abs,
        peff_cm_per_s=peff,
        bcs_class=bcs_class,
        absorption_class=absorption_class,
        rate_limiting_factor=rate_limiting_factor,
        segmental_fa=segmental_fa,
        notes=notes,
    )


def compare_ionization_types(
    drug_name: str,
    pka: float,
    logp: float,
    mw: float,
    dose_mg: float,
    solubility_mg_mL_at_neutral: float,
) -> list[GIAbsorptionResult]:
    """
    Compare GI absorption for all three ionization types (acid, base, neutral).

    Returns
    -------
    list of GIAbsorptionResult
        Three results sorted descending by f_abs.
    """
    results = []
    for ionization_type in ("acid", "base", "neutral"):
        result = predict_gi_absorption(
            drug_name=drug_name,
            pka=pka,
            ionization_type=ionization_type,
            logp=logp,
            mw=mw,
            dose_mg=dose_mg,
            solubility_mg_mL_at_neutral=solubility_mg_mL_at_neutral,
        )
        results.append(result)
    results.sort(key=lambda r: r.f_abs, reverse=True)
    return results
