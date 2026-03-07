"""Predict Caco-2 apparent permeability (Papp) from molecular descriptors (Phase 184).

Uses a rule-based Pade-like approach to estimate Papp (nm/s) from MW, logP,
PSA, HBD, and HBA. Permeability class, fraction absorbed, and efflux ratio are
derived from the predicted Papp.

Formulas
--------
log_Papp = 0.5 * logP - 0.01 * psa - 0.3 * hbd
           - 0.3 * max(0, mw - 500) / 100 + base_const
base_const = 0.0  (positions output in typical nm/s range)

Papp_nm_s = 10 ** log_Papp  (clamped to [0.01, 1000] nm/s)

Permeability class:
    "high"   : Papp > 100 nm/s
    "medium" : 10 <= Papp <= 100 nm/s
    "low"    : Papp < 10 nm/s

Fraction absorbed (simplified Michaelis-Menten-like):
    fa = Papp / (Papp + Km)  where Km = 50 nm/s

Absorption prediction:
    fa > 0.90 -> "complete"
    fa > 0.70 -> "high"
    fa > 0.40 -> "moderate"
    else      -> "poor"
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass (frozen — no list fields)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Caco2Result:
    """Result of a Caco-2 permeability prediction (Phase 184)."""

    drug_name: str
    mw_Da: float
    logP: float
    psa_A2: float
    hbd: int
    hba: int
    papp_nm_s: float
    log_papp: float
    permeability_class: str
    fa_fraction: float
    efflux_ratio: float
    absorption_prediction: str
    notes: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_CONST: float = 0.0
_KM_NM_S: float = 50.0  # empirical Michaelis constant for fa estimation
_PAPP_MIN: float = 0.01
_PAPP_MAX: float = 1000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_inputs(mw_Da: float, psa_A2: float, hbd: int) -> None:
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if psa_A2 < 0:
        raise ValueError(f"psa_A2 must be >= 0, got {psa_A2}")
    if hbd < 0:
        raise ValueError(f"hbd must be >= 0, got {hbd}")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _log10(x: float) -> float:
    """Safe base-10 logarithm (x must be positive)."""
    import math

    return math.log10(x)


def _pow10(x: float) -> float:

    return 10.0**x


def _permeability_class(papp: float) -> str:
    if papp > 100.0:
        return "high"
    if papp >= 10.0:
        return "medium"
    return "low"


def _fa_fraction(papp: float) -> float:
    """Fraction absorbed from simplified Km model: fa = Papp / (Papp + Km)."""
    return papp / (papp + _KM_NM_S)


def _absorption_prediction(fa: float) -> str:
    if fa > 0.90:
        return "complete"
    if fa > 0.70:
        return "high"
    if fa > 0.40:
        return "moderate"
    return "poor"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_caco2(
    drug_name: str,
    mw_Da: float,
    logP: float,
    psa_A2: float,
    hbd: int = 0,
    hba: int = 0,
    efflux_ratio: float = 1.0,
) -> Caco2Result:
    """Predict Caco-2 apparent permeability from molecular descriptors.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    mw_Da:
        Molecular weight (Da). Must be > 0.
    logP:
        Octanol-water partition coefficient.
    psa_A2:
        Polar surface area (Å²). Must be >= 0.
    hbd:
        Hydrogen bond donor count. Must be >= 0.
    hba:
        Hydrogen bond acceptor count.
    efflux_ratio:
        Basolateral-to-apical / apical-to-basolateral permeability ratio.
        Default 1.0 (no net efflux). Values > 2 indicate P-gp substrate.

    Returns
    -------
    Caco2Result (frozen dataclass)

    Raises
    ------
    ValueError
        If mw_Da <= 0, psa_A2 < 0, or hbd < 0.
    """
    _validate_inputs(mw_Da, psa_A2, hbd)

    # Compute log Papp
    mw_penalty = max(0.0, mw_Da - 500.0) / 100.0
    log_papp = 0.5 * logP - 0.01 * psa_A2 - 0.3 * hbd - 0.3 * mw_penalty + _BASE_CONST

    # Convert to nm/s and clamp
    papp_raw = _pow10(log_papp)
    papp_nm_s = _clamp(papp_raw, _PAPP_MIN, _PAPP_MAX)

    # Derive metrics
    perm_class = _permeability_class(papp_nm_s)
    fa = _fa_fraction(papp_nm_s)
    abs_pred = _absorption_prediction(fa)

    notes = (
        f"Caco-2 prediction (Phase 184): log_Papp={log_papp:.3f}, "
        f"Papp={papp_nm_s:.3f} nm/s ({perm_class}). "
        f"fa={fa:.3f} ({abs_pred}). "
        f"Efflux ratio={efflux_ratio:.2f}. "
        f"Descriptors: MW={mw_Da}, logP={logP}, PSA={psa_A2}, HBD={hbd}, HBA={hba}."
    )

    return Caco2Result(
        drug_name=drug_name,
        mw_Da=mw_Da,
        logP=logP,
        psa_A2=psa_A2,
        hbd=hbd,
        hba=hba,
        papp_nm_s=papp_nm_s,
        log_papp=log_papp,
        permeability_class=perm_class,
        fa_fraction=fa,
        efflux_ratio=efflux_ratio,
        absorption_prediction=abs_pred,
        notes=notes,
    )


def screen_permeability(
    drugs: list,
    efflux_ratio: float = 1.0,
) -> list:
    """Screen multiple compounds and rank by Papp descending.

    Parameters
    ----------
    drugs:
        List of dicts, each containing keys:
        ``drug_name``, ``mw_Da``, ``logP``, ``psa_A2``, ``hbd``, ``hba``.
        ``hbd`` and ``hba`` are optional (default 0).
    efflux_ratio:
        Default efflux ratio applied to all compounds unless overridden per
        drug via an ``efflux_ratio`` key.

    Returns
    -------
    list of Caco2Result sorted by papp_nm_s descending.
    """
    results = []
    for drug in drugs:
        result = predict_caco2(
            drug_name=drug["drug_name"],
            mw_Da=float(drug["mw_Da"]),
            logP=float(drug["logP"]),
            psa_A2=float(drug["psa_A2"]),
            hbd=int(drug.get("hbd", 0)),
            hba=int(drug.get("hba", 0)),
            efflux_ratio=float(drug.get("efflux_ratio", efflux_ratio)),
        )
        results.append(result)

    results.sort(key=lambda r: r.papp_nm_s, reverse=True)
    return results


__all__ = [
    "Caco2Result",
    "predict_caco2",
    "screen_permeability",
]
