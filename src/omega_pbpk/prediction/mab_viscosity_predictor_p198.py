"""
Phase 198 — mAb Solution Viscosity Predictor

Predicts mAb solution viscosity at high concentrations using the
Ross-Minton model for SC formulation development.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MABViscosityResult:
    """Result of mAb viscosity prediction."""

    drug_name: str
    conc_mg_mL: float
    mw_kDa: float
    pi_isoelectric: float
    ph: float
    ionic_strength_mM: float
    viscosity_mPas: float
    viscosity_class: str
    max_injectable_conc_mg_mL: float
    k_rm: float
    notes: str


# Ross-Minton model constants
_ETA_0 = 1.0  # mPa·s (water at 25°C)
_CONC_MAX = 500.0  # mg/mL (theoretical maximum)
_BASE_K = 0.003  # mL/mg


def _charge_factor(ph: float, pi_isoelectric: float) -> float:
    """
    Compute viscosity charge factor based on distance from pI.

    When pH is near pI (|pH - pI| < 1), charge is minimised and
    intermolecular interactions/viscosity are maximised → factor 1.5.
    As charge increases (pH moves away from pI) viscosity drops.
    """
    delta = abs(ph - pi_isoelectric)
    if delta < 1.0:
        return 1.5
    elif delta < 2.0:
        return 1.2
    elif delta < 4.0:
        return 1.0
    else:
        return 0.8


def _mw_factor(mw_kDa: float) -> float:
    """MW factor normalised to typical IgG (150 kDa)."""
    return mw_kDa / 150.0


def _compute_k_rm(mw_kDa: float, pi_isoelectric: float, ph: float) -> float:
    """Compute effective Ross-Minton k parameter."""
    return _BASE_K * _charge_factor(ph, pi_isoelectric) * _mw_factor(mw_kDa)


def _ross_minton_viscosity(conc_mg_mL: float, k_rm: float) -> float:
    """
    Ross-Minton equation:
        eta = eta_0 * exp(k_RM * c / (1 - c / c_max))
    """
    if conc_mg_mL >= _CONC_MAX:
        return float("inf")
    exponent = k_rm * conc_mg_mL / (1.0 - conc_mg_mL / _CONC_MAX)
    return _ETA_0 * math.exp(exponent)


def _classify_viscosity(viscosity_mPas: float) -> str:
    """Classify viscosity for injectability."""
    if viscosity_mPas < 20.0:
        return "injectable"
    elif viscosity_mPas <= 50.0:
        return "challenging"
    else:
        return "problematic"


def _find_max_injectable_conc(k_rm: float, target_viscosity: float = 20.0) -> float:
    """
    Binary search for the maximum concentration giving
    viscosity <= target_viscosity mPa·s.
    """
    lo = 0.0
    hi = _CONC_MAX - 1e-6

    # Quick check: even at very low concentration it might exceed target
    if _ross_minton_viscosity(1.0, k_rm) > target_viscosity:
        return 0.0

    for _ in range(60):
        mid = (lo + hi) / 2.0
        eta = _ross_minton_viscosity(mid, k_rm)
        if eta <= target_viscosity:
            lo = mid
        else:
            hi = mid

    return lo


def predict_mab_viscosity(
    drug_name: str,
    conc_mg_mL: float,
    mw_kDa: float,
    pi_isoelectric: float,
    ph: float,
    ionic_strength_mM: float = 150.0,
) -> MABViscosityResult:
    """
    Predict mAb solution viscosity using the Ross-Minton model.

    Parameters
    ----------
    drug_name : str
        Name of the mAb.
    conc_mg_mL : float
        Protein concentration (mg/mL).
    mw_kDa : float
        Molecular weight (kDa).
    pi_isoelectric : float
        Isoelectric point (pI).
    ph : float
        Formulation pH.
    ionic_strength_mM : float
        Ionic strength (mM), default 150 mM.

    Returns
    -------
    MABViscosityResult
    """
    if conc_mg_mL <= 0:
        raise ValueError("conc_mg_mL must be > 0")
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be > 0")
    if ionic_strength_mM < 0:
        raise ValueError("ionic_strength_mM must be >= 0")

    k_rm = _compute_k_rm(mw_kDa, pi_isoelectric, ph)
    viscosity = _ross_minton_viscosity(conc_mg_mL, k_rm)
    viscosity_class = _classify_viscosity(viscosity)
    max_injectable = _find_max_injectable_conc(k_rm)

    charge_dist = abs(ph - pi_isoelectric)
    notes = (
        f"Ross-Minton model | k_RM={k_rm:.5f} mL/mg | "
        f"|pH-pI|={charge_dist:.1f} | "
        f"charge_factor={_charge_factor(ph, pi_isoelectric):.2f} | "
        f"MW_factor={_mw_factor(mw_kDa):.3f} | "
        f"IS={ionic_strength_mM:.0f} mM"
    )

    return MABViscosityResult(
        drug_name=drug_name,
        conc_mg_mL=conc_mg_mL,
        mw_kDa=mw_kDa,
        pi_isoelectric=pi_isoelectric,
        ph=ph,
        ionic_strength_mM=ionic_strength_mM,
        viscosity_mPas=viscosity,
        viscosity_class=viscosity_class,
        max_injectable_conc_mg_mL=max_injectable,
        k_rm=k_rm,
        notes=notes,
    )


def screen_mab_concentrations(
    drug_name: str,
    mw_kDa: float,
    pi_isoelectric: float,
    ph: float,
    concs_mg_mL: list[float],
    ionic_strength_mM: float = 150.0,
) -> list[MABViscosityResult]:
    """
    Screen multiple concentrations and return results sorted by
    concentration ascending.

    Parameters
    ----------
    drug_name : str
        Name of the mAb.
    mw_kDa : float
        Molecular weight (kDa).
    pi_isoelectric : float
        Isoelectric point (pI).
    ph : float
        Formulation pH.
    concs_mg_mL : list of float
        Concentrations to evaluate (mg/mL).
    ionic_strength_mM : float
        Ionic strength (mM).

    Returns
    -------
    list of MABViscosityResult sorted by conc_mg_mL ascending
    """
    results = [
        predict_mab_viscosity(
            drug_name=drug_name,
            conc_mg_mL=c,
            mw_kDa=mw_kDa,
            pi_isoelectric=pi_isoelectric,
            ph=ph,
            ionic_strength_mM=ionic_strength_mM,
        )
        for c in concs_mg_mL
    ]
    results.sort(key=lambda r: r.conc_mg_mL)
    return results
