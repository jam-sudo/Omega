"""PK parameter estimation from molecular descriptors — Phase 500."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PKEstimateResult:
    """Result of PK parameter estimation from molecular descriptors."""

    compound_name: str
    logP: float
    mw: float
    psa_A2: float
    cl_L_per_h: float
    vd_L: float
    t_half_h: float
    f_oral: float
    mrt_h: float
    lipinski_compliant: bool
    bcs_class: str
    confidence: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def _validate_inputs(mw: float, psa_A2: float) -> None:
    if mw <= 0:
        raise ValueError(f"mw must be positive, got {mw}")
    if psa_A2 < 0:
        raise ValueError(f"psa_A2 must be non-negative, got {psa_A2}")


# ---------------------------------------------------------------------------
# QSPKR model functions
# ---------------------------------------------------------------------------


def estimate_cl_human(
    logP: float,
    mw: float,
    fu_plasma: float,
    psa_A2: float,
    n_hbd: float,
    logD_74: float | None = None,
) -> float:
    """Estimate human hepatic clearance (CL) from molecular descriptors.

    QSPKR model (empirical, calibrated to human data):
        CL = 10^(-0.3*logP + 0.02*PSA - 0.5*log10(MW) + 0.8*log10(fu+0.01) + 1.5)

    Result is clamped to [0.1, 200] L/h.

    Parameters
    ----------
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight (g/mol).
    fu_plasma:
        Fraction unbound in plasma (0-1).
    psa_A2:
        Polar surface area (Angstrom^2).
    n_hbd:
        Number of hydrogen bond donors.
    logD_74:
        Optional logD at pH 7.4; not used in current model but accepted for
        future use.

    Returns
    -------
    float
        Estimated CL in L/h, clamped to [0.1, 200].
    """
    _validate_inputs(mw, psa_A2)
    if not (0.0 <= fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in [0, 1], got {fu_plasma}")

    log_cl = (
        -0.3 * logP
        + 0.02 * psa_A2
        - 0.5 * math.log10(mw)
        + 0.8 * math.log10(fu_plasma + 0.01)
        + 1.5
    )
    cl = 10.0**log_cl
    return _clamp(cl, 0.1, 200.0)


def estimate_vd_human(
    logP: float,
    mw: float,
    fu_plasma: float,
    psa_A2: float,
    n_hbd: float,
    pKa: float | None = None,
) -> float:
    """Estimate human volume of distribution (Vd) from molecular descriptors.

    QSPKR model:
        Vd = 10^(0.5*logP - 0.01*PSA + 0.3*log10(MW) - 0.3*log10(fu+0.01) + 1.2)

    Result is clamped to [1, 1000] L.

    Parameters
    ----------
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight (g/mol).
    fu_plasma:
        Fraction unbound in plasma (0-1).
    psa_A2:
        Polar surface area (Angstrom^2).
    n_hbd:
        Number of hydrogen bond donors.
    pKa:
        Optional ionisation pKa; accepted for future use.

    Returns
    -------
    float
        Estimated Vd in L, clamped to [1, 1000].
    """
    _validate_inputs(mw, psa_A2)
    if not (0.0 <= fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in [0, 1], got {fu_plasma}")

    log_vd = (
        0.5 * logP - 0.01 * psa_A2 + 0.3 * math.log10(mw) - 0.3 * math.log10(fu_plasma + 0.01) + 1.2
    )
    vd = 10.0**log_vd
    return _clamp(vd, 1.0, 1000.0)


def estimate_oral_bioavailability(
    logP: float,
    mw: float,
    psa_A2: float,
    n_hbd: float,
    solubility_mg_mL: float = 1.0,
    fu_plasma: float = 0.1,
) -> float:
    """Estimate oral bioavailability (F) from molecular descriptors.

    Model:
        f_abs  = 1 / (1 + exp(0.5*(PSA - 100))) * (1 if MW < 500 else 0.7)
        f_gut  = 1 / (1 + 0.3*logP)  if logP > 2 else 1.0
        CL_approx (for hepatic FPE) = estimate_cl_human(...)
        f_liver = 1 - CL / (CL + 100)
        F      = f_abs * f_gut * f_liver, clamped to [0.01, 0.95]

    Parameters
    ----------
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight (g/mol).
    psa_A2:
        Polar surface area (Angstrom^2).
    n_hbd:
        Number of hydrogen bond donors.
    solubility_mg_mL:
        Aqueous solubility (mg/mL); accepted for future refinement.
    fu_plasma:
        Fraction unbound in plasma.

    Returns
    -------
    float
        Estimated oral bioavailability (0.01 – 0.95).
    """
    _validate_inputs(mw, psa_A2)

    # Absorption fraction
    f_abs = 1.0 / (1.0 + math.exp(0.5 * (psa_A2 - 100.0)))
    if mw >= 500.0:
        f_abs *= 0.7

    # Gut wall extraction
    if logP > 2.0:
        denom = 1.0 + 0.3 * logP
        f_gut = 1.0 / denom if denom > 0 else 1.0
    else:
        f_gut = 1.0

    # Hepatic first-pass extraction
    cl = estimate_cl_human(logP, mw, fu_plasma, psa_A2, n_hbd)
    f_liver = 1.0 - cl / (cl + 100.0)

    f_oral = f_abs * f_gut * f_liver
    return _clamp(f_oral, 0.01, 0.95)


def _lipinski_compliant(logP: float, mw: float, n_hbd: float, psa_A2: float) -> bool:
    """Check Lipinski Rule of Five compliance."""
    return mw < 500.0 and logP < 5.0 and n_hbd < 5.0 and psa_A2 < 140.0


def _bcs_class(f_oral: float, solubility_mg_mL: float, mw: float) -> str:
    """Assign BCS class based on bioavailability and solubility.

    BCS classification:
        High F (>= 0.7) + High solubility (>= 0.1 mg/mL) → Class I
        High F (>= 0.7) + Low solubility                  → Class II
        Low F  (< 0.7)  + High solubility                 → Class III
        Low F  (< 0.7)  + Low solubility                  → Class IV
    """
    high_f = f_oral >= 0.7
    high_sol = solubility_mg_mL >= 0.1
    if high_f and high_sol:
        return "BCS I"
    if high_f and not high_sol:
        return "BCS II"
    if not high_f and high_sol:
        return "BCS III"
    return "BCS IV"


def _confidence_level(logP: float, mw: float, psa_A2: float) -> str:
    """Determine model confidence based on descriptor applicability domain.

    High:   logP in [0,4], MW < 500, PSA < 100
    Medium: exactly one violation
    Low:    two or more violations
    """
    violations = 0
    if not (0.0 <= logP <= 4.0):
        violations += 1
    if mw >= 500.0:
        violations += 1
    if psa_A2 >= 100.0:
        violations += 1

    if violations == 0:
        return "high"
    if violations == 1:
        return "medium"
    return "low"


def estimate_full_pk(
    compound_name: str,
    logP: float,
    mw: float,
    psa_A2: float,
    n_hbd: float,
    fu_plasma: float = 0.1,
    pKa: float | None = None,
    solubility_mg_mL: float = 1.0,
    logD_74: float | None = None,
) -> PKEstimateResult:
    """Estimate a full set of human PK parameters from molecular descriptors.

    Parameters
    ----------
    compound_name:
        Identifier for the compound.
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight (g/mol). Must be > 0.
    psa_A2:
        Polar surface area (Angstrom^2). Must be >= 0.
    n_hbd:
        Number of hydrogen bond donors.
    fu_plasma:
        Fraction unbound in plasma (default 0.1).
    pKa:
        Optional ionisation pKa.
    solubility_mg_mL:
        Aqueous solubility (mg/mL, default 1.0).
    logD_74:
        Optional logD at pH 7.4.

    Returns
    -------
    PKEstimateResult
    """
    _validate_inputs(mw, psa_A2)
    if not (0.0 <= fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in [0, 1], got {fu_plasma}")

    cl = estimate_cl_human(logP, mw, fu_plasma, psa_A2, n_hbd, logD_74=logD_74)
    vd = estimate_vd_human(logP, mw, fu_plasma, psa_A2, n_hbd, pKa=pKa)
    t_half = math.log(2.0) * vd / cl
    f_oral = estimate_oral_bioavailability(
        logP, mw, psa_A2, n_hbd, solubility_mg_mL=solubility_mg_mL, fu_plasma=fu_plasma
    )
    mrt = vd / cl  # MRT = Vd / CL for a one-compartment model

    lipinski = _lipinski_compliant(logP, mw, n_hbd, psa_A2)
    bcs = _bcs_class(f_oral, solubility_mg_mL, mw)
    confidence = _confidence_level(logP, mw, psa_A2)

    notes = (
        f"QSPKR estimates; empirically calibrated. "
        f"logP={logP:.2f}, MW={mw:.1f}, PSA={psa_A2:.1f}, fu={fu_plasma:.3f}. "
        f"Confidence: {confidence}."
    )

    return PKEstimateResult(
        compound_name=compound_name,
        logP=logP,
        mw=mw,
        psa_A2=psa_A2,
        cl_L_per_h=cl,
        vd_L=vd,
        t_half_h=t_half,
        f_oral=f_oral,
        mrt_h=mrt,
        lipinski_compliant=lipinski,
        bcs_class=bcs,
        confidence=confidence,
        notes=notes,
    )
