"""Brain-to-blood concentration ratio (logBB) predictor (Phase 599).

Implements Young et al. 1988 model with Norinder corrections to predict
logBB from physicochemical properties.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "LogBBResult",
    "predict_logbb",
    "screen_cns_compounds",
]


@dataclass(frozen=True)
class LogBBResult:
    """Result from logBB prediction."""

    drug_name: str
    logP: float
    mw: float
    psa: float
    logBB: float
    bb_ratio: float
    cns_penetration: str
    pgp_efflux_risk: str
    bbb_permeability_class: str
    predicted_brain_conc_mg_L: float
    notes: str


def predict_logbb(
    drug_name: str,
    logP: float,
    mw: float,
    psa: float,
    plasma_conc_mg_L: float = 1.0,
    n_hbd: int = 0,
    pka_base: float | None = None,
) -> LogBBResult:
    """Predict brain-to-blood concentration ratio (logBB).

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Octanol-water partition coefficient (log scale).
    mw:
        Molecular weight in Da.
    psa:
        Polar surface area in Å².
    plasma_conc_mg_L:
        Plasma concentration in mg/L (default 1.0).
    n_hbd:
        Number of hydrogen bond donors.
    pka_base:
        Basic pKa if the drug is a base (None if neutral/acidic).

    Returns
    -------
    LogBBResult
    """
    # Validation
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if psa < 0:
        raise ValueError(f"psa must be >= 0, got {psa}")
    if plasma_conc_mg_L <= 0:
        raise ValueError(f"plasma_conc_mg_L must be > 0, got {plasma_conc_mg_L}")
    if n_hbd < 0:
        raise ValueError(f"n_hbd must be >= 0, got {n_hbd}")

    # Base logBB (Young et al. 1988 + Norinder)
    logbb = 0.152 * logP - 0.0148 * psa + 0.139

    # MW correction
    if mw > 450:
        logbb -= 0.5

    # HBD correction
    logbb -= 0.2 * n_hbd

    # Basic drug bonus
    if pka_base is not None and pka_base > 7:
        logbb += 0.3

    bb_ratio = 10.0**logbb

    # CNS penetration classification
    if logbb > 0.3:
        cns_penetration = "high"
    elif logbb > -0.3:
        cns_penetration = "moderate"
    elif logbb > -1.0:
        cns_penetration = "low"
    else:
        cns_penetration = "negligible"

    # P-gp efflux risk
    if mw > 400 and psa > 90:
        pgp_efflux_risk = "high"
    else:
        pgp_efflux_risk = "low"

    # BBB permeability class
    if psa < 60 and mw < 400:
        bbb_permeability_class = "permeable"
    elif psa < 90:
        bbb_permeability_class = "moderate"
    else:
        bbb_permeability_class = "impermeable"

    predicted_brain_conc = plasma_conc_mg_L * bb_ratio

    # Build notes
    note_parts = [
        f"logBB={logbb:.3f} (Young 1988 + Norinder)",
        f"CNS penetration: {cns_penetration}",
        f"P-gp efflux risk: {pgp_efflux_risk}",
        f"BBB class: {bbb_permeability_class}",
    ]
    if mw > 450:
        note_parts.append("MW>450 penalty applied")
    if n_hbd > 0:
        note_parts.append(f"HBD penalty: {0.2 * n_hbd:.2f}")
    if pka_base is not None and pka_base > 7:
        note_parts.append("Basic drug bonus applied")
    notes = "; ".join(note_parts)

    return LogBBResult(
        drug_name=drug_name,
        logP=logP,
        mw=mw,
        psa=psa,
        logBB=logbb,
        bb_ratio=bb_ratio,
        cns_penetration=cns_penetration,
        pgp_efflux_risk=pgp_efflux_risk,
        bbb_permeability_class=bbb_permeability_class,
        predicted_brain_conc_mg_L=predicted_brain_conc,
        notes=notes,
    )


def screen_cns_compounds(compounds: list) -> list:
    """Screen a list of compounds for CNS penetration.

    Parameters
    ----------
    compounds:
        List of dicts with keys: name, logP, mw, psa, n_hbd
        Optional keys: plasma_conc_mg_L, pka_base

    Returns
    -------
    list of LogBBResult sorted by logBB descending.
    """
    results = []
    for comp in compounds:
        result = predict_logbb(
            drug_name=comp["name"],
            logP=comp["logP"],
            mw=comp["mw"],
            psa=comp["psa"],
            plasma_conc_mg_L=comp.get("plasma_conc_mg_L", 1.0),
            n_hbd=comp.get("n_hbd", 0),
            pka_base=comp.get("pka_base", None),
        )
        results.append(result)

    results.sort(key=lambda r: r.logBB, reverse=True)
    return results
