"""
Phase 952 — Drug excretion route prediction.

Predicts renal, biliary, pulmonary, and fecal excretion fractions from
physicochemical properties using rule-based scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DrugExcretionPredictionResult",
    "predict_excretion",
    "screen_excretion_routes",
]

_VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class DrugExcretionPredictionResult:
    """Predicted drug excretion routes (frozen dataclass)."""

    drug_name: str
    logp: float
    mw: float
    pka: float
    ionization_type: str
    fu_plasma: float
    fe_renal: float
    fe_biliary: float
    fe_pulmonary: float
    fe_fecal: float
    primary_excretion_route: str
    renal_sensitivity: str
    cl_renal_predicted_L_per_h: float
    cl_biliary_predicted_L_per_h: float
    notes: str


def _compute_ionization_fraction(pka: float, ionization_type: str) -> float:
    """
    Approximate ionized fraction at physiological pH 7.4.
    Acid: f_ionized = 1 / (1 + 10^(pKa - 7.4))
    Base: f_ionized = 1 / (1 + 10^(7.4 - pKa))
    Neutral: 0
    """
    if ionization_type == "acid":
        # Henderson-Hasselbalch: [A-]/[HA] = 10^(pH-pKa)
        ratio = 10.0 ** (7.4 - pka)
        return ratio / (1.0 + ratio)
    elif ionization_type == "base":
        # [BH+]/[B] = 10^(pKa-pH)
        ratio = 10.0 ** (pka - 7.4)
        return ratio / (1.0 + ratio)
    else:
        return 0.0


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _predict_fractions(
    logp: float,
    mw: float,
    pka: float,
    ionization_type: str,
    fu_plasma: float,
) -> tuple:
    """Return (fe_renal, fe_biliary, fe_pulmonary) before normalisation."""

    ionization_fraction = _compute_ionization_fraction(pka, ionization_type)

    # --- Renal ---
    fe_renal = 0.0
    if mw < 200:
        fe_renal += 0.6
    elif mw < 300:
        fe_renal += 0.5
    if logp < 0:
        fe_renal += 0.3
    if ionization_fraction > 0.5:
        fe_renal += 0.2
    if fu_plasma > 0.3:
        fe_renal += 0.1
    fe_renal = _clamp(fe_renal)

    # --- Biliary ---
    fe_biliary = 0.0
    if mw > 500:
        fe_biliary += 0.5
    if logp > 2:
        fe_biliary += 0.2
    if mw > 300 and ionization_type == "acid":
        fe_biliary += 0.2
    fe_biliary = _clamp(fe_biliary)

    # --- Pulmonary ---
    fe_pulmonary = 0.0
    if logp >= -2 and logp <= 3 and mw < 200:
        fe_pulmonary += 0.3
    else:
        fe_pulmonary += 0.02
    fe_pulmonary = _clamp(fe_pulmonary)

    return fe_renal, fe_biliary, fe_pulmonary


def predict_excretion(
    drug_name: str,
    logp: float,
    mw: float,
    pka: float,
    ionization_type: str,
    fu_plasma: float = 0.5,
    cl_total_L_per_h: float = 10.0,
) -> DrugExcretionPredictionResult:
    """
    Predict excretion routes for a single compound.

    Parameters
    ----------
    drug_name : str
    logp : float
    mw : float  — molecular weight in Da
    pka : float
    ionization_type : str — "acid", "base", or "neutral"
    fu_plasma : float — unbound fraction in plasma (0, 1]
    cl_total_L_per_h : float — total clearance used to derive route-specific CLs

    Returns
    -------
    DrugExcretionPredictionResult (frozen)
    """
    # Validate
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if fu_plasma <= 0 or fu_plasma > 1:
        raise ValueError("fu_plasma must be in (0, 1]")
    if cl_total_L_per_h <= 0:
        raise ValueError("cl_total_L_per_h must be > 0")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {sorted(_VALID_IONIZATION_TYPES)}, "
            f"got '{ionization_type}'"
        )

    fe_renal, fe_biliary, fe_pulmonary = _predict_fractions(
        logp, mw, pka, ionization_type, fu_plasma
    )

    # Fecal = remainder
    fe_fecal = _clamp(1.0 - fe_renal - fe_biliary - fe_pulmonary)

    # Normalize so fractions sum to 1
    total = fe_renal + fe_biliary + fe_pulmonary + fe_fecal
    if total > 0:
        fe_renal /= total
        fe_biliary /= total
        fe_pulmonary /= total
        fe_fecal /= total
    else:
        fe_fecal = 1.0

    # Primary route
    route_map = {
        "renal": fe_renal,
        "biliary": fe_biliary,
        "pulmonary": fe_pulmonary,
        "fecal": fe_fecal,
    }
    primary_route = max(route_map, key=lambda k: route_map[k])

    # Renal sensitivity
    if fe_renal > 0.7:
        renal_sensitivity = "high"
    elif fe_renal > 0.3:
        renal_sensitivity = "moderate"
    else:
        renal_sensitivity = "low"

    # Predicted clearances
    cl_renal = fe_renal * cl_total_L_per_h
    cl_biliary = fe_biliary * cl_total_L_per_h

    ionization_frac = _compute_ionization_fraction(pka, ionization_type)
    notes = (
        f"Excretion prediction for {drug_name}: MW={mw:.1f}, logP={logp:.2f}, "
        f"ionization_type={ionization_type}, ionized_frac@pH7.4={ionization_frac:.2f}. "
        f"Primary route: {primary_route} (fe={route_map[primary_route]:.2f}). "
        f"Renal sensitivity: {renal_sensitivity}."
    )

    return DrugExcretionPredictionResult(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        pka=pka,
        ionization_type=ionization_type,
        fu_plasma=fu_plasma,
        fe_renal=fe_renal,
        fe_biliary=fe_biliary,
        fe_pulmonary=fe_pulmonary,
        fe_fecal=fe_fecal,
        primary_excretion_route=primary_route,
        renal_sensitivity=renal_sensitivity,
        cl_renal_predicted_L_per_h=cl_renal,
        cl_biliary_predicted_L_per_h=cl_biliary,
        notes=notes,
    )


def screen_excretion_routes(compounds: list) -> list:
    """
    Screen a list of compounds and return excretion predictions sorted by fe_renal descending.

    Each element of ``compounds`` must be a dict with keys accepted by ``predict_excretion``:
        drug_name, logp, mw, pka, ionization_type,
        (optional) fu_plasma, cl_total_L_per_h

    Returns
    -------
    list[DrugExcretionPredictionResult] sorted by fe_renal descending
    """
    results = []
    for cmpd in compounds:
        cmpd = dict(cmpd)  # shallow copy to avoid mutation
        drug_name = cmpd.pop("drug_name")
        logp = cmpd.pop("logp")
        mw = cmpd.pop("mw")
        pka = cmpd.pop("pka")
        ionization_type = cmpd.pop("ionization_type")
        # remaining optional kwargs
        r = predict_excretion(
            drug_name=drug_name,
            logp=logp,
            mw=mw,
            pka=pka,
            ionization_type=ionization_type,
            **cmpd,
        )
        results.append(r)

    results.sort(key=lambda x: x.fe_renal, reverse=True)
    return results
