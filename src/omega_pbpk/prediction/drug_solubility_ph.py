"""
Phase 941: pH-solubility profile prediction for ionizable drugs.

Uses Henderson-Hasselbalch ionization and simplified Yalkowsky intrinsic solubility.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "SolubilityPhProfileResult",
    "predict_solubility_ph_profile",
    "screen_solubility_profiles",
]

_VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}

_GASTRIC_PH = 1.2
_INTESTINAL_PH = 6.8
_COLONIC_PH = 7.4
_MAX_SOLUBILITY = 1000.0  # mg/mL cap
_MIN_S0 = 1e-6
_MAX_S0 = 1000.0


@dataclass
class SolubilityPhProfileResult:
    """Result of pH-solubility profile prediction."""

    drug_name: str
    logp: float
    pka: float
    ionization_type: str
    intrinsic_solubility_mg_mL: float
    ph_values: list
    solubility_mg_mL: list
    ph_max_solubility: float
    max_solubility_mg_mL: float
    ph_min_solubility: float
    min_solubility_mg_mL: float
    solubility_at_gastric_ph: float
    solubility_at_intestinal_ph: float
    solubility_at_colonic_ph: float
    gut_solubility_ratio: float
    bcs_solubility_class: str
    notes: str


def _intrinsic_solubility(logp: float) -> float:
    """Simplified Yalkowsky: log10(S0) = -logP + 0.5"""
    log_s0 = -logp + 0.5
    s0 = 10.0**log_s0
    return max(_MIN_S0, min(_MAX_S0, s0))


def _solubility_at_ph(s0: float, ph: float, pka: float, ionization_type: str) -> float:
    """Compute solubility at given pH using Henderson-Hasselbalch."""
    if ionization_type == "neutral":
        return s0
    elif ionization_type == "acid":
        # S = S0 * (1 + 10^(pH - pKa))
        exponent = ph - pka
        # Clamp exponent to avoid overflow
        exponent = max(-300.0, min(300.0, exponent))
        s = s0 * (1.0 + 10.0**exponent)
    elif ionization_type == "base":
        # S = S0 * (1 + 10^(pKa - pH))
        exponent = pka - ph
        exponent = max(-300.0, min(300.0, exponent))
        s = s0 * (1.0 + 10.0**exponent)
    else:
        raise ValueError(f"ionization_type must be one of {_VALID_IONIZATION_TYPES}")

    # Cap at max and floor at s0
    s = min(s, _MAX_SOLUBILITY)
    s = max(s, s0)
    return s


def predict_solubility_ph_profile(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str,
    dose_mg: float = 100.0,
    ph_range: tuple = (1.0, 10.0),
    n_points: int = 37,
) -> SolubilityPhProfileResult:
    """
    Predict pH-solubility profile for an ionizable drug.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    logp : float
        Octanol-water partition coefficient.
    pka : float
        Acid or base dissociation constant.
    ionization_type : str
        One of "acid", "base", "neutral".
    dose_mg : float
        Dose in mg (used for BCS classification). Default 100.0.
    ph_range : tuple
        (pH_min, pH_max). Default (1.0, 10.0).
    n_points : int
        Number of pH points. Default 37 (1.0 to 10.0 in steps of 0.25).

    Returns
    -------
    SolubilityPhProfileResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION_TYPES}, got '{ionization_type}'"
        )
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    if ph_range[0] >= ph_range[1]:
        raise ValueError("ph_range must be increasing: ph_range[0] < ph_range[1]")

    # Build pH grid
    ph_min, ph_max = ph_range
    step = (ph_max - ph_min) / (n_points - 1)
    ph_values = [ph_min + i * step for i in range(n_points)]

    # Intrinsic solubility
    s0 = _intrinsic_solubility(logp)

    # Solubility at each pH
    solubility = [_solubility_at_ph(s0, ph, pka, ionization_type) for ph in ph_values]

    # Max/min pH and solubility
    max_sol = max(solubility)
    min_sol = min(solubility)
    ph_max_sol = ph_values[solubility.index(max_sol)]
    ph_min_sol = ph_values[solubility.index(min_sol)]

    # Solubility at physiologically relevant pH values
    sol_gastric = _solubility_at_ph(s0, _GASTRIC_PH, pka, ionization_type)
    sol_intestinal = _solubility_at_ph(s0, _INTESTINAL_PH, pka, ionization_type)
    sol_colonic = _solubility_at_ph(s0, _COLONIC_PH, pka, ionization_type)

    # Gut solubility ratio
    if sol_gastric > 0:
        gut_ratio = sol_intestinal / sol_gastric
    else:
        gut_ratio = 1.0

    # BCS classification: high if sol_intestinal * 250 mL >= dose_mg
    # (250 mL = volume of water for BCS classification)
    bcs_class = "high" if sol_intestinal * 250.0 >= dose_mg else "low"

    # Notes
    notes_parts = [
        f"Intrinsic solubility (S0) = {s0:.4g} mg/mL (logP={logp:.2f}).",
        f"Ionization type: {ionization_type} (pKa={pka:.2f}).",
        f"Max solubility {max_sol:.4g} mg/mL at pH {ph_max_sol:.2f}.",
        f"BCS solubility: {bcs_class} (dose={dose_mg} mg, S_intestinal*250mL={sol_intestinal * 250:.2f} mg).",
    ]
    notes = " ".join(notes_parts)

    return SolubilityPhProfileResult(
        drug_name=drug_name,
        logp=logp,
        pka=pka,
        ionization_type=ionization_type,
        intrinsic_solubility_mg_mL=s0,
        ph_values=ph_values,
        solubility_mg_mL=solubility,
        ph_max_solubility=ph_max_sol,
        max_solubility_mg_mL=max_sol,
        ph_min_solubility=ph_min_sol,
        min_solubility_mg_mL=min_sol,
        solubility_at_gastric_ph=sol_gastric,
        solubility_at_intestinal_ph=sol_intestinal,
        solubility_at_colonic_ph=sol_colonic,
        gut_solubility_ratio=gut_ratio,
        bcs_solubility_class=bcs_class,
        notes=notes,
    )


def screen_solubility_profiles(
    compounds: list,
    target_ph: float = 6.8,
) -> list:
    """
    Screen a list of compounds and return pH-solubility profiles sorted by
    solubility at target_ph (descending).

    Parameters
    ----------
    compounds : list[dict]
        Each dict must contain keys: drug_name, logp, pka, ionization_type.
        Optional keys: dose_mg, ph_range, n_points.
    target_ph : float
        pH at which to evaluate solubility for ranking. Default 6.8.

    Returns
    -------
    list[SolubilityPhProfileResult] sorted by solubility at target_ph descending.
    """
    results = []
    for cmpd in compounds:
        result = predict_solubility_ph_profile(
            drug_name=cmpd["drug_name"],
            logp=cmpd["logp"],
            pka=cmpd["pka"],
            ionization_type=cmpd["ionization_type"],
            dose_mg=cmpd.get("dose_mg", 100.0),
            ph_range=cmpd.get("ph_range", (1.0, 10.0)),
            n_points=cmpd.get("n_points", 37),
        )
        results.append(result)

    # Sort by solubility at target_ph descending
    def _sol_at_target(r: SolubilityPhProfileResult) -> float:
        s0 = r.intrinsic_solubility_mg_mL
        return _solubility_at_ph(s0, target_ph, r.pka, r.ionization_type)

    results.sort(key=_sol_at_target, reverse=True)
    return results
