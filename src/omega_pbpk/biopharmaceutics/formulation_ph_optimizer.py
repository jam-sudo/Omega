"""Formulation pH optimization for drug solubility — Phase 480.

Optimizes formulation pH to maximize drug solubility while considering
stability constraints. Uses the Henderson-Hasselbalch equation to compute
pH-dependent solubility for ionizable drugs.

Henderson-Hasselbalch solubility:
    Acid:  S(pH) = S0 * (1 + 10^(pH - pKa))
    Base:  S(pH) = S0 * (1 + 10^(pKa - pH))

where S0 is the intrinsic (un-ionized) solubility.

Buffer capacity:
    beta = 2.303 * C * Ka * [H+] / (Ka + [H+])^2

References
----------
- Avdeef A, Absorption and Drug Development, 2nd ed. Wiley 2012
- Dressman JB & Reppas C (eds), Oral Drug Absorption, 2nd ed. Informa 2010
- Sorby DL, J Pharm Sci. 1965;54:677-680
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class pHOptimizationResult:
    """Result of pH optimization for drug solubility.

    Attributes
    ----------
    drug_type : 'acid' or 'base'.
    pka : Drug pKa.
    intrinsic_solubility_mg_mL : S0, solubility of un-ionized form (mg/mL).
    target_solubility_mg_mL : Desired solubility (mg/mL).
    optimal_ph : pH at which maximum solubility is achieved within ph_range.
    max_solubility_mg_mL : Maximum achievable solubility across full scan.
    solubility_at_optimal_ph : Solubility at optimal_ph (mg/mL).
    target_achievable : Whether target_solubility can be reached.
    stability_concern : Whether optimal_ph falls outside stability_range.
    notes : Human-readable summary.
    """

    drug_type: str
    pka: float
    intrinsic_solubility_mg_mL: float
    target_solubility_mg_mL: float
    optimal_ph: float
    max_solubility_mg_mL: float
    solubility_at_optimal_ph: float
    target_achievable: bool
    stability_concern: bool
    notes: str


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _hh_solubility(
    ph: float,
    pka: float,
    drug_type: str,
    intrinsic_solubility_mg_mL: float,
) -> float:
    """Henderson-Hasselbalch solubility at a given pH.

    Parameters
    ----------
    ph : Solution pH.
    pka : Drug pKa.
    drug_type : 'acid' or 'base'.
    intrinsic_solubility_mg_mL : S0 (mg/mL).

    Returns
    -------
    float : Solubility (mg/mL).
    """
    if drug_type == "acid":
        return intrinsic_solubility_mg_mL * (1.0 + 10.0 ** (ph - pka))
    elif drug_type == "base":
        return intrinsic_solubility_mg_mL * (1.0 + 10.0 ** (pka - ph))
    else:
        raise ValueError(f"drug_type must be 'acid' or 'base', got {drug_type!r}")


def solubility_vs_ph(
    pka: float,
    logP: float,  # noqa: N803  (conventional pharmaceutical naming)
    mw: float,
    drug_type: str = "acid",
    intrinsic_solubility_mg_mL: float = 0.1,
    ph_range: tuple[float, float] = (1.0, 10.0),
    n_points: int = 50,
) -> dict:
    """Compute pH-solubility profile over a pH range.

    Parameters
    ----------
    pka : Drug pKa.
    logP : Lipophilicity (octanol-water log partition).
    mw : Molecular weight (g/mol).
    drug_type : 'acid' or 'base'.
    intrinsic_solubility_mg_mL : S0 for un-ionized species (mg/mL).
    ph_range : (pH_min, pH_max) tuple.
    n_points : Number of pH points to evaluate.

    Returns
    -------
    dict with keys 'ph_values' (list[float]) and 'solubility_mg_mL' (list[float]).
    """
    if pka <= 0:
        raise ValueError(f"pka must be > 0, got {pka}")
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError(
            f"intrinsic_solubility_mg_mL must be > 0, got {intrinsic_solubility_mg_mL}"
        )
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if ph_range[0] >= ph_range[1]:
        raise ValueError(f"ph_range[0] must be < ph_range[1], got {ph_range}")

    ph_min, ph_max = ph_range
    step = (ph_max - ph_min) / (n_points - 1)
    ph_values = [ph_min + i * step for i in range(n_points)]
    solubilities = [
        _hh_solubility(ph, pka, drug_type, intrinsic_solubility_mg_mL) for ph in ph_values
    ]

    return {"ph_values": ph_values, "solubility_mg_mL": solubilities}


def optimal_ph_for_solubility(
    pka: float,
    logP: float,  # noqa: N803
    mw: float,
    drug_type: str = "acid",
    intrinsic_solubility_mg_mL: float = 0.1,
    target_solubility_mg_mL: float = 1.0,
) -> pHOptimizationResult:
    """Find the pH that maximises solubility within pharmaceutical pH limits.

    Scans pH 1–12 in 0.1 increments. The optimal pH is chosen as the pH that
    achieves the highest solubility (for acids: highest feasible pH; for bases:
    lowest feasible pH). Stability is assessed against the range (3, 8).

    Parameters
    ----------
    pka : Drug pKa.
    logP : Lipophilicity.
    mw : Molecular weight (g/mol).
    drug_type : 'acid' or 'base'.
    intrinsic_solubility_mg_mL : S0 (mg/mL).
    target_solubility_mg_mL : Desired minimum solubility (mg/mL).

    Returns
    -------
    pHOptimizationResult
    """
    if pka <= 0:
        raise ValueError(f"pka must be > 0, got {pka}")
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError(
            f"intrinsic_solubility_mg_mL must be > 0, got {intrinsic_solubility_mg_mL}"
        )
    if target_solubility_mg_mL <= 0:
        raise ValueError(f"target_solubility_mg_mL must be > 0, got {target_solubility_mg_mL}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")

    # Practical pharmaceutical pH range
    _PH_MIN = 1.0
    _PH_MAX = 12.0
    _STEP = 0.1
    _STABILITY_MIN = 3.0
    _STABILITY_MAX = 8.0

    n = int(round((_PH_MAX - _PH_MIN) / _STEP)) + 1
    ph_values = [_PH_MIN + i * _STEP for i in range(n)]
    solubilities = [
        _hh_solubility(ph, pka, drug_type, intrinsic_solubility_mg_mL) for ph in ph_values
    ]

    max_sol = max(solubilities)
    max_idx = solubilities.index(max_sol)
    optimal_ph = ph_values[max_idx]
    sol_at_optimal = solubilities[max_idx]

    target_achievable = max_sol >= target_solubility_mg_mL
    stability_concern = not (_STABILITY_MIN <= optimal_ph <= _STABILITY_MAX)

    if stability_concern and target_achievable:
        # Look for best pH within stability window that still meets target
        best_stable_sol = -1.0
        best_stable_ph = optimal_ph
        for ph, sol in zip(ph_values, solubilities, strict=False):
            if _STABILITY_MIN <= ph <= _STABILITY_MAX and sol > best_stable_sol:
                best_stable_sol = sol
                best_stable_ph = ph
        if best_stable_sol >= target_solubility_mg_mL:
            optimal_ph = best_stable_ph
            sol_at_optimal = best_stable_sol
            stability_concern = False

    notes_parts = [
        f"Drug type: {drug_type}, pKa = {pka:.1f}",
        f"Optimal pH = {optimal_ph:.1f}, solubility = {sol_at_optimal:.4f} mg/mL",
        f"Target {'achievable' if target_achievable else 'NOT achievable'}",
    ]
    if stability_concern:
        notes_parts.append("Stability concern: optimal pH outside 3–8 range")

    return pHOptimizationResult(
        drug_type=drug_type,
        pka=pka,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        target_solubility_mg_mL=target_solubility_mg_mL,
        optimal_ph=round(optimal_ph, 1),
        max_solubility_mg_mL=max_sol,
        solubility_at_optimal_ph=sol_at_optimal,
        target_achievable=target_achievable,
        stability_concern=stability_concern,
        notes="; ".join(notes_parts),
    )


def buffer_capacity(
    ph: float,
    pka: float,
    concentration_M: float = 0.05,
) -> float:
    """Compute buffer capacity (beta) at a given pH.

    beta = 2.303 * C * Ka * [H+] / (Ka + [H+])^2

    Parameters
    ----------
    ph : Solution pH.
    pka : Buffer pKa.
    concentration_M : Total buffer concentration (mol/L); default 0.05 M.

    Returns
    -------
    float : Buffer capacity (mol/L/pH unit).
    """
    if pka <= 0:
        raise ValueError(f"pka must be > 0, got {pka}")
    if concentration_M <= 0:
        raise ValueError(f"concentration_M must be > 0, got {concentration_M}")

    h_plus = 10.0 ** (-ph)
    ka = 10.0 ** (-pka)
    numerator = ka * h_plus
    denominator = (ka + h_plus) ** 2
    return 2.303 * concentration_M * numerator / denominator


def predict_stability_ph(
    drug_type: str,
    optimal_ph: float,
    stability_range: tuple[float, float] = (3.0, 8.0),
) -> dict:
    """Predict formulation stability relative to an optimal pH.

    Parameters
    ----------
    drug_type : 'acid' or 'base'.
    optimal_ph : pH value to assess.
    stability_range : (pH_min, pH_max) acceptable stability window.

    Returns
    -------
    dict with keys:
        'optimal_ph'         : float
        'stability_range'    : tuple[float, float]
        'stability_concern'  : bool
        'recommendation'     : str
        'risk_level'         : str  ('low', 'moderate', 'high')
    """
    if drug_type not in {"acid", "base"}:
        raise ValueError(f"drug_type must be 'acid' or 'base', got {drug_type!r}")

    ph_min, ph_max = stability_range
    if ph_min >= ph_max:
        raise ValueError(f"stability_range[0] must be < stability_range[1], got {stability_range}")

    in_range = ph_min <= optimal_ph <= ph_max
    stability_concern = not in_range

    if in_range:
        risk_level = "low"
        recommendation = (
            f"Optimal pH {optimal_ph:.1f} is within the stability window "
            f"({ph_min}–{ph_max}). No formulation adjustment needed."
        )
    else:
        distance = min(abs(optimal_ph - ph_min), abs(optimal_ph - ph_max))
        if distance < 1.0:
            risk_level = "moderate"
            recommendation = (
                f"Optimal pH {optimal_ph:.1f} is slightly outside the stability window "
                f"({ph_min}–{ph_max}). Consider buffering near the boundary "
                f"or adding antioxidants/stabilisers."
            )
        else:
            risk_level = "high"
            recommendation = (
                f"Optimal pH {optimal_ph:.1f} is significantly outside the stability "
                f"window ({ph_min}–{ph_max}). Formulation excipients, solid-state "
                f"form, or salt selection should be reconsidered."
            )

    return {
        "optimal_ph": optimal_ph,
        "stability_range": stability_range,
        "stability_concern": stability_concern,
        "recommendation": recommendation,
        "risk_level": risk_level,
    }
