"""Phase 269: Protein stability predictor for therapeutic proteins during
manufacturing and storage.

Pure Python (stdlib: math, dataclasses only).
"""

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Stress condition lookup table
# ---------------------------------------------------------------------------

_STRESS_CONDITIONS = {
    "room_temp_open": {
        "temp_C": 25.0,
        "oxidation_risk": 0.8,
        "hydrolysis_risk": 0.5,
        "agitation": 0.3,
    },
    "refrigerated": {
        "temp_C": 5.0,
        "oxidation_risk": 0.2,
        "hydrolysis_risk": 0.1,
        "agitation": 0.0,
    },
    "frozen": {
        "temp_C": -20.0,
        "oxidation_risk": 0.05,
        "hydrolysis_risk": 0.02,
        "agitation": 0.0,
    },
    "freeze_thaw": {
        "temp_C": 4.0,
        "oxidation_risk": 0.3,
        "hydrolysis_risk": 0.2,
        "agitation": 0.9,
    },
    "light_exposed": {
        "temp_C": 25.0,
        "oxidation_risk": 1.0,
        "hydrolysis_risk": 0.3,
        "agitation": 0.1,
    },
}

_VALID_CONDITIONS = list(_STRESS_CONDITIONS.keys())


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProteinStabilityResult:
    """Result of a protein stability prediction."""

    protein_name: str
    stress_condition: str
    mw_kDa: float
    pi: float
    storage_ph: float
    aggregation_rate_per_day: float
    oxidation_rate_per_day: float
    hydrolysis_rate_per_day: float
    predicted_monomer_pct_30days: float
    predicted_monomer_pct_12months: float
    shelf_life_days: float
    dominant_degradation_pathway: str
    stability_grade: str
    notes: str


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------


def predict_protein_stability(
    protein_name: str,
    mw_kDa: float,
    pi: float,
    stress_condition: str,
    storage_ph: float = 7.0,
) -> ProteinStabilityResult:
    """Predict stability of a therapeutic protein under a given stress condition.

    Parameters
    ----------
    protein_name:
        Human-readable name for the protein.
    mw_kDa:
        Molecular weight in kDa (must be > 0).
    pi:
        Isoelectric point (must be in [0, 14]).
    stress_condition:
        One of the five pre-defined stress conditions.
    storage_ph:
        Storage formulation pH (must be in [4, 10]).

    Returns
    -------
    ProteinStabilityResult
    """
    # --- Validation ---
    if mw_kDa <= 0:
        raise ValueError(f"mw_kDa must be > 0, got {mw_kDa}")
    if not (0 <= pi <= 14):
        raise ValueError(f"pi must be in [0, 14], got {pi}")
    if not (4 <= storage_ph <= 10):
        raise ValueError(f"storage_ph must be in [4, 10], got {storage_ph}")
    if stress_condition not in _STRESS_CONDITIONS:
        raise ValueError(
            f"stress_condition must be one of {_VALID_CONDITIONS}, got '{stress_condition}'"
        )

    cond = _STRESS_CONDITIONS[stress_condition]
    temp_C: float = cond["temp_C"]
    oxidation_risk: float = cond["oxidation_risk"]
    hydrolysis_risk: float = cond["hydrolysis_risk"]
    agitation: float = cond["agitation"]

    # --- Rate calculations ---
    # Aggregation: Arrhenius-like, modulated by agitation and pH proximity to pI
    base_agg = 0.001  # /day at 25 deg C
    k_agg_raw = (
        base_agg * math.exp(0.08 * (temp_C - 25.0)) * agitation * (1.0 + 0.1 * abs(storage_ph - pi))
    )
    k_agg = max(0.0, min(0.1, k_agg_raw))

    k_ox = 0.0005 * oxidation_risk  # /day
    k_hyd = 0.0003 * hydrolysis_risk  # /day

    k_total = k_agg + k_ox + k_hyd

    # --- Monomer fraction over time ---
    t_30d = 30.0
    t_12m = 365.0

    predicted_monomer_pct_30days = 100.0 * math.exp(-k_total * t_30d)
    predicted_monomer_pct_12months = 100.0 * math.exp(-k_total * t_12m)

    # --- Shelf life (time to 90% monomer) ---
    if k_total > 0.0:
        raw_shelf = -math.log(0.9) / k_total
        shelf_life_days = min(raw_shelf, 3650.0)
    else:
        shelf_life_days = 3650.0

    # --- Dominant degradation pathway ---
    rates = {
        "aggregation": k_agg,
        "oxidation": k_ox,
        "hydrolysis": k_hyd,
    }
    dominant_degradation_pathway = max(rates, key=lambda k: rates[k])

    # --- Stability grade ---
    if predicted_monomer_pct_12months >= 95.0:
        stability_grade = "A"
    elif predicted_monomer_pct_12months >= 90.0:
        stability_grade = "B"
    else:
        stability_grade = "C"

    # --- Notes ---
    note_parts: list[str] = []
    note_parts.append(
        f"Dominant pathway: {dominant_degradation_pathway} "
        f"(k={rates[dominant_degradation_pathway]:.5f}/day)"
    )
    if abs(storage_ph - pi) < 0.5:
        note_parts.append("Storage pH near pI increases aggregation risk")
    if agitation > 0.5:
        note_parts.append("High agitation contributes to aggregation")
    notes = "; ".join(note_parts) if note_parts else "No notable concerns"

    return ProteinStabilityResult(
        protein_name=protein_name,
        stress_condition=stress_condition,
        mw_kDa=mw_kDa,
        pi=pi,
        storage_ph=storage_ph,
        aggregation_rate_per_day=k_agg,
        oxidation_rate_per_day=k_ox,
        hydrolysis_rate_per_day=k_hyd,
        predicted_monomer_pct_30days=predicted_monomer_pct_30days,
        predicted_monomer_pct_12months=predicted_monomer_pct_12months,
        shelf_life_days=shelf_life_days,
        dominant_degradation_pathway=dominant_degradation_pathway,
        stability_grade=stability_grade,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Screen all storage conditions
# ---------------------------------------------------------------------------


def screen_storage_conditions(
    protein_name: str,
    mw_kDa: float,
    pi: float,
    storage_ph: float = 7.0,
) -> list[ProteinStabilityResult]:
    """Screen all 5 stress conditions and return results sorted by
    predicted_monomer_pct_12months descending (most stable first).

    Parameters
    ----------
    protein_name:
        Human-readable name for the protein.
    mw_kDa:
        Molecular weight in kDa (must be > 0).
    pi:
        Isoelectric point (must be in [0, 14]).
    storage_ph:
        Storage formulation pH (must be in [4, 10]).

    Returns
    -------
    List of ProteinStabilityResult sorted most-stable first.
    """
    results: list[ProteinStabilityResult] = []
    for condition in _VALID_CONDITIONS:
        result = predict_protein_stability(
            protein_name=protein_name,
            mw_kDa=mw_kDa,
            pi=pi,
            stress_condition=condition,
            storage_ph=storage_ph,
        )
        results.append(result)

    results.sort(key=lambda r: r.predicted_monomer_pct_12months, reverse=True)
    return results
