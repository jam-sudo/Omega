"""Phase 1142 — Freeze-drying (lyophilization) stability prediction.

Model for protein/peptide drug lyophilization stability based on
residual moisture, glass transition temperature, and excipient properties.
"""

from __future__ import annotations

from dataclasses import dataclass

_EXCIPIENTS: dict[str, dict[str, float]] = {
    "sucrose": {
        "tg_prime_C": -32.0,
        "tc_C": -28.0,
        "cryoprotection": 0.9,
        "lyoprotection": 0.9,
    },
    "trehalose": {
        "tg_prime_C": -29.0,
        "tc_C": -25.0,
        "cryoprotection": 0.95,
        "lyoprotection": 0.95,
    },
    "mannitol": {
        "tg_prime_C": -25.0,
        "tc_C": -20.0,
        "cryoprotection": 0.4,
        "lyoprotection": 0.7,
    },
    "glycine": {
        "tg_prime_C": -35.0,
        "tc_C": -30.0,
        "cryoprotection": 0.3,
        "lyoprotection": 0.6,
    },
}


@dataclass(frozen=True)
class LyophilizationResult:
    drug_name: str
    excipient_type: str
    protein_concentration_mg_mL: float
    residual_moisture_pct: float
    storage_temp_C: float
    tg_prime_C: float
    tc_C: float
    stability_score: float
    shelf_life_estimate_years: float
    moisture_score: float
    tg_margin_score: float
    lyoprotection_score: float
    protein_conc_score: float
    collapse_risk: bool
    stability_class: str
    notes: str


def _compute_moisture_score(moisture_pct: float) -> float:
    if moisture_pct < 1.0:
        return 30.0
    if moisture_pct < 2.0:
        return 20.0
    if moisture_pct < 3.0:
        return 10.0
    return 0.0


def _compute_tg_margin_score(tg_prime_C: float, storage_temp_C: float) -> float:
    margin = tg_prime_C - storage_temp_C
    if margin > 20.0:
        return 30.0
    if margin > 10.0:
        return 20.0
    if margin > 0.0:
        return 10.0
    return 0.0


def _compute_protein_conc_score(conc: float) -> float:
    if conc < 10.0:
        return 20.0
    if conc < 50.0:
        return 15.0
    if conc < 100.0:
        return 10.0
    return 5.0


def _stability_class(score: float) -> str:
    if score >= 75.0:
        return "excellent"
    if score >= 60.0:
        return "good"
    if score >= 40.0:
        return "moderate"
    return "poor"


def _shelf_life(score: float) -> float:
    if score >= 75.0:
        return 3.0
    if score >= 60.0:
        return 2.0
    if score >= 40.0:
        return 1.0
    return 0.5


def _validate(
    protein_concentration_mg_mL: float,
    residual_moisture_pct: float,
    storage_temp_C: float,
    excipient_type: str,
) -> None:
    if protein_concentration_mg_mL <= 0:
        raise ValueError("protein_concentration_mg_mL must be > 0")
    if not (0.0 <= residual_moisture_pct <= 10.0):
        raise ValueError("residual_moisture_pct must be in [0, 10]")
    if not (-80.0 <= storage_temp_C <= 30.0):
        raise ValueError("storage_temp_C must be in [-80, 30]")
    if excipient_type not in _EXCIPIENTS:
        raise ValueError(
            f"excipient_type must be one of {sorted(_EXCIPIENTS)}; got {excipient_type!r}"
        )


def assess_lyophilization_stability(
    drug_name: str,
    protein_concentration_mg_mL: float,
    excipient_type: str,
    residual_moisture_pct: float,
    storage_temp_C: float,
    reconstitution_time_min: float = 5.0,
) -> LyophilizationResult:
    """Predict lyophilization stability for a protein/peptide drug.

    Parameters
    ----------
    drug_name:
        Drug or molecule name.
    protein_concentration_mg_mL:
        Protein concentration in the formulation (mg/mL).
    excipient_type:
        One of "sucrose", "trehalose", "mannitol", "glycine".
    residual_moisture_pct:
        Residual moisture content after lyophilization (%).
    storage_temp_C:
        Storage temperature (°C).
    reconstitution_time_min:
        Time required to reconstitute the lyophilized cake (min).

    Returns
    -------
    LyophilizationResult
    """
    _validate(protein_concentration_mg_mL, residual_moisture_pct, storage_temp_C, excipient_type)

    exc = _EXCIPIENTS[excipient_type]
    tg_prime = exc["tg_prime_C"]
    tc = exc["tc_C"]
    lyoprotection = exc["lyoprotection"]

    moisture_score = _compute_moisture_score(residual_moisture_pct)
    tg_margin_score = _compute_tg_margin_score(tg_prime, storage_temp_C)
    lyoprotection_score = lyoprotection * 20.0
    protein_conc_score = _compute_protein_conc_score(protein_concentration_mg_mL)

    total_score = moisture_score + tg_margin_score + lyoprotection_score + protein_conc_score
    shelf_life = _shelf_life(total_score)
    stab_class = _stability_class(total_score)
    collapse_risk = storage_temp_C >= tc

    note_parts = []
    if collapse_risk:
        note_parts.append("Collapse risk: storage_temp >= Tc")
    if residual_moisture_pct >= 3.0:
        note_parts.append("High moisture may accelerate degradation")
    if stab_class == "excellent":
        note_parts.append("Excellent stability predicted")
    if reconstitution_time_min > 10.0:
        note_parts.append(f"Slow reconstitution ({reconstitution_time_min:.1f} min)")
    notes = "; ".join(note_parts) if note_parts else "Acceptable lyophilization parameters"

    return LyophilizationResult(
        drug_name=drug_name,
        excipient_type=excipient_type,
        protein_concentration_mg_mL=protein_concentration_mg_mL,
        residual_moisture_pct=residual_moisture_pct,
        storage_temp_C=storage_temp_C,
        tg_prime_C=tg_prime,
        tc_C=tc,
        stability_score=total_score,
        shelf_life_estimate_years=shelf_life,
        moisture_score=moisture_score,
        tg_margin_score=tg_margin_score,
        lyoprotection_score=lyoprotection_score,
        protein_conc_score=protein_conc_score,
        collapse_risk=collapse_risk,
        stability_class=stab_class,
        notes=notes,
    )


def screen_excipients(
    drug_name: str,
    protein_concentration_mg_mL: float,
    residual_moisture_pct: float,
    storage_temp_C: float,
) -> list[LyophilizationResult]:
    """Screen all four excipients and return results sorted by stability_score descending.

    Parameters
    ----------
    drug_name:
        Drug or molecule name.
    protein_concentration_mg_mL:
        Protein concentration in the formulation (mg/mL).
    residual_moisture_pct:
        Residual moisture content after lyophilization (%).
    storage_temp_C:
        Storage temperature (°C).

    Returns
    -------
    list[LyophilizationResult]
        Four results, one per excipient, sorted by stability_score descending.
    """
    results = [
        assess_lyophilization_stability(
            drug_name=drug_name,
            protein_concentration_mg_mL=protein_concentration_mg_mL,
            excipient_type=exc,
            residual_moisture_pct=residual_moisture_pct,
            storage_temp_C=storage_temp_C,
        )
        for exc in _EXCIPIENTS
    ]
    results.sort(key=lambda r: r.stability_score, reverse=True)
    return results
