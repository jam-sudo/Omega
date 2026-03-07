"""Drug Lyophilization Stability Predictor — Phase 845.

Predicts lyophilized (freeze-dried) formulation stability and recommends
optimal lyophilization cycle parameters based on excipient properties,
drug loading, and protein content.

References
----------
- Pikal MJ, J Pharm Sci. 2010 (lyophilization principles)
- Carpenter JF, Pharm Res. 1997 (lyoprotectant mechanisms)
- ICH Q1A(R2): Stability testing of new drug substances and drug products
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "LyophilizationResult",
    "predict_lyophilization_stability",
    "optimize_lyophilization_cycle",
]

# ---------------------------------------------------------------------------
# Excipient database: name → (Tg_prime_celsius, Tg_dry_celsius)
# Tg_prime: glass transition of maximally freeze-concentrated solution
# Tg_dry:   glass transition of the dried amorphous cake
# ---------------------------------------------------------------------------
_EXCIPIENT_DB: dict[str, tuple[float, float]] = {
    "sucrose": (-32.0, 70.0),
    "trehalose": (-28.0, 120.0),
    "mannitol": (-25.0, 13.0),  # crystalline — lower Tg_dry
    "pvp": (-20.0, 175.0),
    "dextran": (-15.0, 200.0),
}

# Reconstitution time (seconds) per excipient
_RECONSTITUTION_TIME_S: dict[str, float] = {
    "sucrose": 30.0,
    "trehalose": 45.0,
    "mannitol": 15.0,
    "pvp": 60.0,
    "dextran": 90.0,
}

# Base shelf-life (months) per excipient class
_BASE_SHELF_LIFE_MONTHS: dict[str, float] = {
    "trehalose": 36.0,
    "pvp": 36.0,
    "dextran": 36.0,
    "sucrose": 24.0,
    "mannitol": 24.0,
}


@dataclass(frozen=True)
class LyophilizationResult:
    """Result of lyophilization stability prediction.

    All temperature values are in degrees Celsius.
    """

    drug_name: str
    excipient_name: str
    drug_loading_pct: float
    tg_prime_celsius: float
    tg_dry_celsius: float
    collapse_temperature_celsius: float
    primary_drying_temp_celsius: float
    secondary_drying_temp_celsius: float
    residual_moisture_target_pct: float
    reconstitution_time_s: float
    cake_appearance: str
    shelf_life_months: float
    ich_stability_compliant: bool
    notes: str


def predict_lyophilization_stability(
    drug_name: str,
    drug_loading_pct: float,
    protein_content_pct: float,
    excipient_name: str,
    drug_mw_kDa: float = 10.0,
) -> LyophilizationResult:
    """Predict lyophilization stability for a drug-excipient combination.

    Parameters
    ----------
    drug_name:
        Name of the drug substance.
    drug_loading_pct:
        Drug loading as weight percent (0–100, exclusive).
    protein_content_pct:
        Protein content of the drug as weight percent (0–100, inclusive).
    excipient_name:
        Lyoprotectant name; one of sucrose, trehalose, mannitol, PVP, dextran.
    drug_mw_kDa:
        Molecular weight of the drug in kDa (informational; affects notes).

    Returns
    -------
    LyophilizationResult

    Raises
    ------
    ValueError
        If drug_loading_pct is not in (0, 100) or protein_content_pct is not in [0, 100].
    """
    if not (0.0 < drug_loading_pct < 100.0):
        raise ValueError(f"drug_loading_pct must be in (0, 100); got {drug_loading_pct}")
    if not (0.0 <= protein_content_pct <= 100.0):
        raise ValueError(f"protein_content_pct must be in [0, 100]; got {protein_content_pct}")

    exc_key = excipient_name.lower() if excipient_name.lower() in _EXCIPIENT_DB else None
    if exc_key is None:
        tg_prime, tg_dry_exc = -30.0, 80.0
    else:
        tg_prime, tg_dry_exc = _EXCIPIENT_DB[exc_key]

    # Drug Tg based on protein content
    if protein_content_pct > 50.0:
        drug_tg = 150.0
    elif protein_content_pct > 20.0:
        drug_tg = 120.0
    else:
        drug_tg = 80.0

    # Gordon-Taylor blend (linear simplified)
    w_drug = drug_loading_pct / 100.0
    w_exc = 1.0 - w_drug
    tg_blend = (w_drug * drug_tg + w_exc * tg_dry_exc) / (w_drug + w_exc)

    # Derived cycle parameters
    collapse_temperature = tg_prime + 2.0
    primary_drying_temp = collapse_temperature - 5.0
    secondary_drying_temp = min(50.0, tg_blend - 10.0)

    # Residual moisture target
    residual_moisture_target_pct = 0.5 if tg_blend > 100.0 else 1.0

    # Reconstitution time
    reconstitution_time_s = _RECONSTITUTION_TIME_S.get(exc_key if exc_key else "", 45.0)

    # Cake appearance
    if tg_blend > 80.0 and primary_drying_temp <= collapse_temperature:
        cake_appearance = "elegant"
    elif tg_blend > 40.0:
        cake_appearance = "acceptable"
    else:
        cake_appearance = "collapsed"

    # Shelf life calculation
    if exc_key is None:
        base_shelf_life = 12.0
    else:
        base_shelf_life = _BASE_SHELF_LIFE_MONTHS.get(exc_key, 24.0)

    shelf_life = base_shelf_life
    if drug_loading_pct > 50.0:
        shelf_life *= 0.7
    if protein_content_pct > 60.0:
        shelf_life *= 0.8
    shelf_life = max(1.0, min(60.0, shelf_life))

    ich_stability_compliant = shelf_life >= 24.0

    # Notes
    note_parts: list[str] = []
    if exc_key is None:
        note_parts.append(f"Unknown excipient '{excipient_name}'; default values used.")
    if cake_appearance == "collapsed":
        note_parts.append("Collapse risk: consider lower drug loading or alternative excipient.")
    if drug_mw_kDa > 100.0:
        note_parts.append("High-MW biologic: steric protection by lyoprotectant is critical.")
    if not ich_stability_compliant:
        note_parts.append("ICH Q1A non-compliant: predicted shelf life < 24 months.")
    if not note_parts:
        note_parts.append("Formulation within acceptable lyophilization parameters.")
    notes = " ".join(note_parts)

    return LyophilizationResult(
        drug_name=drug_name,
        excipient_name=excipient_name,
        drug_loading_pct=drug_loading_pct,
        tg_prime_celsius=tg_prime,
        tg_dry_celsius=tg_blend,
        collapse_temperature_celsius=collapse_temperature,
        primary_drying_temp_celsius=primary_drying_temp,
        secondary_drying_temp_celsius=secondary_drying_temp,
        residual_moisture_target_pct=residual_moisture_target_pct,
        reconstitution_time_s=reconstitution_time_s,
        cake_appearance=cake_appearance,
        shelf_life_months=shelf_life,
        ich_stability_compliant=ich_stability_compliant,
        notes=notes,
    )


def optimize_lyophilization_cycle(
    drug_name: str,
    drug_loading_pct: float,
    protein_content_pct: float,
    drug_mw_kDa: float = 10.0,
) -> list[LyophilizationResult]:
    """Screen all known excipients and return results sorted by shelf life (descending).

    Parameters
    ----------
    drug_name:
        Name of the drug substance.
    drug_loading_pct:
        Drug loading as weight percent (0–100, exclusive).
    protein_content_pct:
        Protein content as weight percent (0–100, inclusive).
    drug_mw_kDa:
        Molecular weight of the drug in kDa.

    Returns
    -------
    list[LyophilizationResult]
        Five results, one per excipient, sorted by shelf_life_months descending.
    """
    excipients = list(_EXCIPIENT_DB.keys())
    results = [
        predict_lyophilization_stability(
            drug_name=drug_name,
            drug_loading_pct=drug_loading_pct,
            protein_content_pct=protein_content_pct,
            excipient_name=exc,
            drug_mw_kDa=drug_mw_kDa,
        )
        for exc in excipients
    ]
    results.sort(key=lambda r: r.shelf_life_months, reverse=True)
    return results
