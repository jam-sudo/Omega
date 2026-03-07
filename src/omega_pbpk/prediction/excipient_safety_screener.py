"""Phase 1144: Excipient safety screener for formulation development.

Screens excipients against regulatory status (GRAS, ECHA), acceptable daily
intake (ADI), and route suitability.
"""

from dataclasses import dataclass

_EXCIPIENTS: dict = {
    "microcrystalline_cellulose": {
        "adi_mg_per_kg": None,
        "gras": True,
        "echa": False,
        "route": ["oral"],
        "function": "diluent",
    },
    "lactose": {
        "adi_mg_per_kg": None,
        "gras": True,
        "echa": False,
        "route": ["oral"],
        "function": "diluent",
    },
    "pvp": {
        "adi_mg_per_kg": 50.0,
        "gras": True,
        "echa": False,
        "route": ["oral", "iv", "topical"],
        "function": "binder",
    },
    "peg400": {
        "adi_mg_per_kg": 10.0,
        "gras": True,
        "echa": False,
        "route": ["oral", "iv", "topical"],
        "function": "cosolvent",
    },
    "polysorbate80": {
        "adi_mg_per_kg": 25.0,
        "gras": True,
        "echa": False,
        "route": ["oral", "iv"],
        "function": "surfactant",
    },
    "benzalkonium_cl": {
        "adi_mg_per_kg": None,
        "gras": False,
        "echa": True,
        "route": ["topical", "nasal"],
        "function": "preservative",
    },
    "propylene_glycol": {
        "adi_mg_per_kg": 25.0,
        "gras": True,
        "echa": False,
        "route": ["oral", "iv", "topical"],
        "function": "cosolvent",
    },
    "hydroxypropyl_methylcellulose": {
        "adi_mg_per_kg": None,
        "gras": True,
        "echa": False,
        "route": ["oral"],
        "function": "matrix_former",
    },
    "ethanol": {
        "adi_mg_per_kg": None,
        "gras": True,
        "echa": False,
        "route": ["oral", "topical"],
        "function": "cosolvent",
    },
    "dmso": {
        "adi_mg_per_kg": None,
        "gras": False,
        "echa": True,
        "route": ["topical"],
        "function": "penetration_enhancer",
    },
    "sodium_lauryl_sulfate": {
        "adi_mg_per_kg": None,
        "gras": False,
        "echa": True,
        "route": ["topical", "oral"],
        "function": "surfactant",
    },
    "magnesium_stearate": {
        "adi_mg_per_kg": None,
        "gras": True,
        "echa": False,
        "route": ["oral"],
        "function": "lubricant",
    },
    "titanium_dioxide": {
        "adi_mg_per_kg": None,
        "gras": False,
        "echa": True,
        "route": ["oral", "topical"],
        "function": "colorant",
    },
    "citric_acid": {
        "adi_mg_per_kg": None,
        "gras": True,
        "echa": False,
        "route": ["oral", "iv", "topical"],
        "function": "ph_modifier",
    },
    "mannitol": {
        "adi_mg_per_kg": None,
        "gras": True,
        "echa": False,
        "route": ["oral", "iv"],
        "function": "diluent",
    },
    "sucrose": {
        "adi_mg_per_kg": None,
        "gras": True,
        "echa": False,
        "route": ["oral", "iv"],
        "function": "cryoprotectant",
    },
    "trehalose": {
        "adi_mg_per_kg": None,
        "gras": True,
        "echa": False,
        "route": ["oral", "iv"],
        "function": "cryoprotectant",
    },
    "cyclodextrin": {
        "adi_mg_per_kg": None,
        "gras": True,
        "echa": False,
        "route": ["oral", "iv"],
        "function": "solubilizer",
    },
    "parabens": {
        "adi_mg_per_kg": 10.0,
        "gras": False,
        "echa": True,
        "route": ["topical", "oral"],
        "function": "preservative",
    },
    "benzyl_alcohol": {
        "adi_mg_per_kg": 5.0,
        "gras": False,
        "echa": True,
        "route": ["iv", "topical"],
        "function": "preservative",
    },
    "cremophor_el": {
        "adi_mg_per_kg": None,
        "gras": False,
        "echa": False,
        "route": ["iv"],
        "function": "solubilizer",
    },
    "sorbitol": {
        "adi_mg_per_kg": None,
        "gras": True,
        "echa": False,
        "route": ["oral"],
        "function": "sweetener",
    },
    "sodium_chloride": {
        "adi_mg_per_kg": None,
        "gras": True,
        "echa": False,
        "route": ["oral", "iv", "topical"],
        "function": "tonicity_agent",
    },
    "carbomer": {
        "adi_mg_per_kg": None,
        "gras": False,
        "echa": False,
        "route": ["topical"],
        "function": "gelling_agent",
    },
    "talc": {
        "adi_mg_per_kg": None,
        "gras": False,
        "echa": True,
        "route": ["oral"],
        "function": "glidant",
    },
}

# Alias map for user-facing names → internal keys
_ALIAS: dict = {
    "pvp": "pvp",
    "peg400": "peg400",
    "benzalkonium_cl": "benzalkonium_cl",
    "dmso": "dmso",
    "cremophor_el": "cremophor_el",
}


def _lookup_key(excipient_name: str) -> str | None:
    """Return normalised database key or None if not found."""
    key = excipient_name.strip().lower().replace(" ", "_")
    if key in _EXCIPIENTS:
        return key
    return None


@dataclass(frozen=True)
class ExcipientSafetyResult:
    excipient_name: str
    route_of_administration: str
    level_mg_per_day: float
    body_weight_kg: float
    gras_status: bool
    echa_concern: bool
    adi_mg_per_kg: float | None
    daily_intake_mg_per_kg: float
    adi_fraction: float | None
    route_approved: bool
    safety_concern: bool
    safety_flag: str
    regulatory_note: str
    notes: str


def screen_excipient(
    excipient_name: str,
    route_of_administration: str,
    level_mg_per_day: float,
    body_weight_kg: float = 60.0,
) -> ExcipientSafetyResult:
    """Screen a single excipient for safety and regulatory compliance.

    Parameters
    ----------
    excipient_name : str
        Name of the excipient (case-insensitive, underscores or spaces).
    route_of_administration : str
        Intended administration route (e.g., "oral", "iv", "topical").
    level_mg_per_day : float
        Proposed daily excipient level (mg/day).
    body_weight_kg : float
        Patient body weight in kg (default 60.0).

    Returns
    -------
    ExcipientSafetyResult
    """
    # --- Validation ---
    if not excipient_name or not excipient_name.strip():
        raise ValueError("excipient_name must not be empty")
    if not route_of_administration or not route_of_administration.strip():
        raise ValueError("route_of_administration must be a non-empty string")
    if level_mg_per_day <= 0:
        raise ValueError("level_mg_per_day must be > 0")
    if body_weight_kg <= 0:
        raise ValueError("body_weight_kg must be > 0")

    key = _lookup_key(excipient_name)
    if key is None:
        raise ValueError(
            f"Unknown excipient {excipient_name!r}. Known excipients: {sorted(_EXCIPIENTS.keys())}"
        )

    entry = _EXCIPIENTS[key]
    gras = entry["gras"]
    echa = entry["echa"]
    adi = entry["adi_mg_per_kg"]
    approved_routes: list = entry["route"]
    function = entry["function"]

    route_norm = route_of_administration.strip().lower()
    route_approved = route_norm in approved_routes

    daily_intake_per_kg = level_mg_per_day / body_weight_kg
    adi_fraction: float | None = None
    if adi is not None:
        adi_fraction = daily_intake_per_kg / adi

    # safety_concern: ECHA flag OR ADI exceeded
    adi_exceeded = adi_fraction is not None and adi_fraction > 1.0
    safety_concern = echa or adi_exceeded

    # safety_flag
    if safety_concern:
        safety_flag = "red"
    elif adi_fraction is not None and adi_fraction > 0.5:
        safety_flag = "yellow"
    else:
        safety_flag = "green"

    # Regulatory note
    reg_parts = []
    if gras:
        reg_parts.append("GRAS")
    else:
        reg_parts.append("Not GRAS")
    if echa:
        reg_parts.append("ECHA concern")
    if not route_approved:
        reg_parts.append(f"route '{route_norm}' not in approved routes {approved_routes}")
    if adi_fraction is not None:
        reg_parts.append(f"ADI fraction={adi_fraction:.3f}")
    regulatory_note = "; ".join(reg_parts)

    # Notes
    note_parts = [f"function={function}"]
    if adi is not None:
        note_parts.append(f"ADI={adi} mg/kg/day")
    note_parts.append(f"daily intake={daily_intake_per_kg:.4f} mg/kg/day")
    notes = "; ".join(note_parts)

    return ExcipientSafetyResult(
        excipient_name=excipient_name,
        route_of_administration=route_of_administration,
        level_mg_per_day=level_mg_per_day,
        body_weight_kg=body_weight_kg,
        gras_status=gras,
        echa_concern=echa,
        adi_mg_per_kg=adi,
        daily_intake_mg_per_kg=daily_intake_per_kg,
        adi_fraction=adi_fraction,
        route_approved=route_approved,
        safety_concern=safety_concern,
        safety_flag=safety_flag,
        regulatory_note=regulatory_note,
        notes=notes,
    )


def screen_formulation(
    excipients: list,
    route: str,
    body_weight_kg: float = 60.0,
) -> list:
    """Screen all excipients in a formulation.

    Parameters
    ----------
    excipients : list[dict]
        List of dicts with keys "name" and "level_mg_per_day".
    route : str
        Intended route of administration for all excipients.
    body_weight_kg : float
        Patient body weight in kg.

    Returns
    -------
    list[ExcipientSafetyResult]
        Results sorted by safety_concern descending (concerns first).
    """
    results = []
    for exc in excipients:
        result = screen_excipient(
            excipient_name=exc["name"],
            route_of_administration=route,
            level_mg_per_day=exc["level_mg_per_day"],
            body_weight_kg=body_weight_kg,
        )
        results.append(result)

    # Sort: safety_concern=True first, then by safety_flag (red > yellow > green)
    _flag_order = {"red": 2, "yellow": 1, "green": 0}
    results.sort(key=lambda r: _flag_order[r.safety_flag], reverse=True)
    return results
