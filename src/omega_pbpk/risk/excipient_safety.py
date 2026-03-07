"""Pharmaceutical excipient safety and acceptability assessment.

Provides rule-based safety ratings for common pharmaceutical excipients
based on GRAS limits, route-specific concerns, and regulatory guidance
(FDA Inactive Ingredient Database, EMA excipient guidelines).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ExcipientSafetyResult",
    "assess_excipient",
    "screen_formulation",
    "suggest_alternatives",
    "EXCIPIENT_DB",
]

# ---------------------------------------------------------------------------
# Excipient database with GRAS limits
# ---------------------------------------------------------------------------

EXCIPIENT_DB: dict[str, dict] = {
    "peg400": {
        "oral_max_pct": 30.0,
        "iv_max_g_per_day": 10.0,
        "category": "polymer",
        "concerns": ["renal_toxicity_iv"],
    },
    "propylene_glycol": {
        "oral_max_pct": 5.0,
        "iv_max_g_per_day": 1.0,
        "category": "solvent",
        "concerns": ["toxicity_iv"],
    },
    "ethanol": {
        "oral_max_pct": 10.0,
        "iv_max_g_per_day": 5.0,
        "category": "solvent",
        "concerns": ["CNS_effects"],
    },
    "polysorbate_80": {
        "oral_max_pct": 1.0,
        "iv_max_g_per_day": 0.1,
        "category": "surfactant",
        "concerns": ["anaphylaxis"],
    },
    "hpbcd": {
        "oral_max_pct": 40.0,
        "iv_max_g_per_day": 4.0,
        "category": "cyclodextrin",
        "concerns": ["renal_toxicity_iv"],
    },
    "cremophor_el": {
        "oral_max_pct": 5.0,
        "iv_max_g_per_day": 0.5,
        "category": "surfactant",
        "concerns": ["hypersensitivity"],
    },
    "benzyl_alcohol": {
        "oral_max_pct": 0.0,
        "iv_max_g_per_day": 0.1,
        "category": "preservative",
        "concerns": ["neurotoxicity_neonates"],
    },
    "sucrose": {
        "oral_max_pct": 99.0,
        "iv_max_g_per_day": 100.0,
        "category": "filler",
        "concerns": [],
    },
    "mannitol": {
        "oral_max_pct": 20.0,
        "iv_max_g_per_day": 200.0,
        "category": "filler",
        "concerns": [],
    },
    "hpmc": {
        "oral_max_pct": 25.0,
        "iv_max_g_per_day": 0.0,
        "category": "polymer",
        "concerns": ["not_for_iv"],
    },
}

_SUPPORTED_ROUTES = {"oral", "iv", "topical", "sc", "im", "inhalation"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExcipientSafetyResult:
    """Safety assessment result for a single excipient."""

    name: str
    route: str
    concentration_pct: float
    within_limits: bool
    concerns: list
    safety_rating: str  # 'acceptable' / 'caution' / 'avoid'
    recommendation: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _assess_within_limits(
    db_entry: dict,
    concentration_pct: float,
    route: str,
    daily_dose_g_per_day: float | None,
) -> bool:
    """Return True if the excipient usage is within known limits."""
    if route == "oral":
        max_pct = db_entry.get("oral_max_pct", 0.0)
        if max_pct == 0.0:
            return False
        return concentration_pct <= max_pct
    elif route == "iv":
        # Check concentration limit
        iv_max_g = db_entry.get("iv_max_g_per_day", 0.0)
        if iv_max_g == 0.0:
            return False
        if daily_dose_g_per_day is not None:
            return daily_dose_g_per_day <= iv_max_g
        # If no daily dose provided, can only check if any IV is allowed
        return iv_max_g > 0.0
    else:
        # For other routes (topical, sc, im, inhalation), use oral_max_pct as guide
        max_pct = db_entry.get("oral_max_pct", 0.0)
        if max_pct == 0.0:
            return False
        return concentration_pct <= max_pct


def _safety_rating_from_concerns(
    within_limits: bool,
    concerns: list[str],
    route: str,
    db_entry: dict,
) -> str:
    """Determine safety rating: 'acceptable', 'caution', or 'avoid'."""
    if not within_limits:
        return "avoid"

    # Certain concerns automatically escalate
    high_concern_keywords = {"neurotoxicity_neonates", "not_for_iv", "toxicity_iv"}
    if any(c in high_concern_keywords for c in concerns):
        if route == "iv":
            return "avoid"

    if concerns:
        return "caution"
    return "acceptable"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_excipient(
    name: str,
    concentration_pct: float,
    route: str,
    daily_dose_g_per_day: float | None = None,
) -> ExcipientSafetyResult:
    """Assess safety and acceptability of a single excipient.

    Parameters
    ----------
    name:
        Excipient name (case-insensitive, spaces/dashes normalised).
    concentration_pct:
        Concentration in the formulation (% w/v or % w/w, must be >= 0).
    route:
        Administration route: 'oral', 'iv', 'topical', 'sc', 'im', 'inhalation'.
    daily_dose_g_per_day:
        Total daily dose of the excipient in g/day (optional, used for IV limit checks).

    Returns
    -------
    ExcipientSafetyResult

    Raises
    ------
    ValueError
        If concentration_pct < 0 or route is not supported.
    """
    if concentration_pct < 0.0:
        raise ValueError(f"concentration_pct must be >= 0, got {concentration_pct}")

    route = route.strip().lower()
    if route not in _SUPPORTED_ROUTES:
        raise ValueError(f"route must be one of {sorted(_SUPPORTED_ROUTES)}, got '{route}'")

    norm_name = _normalise_name(name)

    if norm_name not in EXCIPIENT_DB:
        # Unknown excipient: return conservative 'unknown' rating
        return ExcipientSafetyResult(
            name=name,
            route=route,
            concentration_pct=concentration_pct,
            within_limits=False,
            concerns=["unknown_excipient"],
            safety_rating="caution",
            recommendation=(
                f"Excipient '{name}' not in database. "
                "Consult FDA Inactive Ingredient Database for limits."
            ),
            notes="Not in built-in EXCIPIENT_DB; manual review required.",
        )

    db_entry = EXCIPIENT_DB[norm_name]
    concerns: list[str] = list(db_entry.get("concerns", []))

    within_limits = _assess_within_limits(db_entry, concentration_pct, route, daily_dose_g_per_day)

    # Additional route-specific concern: hpmc is not for IV
    if norm_name == "hpmc" and route == "iv":
        if "not_for_iv" not in concerns:
            concerns.append("not_for_iv")
        within_limits = False

    # benzyl_alcohol oral_max_pct = 0: oral use not acceptable
    if norm_name == "benzyl_alcohol" and route == "oral":
        within_limits = False

    safety_rating = _safety_rating_from_concerns(within_limits, concerns, route, db_entry)

    # Build recommendation
    if safety_rating == "acceptable":
        recommendation = f"{name} at {concentration_pct}% via {route} is within accepted limits."
    elif safety_rating == "caution":
        concern_str = ", ".join(concerns) if concerns else "none"
        recommendation = (
            f"Use {name} with caution via {route}. "
            f"Known concerns: {concern_str}. "
            "Monitor patient and consider alternatives if sensitised."
        )
    else:  # avoid
        if not within_limits:
            recommendation = (
                f"Avoid {name} via {route}: concentration {concentration_pct}% "
                "exceeds accepted limit or route is not supported."
            )
        else:
            recommendation = (
                f"Avoid {name} via {route} due to safety concerns: " + ", ".join(concerns) + "."
            )

    # Notes
    notes_parts: list[str] = [f"category={db_entry['category']}"]
    if route == "oral":
        notes_parts.append(f"oral_max_pct={db_entry.get('oral_max_pct', 'N/A')}")
    elif route == "iv":
        notes_parts.append(f"iv_max_g_per_day={db_entry.get('iv_max_g_per_day', 'N/A')}")
    if daily_dose_g_per_day is not None:
        notes_parts.append(f"daily_dose_g_per_day={daily_dose_g_per_day}")
    notes = "; ".join(notes_parts)

    return ExcipientSafetyResult(
        name=name,
        route=route,
        concentration_pct=concentration_pct,
        within_limits=within_limits,
        concerns=concerns,
        safety_rating=safety_rating,
        recommendation=recommendation,
        notes=notes,
    )


def screen_formulation(
    excipients: list[dict],
    route: str,
    total_dose_g: float | None = None,
) -> list[ExcipientSafetyResult]:
    """Assess safety of all excipients in a formulation.

    Parameters
    ----------
    excipients:
        List of dicts with keys 'name' and 'concentration_pct'
        (optionally 'daily_dose_g_per_day' per excipient).
    route:
        Administration route for all excipients.
    total_dose_g:
        Optional total formulation dose in g (informational only).

    Returns
    -------
    list[ExcipientSafetyResult]
        One result per excipient, in input order.
    """
    results: list[ExcipientSafetyResult] = []

    for exc in excipients:
        exc_name = exc["name"]
        conc_pct = exc["concentration_pct"]
        daily_dose = exc.get("daily_dose_g_per_day", None)

        result = assess_excipient(
            name=exc_name,
            concentration_pct=conc_pct,
            route=route,
            daily_dose_g_per_day=daily_dose,
        )
        results.append(result)

    # Additional check: combined surfactant load for IV
    if route == "iv":
        surfactant_names = {"polysorbate_80", "cremophor_el"}
        total_surfactant_pct = 0.0
        for exc in excipients:
            if _normalise_name(exc["name"]) in surfactant_names:
                total_surfactant_pct += exc["concentration_pct"]

        if total_surfactant_pct > 1.0:
            # Append a combined surfactant warning as a synthetic entry
            warning = ExcipientSafetyResult(
                name="combined_surfactant_load",
                route=route,
                concentration_pct=total_surfactant_pct,
                within_limits=False,
                concerns=["combined_surfactant_exceeds_1pct_iv"],
                safety_rating="avoid",
                recommendation=(
                    f"Combined IV surfactant load ({total_surfactant_pct:.2f}%) "
                    "exceeds 1% threshold. Risk of hypersensitivity and anaphylaxis."
                ),
                notes="Combined polysorbate_80 + cremophor_el > 1% for IV",
            )
            results.append(warning)

    return results


def suggest_alternatives(excipient_name: str, route: str) -> list[str]:
    """Suggest alternative excipients for a given excipient and route.

    Parameters
    ----------
    excipient_name:
        Name of the excipient to replace.
    route:
        Target administration route.

    Returns
    -------
    list[str]
        Names of suggested alternative excipients. Empty list if no
        alternatives are known.
    """
    route = route.strip().lower()
    norm_name = _normalise_name(excipient_name)

    # Rule-based alternatives for IV route
    _IV_ALTERNATIVES: dict[str, list[str]] = {
        "propylene_glycol": ["peg400", "hpbcd"],
        "cremophor_el": ["polysorbate_80", "hpbcd"],
        "benzyl_alcohol": ["phenol"],
        "ethanol": ["hpbcd", "peg400"],
        "hpmc": [],  # not suitable for IV regardless
    }

    # Rule-based alternatives for oral route
    _ORAL_ALTERNATIVES: dict[str, list[str]] = {
        "cremophor_el": ["polysorbate_80"],
        "propylene_glycol": ["peg400"],
        "ethanol": ["peg400"],
    }

    if route == "iv":
        return list(_IV_ALTERNATIVES.get(norm_name, []))
    elif route == "oral":
        return list(_ORAL_ALTERNATIVES.get(norm_name, []))
    else:
        # Generic: combine both if available
        iv_alts = _IV_ALTERNATIVES.get(norm_name, [])
        oral_alts = _ORAL_ALTERNATIVES.get(norm_name, [])
        combined = list(dict.fromkeys(iv_alts + oral_alts))  # deduplicate, preserve order
        return combined
