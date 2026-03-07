"""Phase 669 — Protein Drug Dosing: dose adjustment for altered protein binding states."""

from __future__ import annotations

from dataclasses import dataclass

# Normal albumin reference level (g/dL)
_ALBUMIN_NORMAL = 4.0

# Default albumin levels per clinical scenario
_ALBUMIN_LEVELS: dict[str, float] = {
    "normal": 4.0,
    "hypoalbuminemia": 2.0,
    "cirrhosis": 2.5,
    "pregnancy": 3.0,
    "uremia": 4.0,  # albumin may be normal but AGP elevated
}

_VALID_SCENARIOS = {"normal", "hypoalbuminemia", "uremia", "pregnancy", "cirrhosis"}


def _compute_fup_adjusted(fup_normal: float, clinical_scenario: str, albumin_g_dL: float) -> float:
    """Compute adjusted fraction unbound for a given clinical scenario."""
    if clinical_scenario == "normal":
        return fup_normal
    elif clinical_scenario in ("hypoalbuminemia", "cirrhosis"):
        fup_adj = fup_normal * (_ALBUMIN_NORMAL / albumin_g_dL)
        return min(fup_adj, 1.0)
    elif clinical_scenario == "uremia":
        # Elevated AGP for basic drugs → more bound → less unbound
        fup_adj = fup_normal * 0.7
        return fup_adj
    elif clinical_scenario == "pregnancy":
        fup_adj = fup_normal * (_ALBUMIN_NORMAL / 3.0)
        return min(fup_adj, 1.0)
    else:
        raise ValueError(f"Unknown clinical_scenario: {clinical_scenario!r}")


def _build_recommendation(
    clinical_scenario: str,
    dose_adjustment_factor: float,
    fup_normal: float,
    fup_adjusted: float,
    risk_of_toxicity: bool,
    risk_of_subtherapeutic: bool,
) -> str:
    """Build actionable clinical recommendation string."""
    lines: list[str] = []

    if clinical_scenario == "normal":
        lines.append("No dose adjustment required; protein binding within normal limits.")
    elif clinical_scenario == "hypoalbuminemia":
        lines.append(
            f"Hypoalbuminemia detected. Free drug fraction increased "
            f"({fup_normal:.3f} → {fup_adjusted:.3f})."
        )
        if risk_of_toxicity:
            lines.append(
                f"Reduce dose by factor {dose_adjustment_factor:.2f} to avoid toxicity. "
                "Monitor free drug levels closely."
            )
        else:
            lines.append(
                "Consider total drug level interpretation cautiously; monitor free levels."
            )
    elif clinical_scenario == "cirrhosis":
        lines.append(
            f"Hepatic impairment (cirrhosis) reduces albumin. "
            f"Free fraction increased ({fup_normal:.3f} → {fup_adjusted:.3f})."
        )
        if risk_of_toxicity:
            lines.append(
                f"Dose reduction by factor {dose_adjustment_factor:.2f} recommended. "
                "Also consider reduced hepatic metabolism."
            )
    elif clinical_scenario == "uremia":
        lines.append(
            f"Uremia elevates AGP, increasing protein binding "
            f"({fup_normal:.3f} → {fup_adjusted:.3f})."
        )
        if risk_of_subtherapeutic:
            lines.append(
                f"Dose increase by factor {dose_adjustment_factor:.2f} may be needed. "
                "Monitor therapeutic levels and adjust based on clinical response."
            )
    elif clinical_scenario == "pregnancy":
        lines.append(
            f"Pregnancy-related albumin dilution increases free fraction "
            f"({fup_normal:.3f} → {fup_adjusted:.3f})."
        )
        if risk_of_toxicity:
            lines.append(
                f"Consider dose reduction by factor {dose_adjustment_factor:.2f}. "
                "Frequent monitoring during all trimesters recommended."
            )

    if not lines:
        lines.append(
            f"Dose adjustment factor: {dose_adjustment_factor:.2f}. Clinical judgment required."
        )

    return " ".join(lines)


def _build_notes(
    clinical_scenario: str,
    fup_normal: float,
    fup_adjusted: float,
    albumin_g_dL: float,
) -> str:
    """Build supplementary notes string."""
    return (
        f"Scenario: {clinical_scenario}. "
        f"Albumin: {albumin_g_dL:.1f} g/dL (normal: {_ALBUMIN_NORMAL:.1f} g/dL). "
        f"fup: {fup_normal:.4f} (normal) → {fup_adjusted:.4f} (adjusted). "
        "Dose adjustment preserves free drug CSS. "
        "For drugs with narrow therapeutic windows, free drug monitoring is preferred."
    )


@dataclass(frozen=True)
class ProteinDrugDosingResult:
    """Result of protein-binding-based dose adjustment."""

    drug_name: str
    clinical_scenario: str
    albumin_g_dL: float
    fup_normal: float
    fup_adjusted: float
    dose_normal_mg: float
    dose_adjusted_mg: float
    dose_adjustment_factor: float
    css_free_target_mg_L: float
    css_total_adjusted_mg_L: float
    risk_of_toxicity: bool
    risk_of_subtherapeutic: bool
    clinical_recommendation: str
    notes: str


def adjust_dose_for_protein_binding(
    drug_name: str,
    dose_normal_mg: float,
    fup_normal: float,
    cl_intrinsic_L_per_h: float = 20.0,
    vd_normal_L: float = 50.0,
    clinical_scenario: str = "hypoalbuminemia",
    albumin_g_dL: float = 2.0,
    target_css_free_mg_L: float = 0.5,
) -> ProteinDrugDosingResult:
    """Adjust dose to maintain the same free drug exposure in altered protein binding states.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_normal_mg:
        Standard dose in mg.
    fup_normal:
        Fraction unbound at normal albumin (4.0 g/dL). Must be in (0, 1].
    cl_intrinsic_L_per_h:
        Intrinsic (unbound) clearance in L/h.
    vd_normal_L:
        Volume of distribution at normal protein binding.
    clinical_scenario:
        One of "normal", "hypoalbuminemia", "uremia", "pregnancy", "cirrhosis".
    albumin_g_dL:
        Patient albumin level in g/dL.
    target_css_free_mg_L:
        Target free (unbound) drug concentration at steady state.

    Returns
    -------
    ProteinDrugDosingResult
    """
    # --- Validation ---
    if fup_normal <= 0 or fup_normal > 1:
        raise ValueError(f"fup_normal must be in (0, 1], got {fup_normal}")
    if dose_normal_mg <= 0:
        raise ValueError(f"dose_normal_mg must be > 0, got {dose_normal_mg}")
    if albumin_g_dL <= 0:
        raise ValueError(f"albumin_g_dL must be > 0, got {albumin_g_dL}")
    if clinical_scenario not in _VALID_SCENARIOS:
        raise ValueError(
            f"clinical_scenario must be one of {_VALID_SCENARIOS}, got {clinical_scenario!r}"
        )

    # --- Compute adjusted fup ---
    fup_adjusted = _compute_fup_adjusted(fup_normal, clinical_scenario, albumin_g_dL)

    # --- Dose adjustment to preserve free CSS ---
    # CSS_free proportional to (F * dose * fup) / (CLint * fup) = F * dose / CLint for high-ER
    # For simplicity: dose_adjusted = dose_normal * (fup_normal / fup_adjusted)
    dose_adjustment_factor = fup_normal / fup_adjusted
    dose_adjusted_mg = dose_normal_mg * dose_adjustment_factor

    # --- Total CSS after adjustment ---
    # CSS_total = target_css_free / fup_adjusted
    css_total_adjusted_mg_L = target_css_free_mg_L / fup_adjusted

    # --- Risk flags ---
    # Unadjusted dose with increased fup → higher free drug → toxicity risk
    risk_of_toxicity = fup_adjusted > fup_normal
    # Unadjusted dose with decreased fup → lower free drug → subtherapeutic risk
    risk_of_subtherapeutic = fup_adjusted < fup_normal

    # --- Clinical recommendation ---
    recommendation = _build_recommendation(
        clinical_scenario,
        dose_adjustment_factor,
        fup_normal,
        fup_adjusted,
        risk_of_toxicity,
        risk_of_subtherapeutic,
    )

    notes = _build_notes(clinical_scenario, fup_normal, fup_adjusted, albumin_g_dL)

    return ProteinDrugDosingResult(
        drug_name=drug_name,
        clinical_scenario=clinical_scenario,
        albumin_g_dL=albumin_g_dL,
        fup_normal=fup_normal,
        fup_adjusted=fup_adjusted,
        dose_normal_mg=dose_normal_mg,
        dose_adjusted_mg=dose_adjusted_mg,
        dose_adjustment_factor=dose_adjustment_factor,
        css_free_target_mg_L=target_css_free_mg_L,
        css_total_adjusted_mg_L=css_total_adjusted_mg_L,
        risk_of_toxicity=risk_of_toxicity,
        risk_of_subtherapeutic=risk_of_subtherapeutic,
        clinical_recommendation=recommendation,
        notes=notes,
    )


def screen_protein_binding_scenarios(
    drug_name: str,
    dose_mg: float,
    fup_normal: float,
    scenarios: list[str] | None = None,
) -> list[ProteinDrugDosingResult]:
    """Screen multiple clinical scenarios for protein-binding-driven dose adjustment.

    Returns results sorted by dose_adjustment_factor descending (largest adjustment first).
    """
    if scenarios is None:
        scenarios = ["hypoalbuminemia", "uremia", "pregnancy", "cirrhosis"]

    results: list[ProteinDrugDosingResult] = []
    for scenario in scenarios:
        albumin = _ALBUMIN_LEVELS.get(scenario, _ALBUMIN_NORMAL)
        result = adjust_dose_for_protein_binding(
            drug_name=drug_name,
            dose_normal_mg=dose_mg,
            fup_normal=fup_normal,
            clinical_scenario=scenario,
            albumin_g_dL=albumin,
        )
        results.append(result)

    results.sort(key=lambda r: r.dose_adjustment_factor, reverse=True)
    return results
