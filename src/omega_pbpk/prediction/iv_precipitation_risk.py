"""Phase 335 — Drug Precipitation in IV Formulations.

Assesses precipitation risk when an IV drug formulation is diluted
in physiological conditions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IVPrecipitationResult:
    """Result of IV precipitation risk assessment."""

    drug_name: str
    drug_conc_mg_mL: float
    formulation_pH: float
    solubility_physiological_mg_mL: float
    effective_solubility_in_formulation_mg_mL: float
    precipitation_conc_after_dilution_mg_mL: float
    precipitation_risk: bool
    precipitation_risk_score: float
    hemolysis_risk: bool
    cosolvent_contribution_fold: float
    risk_class: str
    recommendations: str
    notes: str


def _solubility_at_ph(
    solubility_physiological: float,
    formulation_ph: float,
    pka: float,
    drug_type: str,
) -> float:
    """Calculate solubility at formulation pH using Henderson-Hasselbalch.

    For acids: S(pH) = S_neutral * (1 + 10^(pH - pKa))
    For bases: S(pH) = S_neutral * (1 + 10^(pKa - pH))
    Neutral: S(pH) = S_neutral

    We normalise back using physiol pH = 7.4 to get intrinsic (neutral) solubility,
    then project to formulation pH.
    """
    if drug_type == "neutral":
        return solubility_physiological

    if drug_type == "acid":
        # Intrinsic solubility from physiological
        s_neutral = solubility_physiological / (1.0 + 10.0 ** (7.4 - pka))
        s_at_formulation = s_neutral * (1.0 + 10.0 ** (formulation_ph - pka))
    else:  # base
        s_neutral = solubility_physiological / (1.0 + 10.0 ** (pka - 7.4))
        s_at_formulation = s_neutral * (1.0 + 10.0 ** (pka - formulation_ph))

    return max(s_at_formulation, 1e-9)


def assess_iv_precipitation_risk(
    drug_name: str,
    drug_conc_mg_mL: float,
    formulation_pH: float,
    solubility_mg_mL: float,
    pka: float,
    drug_type: str,
    cosolvent_pct: float = 0.0,
    osmolality_mOsm_kg: float = 290.0,
) -> IVPrecipitationResult:
    """Assess IV precipitation risk for a drug formulation.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    drug_conc_mg_mL:
        Drug concentration in the formulation (mg/mL).
    formulation_pH:
        pH of the formulation.
    solubility_mg_mL:
        Aqueous solubility at physiological pH 7.4 (mg/mL), no cosolvent.
    pka:
        pKa of the drug.
    drug_type:
        One of "acid", "base", or "neutral".
    cosolvent_pct:
        Percentage of cosolvent (e.g., PEG400) in the formulation.
    osmolality_mOsm_kg:
        Osmolality of the formulation (mOsm/kg). Normal = 290.
    """
    # --- Input validation ---
    if drug_conc_mg_mL <= 0:
        raise ValueError("drug_conc_mg_mL must be > 0")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if cosolvent_pct < 0:
        raise ValueError("cosolvent_pct must be >= 0")
    if drug_type not in ("acid", "base", "neutral"):
        raise ValueError("drug_type must be 'acid', 'base', or 'neutral'")

    # Physiological solubility (at pH 7.4, no cosolvent)
    sol_physiological = solubility_mg_mL

    # Effective solubility in the formulation with cosolvent enhancement
    # log-linear: effective_sol = sol_physiological * 10^(0.01 * cosolvent_pct)
    effective_sol_formulation = sol_physiological * 10.0 ** (0.01 * cosolvent_pct)

    # After 10x dilution in blood, cosolvent is lost → use physiological solubility
    # (blood is at pH 7.4, no cosolvent)
    effective_sol_blood = sol_physiological

    # Precipitation concentration = drug_conc / 10 (10x dilution)
    precipitation_conc = drug_conc_mg_mL / 10.0

    # Precipitation occurs if precipitation_conc > effective_sol_blood
    precipitation_risk = precipitation_conc > effective_sol_blood

    # Risk score: precipitation_conc / effective_sol_blood * 50, capped at 100
    risk_score_raw = (precipitation_conc / effective_sol_blood) * 50.0
    precipitation_risk_score = min(risk_score_raw, 100.0)

    # Hemolysis risk: |osmolality - 290| > 100
    hemolysis_risk = abs(osmolality_mOsm_kg - 290.0) > 100.0

    # Cosolvent contribution fold
    cosolvent_contribution_fold = effective_sol_formulation / sol_physiological

    # Risk class
    if precipitation_risk_score < 25.0:
        risk_class = "low"
    elif precipitation_risk_score <= 75.0:
        risk_class = "moderate"
    else:
        risk_class = "high"

    # Recommendations
    rec_parts: list[str] = []
    if precipitation_risk:
        rec_parts.append("Dilute formulation slowly to minimise precipitation risk.")
        rec_parts.append("Consider reducing drug concentration or increasing solubility.")
    else:
        rec_parts.append("Precipitation risk is acceptable under standard dilution.")

    if hemolysis_risk:
        rec_parts.append("Adjust osmolality toward 290 mOsm/kg to reduce hemolysis risk.")

    if cosolvent_pct > 0:
        rec_parts.append(
            f"Cosolvent ({cosolvent_pct:.1f}%) enhances formulation solubility "
            f"{cosolvent_contribution_fold:.1f}-fold; monitor upon dilution."
        )

    recommendations = " ".join(rec_parts)

    # Notes
    notes_parts = [
        f"Solubility at physiological pH 7.4: {sol_physiological:.3f} mg/mL.",
        f"Effective solubility in formulation (with cosolvent): "
        f"{effective_sol_formulation:.3f} mg/mL.",
        f"After 10x dilution drug conc: {precipitation_conc:.3f} mg/mL.",
        f"Risk score: {precipitation_risk_score:.1f}/100 ({risk_class}).",
    ]
    notes = " ".join(notes_parts)

    return IVPrecipitationResult(
        drug_name=drug_name,
        drug_conc_mg_mL=drug_conc_mg_mL,
        formulation_pH=formulation_pH,
        solubility_physiological_mg_mL=sol_physiological,
        effective_solubility_in_formulation_mg_mL=effective_sol_formulation,
        precipitation_conc_after_dilution_mg_mL=precipitation_conc,
        precipitation_risk=precipitation_risk,
        precipitation_risk_score=precipitation_risk_score,
        hemolysis_risk=hemolysis_risk,
        cosolvent_contribution_fold=cosolvent_contribution_fold,
        risk_class=risk_class,
        recommendations=recommendations,
        notes=notes,
    )
