"""Phase 861 — Drug Precipitation Risk prediction for IV formulations and GI tract."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PrecipitationRiskResult:
    """Result of precipitation risk prediction."""

    drug_name: str
    ionization_type: str
    pka: float
    logp: float
    s0_intrinsic_mg_mL: float
    cs_gastric_mg_mL: float
    cs_intestinal_mg_mL: float
    cs_plasma_mg_mL: float
    iv_precipitation_risk_score: float
    gi_precipitation_risk_score: float
    overall_risk: str
    supersaturation_ratio_gi: float
    recommended_action: str
    notes: str


def _hh_solubility(s0: float, pka: float, ph: float, ionization_type: str) -> float:
    """Compute Henderson-Hasselbalch solubility at a given pH.

    For acids:  Cs(pH) = S0 * (1 + 10^(pH - pKa))
    For bases:  Cs(pH) = S0 * (1 + 10^(pKa - pH))
    For neutrals: Cs(pH) = S0
    """
    if ionization_type == "acid":
        return s0 * (1.0 + 10.0 ** (ph - pka))
    elif ionization_type == "base":
        return s0 * (1.0 + 10.0 ** (pka - ph))
    else:
        return s0


def predict_precipitation_risk(
    drug_name: str,
    pka: float,
    logp: float,
    ionization_type: str,
    iv_conc_mg_mL: float = 1.0,
) -> PrecipitationRiskResult:
    """Predict drug precipitation risk in IV formulations and GI tract.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    pka:
        Acid/base dissociation constant (0-14).
    logp:
        Octanol-water partition coefficient.
    ionization_type:
        "acid", "base", or "neutral".
    iv_conc_mg_mL:
        Concentration in the IV solution (mg/mL). Must be > 0.

    Returns
    -------
    PrecipitationRiskResult
    """
    if not (0.0 <= pka <= 14.0):
        raise ValueError("pka must be in [0, 14]")
    if iv_conc_mg_mL <= 0.0:
        raise ValueError("iv_conc_mg_mL must be > 0")
    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError("ionization_type must be 'acid', 'base', or 'neutral'")

    # Intrinsic solubility S0 = 10^(-logP + 0.5) mg/mL
    s0 = 10.0 ** (-logp + 0.5)

    # Solubility at physiological pH values
    ph_gastric = 1.5
    ph_intestinal = 6.8
    ph_plasma = 7.4

    cs_gastric = _hh_solubility(s0, pka, ph_gastric, ionization_type)
    cs_intestinal = _hh_solubility(s0, pka, ph_intestinal, ionization_type)
    cs_plasma = _hh_solubility(s0, pka, ph_plasma, ionization_type)

    # IV precipitation risk: SI = iv_conc / cs_plasma
    si = iv_conc_mg_mL / cs_plasma
    if si > 1.0:
        iv_risk_score = min(100.0, max(0.0, (si - 1.0) * 50.0))
    else:
        iv_risk_score = 0.0

    # GI precipitation risk
    if ionization_type == "base":
        # Base drugs dissolve well at low pH (gastric), may precipitate at intestinal pH
        ratio = cs_gastric / cs_intestinal if cs_intestinal > 0.0 else 1.0
        gi_risk_score = min(100.0, max(0.0, (ratio - 1.0) * 10.0))
        supersaturation_ratio_gi = ratio
    elif ionization_type == "acid":
        # Acid drugs: cs_intestinal > cs_gastric (higher pH = more ionized = more soluble)
        ratio = cs_intestinal / cs_gastric if cs_gastric > 0.0 else 1.0
        gi_risk_score = min(100.0, max(0.0, (ratio - 1.0) * 5.0))
        # For acids, supersaturation ratio still defined as cs_gastric/cs_intestinal
        supersaturation_ratio_gi = cs_gastric / cs_intestinal if cs_intestinal > 0.0 else 1.0
    else:
        gi_risk_score = 0.0
        supersaturation_ratio_gi = 1.0

    # Overall risk classification
    max_risk = max(iv_risk_score, gi_risk_score)
    if max_risk >= 70.0:
        overall_risk = "high"
    elif max_risk >= 40.0:
        overall_risk = "moderate"
    else:
        overall_risk = "low"

    # Recommended action
    if overall_risk == "high":
        recommended_action = (
            "Consider solubilizing excipients (co-solvents, cyclodextrins, pH adjustment) "
            "or reformulation to reduce precipitation risk."
        )
    elif overall_risk == "moderate":
        recommended_action = (
            "Monitor for precipitation in IV lines; consider pH adjustment or "
            "reduced infusion concentration."
        )
    else:
        recommended_action = "Precipitation risk is acceptable under standard conditions."

    # Notes
    notes_parts = [
        f"S0_intrinsic={s0:.4f} mg/mL",
        f"cs_plasma={cs_plasma:.4f} mg/mL (pH 7.4)",
        f"SI_iv={si:.3f}",
    ]
    if ionization_type == "base":
        notes_parts.append(
            f"GI supersaturation ratio (gastric/intestinal)={supersaturation_ratio_gi:.2f}"
        )
    elif ionization_type == "acid":
        notes_parts.append(f"Acid: intestinal solubility > gastric (ratio={ratio:.2f})")
    notes = "; ".join(notes_parts)

    return PrecipitationRiskResult(
        drug_name=drug_name,
        ionization_type=ionization_type,
        pka=pka,
        logp=logp,
        s0_intrinsic_mg_mL=s0,
        cs_gastric_mg_mL=cs_gastric,
        cs_intestinal_mg_mL=cs_intestinal,
        cs_plasma_mg_mL=cs_plasma,
        iv_precipitation_risk_score=iv_risk_score,
        gi_precipitation_risk_score=gi_risk_score,
        overall_risk=overall_risk,
        supersaturation_ratio_gi=supersaturation_ratio_gi,
        recommended_action=recommended_action,
        notes=notes,
    )


def screen_formulation_concentrations(
    drug_name: str,
    pka: float,
    logp: float,
    ionization_type: str,
    iv_concentrations_mg_mL: list,
) -> list:
    """Screen multiple IV concentrations and return results sorted by iv_precipitation_risk_score ascending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    pka:
        Acid/base dissociation constant (0-14).
    logp:
        Octanol-water partition coefficient.
    ionization_type:
        "acid", "base", or "neutral".
    iv_concentrations_mg_mL:
        List of IV concentrations to evaluate (mg/mL).

    Returns
    -------
    list[PrecipitationRiskResult]
        Results sorted by iv_precipitation_risk_score ascending (safest first).
    """
    results = [
        predict_precipitation_risk(drug_name, pka, logp, ionization_type, conc)
        for conc in iv_concentrations_mg_mL
    ]
    # Sort ascending by iv_precipitation_risk_score (safest first)
    results.sort(key=lambda r: r.iv_precipitation_risk_score)
    return results
