"""Placebo-controlled PK/PD model with nocebo and placebo components."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlaceboEffectResult:
    drug_name: str
    drug_effect_pct: float
    placebo_effect_pct: float
    nocebo_effect_pct: float
    total_observed_effect_pct: float
    placebo_responder_rate: float
    nnt: float
    net_benefit_pct: float
    therapeutic_significance: str
    notes: str


def calculate_placebo_adjusted_effect(
    drug_name: str,
    drug_conc_mg_L: float,
    emax_pct: float = 80.0,
    ec50_mg_L: float = 1.0,
    hill_coef: float = 1.0,
    placebo_effect_pct: float = 20.0,
    nocebo_effect_pct: float = 5.0,
    disease_severity_pct: float = 60.0,
    therapeutic_threshold_pct: float = 30.0,
) -> PlaceboEffectResult:
    """Calculate placebo-adjusted drug effect using sigmoid Emax model.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    drug_conc_mg_L:
        Drug plasma concentration in mg/L.
    emax_pct:
        Maximum pharmacological effect (%).
    ec50_mg_L:
        Concentration producing 50% of Emax (mg/L).
    hill_coef:
        Hill coefficient for sigmoidicity.
    placebo_effect_pct:
        Placebo response (%).
    nocebo_effect_pct:
        Negative placebo / nocebo effect (%).
    disease_severity_pct:
        Baseline disease severity (%).
    therapeutic_threshold_pct:
        Minimum clinically important difference (MCID) threshold.

    Returns
    -------
    PlaceboEffectResult
    """
    if drug_conc_mg_L < 0:
        raise ValueError("drug_conc_mg_L must be >= 0")
    if emax_pct <= 0:
        raise ValueError("emax_pct must be > 0")
    if ec50_mg_L <= 0:
        raise ValueError("ec50_mg_L must be > 0")
    if hill_coef <= 0:
        raise ValueError("hill_coef must be > 0")

    # Sigmoid Emax model
    if drug_conc_mg_L == 0.0:
        drug_effect = 0.0
    else:
        c_n = drug_conc_mg_L**hill_coef
        ec50_n = ec50_mg_L**hill_coef
        drug_effect = emax_pct * c_n / (ec50_n + c_n)

    total_observed = min(100.0, drug_effect + placebo_effect_pct - nocebo_effect_pct)
    total_observed = max(0.0, total_observed)

    net_benefit = drug_effect - placebo_effect_pct

    if net_benefit > 0:
        nnt = 100.0 / net_benefit
    else:
        nnt = float("inf")

    placebo_responder_rate = placebo_effect_pct / 100.0

    if net_benefit >= therapeutic_threshold_pct:
        therapeutic_significance = "clinically_meaningful"
    elif net_benefit > 0:
        therapeutic_significance = "marginal"
    else:
        therapeutic_significance = "not_significant"

    notes = (
        f"Drug effect: {drug_effect:.1f}%, "
        f"Net benefit over placebo: {net_benefit:.1f}%, "
        f"Disease severity: {disease_severity_pct:.1f}%"
    )

    return PlaceboEffectResult(
        drug_name=drug_name,
        drug_effect_pct=drug_effect,
        placebo_effect_pct=placebo_effect_pct,
        nocebo_effect_pct=nocebo_effect_pct,
        total_observed_effect_pct=total_observed,
        placebo_responder_rate=placebo_responder_rate,
        nnt=nnt,
        net_benefit_pct=net_benefit,
        therapeutic_significance=therapeutic_significance,
        notes=notes,
    )
