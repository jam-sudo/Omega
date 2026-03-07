"""Biliary excretion predictor.

Predicts biliary excretion fraction and enterohepatic cycling (EHC) potential
based on physicochemical properties.

References:
    - Watanabe T et al. Drug Metab Pharmacokinet. 2011;26(1):20-35.
    - Vlaming ML et al. Mol Pharmacol. 2009;75(4):986-94.
    - Smith DA et al. Pharmacokinetics and Metabolism in Drug Design. 2012.
"""

from __future__ import annotations

from dataclasses import dataclass

# Human MW threshold for significant biliary excretion
_MW_THRESHOLD_DA: float = 500.0

# Valid ionization types
_VALID_IONIZATION = frozenset({"acid", "base", "neutral", "zwitterion"})


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BiliaryExcretionResult:
    """Result from biliary excretion prediction.

    Attributes:
        drug_name: Compound identifier.
        molecular_weight_da: Molecular weight (Da).
        logp: Octanol-water partition coefficient.
        f_biliary: Predicted fraction of dose excreted via bile (0–1).
        bil_cl_mL_per_min_per_kg: Biliary clearance (mL/min/kg).
        ehc_potential: Enterohepatic cycling potential ("high","moderate","low","none").
        conjugation_type: Predominant conjugation ("glucuronide","sulfate","unchanged","none").
        molecular_weight_threshold: MW above which biliary excretion is significant (Da).
        ionization_favorable: True if ionization type favors biliary excretion.
        predicted_ehc_fraction: Fraction of biliary dose reabsorbed (simplified).
        secondary_peak_likelihood: "likely", "possible", or "unlikely".
        dose_adjustment_hepatic: "required" if f_biliary > 0.4, else "not_required".
        notes: Textual summary.
    """

    drug_name: str
    molecular_weight_da: float
    logp: float
    f_biliary: float
    bil_cl_mL_per_min_per_kg: float
    ehc_potential: str
    conjugation_type: str
    molecular_weight_threshold: float
    ionization_favorable: bool
    predicted_ehc_fraction: float
    secondary_peak_likelihood: str
    dose_adjustment_hepatic: str
    notes: str


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------


def predict_biliary_excretion(
    drug_name: str,
    mw_da: float,
    logp: float,
    pka: float = 7.4,
    ionization_type: str = "neutral",
    fu_plasma: float = 0.5,
    hepatic_cl_mL_per_min_per_kg: float = 10.0,
) -> BiliaryExcretionResult:
    """Predict biliary excretion fraction and EHC potential.

    Args:
        drug_name: Compound identifier.
        mw_da: Molecular weight (Da).
        logp: Octanol-water partition coefficient.
        pka: Acid/base pKa (informational; affects scoring via ionization_type).
        ionization_type: One of "acid", "base", "neutral", "zwitterion".
        fu_plasma: Unbound fraction in plasma (0, 1].
        hepatic_cl_mL_per_min_per_kg: Total hepatic clearance (mL/min/kg, >= 0).

    Returns:
        BiliaryExcretionResult with all derived parameters.

    Raises:
        ValueError: If mw_da <= 0, fu_plasma not in (0, 1], or hepatic_cl < 0.
    """
    if mw_da <= 0:
        raise ValueError(f"mw_da must be > 0, got {mw_da}")
    if fu_plasma <= 0 or fu_plasma > 1.0:
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if hepatic_cl_mL_per_min_per_kg < 0:
        raise ValueError(
            f"hepatic_cl_mL_per_min_per_kg must be >= 0, got {hepatic_cl_mL_per_min_per_kg}"
        )

    # ------------------------------------------------------------------
    # f_biliary scoring
    # ------------------------------------------------------------------
    score = 0.1  # base

    if mw_da > 500:
        score += 0.3
    if mw_da > 750:
        score += 0.2  # additional on top of the >500 bonus

    if 0.0 <= logp <= 4.0:
        score += 0.1

    if ionization_type in ("acid", "zwitterion"):
        score += 0.15
    elif ionization_type == "base":
        score -= 0.05

    if fu_plasma < 0.2:
        score -= 0.1

    # Clamp to [0.01, 0.95]
    f_biliary = max(0.01, min(0.95, score))

    # ------------------------------------------------------------------
    # Biliary clearance (simplified: biliary fraction of hepatic CL)
    # ------------------------------------------------------------------
    bil_cl = hepatic_cl_mL_per_min_per_kg * f_biliary

    # ------------------------------------------------------------------
    # Conjugation type
    # ------------------------------------------------------------------
    if mw_da > 500 and logp < 2:
        conjugation_type = "glucuronide"
    elif mw_da < 400 and ionization_type == "acid":
        conjugation_type = "sulfate"
    elif f_biliary > 0.5:
        conjugation_type = "glucuronide"
    elif f_biliary > 0.2:
        conjugation_type = "unchanged"
    else:
        conjugation_type = "none"

    # ------------------------------------------------------------------
    # EHC potential
    # ------------------------------------------------------------------
    if f_biliary > 0.5 and conjugation_type == "glucuronide":
        ehc_potential = "high"
    elif f_biliary > 0.3:
        ehc_potential = "moderate"
    elif f_biliary > 0.1:
        ehc_potential = "low"
    else:
        ehc_potential = "none"

    # ------------------------------------------------------------------
    # Predicted EHC fraction (fraction of dose reabsorbed via EHC)
    # ------------------------------------------------------------------
    predicted_ehc_fraction = f_biliary * 0.5

    # ------------------------------------------------------------------
    # Secondary peak likelihood
    # ------------------------------------------------------------------
    if ehc_potential == "high":
        secondary_peak_likelihood = "likely"
    elif ehc_potential == "moderate":
        secondary_peak_likelihood = "possible"
    else:
        secondary_peak_likelihood = "unlikely"

    # ------------------------------------------------------------------
    # Dose adjustment for hepatic impairment
    # ------------------------------------------------------------------
    dose_adjustment_hepatic = "required" if f_biliary > 0.4 else "not_required"

    # ------------------------------------------------------------------
    # Ionization favorable for biliary excretion
    # ------------------------------------------------------------------
    ionization_favorable = ionization_type in ("acid", "zwitterion")

    notes = (
        f"MW={mw_da:.1f} Da (threshold={_MW_THRESHOLD_DA:.0f} Da); "
        f"logP={logp:.2f}; ionization={ionization_type}; "
        f"fu_plasma={fu_plasma:.3f}; "
        f"f_biliary={f_biliary:.3f}; "
        f"bil_CL={bil_cl:.2f} mL/min/kg; "
        f"EHC={ehc_potential}; "
        f"conjugation={conjugation_type}"
    )

    return BiliaryExcretionResult(
        drug_name=drug_name,
        molecular_weight_da=mw_da,
        logp=logp,
        f_biliary=f_biliary,
        bil_cl_mL_per_min_per_kg=bil_cl,
        ehc_potential=ehc_potential,
        conjugation_type=conjugation_type,
        molecular_weight_threshold=_MW_THRESHOLD_DA,
        ionization_favorable=ionization_favorable,
        predicted_ehc_fraction=predicted_ehc_fraction,
        secondary_peak_likelihood=secondary_peak_likelihood,
        dose_adjustment_hepatic=dose_adjustment_hepatic,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Batch screening function
# ---------------------------------------------------------------------------


def screen_biliary_risk(
    drug_candidates: list[dict],
) -> list[BiliaryExcretionResult]:
    """Screen a list of drug candidates for biliary excretion risk.

    Args:
        drug_candidates: List of dicts, each containing keyword arguments for
            predict_biliary_excretion (drug_name, mw_da, logp are required;
            other fields are optional and use their defaults).

    Returns:
        List of BiliaryExcretionResult sorted by f_biliary descending.
    """
    results = [predict_biliary_excretion(**candidate) for candidate in drug_candidates]
    results.sort(key=lambda r: r.f_biliary, reverse=True)
    return results
