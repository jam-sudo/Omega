"""
Phase 209 — Biowaiver Eligibility Assessment

Evaluates biowaiver eligibility for solid oral dosage forms per FDA/ICH guidance.

BCS Classification:
    Class I:   High solubility, High permeability
    Class II:  Low solubility,  High permeability
    Class III: High solubility, Low permeability
    Class IV:  Low solubility,  Low permeability

FDA Biowaiver Criteria (2015 guidance + ICH M9):
    BCS I  : highly soluble + highly permeable + rapid dissolution (>85% / 30 min) → ELIGIBLE
    BCS III: highly soluble + rapid dissolution → ELIGIBLE (low permeability risk)
    BCS II/IV: NOT eligible (standard bioequivalence required)
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass (frozen – no list fields)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BiowaiverResult:
    """Biowaiver eligibility result for a single formulation / dose."""

    drug_name: str
    bcs_class: int  # 1, 2, 3, or 4
    dose_mg: float
    dose_number: float  # Do = dose / (solubility * 250 mL)
    solubility_mg_mL: float
    permeability_high: bool
    dissolution_85pct_in_30min: bool
    eligible: bool
    regulatory_path: str  # "no_be_study" | "reduced_be" | "full_be"
    limiting_factor: str  # Human-readable explanation of the bottleneck
    confidence_level: str  # "high" | "medium" | "low"
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VALID_BCS = {1, 2, 3, 4}
_VALID_REG = {"no_be_study", "reduced_be", "full_be"}


def _dose_number(dose_mg: float, solubility_mg_mL: float) -> float:
    """Do = dose_mg / (solubility_mg_mL * 250 mL)."""
    return dose_mg / (solubility_mg_mL * 250.0)


def _validate_inputs(
    bcs_class: int,
    dose_mg: float,
    solubility_mg_mL: float,
) -> None:
    if bcs_class not in _VALID_BCS:
        raise ValueError(f"bcs_class must be 1, 2, 3, or 4; got {bcs_class!r}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0; got {dose_mg}")
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be > 0; got {solubility_mg_mL}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_biowaiver(
    drug_name: str,
    bcs_class: int,
    solubility_mg_mL: float,
    dose_mg: float,
    permeability_high: bool,
    dissolution_85pct_in_30min: bool,
    *,
    notes: str = "",
) -> BiowaiverResult:
    """
    Assess FDA biowaiver eligibility for a solid oral dosage form.

    Parameters
    ----------
    drug_name : str
    bcs_class : int
        BCS class (1–4).
    solubility_mg_mL : float
        Aqueous solubility in mg/mL (worst-case pH 1–7.5).
    dose_mg : float
        Highest dose strength being evaluated.
    permeability_high : bool
        True if drug is highly permeable (human Peff > 2×10⁻⁴ cm/s or
        ≥85% absorbed in mass-balance study).
    dissolution_85pct_in_30min : bool
        True if ≥85% dissolved in 30 min (paddle 75 rpm or basket 100 rpm,
        pH 1.2 / 4.5 / 6.8 media, 900 mL).
    notes : str
        Optional free-text notes (passed through to result).

    Returns
    -------
    BiowaiverResult
    """
    _validate_inputs(bcs_class, dose_mg, solubility_mg_mL)

    do = _dose_number(dose_mg, solubility_mg_mL)
    highly_soluble = do <= 1.0

    # ---- BCS Class I -------------------------------------------------------
    if bcs_class == 1:
        if highly_soluble and permeability_high and dissolution_85pct_in_30min:
            return BiowaiverResult(
                drug_name=drug_name,
                bcs_class=bcs_class,
                dose_mg=dose_mg,
                dose_number=do,
                solubility_mg_mL=solubility_mg_mL,
                permeability_high=permeability_high,
                dissolution_85pct_in_30min=dissolution_85pct_in_30min,
                eligible=True,
                regulatory_path="no_be_study",
                limiting_factor="none",
                confidence_level="high",
                notes=notes
                or "BCS I with rapid dissolution: biowaiver granted per FDA 2015 guidance.",
            )
        # Determine which criterion fails
        if not highly_soluble:
            limiting = f"dose number {do:.2f} > 1.0 (not highly soluble)"
            confidence = "low"
        elif not permeability_high:
            limiting = "permeability not confirmed high"
            confidence = "medium"
        else:
            limiting = "dissolution < 85% in 30 min"
            confidence = "medium"
        return BiowaiverResult(
            drug_name=drug_name,
            bcs_class=bcs_class,
            dose_mg=dose_mg,
            dose_number=do,
            solubility_mg_mL=solubility_mg_mL,
            permeability_high=permeability_high,
            dissolution_85pct_in_30min=dissolution_85pct_in_30min,
            eligible=False,
            regulatory_path="full_be",
            limiting_factor=limiting,
            confidence_level=confidence,
            notes=notes or "BCS I criteria not fully met; full BE study required.",
        )

    # ---- BCS Class III -----------------------------------------------------
    if bcs_class == 3:
        # Highly soluble + rapid dissolution; permeability is the accepted risk
        if highly_soluble and dissolution_85pct_in_30min:
            return BiowaiverResult(
                drug_name=drug_name,
                bcs_class=bcs_class,
                dose_mg=dose_mg,
                dose_number=do,
                solubility_mg_mL=solubility_mg_mL,
                permeability_high=permeability_high,
                dissolution_85pct_in_30min=dissolution_85pct_in_30min,
                eligible=True,
                regulatory_path="no_be_study",
                limiting_factor="none",
                confidence_level="medium",
                notes=(
                    notes
                    or "BCS III: highly soluble + rapid dissolution biowaiver per ICH M9. "
                    "Low permeability risk is accepted; excipient caution applies."
                ),
            )
        if not highly_soluble:
            limiting = f"dose number {do:.2f} > 1.0 (not highly soluble)"
        else:
            limiting = "dissolution < 85% in 30 min"
        return BiowaiverResult(
            drug_name=drug_name,
            bcs_class=bcs_class,
            dose_mg=dose_mg,
            dose_number=do,
            solubility_mg_mL=solubility_mg_mL,
            permeability_high=permeability_high,
            dissolution_85pct_in_30min=dissolution_85pct_in_30min,
            eligible=False,
            regulatory_path="full_be",
            limiting_factor=limiting,
            confidence_level="low",
            notes=notes or "BCS III dissolution/solubility criteria not met.",
        )

    # ---- BCS Class II / IV -------------------------------------------------
    # Neither class is eligible for a waiver; II may use reduced_be in some
    # regions but FDA 2015 guidance does not support it for Class II.
    limiting = (
        "low solubility (BCS II)"
        if bcs_class == 2
        else "low solubility and low permeability (BCS IV)"
    )
    return BiowaiverResult(
        drug_name=drug_name,
        bcs_class=bcs_class,
        dose_mg=dose_mg,
        dose_number=do,
        solubility_mg_mL=solubility_mg_mL,
        permeability_high=permeability_high,
        dissolution_85pct_in_30min=dissolution_85pct_in_30min,
        eligible=False,
        regulatory_path="full_be",
        limiting_factor=limiting,
        confidence_level="high",
        notes=(
            notes
            or f"BCS Class {bcs_class}: biowaiver not supported by FDA 2015 guidance. "
            "Full in vivo bioequivalence study required."
        ),
    )


def screen_formulation_variants(
    drug_name: str,
    bcs_class: int,
    solubility_mg_mL: float,
    dose_options_mg: list[float],
    target_dissolution_pct: float,
    permeability_high: bool = True,
) -> list[BiowaiverResult]:
    """
    Evaluate biowaiver eligibility across multiple dose strengths.

    Parameters
    ----------
    drug_name : str
    bcs_class : int
    solubility_mg_mL : float
    dose_options_mg : list of float
        Candidate dose strengths to screen.
    target_dissolution_pct : float
        Measured or predicted dissolution at 30 min (0–100 %).
    permeability_high : bool
        Assumed permeability category for all variants.

    Returns
    -------
    list of BiowaiverResult, one per dose option (same order as input).
    """
    dissolution_85 = target_dissolution_pct >= 85.0
    results: list[BiowaiverResult] = []
    for dose in dose_options_mg:
        result = assess_biowaiver(
            drug_name=drug_name,
            bcs_class=bcs_class,
            solubility_mg_mL=solubility_mg_mL,
            dose_mg=dose,
            permeability_high=permeability_high,
            dissolution_85pct_in_30min=dissolution_85,
        )
        results.append(result)
    return results
