"""Organ reserve capacity and safety margin assessment.

Models the remaining functional capacity of organs and how drug exposure affects it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "OrganReserveResult",
    "assess_hepatic_reserve",
    "assess_renal_reserve",
    "screen_organ_safety",
    "estimate_organ_toxicity_dose",
]


@dataclass(frozen=True)
class OrganReserveResult:
    """Result of organ reserve capacity assessment."""

    organ: str
    drug_auc: float
    threshold: float
    exposure_fraction: float
    safety_margin_fold: float
    reserve_remaining: float
    risk_level: str
    notes: str


def _risk_level(reserve_remaining: float) -> str:
    if reserve_remaining > 0.5:
        return "adequate"
    if reserve_remaining >= 0.2:
        return "reduced"
    return "critically_low"


def assess_hepatic_reserve(
    baseline_enzymes_uL: float,
    drug_auc_mg_h_per_L: float,
    hepatotoxicity_threshold_mg_h_per_L: float = 50.0,
    f_hepatic_reserve: float = 0.7,
) -> OrganReserveResult:
    """Assess hepatic reserve capacity given drug AUC.

    Args:
        baseline_enzymes_uL: Baseline liver enzyme level (U/L), used for context.
        drug_auc_mg_h_per_L: Drug AUC (mg·h/L).
        hepatotoxicity_threshold_mg_h_per_L: AUC threshold for hepatotoxicity.
        f_hepatic_reserve: Fractional reserve of liver (default 0.7).

    Returns:
        OrganReserveResult for hepatic organ.
    """
    if drug_auc_mg_h_per_L < 0:
        raise ValueError("drug_auc_mg_h_per_L must be >= 0")
    if hepatotoxicity_threshold_mg_h_per_L <= 0:
        raise ValueError("hepatotoxicity_threshold must be > 0")
    if not (0.0 <= f_hepatic_reserve <= 1.0):
        raise ValueError("f_hepatic_reserve must be in [0, 1]")

    ef = drug_auc_mg_h_per_L / hepatotoxicity_threshold_mg_h_per_L
    ef = min(1.0, ef)

    reserve_remaining = max(0.0, f_hepatic_reserve - ef * f_hepatic_reserve)
    safety_margin_fold = hepatotoxicity_threshold_mg_h_per_L / max(drug_auc_mg_h_per_L, 1e-9)

    risk = _risk_level(reserve_remaining)

    notes = (
        f"Baseline enzymes {baseline_enzymes_uL:.1f} U/L; "
        f"exposure fraction {ef:.3f}; hepatic reserve fraction {f_hepatic_reserve}; "
        f"safety margin {safety_margin_fold:.2f}x"
    )

    return OrganReserveResult(
        organ="hepatic",
        drug_auc=drug_auc_mg_h_per_L,
        threshold=hepatotoxicity_threshold_mg_h_per_L,
        exposure_fraction=ef,
        safety_margin_fold=safety_margin_fold,
        reserve_remaining=reserve_remaining,
        risk_level=risk,
        notes=notes,
    )


def assess_renal_reserve(
    egfr_mL_per_min: float,
    drug_auc_mg_h_per_L: float,
    nephrotoxicity_threshold: float = 30.0,
    baseline_egfr: float = 100.0,
) -> OrganReserveResult:
    """Assess renal reserve capacity given drug AUC.

    Args:
        egfr_mL_per_min: Patient eGFR (mL/min).
        drug_auc_mg_h_per_L: Drug AUC (mg·h/L).
        nephrotoxicity_threshold: AUC threshold for nephrotoxicity.
        baseline_egfr: Normal/baseline eGFR (mL/min).

    Returns:
        OrganReserveResult for renal organ.
    """
    if drug_auc_mg_h_per_L < 0:
        raise ValueError("drug_auc_mg_h_per_L must be >= 0")
    if nephrotoxicity_threshold <= 0:
        raise ValueError("nephrotoxicity_threshold must be > 0")
    if baseline_egfr <= 0:
        raise ValueError("baseline_egfr must be > 0")

    renal_reserve = egfr_mL_per_min / baseline_egfr

    ef = drug_auc_mg_h_per_L / nephrotoxicity_threshold
    ef = min(1.0, ef)

    reserve_remaining = max(0.0, renal_reserve - ef * renal_reserve)
    safety_margin_fold = nephrotoxicity_threshold / max(drug_auc_mg_h_per_L, 1e-9)

    risk = _risk_level(reserve_remaining)

    notes = (
        f"eGFR {egfr_mL_per_min:.1f} mL/min (baseline {baseline_egfr:.1f}); "
        f"renal reserve {renal_reserve:.3f}; exposure fraction {ef:.3f}; "
        f"safety margin {safety_margin_fold:.2f}x"
    )

    return OrganReserveResult(
        organ="renal",
        drug_auc=drug_auc_mg_h_per_L,
        threshold=nephrotoxicity_threshold,
        exposure_fraction=ef,
        safety_margin_fold=safety_margin_fold,
        reserve_remaining=reserve_remaining,
        risk_level=risk,
        notes=notes,
    )


def screen_organ_safety(
    compounds: list[dict],
    organ: str = "hepatic",
) -> list[OrganReserveResult]:
    """Screen multiple compounds for organ safety, sorted by safety margin ascending.

    Each compound dict should have:
        - drug_auc (float): AUC in mg·h/L
        - For hepatic: optionally baseline_enzymes, threshold, f_hepatic_reserve
        - For renal: optionally egfr, threshold, baseline_egfr

    Args:
        compounds: List of compound parameter dicts.
        organ: "hepatic" or "renal".

    Returns:
        List of OrganReserveResult sorted by safety_margin_fold ascending.
    """
    results: list[OrganReserveResult] = []

    for c in compounds:
        if organ == "hepatic":
            result = assess_hepatic_reserve(
                baseline_enzymes_uL=c.get("baseline_enzymes", 40.0),
                drug_auc_mg_h_per_L=c["drug_auc"],
                hepatotoxicity_threshold_mg_h_per_L=c.get("threshold", 50.0),
                f_hepatic_reserve=c.get("f_hepatic_reserve", 0.7),
            )
        elif organ == "renal":
            result = assess_renal_reserve(
                egfr_mL_per_min=c.get("egfr", 100.0),
                drug_auc_mg_h_per_L=c["drug_auc"],
                nephrotoxicity_threshold=c.get("threshold", 30.0),
                baseline_egfr=c.get("baseline_egfr", 100.0),
            )
        else:
            raise ValueError(f"Unknown organ: {organ}. Use 'hepatic' or 'renal'.")
        results.append(result)

    results.sort(key=lambda r: r.safety_margin_fold)
    return results


def estimate_organ_toxicity_dose(
    cl_L_per_h: float,
    threshold_auc: float,
) -> float:
    """Estimate dose that would produce the toxicity threshold AUC.

    Uses the relationship: Dose = AUC × CL (for linear PK).

    Args:
        cl_L_per_h: Clearance (L/h).
        threshold_auc: Toxicity threshold AUC (mg·h/L).

    Returns:
        Estimated toxic dose in mg.
    """
    return threshold_auc * cl_L_per_h
