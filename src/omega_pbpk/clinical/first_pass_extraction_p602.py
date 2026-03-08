"""First-Pass Hepatic Extraction — Phase 602.

Models first-pass hepatic extraction for oral drugs using the well-stirred
(venous equilibrium) liver model. Predicts oral bioavailability, extraction
ratio, and the impact of enzyme induction, protein binding, and hepatic
impairment on drug exposure.

Well-Stirred (Venous Equilibrium) Model
----------------------------------------
  CLh = Qh * fu * CLint / (Qh + fu * CLint)
  E   = CLh / Qh
  Fh  = 1 - E
  F   = Fg * Fh

References
----------
- Rowland M et al., J Pharmacokinet Biopharm. 1973;1(2):123-36
- Wilkinson GR & Shand DG, Clin Pharmacol Ther. 1975;18(4):377-90
- Houston JB, Biochem Pharmacol. 1994;47(9):1469-79
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "FirstPassExtractionResult",
    "predict_first_pass_extraction",
    "extraction_sensitivity",
]


@dataclass(frozen=True)
class FirstPassExtractionResult:
    """Result from first-pass hepatic extraction prediction.

    Attributes
    ----------
    drug_name : Name of the drug.
    fu_plasma : Unbound fraction in plasma (0–1].
    cl_intrinsic_L_per_h : Intrinsic hepatic clearance (L/h).
    cl_hepatic_L_per_h : Predicted hepatic clearance (L/h).
    extraction_ratio : Hepatic extraction ratio E = CLh/Qh (0–1).
    bioavailability_oral : Oral bioavailability fraction Fg * Fh (0–1).
    f_gut : Gut wall availability (fraction not extracted in gut wall).
    bioavailability_total_pct : Oral bioavailability as percentage (%).
    effect_of_enzyme_induction : Impact classification ('high impact'/'low impact').
    effect_of_protein_binding : PPB classification ('restrictive'/'non-restrictive'/'intermediate').
    dose_adjustment_for_hepatic_imp : Dose factor for Child-Pugh B impairment (≤1 = reduce dose).
    notes : Summary text.
    """

    drug_name: str
    fu_plasma: float
    cl_intrinsic_L_per_h: float
    cl_hepatic_L_per_h: float
    extraction_ratio: float
    bioavailability_oral: float
    f_gut: float
    bioavailability_total_pct: float
    effect_of_enzyme_induction: str
    effect_of_protein_binding: str
    dose_adjustment_for_hepatic_imp: float
    notes: str


def predict_first_pass_extraction(
    drug_name: str,
    cl_int_L_per_h: float,
    fu_plasma: float,
    qh_L_per_h: float = 90.0,
    f_gut: float = 0.85,
) -> FirstPassExtractionResult:
    """Predict hepatic first-pass extraction using the well-stirred model.

    Parameters
    ----------
    drug_name : Name of the drug.
    cl_int_L_per_h : In vitro–scaled intrinsic hepatic clearance (L/h).
    fu_plasma : Unbound fraction in plasma (0–1].
    qh_L_per_h : Hepatic blood flow (L/h). Default 90 L/h (typical human).
    f_gut : Gut wall availability, i.e., fraction escaping gut wall extraction (0–1].
              f_gut = 1 − Fg_extraction. Default 0.85 (15% gut wall extraction).

    Returns
    -------
    FirstPassExtractionResult
    """
    # --- Validation ---
    if cl_int_L_per_h <= 0:
        raise ValueError(f"cl_int_L_per_h must be positive, got {cl_int_L_per_h}")
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if qh_L_per_h <= 0:
        raise ValueError(f"qh_L_per_h must be positive, got {qh_L_per_h}")
    if not (0 < f_gut <= 1):
        raise ValueError(f"f_gut must be in (0, 1], got {f_gut}")

    # --- Well-stirred model ---
    # CLh = Qh * fu * CLint / (Qh + fu * CLint)
    fu_cl_int = fu_plasma * cl_int_L_per_h
    cl_hepatic = qh_L_per_h * fu_cl_int / (qh_L_per_h + fu_cl_int)

    extraction_ratio = cl_hepatic / qh_L_per_h
    # Clamp to valid range [0, 1]
    extraction_ratio = max(0.0, min(1.0, extraction_ratio))

    fh = 1.0 - extraction_ratio
    bioavailability_oral = f_gut * fh
    bioavailability_total_pct = bioavailability_oral * 100.0

    # --- Enzyme induction impact ---
    # High-extraction drugs: induction has large effect (E > 0.7)
    if extraction_ratio > 0.7:
        effect_of_enzyme_induction = "high impact"
    else:
        effect_of_enzyme_induction = "low impact"

    # --- Protein binding sensitivity ---
    # Restrictive elimination: only free drug cleared (low E, fu-sensitive)
    if extraction_ratio < 0.3:
        effect_of_protein_binding = "restrictive"
    elif extraction_ratio > 0.7:
        effect_of_protein_binding = "non-restrictive"
    else:
        effect_of_protein_binding = "intermediate"

    # --- Hepatic impairment adjustment (Child-Pugh B: ~50% CLint reduction) ---
    cl_int_reduced = cl_int_L_per_h * 0.5
    fu_cl_int_red = fu_plasma * cl_int_reduced
    cl_hepatic_reduced = qh_L_per_h * fu_cl_int_red / (qh_L_per_h + fu_cl_int_red)
    e_reduced = max(0.0, min(1.0, cl_hepatic_reduced / qh_L_per_h))
    fh_reduced = 1.0 - e_reduced
    bioav_reduced = f_gut * fh_reduced

    # dose_adjustment: ratio of normal to impaired bioavailability
    # < 1 means reduce dose (drug exposure increases in impairment)
    if bioav_reduced > 0.0:
        dose_adjustment = bioavailability_oral / bioav_reduced
    else:
        dose_adjustment = 1.0
    # Clamp to [0.1, 1.0]
    dose_adjustment = max(0.1, min(1.0, dose_adjustment))

    notes = (
        f"Drug '{drug_name}': CLint={cl_int_L_per_h:.1f} L/h, "
        f"fu={fu_plasma:.3f}, Qh={qh_L_per_h:.1f} L/h. "
        f"CLh={cl_hepatic:.2f} L/h, E={extraction_ratio:.3f}, "
        f"Fh={fh:.3f}, Fg={f_gut:.2f}, F={bioavailability_oral:.3f} "
        f"({bioavailability_total_pct:.1f}%). "
        f"Child-Pugh B dose factor: {dose_adjustment:.2f}."
    )

    return FirstPassExtractionResult(
        drug_name=drug_name,
        fu_plasma=fu_plasma,
        cl_intrinsic_L_per_h=cl_int_L_per_h,
        cl_hepatic_L_per_h=cl_hepatic,
        extraction_ratio=extraction_ratio,
        bioavailability_oral=bioavailability_oral,
        f_gut=f_gut,
        bioavailability_total_pct=bioavailability_total_pct,
        effect_of_enzyme_induction=effect_of_enzyme_induction,
        effect_of_protein_binding=effect_of_protein_binding,
        dose_adjustment_for_hepatic_imp=dose_adjustment,
        notes=notes,
    )


def extraction_sensitivity(
    drug_name: str,
    cl_int_L_per_h: float,
    fu_values: list,
    qh_L_per_h: float = 90.0,
    f_gut: float = 0.85,
) -> list:
    """Assess sensitivity of extraction to plasma protein binding.

    Parameters
    ----------
    drug_name : Name of the drug.
    cl_int_L_per_h : Intrinsic hepatic clearance (L/h).
    fu_values : List of unbound fraction values (0–1] to evaluate.
    qh_L_per_h : Hepatic blood flow (L/h).
    f_gut : Gut wall availability fraction (0–1].

    Returns
    -------
    list of FirstPassExtractionResult sorted by bioavailability_oral descending.
    """
    results = []
    for fu in fu_values:
        res = predict_first_pass_extraction(
            drug_name=drug_name,
            cl_int_L_per_h=cl_int_L_per_h,
            fu_plasma=fu,
            qh_L_per_h=qh_L_per_h,
            f_gut=f_gut,
        )
        results.append(res)

    results.sort(key=lambda r: r.bioavailability_oral, reverse=True)
    return results
