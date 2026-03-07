"""Dose capping and weight-based dosing for obese patients — Phase 261.

Implements adjusted body weight (ABW), ideal body weight (IBW), and dose-capping
strategies for obese patients. Drugs distribute to lean vs. fat tissue differently.

References
----------
- Janmahasatian S et al., Clin Pharmacokinet. 2005;44(10):1051-65 (lean body weight)
- Pai MP & Paloucek FP, Ann Pharmacother. 2000;34(9):1066-9 (IBW/ABW)
- Green B & Duffull SB, Br J Clin Pharmacol. 2004;58(2):119-33
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BodyWeightResult", "DosingResult", "compute_body_weights", "recommend_dose_obese"]


@dataclass(frozen=True)
class BodyWeightResult:
    """Body weight indices for dosing calculations."""

    total_body_weight_kg: float
    height_cm: float
    sex: str
    bmi: float
    ibw_kg: float  # Ideal body weight (Devine formula)
    abw_kg: float  # Adjusted body weight
    lbw_kg: float  # Lean body weight (Janmahasatian)
    obesity_class: str  # "normal", "overweight", "obese_I", "obese_II", "obese_III"
    notes: str


@dataclass(frozen=True)
class DosingResult:
    """Recommended dose for obese patient."""

    drug_name: str
    dosing_weight_strategy: str  # "ibw", "abw", "lbw", "tbw", "flat"
    dosing_weight_kg: float
    mg_per_kg: float
    recommended_dose_mg: float
    dose_cap_applied: bool
    notes: str


def compute_body_weights(
    total_body_weight_kg: float,
    height_cm: float,
    sex: str,
) -> BodyWeightResult:
    """Compute IBW, ABW, LBW from total body weight and height.

    Parameters
    ----------
    total_body_weight_kg : Total body weight (kg). Must be > 0.
    height_cm : Height (cm). Must be > 0.
    sex : 'male' or 'female'.

    Returns
    -------
    BodyWeightResult
    """
    if total_body_weight_kg <= 0:
        raise ValueError("total_body_weight_kg must be > 0")
    if height_cm <= 0:
        raise ValueError("height_cm must be > 0")
    if sex not in ("male", "female"):
        raise ValueError("sex must be 'male' or 'female'")

    # BMI
    height_m = height_cm / 100.0
    bmi = total_body_weight_kg / height_m**2

    # IBW — Devine formula
    height_in = height_cm / 2.54
    height_above_60in = max(0.0, height_in - 60.0)
    if sex == "male":
        ibw = 50.0 + 2.3 * height_above_60in
    else:
        ibw = 45.5 + 2.3 * height_above_60in
    ibw = max(ibw, 1.0)  # avoid zero for very short individuals

    # ABW — IBW + 0.4 * (TBW - IBW)
    abw = ibw + 0.4 * max(0.0, total_body_weight_kg - ibw)

    # LBW — Janmahasatian 2005
    if sex == "male":
        lbw = 9270 * total_body_weight_kg / (6680 + 216 * bmi)
    else:
        lbw = 9270 * total_body_weight_kg / (8780 + 244 * bmi)

    # Obesity classification
    if bmi < 25:
        obesity_class = "normal"
    elif bmi < 30:
        obesity_class = "overweight"
    elif bmi < 35:
        obesity_class = "obese_I"
    elif bmi < 40:
        obesity_class = "obese_II"
    else:
        obesity_class = "obese_III"

    notes = (
        f"TBW={total_body_weight_kg:.1f}kg, H={height_cm:.0f}cm, BMI={bmi:.1f}. "
        f"IBW={ibw:.1f}kg, ABW={abw:.1f}kg, LBW={lbw:.1f}kg. Class: {obesity_class}."
    )

    return BodyWeightResult(
        total_body_weight_kg=total_body_weight_kg,
        height_cm=height_cm,
        sex=sex,
        bmi=bmi,
        ibw_kg=ibw,
        abw_kg=abw,
        lbw_kg=lbw,
        obesity_class=obesity_class,
        notes=notes,
    )


def recommend_dose_obese(
    drug_name: str,
    total_body_weight_kg: float,
    height_cm: float,
    sex: str,
    mg_per_kg: float,
    dosing_weight_strategy: str = "abw",
    max_dose_mg: float | None = None,
) -> DosingResult:
    """Recommend dose for obese patient using appropriate weight descriptor.

    Parameters
    ----------
    drug_name : Drug name.
    total_body_weight_kg : Total body weight (kg). Must be > 0.
    height_cm : Height (cm). Must be > 0.
    sex : 'male' or 'female'.
    mg_per_kg : Weight-based dose (mg/kg). Must be > 0.
    dosing_weight_strategy : 'ibw', 'abw', 'lbw', or 'tbw'.
    max_dose_mg : Maximum dose cap (mg). If None, no cap.

    Returns
    -------
    DosingResult
    """
    if mg_per_kg <= 0:
        raise ValueError("mg_per_kg must be > 0")
    valid_strategies = ("ibw", "abw", "lbw", "tbw")
    if dosing_weight_strategy not in valid_strategies:
        raise ValueError(f"dosing_weight_strategy must be one of {valid_strategies}")

    bw = compute_body_weights(total_body_weight_kg, height_cm, sex)

    weight_map = {
        "ibw": bw.ibw_kg,
        "abw": bw.abw_kg,
        "lbw": bw.lbw_kg,
        "tbw": total_body_weight_kg,
    }
    dosing_weight = weight_map[dosing_weight_strategy]
    raw_dose = dosing_weight * mg_per_kg

    # Apply cap
    capped = False
    final_dose = raw_dose
    if max_dose_mg is not None and raw_dose > max_dose_mg:
        final_dose = max_dose_mg
        capped = True

    notes = (
        f"{drug_name}: {dosing_weight_strategy.upper()} weight = {dosing_weight:.1f} kg. "
        f"Dose = {dosing_weight:.1f} \u00d7 {mg_per_kg} = {raw_dose:.1f} mg"
        + (f" \u2192 capped at {max_dose_mg:.0f} mg." if capped else ".")
    )

    return DosingResult(
        drug_name=drug_name,
        dosing_weight_strategy=dosing_weight_strategy,
        dosing_weight_kg=dosing_weight,
        mg_per_kg=mg_per_kg,
        recommended_dose_mg=final_dose,
        dose_cap_applied=capped,
        notes=notes,
    )
