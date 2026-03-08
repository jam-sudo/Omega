"""GI Tract Drug Precipitation Predictor (Phase 612).

Predicts drug precipitation risk from a supersaturated state in the GI tract.
Accounts for dose number, ionisation state, amorphous form solubility advantage,
and polymer inhibitor effects.

References
----------
- Brouwers J et al., Pharm Res 2009;26:2240-53
- Bevernage J et al., Mol Pharm 2010;7:1734-44
- Patel S et al., Mol Pharm 2011;8:2119-29
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass (frozen — no list/dict fields)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GIPrecipitationResult:
    """Predicted GI precipitation risk for a drug candidate.

    Attributes
    ----------
    drug_name : Drug identifier.
    dose_mg : Administered dose (mg).
    dose_number : Do = dose / (Cs_intestine * 250 mL).
    cs_ph_intestine_mg_mL : Solubility at intestinal pH 6.8 (mg/mL).
    initial_supersaturation : C_apparent / Cs at intestinal pH.
    precipitation_risk_score : Composite risk score 0-100.
    precipitation_risk_class : "negligible", "low", "moderate", or "high".
    induction_time_h : Estimated onset of precipitation (h).
    solubility_advantage : S_amorphous / S_crystalline ratio.
    polymer_required : True if high risk and polymer inhibitor recommended.
    recommended_polymer : "HPMC-AS", "PVP-VA", "HPMC", or "none".
    notes : Human-readable summary.
    """

    drug_name: str
    dose_mg: float
    dose_number: float
    cs_ph_intestine_mg_mL: float
    initial_supersaturation: float
    precipitation_risk_score: float
    precipitation_risk_class: str
    induction_time_h: float
    solubility_advantage: float
    polymer_required: bool
    recommended_polymer: str
    notes: str


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_VALID_DRUG_TYPES = ("neutral", "acid", "base")


def _validate_inputs(
    dose_mg: float,
    cs_fasted_mg_mL: float,
    mw_Da: float,
    polymer_conc_mg_mL: float,
    drug_type: str,
) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cs_fasted_mg_mL <= 0:
        raise ValueError(f"cs_fasted_mg_mL must be > 0, got {cs_fasted_mg_mL}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if polymer_conc_mg_mL < 0:
        raise ValueError(f"polymer_conc_mg_mL must be >= 0, got {polymer_conc_mg_mL}")
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(f"drug_type must be one of {_VALID_DRUG_TYPES}, got '{drug_type}'")


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------


def predict_gi_precipitation(
    drug_name: str,
    dose_mg: float,
    cs_fasted_mg_mL: float,
    logP: float,
    mw_Da: float,
    pka: float | None = None,
    drug_type: str = "neutral",
    amorphous: bool = False,
    polymer_conc_mg_mL: float = 0.0,
) -> GIPrecipitationResult:
    """Predict GI precipitation risk for a drug.

    Parameters
    ----------
    drug_name : Drug identifier.
    dose_mg : Oral dose (mg).
    cs_fasted_mg_mL : Crystalline solubility in fasted-state simulated intestinal
        fluid at gastric pH (mg/mL).
    logP : Octanol-water partition coefficient.
    mw_Da : Molecular weight (Da).
    pka : Ionisation pKa (required for acid/base; ignored for neutral).
    drug_type : "neutral", "acid", or "base".
    amorphous : If True, apply amorphous solubility advantage.
    polymer_conc_mg_mL : Concentration of polymeric precipitation inhibitor
        (mg/mL). 0 means no polymer present.

    Returns
    -------
    GIPrecipitationResult
    """
    _validate_inputs(dose_mg, cs_fasted_mg_mL, mw_Da, polymer_conc_mg_mL, drug_type)

    # ------------------------------------------------------------------
    # 1. Intestinal solubility at pH 6.8 (Henderson-Hasselbalch)
    # ------------------------------------------------------------------
    if drug_type == "neutral" or pka is None:
        cs_ph_intestine = cs_fasted_mg_mL
    elif drug_type == "acid":
        cs_ph_intestine = cs_fasted_mg_mL * (1.0 + 10.0 ** (6.8 - pka))
    else:  # base
        cs_ph_intestine = cs_fasted_mg_mL * (1.0 + 10.0 ** (pka - 6.8))

    # Cap at 100× fasted solubility
    cs_ph_intestine = min(cs_ph_intestine, 100.0 * cs_fasted_mg_mL)

    # ------------------------------------------------------------------
    # 2. Dose number
    # ------------------------------------------------------------------
    dose_number = dose_mg / (cs_ph_intestine * 250.0)

    # ------------------------------------------------------------------
    # 3. Initial supersaturation
    # ------------------------------------------------------------------
    # Amorphous drugs have higher apparent solubility
    S_amorphous_boost = math.exp(0.05 * max(0.0, logP * 10.0))

    if amorphous:
        apparent_conc = cs_fasted_mg_mL * min(10.0, S_amorphous_boost)
    else:
        apparent_conc = cs_ph_intestine

    supersaturation_from_form = apparent_conc / cs_ph_intestine if cs_ph_intestine > 0 else 1.0
    supersaturation_from_dose = dose_number  # high Do → supersaturation on dilution

    initial_supersaturation = max(1.0, supersaturation_from_form, supersaturation_from_dose)

    # ------------------------------------------------------------------
    # 4. Risk score
    # ------------------------------------------------------------------
    # Component A: dose number risk
    if dose_number < 1.0:
        risk_do = 0.0
    elif dose_number < 5.0:
        risk_do = 20.0
    elif dose_number < 10.0:
        risk_do = 50.0
    else:
        risk_do = 80.0

    # Component B: supersaturation risk
    risk_ss = min(40.0, (initial_supersaturation - 1.0) * 10.0)
    risk_ss = max(0.0, risk_ss)

    # Component C: polymer reduction
    polymer_reduction = 0.0 if polymer_conc_mg_mL == 0.0 else min(30.0, polymer_conc_mg_mL * 10.0)

    risk_score = min(100.0, max(0.0, risk_do + risk_ss - polymer_reduction))

    # ------------------------------------------------------------------
    # 5. Risk classification
    # ------------------------------------------------------------------
    if risk_score < 20.0:
        risk_class = "negligible"
    elif risk_score < 40.0:
        risk_class = "low"
    elif risk_score < 70.0:
        risk_class = "moderate"
    else:
        risk_class = "high"

    # ------------------------------------------------------------------
    # 6. Induction time (h)
    # ------------------------------------------------------------------
    induction_time = 1.0 / max(0.01, (initial_supersaturation - 1.0) * 0.5 + 0.001)
    induction_time = max(0.1, min(24.0, induction_time))

    # ------------------------------------------------------------------
    # 7. Solubility advantage
    # ------------------------------------------------------------------
    solubility_advantage = min(10.0, S_amorphous_boost) if amorphous else 1.0

    # ------------------------------------------------------------------
    # 8. Polymer recommendation
    # ------------------------------------------------------------------
    polymer_required = risk_class == "high"

    if polymer_required:
        if logP > 2.0:
            recommended_polymer = "HPMC-AS"
        elif logP > 0.0:
            recommended_polymer = "PVP-VA"
        else:
            recommended_polymer = "HPMC"
    else:
        recommended_polymer = "none"

    # ------------------------------------------------------------------
    # 9. Notes
    # ------------------------------------------------------------------
    notes = (
        f"Drug: {drug_name}, dose={dose_mg} mg, Do={dose_number:.2f}. "
        f"Cs@pH6.8={cs_ph_intestine:.3f} mg/mL, "
        f"SS={initial_supersaturation:.2f}, "
        f"risk_score={risk_score:.1f} ({risk_class}). "
        f"Induction time={induction_time:.2f} h. "
        + (f"Amorphous boost={solubility_advantage:.2f}x. " if amorphous else "")
        + (f"Polymer: {recommended_polymer}." if polymer_required else "No polymer required.")
    )

    return GIPrecipitationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        dose_number=dose_number,
        cs_ph_intestine_mg_mL=cs_ph_intestine,
        initial_supersaturation=initial_supersaturation,
        precipitation_risk_score=risk_score,
        precipitation_risk_class=risk_class,
        induction_time_h=induction_time,
        solubility_advantage=solubility_advantage,
        polymer_required=polymer_required,
        recommended_polymer=recommended_polymer,
        notes=notes,
    )
