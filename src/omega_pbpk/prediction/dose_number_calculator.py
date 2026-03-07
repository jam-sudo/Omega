"""BCS dose number (Do) calculation and dissolution rate assessment.

Dose number Do = dose / (solubility_at_pH * 250 mL).
Do < 1 => BCS high solubility class (complete dissolution in 250 mL).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DoseNumberResult:
    drug_name: str
    dose_mg: float
    intrinsic_solubility_mg_mL: float
    ionization_type: str
    pka: float
    ph_gi: float
    solubility_at_ph_mg_mL: float
    dose_number: float
    bcs_solubility_class: str
    dissolution_limited: bool
    absorption_risk: str
    notes: str


@dataclass(frozen=True)
class DissolutionRiskResult:
    drug_name: str
    dose_mg: float
    do_gastric: float
    do_duodenum: float
    do_jejunum: float
    do_fasted_si: float
    worst_case_do: float
    primary_risk_location: str
    overall_dissolution_risk: str
    notes: str


def _solubility_at_ph(s0: float, pka: float, ionization_type: str, ph: float) -> float:
    """Henderson-Hasselbalch solubility adjustment at given pH."""
    if ionization_type == "acid":
        return s0 * (1.0 + 10.0 ** (ph - pka))
    if ionization_type == "base":
        return s0 * (1.0 + 10.0 ** (pka - ph))
    return s0  # neutral


def _absorption_risk(do: float) -> str:
    if do < 1.0:
        return "low"
    if do < 5.0:
        return "moderate"
    return "high"


def _validate_inputs(
    dose_mg: float,
    solubility_mg_mL: float,
    pka: float,
    ionization_type: str,
    ph_gi: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if pka < 0 or pka > 14:
        raise ValueError("pka must be in [0, 14]")
    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError("ionization_type must be 'acid', 'base', or 'neutral'")
    if ph_gi < 0 or ph_gi > 14:
        raise ValueError("ph_gi must be in [0, 14]")


def classify_bcs_solubility(do_value: float) -> str:
    """Classify BCS solubility based on dose number.

    Args:
        do_value: dose number (dimensionless)

    Returns:
        'high' if Do < 1 (BCS high solubility), 'low' otherwise.
    """
    return "high" if do_value < 1.0 else "low"


def calculate_dose_number(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL: float,
    pka: float,
    ionization_type: str,
    ph_gi: float = 6.8,
) -> DoseNumberResult:
    """Calculate BCS dose number at a specified GI pH.

    Args:
        drug_name: drug identifier
        dose_mg: administered dose (mg)
        solubility_mg_mL: intrinsic (un-ionized) solubility at pH~pKa or neutral solubility
        pka: acid dissociation constant
        ionization_type: 'acid', 'base', or 'neutral'
        ph_gi: GI pH at which to evaluate solubility (default 6.8 for small intestine)

    Returns:
        DoseNumberResult with Do and BCS classification.
    """
    _validate_inputs(dose_mg, solubility_mg_mL, pka, ionization_type, ph_gi)

    s_at_ph = _solubility_at_ph(solubility_mg_mL, pka, ionization_type, ph_gi)
    do = dose_mg / (s_at_ph * 250.0)
    bcs_class = classify_bcs_solubility(do)
    dissolution_limited = do >= 1.0
    risk = _absorption_risk(do)

    notes = (
        f"S0={solubility_mg_mL:.4f} mg/mL, "
        f"S(pH={ph_gi})={s_at_ph:.4f} mg/mL, "
        f"Do={do:.3f} => BCS solubility {bcs_class}"
    )

    return DoseNumberResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        intrinsic_solubility_mg_mL=solubility_mg_mL,
        ionization_type=ionization_type,
        pka=pka,
        ph_gi=ph_gi,
        solubility_at_ph_mg_mL=s_at_ph,
        dose_number=do,
        bcs_solubility_class=bcs_class,
        dissolution_limited=dissolution_limited,
        absorption_risk=risk,
        notes=notes,
    )


def assess_dissolution_risk(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL: float,
    pka: float,
    ionization_type: str,
) -> DissolutionRiskResult:
    """Assess dissolution risk across GI segments.

    Evaluates Do at four physiologically relevant pH values:
        - Gastric (fasted): pH 1.5
        - Duodenum: pH 5.0
        - Jejunum: pH 6.8
        - Fasted SI: pH 6.5

    Args:
        drug_name: drug identifier
        dose_mg: administered dose (mg)
        solubility_mg_mL: intrinsic solubility (mg/mL)
        pka: acid dissociation constant
        ionization_type: 'acid', 'base', or 'neutral'

    Returns:
        DissolutionRiskResult with per-segment Do values and overall risk.
    """
    _validate_inputs(dose_mg, solubility_mg_mL, pka, ionization_type, ph_gi=6.8)

    ph_map = {
        "gastric": 1.5,
        "duodenum": 5.0,
        "jejunum": 6.8,
        "fasted_si": 6.5,
    }

    do_map: dict[str, float] = {}
    for segment, ph in ph_map.items():
        s_ph = _solubility_at_ph(solubility_mg_mL, pka, ionization_type, ph)
        do_map[segment] = dose_mg / (s_ph * 250.0)

    do_gastric = do_map["gastric"]
    do_duodenum = do_map["duodenum"]
    do_jejunum = do_map["jejunum"]
    do_fasted_si = do_map["fasted_si"]

    worst_case_do = max(do_gastric, do_duodenum, do_jejunum, do_fasted_si)
    primary_risk_location = max(do_map, key=lambda k: do_map[k])

    overall_risk = _absorption_risk(worst_case_do)

    notes = (
        f"Do gastric={do_gastric:.3f}, duodenum={do_duodenum:.3f}, "
        f"jejunum={do_jejunum:.3f}, fasted_SI={do_fasted_si:.3f}; "
        f"worst={worst_case_do:.3f} at {primary_risk_location}"
    )

    return DissolutionRiskResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        do_gastric=do_gastric,
        do_duodenum=do_duodenum,
        do_jejunum=do_jejunum,
        do_fasted_si=do_fasted_si,
        worst_case_do=worst_case_do,
        primary_risk_location=primary_risk_location,
        overall_dissolution_risk=overall_risk,
        notes=notes,
    )
