"""Phase 871: Drug Excretion in Breast Milk and Infant Exposure Prediction."""

from dataclasses import dataclass

PH_MILK = 7.0
PH_PLASMA = 7.4
MILK_VOLUME_ML_PER_KG_PER_DAY = 150.0
MP_MIN = 0.1
MP_MAX = 50.0


@dataclass(frozen=True)
class MilkExcretionResult:
    drug_name: str
    ionization_type: str
    pka: float
    logp: float
    milk_to_plasma_ratio: float
    c_milk_mg_L: float
    infant_daily_dose_mg: float
    relative_infant_dose_pct: float
    safety_assessment: str
    compatible_with_breastfeeding: bool
    notes: str


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _pow10(x: float) -> float:
    """Compute 10^x using exp."""
    import math

    return math.pow(10.0, x)


def _calc_mp_ph(pka: float, ionization_type: str) -> float:
    """Henderson-Hasselbalch pH-partitioning M/P ratio."""
    if ionization_type == "neutral":
        return 1.0

    if ionization_type == "base":
        # Base concentrates in more acidic compartment (milk pH 7.0 < plasma 7.4)
        # M/P_pH = (1 + 10^(pH_milk - pKa)) / (1 + 10^(pH_plasma - pKa))
        numerator = 1.0 + _pow10(PH_MILK - pka)
        denominator = 1.0 + _pow10(PH_PLASMA - pka)
        if denominator == 0.0:
            return 1.0
        return numerator / denominator

    # acid: reverse formula
    # Acid concentrates where pH is higher (plasma)
    numerator = 1.0 + _pow10(pka - PH_MILK)
    denominator = 1.0 + _pow10(pka - PH_PLASMA)
    if denominator == 0.0:
        return 1.0
    return numerator / denominator


def _calc_mp_lipid(pka: float, logp: float, fat_pct: float) -> float:
    """Lipid partitioning contribution to M/P ratio."""
    # M/P_lipid = 1 + (0.04 * fat_pct * logP) / (1 + 10^(pH_plasma - pKa))
    denom = 1.0 + _pow10(PH_PLASMA - pka)
    if denom == 0.0:
        denom = 1e-9
    correction = (0.04 * fat_pct * logp) / denom
    return 1.0 + correction


def predict_milk_excretion(
    drug_name: str,
    pka: float,
    logp: float,
    ionization_type: str,
    maternal_plasma_conc_mg_L: float = 1.0,
    maternal_daily_dose_mg: float = 100.0,
    maternal_weight_kg: float = 65.0,
    infant_weight_kg: float = 5.0,
    milk_fat_pct: float = 3.5,
    fup: float = 0.5,
) -> MilkExcretionResult:
    """Predict drug excretion into breast milk and infant daily exposure."""
    # Validation
    if not (0.0 <= pka <= 14.0):
        raise ValueError("pka must be in [0, 14]")
    if maternal_plasma_conc_mg_L <= 0:
        raise ValueError("maternal_plasma_conc_mg_L must be > 0")
    if infant_weight_kg <= 0:
        raise ValueError("infant_weight_kg must be > 0")
    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError(
            f"ionization_type must be 'acid', 'base', or 'neutral'; got '{ionization_type}'"
        )

    mp_ph = _calc_mp_ph(pka, ionization_type)
    mp_lipid = _calc_mp_lipid(pka, logp, milk_fat_pct)

    if ionization_type == "neutral":
        mp_overall = mp_lipid
    else:
        mp_overall = max(mp_ph, mp_lipid)

    mp_overall = _clamp(mp_overall, MP_MIN, MP_MAX)

    # Milk concentration (simplified: M/P already integrates fup implicitly)
    c_milk = mp_overall * maternal_plasma_conc_mg_L

    # Infant daily dose
    v_milk_ml = MILK_VOLUME_ML_PER_KG_PER_DAY * infant_weight_kg
    v_milk_l = v_milk_ml / 1000.0
    infant_daily_dose = c_milk * v_milk_l

    # Relative infant dose (%)
    maternal_dose_per_kg = maternal_daily_dose_mg / maternal_weight_kg
    infant_dose_per_kg = infant_daily_dose / infant_weight_kg
    if maternal_dose_per_kg > 0:
        rid_pct = (infant_dose_per_kg / maternal_dose_per_kg) * 100.0
    else:
        rid_pct = 0.0

    # Safety assessment
    if rid_pct < 10.0:
        safety = "safe"
        compatible = True
        notes = (
            f"RID {rid_pct:.1f}% < 10%: generally compatible with breastfeeding. "
            f"M/P ratio {mp_overall:.2f}."
        )
    elif rid_pct <= 25.0:
        safety = "caution"
        compatible = False
        notes = (
            f"RID {rid_pct:.1f}% between 10-25%: use with caution; monitor infant. "
            f"M/P ratio {mp_overall:.2f}."
        )
    else:
        safety = "avoid"
        compatible = False
        notes = (
            f"RID {rid_pct:.1f}% > 25%: avoid breastfeeding while on this drug. "
            f"M/P ratio {mp_overall:.2f}."
        )

    return MilkExcretionResult(
        drug_name=drug_name,
        ionization_type=ionization_type,
        pka=pka,
        logp=logp,
        milk_to_plasma_ratio=mp_overall,
        c_milk_mg_L=c_milk,
        infant_daily_dose_mg=infant_daily_dose,
        relative_infant_dose_pct=rid_pct,
        safety_assessment=safety,
        compatible_with_breastfeeding=compatible,
        notes=notes,
    )


def screen_drugs_for_breastfeeding(drugs: list[dict]) -> list[MilkExcretionResult]:
    """Screen a list of drugs for breastfeeding safety, sorted safest first."""
    results = []
    for d in drugs:
        result = predict_milk_excretion(
            drug_name=d["drug_name"],
            pka=d["pka"],
            logp=d["logp"],
            ionization_type=d["ionization_type"],
            maternal_plasma_conc_mg_L=d.get("maternal_plasma_conc_mg_L", 1.0),
            maternal_daily_dose_mg=d.get("maternal_daily_dose_mg", 100.0),
            maternal_weight_kg=d.get("maternal_weight_kg", 65.0),
            infant_weight_kg=d.get("infant_weight_kg", 5.0),
            milk_fat_pct=d.get("milk_fat_pct", 3.5),
            fup=d.get("fup", 0.5),
        )
        results.append(result)

    # Sort by RID ascending (safest first)
    results.sort(key=lambda r: r.relative_infant_dose_pct)
    return results
