"""Pediatric CYP enzyme ontogeny and physiological maturation functions.

Implements sigmoidal maturation functions for key CYP enzymes and GFR
based on published pediatric PBPK references:

  - Bjorkman S, J Pharm Pharmacol, 2005 (CYP maturation in neonates/infants)
  - Salem F et al., Drug Metab Dispos, 2013 (CYP3A4/2D6 ontogeny)
  - Johnson TN et al., Clin Pharmacokinet, 2006 (CYP1A2 ontogeny)
  - Rhodin MM et al., Pediatr Nephrol, 2009 (GFR maturation)

Sigmoidal maturation function (Hill equation):
  f(age) = age^hill / (age^hill + TM50^hill)

  where:
    age   = postnatal age in years
    TM50  = age at 50% of adult activity (years)
    hill  = Hill coefficient (steepness)

  At age >> TM50, f -> 1.0 (adult activity).
  At age = 0 (birth), f -> 0 (or small positive basal value).
"""

from __future__ import annotations


def _hill_maturation(age_years: float, tm50: float, hill: float) -> float:
    """Sigmoidal Hill maturation function.

    Args:
        age_years: Postnatal age in years (>=0).
        tm50: Age at 50% of maximum activity (years).
        hill: Hill coefficient (steepness of the maturation curve).

    Returns:
        Maturation factor in range [0, 1] relative to adult.
    """
    if age_years < 0.0:
        age_years = 0.0
    if tm50 <= 0.0:
        raise ValueError(f"TM50 must be positive, got {tm50}")
    age_h = age_years ** hill
    tm50_h = tm50 ** hill
    return age_h / (age_h + tm50_h)


def cyp3a4_ontogeny(age_years: float) -> float:
    """CYP3A4 activity relative to adult (1.0) as a function of postnatal age.

    CYP3A4 is the most abundant hepatic CYP in adults. At birth, activity is
    ~20-30% of adult values (fetal CYP3A7 predominates). Rapid postnatal
    increase reaches adult levels by approximately 1-2 years.

    Maturation parameters (Bjorkman 2005, Salem 2013):
      TM50 = 0.25 years (approximately 3 months postnatal)
      hill = 2.0 (moderate sigmoidal steepness)
      basal_fraction = 0.1 (10% activity at birth from CYP3A4)

    Args:
        age_years: Postnatal age in years.

    Returns:
        CYP3A4 activity as fraction of adult (0 to 1).
        Capped at 1.0 for adults (age >= 18).
    """
    basal = 0.1  # 10% activity at birth
    tm50 = 0.25
    hill = 2.0
    maturation = _hill_maturation(age_years, tm50, hill)
    activity = basal + (1.0 - basal) * maturation
    return min(activity, 1.0)


def cyp2d6_ontogeny(age_years: float) -> float:
    """CYP2D6 activity relative to adult (1.0) as a function of postnatal age.

    CYP2D6 is essentially absent at birth but develops rapidly in the first
    weeks to months of life. Adult levels reached by approximately 6-12 months.

    Maturation parameters (Bjorkman 2005, Salem 2013):
      TM50 = 0.4 years (approximately 5 months postnatal)
      hill = 3.0 (steep sigmoidal curve — rapid neonatal increase)
      basal_fraction = 0.02 (2% activity at birth)

    Args:
        age_years: Postnatal age in years.

    Returns:
        CYP2D6 activity as fraction of adult (0 to 1).
    """
    basal = 0.02  # 2% activity at birth
    tm50 = 0.4
    hill = 3.0
    maturation = _hill_maturation(age_years, tm50, hill)
    activity = basal + (1.0 - basal) * maturation
    return min(activity, 1.0)


def cyp1a2_ontogeny(age_years: float) -> float:
    """CYP1A2 activity relative to adult (1.0) as a function of postnatal age.

    CYP1A2 is very low at birth, shows a slow postnatal increase, and may not
    reach adult levels until 1-2 years. It is responsible for caffeine
    metabolism, which is markedly reduced in neonates.

    Maturation parameters (Johnson 2006):
      TM50 = 1.0 year
      hill = 2.5
      basal_fraction = 0.01 (1% activity at birth)

    Args:
        age_years: Postnatal age in years.

    Returns:
        CYP1A2 activity as fraction of adult (0 to 1).
    """
    basal = 0.01  # 1% activity at birth
    tm50 = 1.0
    hill = 2.5
    maturation = _hill_maturation(age_years, tm50, hill)
    activity = basal + (1.0 - basal) * maturation
    return min(activity, 1.0)


def gfr_ontogeny(age_years: float) -> float:
    """GFR (renal function) relative to adult (1.0) as a function of postnatal age.

    Neonatal GFR is approximately 10% of adult values (corrected for body
    surface area) due to incomplete glomerular development. Rapid maturation
    occurs in the first 2 years of life.

    Maturation parameters (Rhodin 2009):
      TM50 = 0.25 years (approximately 3 months)
      hill = 3.4
      basal_fraction = 0.1 (10% at birth)

    Args:
        age_years: Postnatal age in years.

    Returns:
        GFR as fraction of adult reference (0 to 1).
    """
    basal = 0.1  # 10% at birth (normalized to BSA)
    tm50 = 0.25
    hill = 3.4
    maturation = _hill_maturation(age_years, tm50, hill)
    activity = basal + (1.0 - basal) * maturation
    return min(activity, 1.0)


def get_pediatric_scaling(age_years: float, weight_kg: float) -> dict[str, float]:
    """Get all ontogeny scaling factors for a pediatric subject.

    Combines enzyme ontogeny (maturation-based) with allometric scaling
    (weight-based) for physiological flow parameters.

    Reference adult body weight is 70 kg.

    Args:
        age_years: Postnatal age in years (0 = term neonate).
        weight_kg: Body weight in kg.

    Returns:
        Dict with ontogeny scaling factors:
          "cyp3a4":           CYP3A4 activity fraction (0-1)
          "cyp2d6":           CYP2D6 activity fraction (0-1)
          "cyp1a2":           CYP1A2 activity fraction (0-1)
          "gfr":              GFR fraction relative to adult (0-1)
          "hepatic_blood_flow": hepatic blood flow relative to adult (BW^0.75/70^0.75)
          "cardiac_output":    cardiac output relative to adult (BW^0.75/70^0.75)

    Example:
        factors = get_pediatric_scaling(2.0, 12.0)
        # 2-year-old, 12 kg: CYP3A4 near-adult, GFR near-adult
    """
    adult_bw = 70.0
    # Allometric scaling exponent 0.75 for flows
    flow_scale = (weight_kg / adult_bw) ** 0.75

    return {
        "cyp3a4": cyp3a4_ontogeny(age_years),
        "cyp2d6": cyp2d6_ontogeny(age_years),
        "cyp1a2": cyp1a2_ontogeny(age_years),
        "gfr": gfr_ontogeny(age_years),
        "hepatic_blood_flow": flow_scale,
        "cardiac_output": flow_scale,
    }
