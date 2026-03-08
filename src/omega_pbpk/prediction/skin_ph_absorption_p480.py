"""Phase 480 — Skin pH Effect on Drug Absorption.

Models the effect of skin surface pH on topical drug absorption and permeation.
"""

from __future__ import annotations

from dataclasses import dataclass

# Body site skin pH values
_BODY_SITE_PH: dict[str, float] = {
    "forehead": 5.5,
    "arm": 5.0,
    "scalp": 4.5,
    "foot": 4.2,
    "groin": 6.5,
    "back": 5.0,
}

_VALID_BODY_SITES = set(_BODY_SITE_PH.keys())
_VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}

# Reference arm Kp for permeation_enhancement_vs_arm computation
_ARM_SITE = "arm"


@dataclass(frozen=True)
class SkinPhAbsorptionResult:
    """Result of skin pH drug absorption prediction."""

    drug_name: str
    body_site: str
    skin_ph: float
    mw_Da: float
    logP: float
    pKa: float
    ionization_type: str
    kp_neutral_cm_per_h: float
    kp_effective_cm_per_h: float
    fraction_unionized: float
    flux_mg_per_cm2_per_h: float
    permeation_rate_mg_per_h: float
    permeation_enhancement_vs_arm: float
    notes: str


def _compute_kp_neutral(logP: float, mw_Da: float) -> float:
    """Compute neutral-form skin permeability coefficient (cm/h)."""
    log_kp = -2.7 + 0.71 * logP - 0.0061 * mw_Da
    return 10.0**log_kp


def _compute_fraction_unionized(ionization_type: str, pKa: float, skin_ph: float) -> float:
    """Compute fraction of drug in unionized form at given skin pH."""
    if ionization_type == "neutral":
        return 1.0
    elif ionization_type == "acid":
        # HH: fraction ionized = 1/(1 + 10^(pKa - pH))
        # fraction unionized = 1 - fi = 1 - 1/(1+10^(pKa-pH)) = 10^(pKa-pH)/(1+10^(pKa-pH))
        exponent = pKa - skin_ph
        # Clamp exponent for numerical stability
        exponent = max(-50.0, min(50.0, exponent))
        fi = 1.0 / (1.0 + 10.0**exponent)
        return 1.0 - fi
    else:  # base
        # For base: fi = 1/(1 + 10^(pH - pKa))
        exponent = skin_ph - pKa
        exponent = max(-50.0, min(50.0, exponent))
        fi = 1.0 / (1.0 + 10.0**exponent)
        return 1.0 - fi


def _compute_kp_effective(
    kp_neutral: float, fraction_unionized: float, ionization_type: str
) -> float:
    """Compute effective Kp adjusted for ionization."""
    if ionization_type == "neutral":
        kp_eff = kp_neutral
    else:
        kp_eff = kp_neutral * fraction_unionized
    # Clamp to [1e-7, 1.0]
    return max(1e-7, min(1.0, kp_eff))


def _build_notes(
    ionization_type: str,
    skin_ph: float,
    pKa: float,
    fraction_unionized: float,
    body_site: str,
) -> str:
    """Build descriptive notes for the result."""
    ion_desc = {
        "neutral": "Drug is neutral; ionization state does not affect permeability.",
        "acid": f"Acidic drug (pKa={pKa:.1f}) at skin pH {skin_ph:.1f}: "
        f"{fraction_unionized * 100:.1f}% unionized.",
        "base": f"Basic drug (pKa={pKa:.1f}) at skin pH {skin_ph:.1f}: "
        f"{fraction_unionized * 100:.1f}% unionized.",
    }
    desc = ion_desc.get(ionization_type, "")
    return f"Body site: {body_site} (pH={skin_ph:.1f}). {desc}"


def predict_skin_ph_absorption(
    drug_name: str,
    body_site: str,
    mw_Da: float,
    logP: float,
    pKa: float = 7.0,
    ionization_type: str = "neutral",
    dose_conc_mg_per_mL: float = 10.0,
    area_cm2: float = 10.0,
) -> SkinPhAbsorptionResult:
    """Predict topical drug absorption considering skin pH effects.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    body_site:
        One of "forehead", "arm", "scalp", "foot", "groin", "back".
    mw_Da:
        Molecular weight in Daltons (must be > 0).
    logP:
        Octanol-water partition coefficient.
    pKa:
        Drug pKa (relevant for acid/base ionization types).
    ionization_type:
        One of "acid", "base", "neutral".
    dose_conc_mg_per_mL:
        Applied drug concentration (mg/mL, must be > 0).
    area_cm2:
        Skin surface area (cm^2, must be > 0).

    Returns
    -------
    SkinPhAbsorptionResult
    """
    if body_site not in _VALID_BODY_SITES:
        raise ValueError(f"body_site must be one of {_VALID_BODY_SITES}, got {body_site!r}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION_TYPES}, got {ionization_type!r}"
        )
    if dose_conc_mg_per_mL <= 0:
        raise ValueError(f"dose_conc_mg_per_mL must be > 0, got {dose_conc_mg_per_mL}")
    if area_cm2 <= 0:
        raise ValueError(f"area_cm2 must be > 0, got {area_cm2}")

    skin_ph = _BODY_SITE_PH[body_site]
    kp_neutral = _compute_kp_neutral(logP, mw_Da)
    fraction_unionized = _compute_fraction_unionized(ionization_type, pKa, skin_ph)
    kp_effective = _compute_kp_effective(kp_neutral, fraction_unionized, ionization_type)

    flux = kp_effective * dose_conc_mg_per_mL
    permeation_rate = flux * area_cm2

    # Compute enhancement vs arm reference
    arm_ph = _BODY_SITE_PH[_ARM_SITE]
    arm_fu = _compute_fraction_unionized(ionization_type, pKa, arm_ph)
    arm_kp_eff = _compute_kp_effective(kp_neutral, arm_fu, ionization_type)
    arm_flux = arm_kp_eff * dose_conc_mg_per_mL
    arm_permeation = arm_flux * area_cm2
    if arm_permeation > 0:
        enhancement = permeation_rate / arm_permeation
    else:
        enhancement = 1.0

    notes = _build_notes(ionization_type, skin_ph, pKa, fraction_unionized, body_site)

    return SkinPhAbsorptionResult(
        drug_name=drug_name,
        body_site=body_site,
        skin_ph=skin_ph,
        mw_Da=mw_Da,
        logP=logP,
        pKa=pKa,
        ionization_type=ionization_type,
        kp_neutral_cm_per_h=kp_neutral,
        kp_effective_cm_per_h=kp_effective,
        fraction_unionized=fraction_unionized,
        flux_mg_per_cm2_per_h=flux,
        permeation_rate_mg_per_h=permeation_rate,
        permeation_enhancement_vs_arm=enhancement,
        notes=notes,
    )


def compare_body_sites(
    drug_name: str,
    mw_Da: float,
    logP: float,
    pKa: float = 7.0,
    ionization_type: str = "neutral",
    dose_conc_mg_per_mL: float = 10.0,
) -> list[SkinPhAbsorptionResult]:
    """Screen all 6 body sites, sorted by permeation_rate_mg_per_h descending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    mw_Da:
        Molecular weight in Daltons.
    logP:
        Octanol-water partition coefficient.
    pKa:
        Drug pKa.
    ionization_type:
        One of "acid", "base", "neutral".
    dose_conc_mg_per_mL:
        Applied drug concentration (mg/mL).

    Returns
    -------
    list[SkinPhAbsorptionResult]
        Results for all 6 body sites sorted by permeation_rate descending.
    """
    results = [
        predict_skin_ph_absorption(
            drug_name=drug_name,
            body_site=site,
            mw_Da=mw_Da,
            logP=logP,
            pKa=pKa,
            ionization_type=ionization_type,
            dose_conc_mg_per_mL=dose_conc_mg_per_mL,
            area_cm2=10.0,
        )
        for site in _VALID_BODY_SITES
    ]
    return sorted(results, key=lambda r: r.permeation_rate_mg_per_h, reverse=True)
