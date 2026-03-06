"""Absorption rate constant (ka) prediction from physicochemical properties.

Predicts first-order oral absorption rate constant ka (1/h) using:
  1. Intestinal permeability (Peff) from logP, MW, PSA — Amidon model
  2. Dissolution rate from Noyes-Whitney equation
  3. Rate-limiting step = min(ka_permeability, ka_dissolution)

References:
  - Amidon GL et al. (1995) Pharm Res 12:413-420
  - Sun D et al. (2002) Pharm Res 19:261-267
  - Noyes AA, Whitney WR (1897) J Am Chem Soc 19:930-934
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KaPredictionResult:
    """Result of ka prediction from physicochemical properties."""

    mw: float
    logP: float
    psa_A2: float
    pka: float
    compound_type: str
    peff_cm_s: float
    ka_permeability_per_h: float
    ka_dissolution_per_h: float
    ka_effective_per_h: float
    rate_limiting_step: str
    bcs_class_predicted: str
    absorption_category: str
    notes: list[str]


# BCS class typical ka values (1/h)
_BCS_KA: dict[str, float] = {
    "I": 1.5,
    "II": 0.3,
    "III": 2.0,
    "IV": 0.1,
}

# Intestinal radius for Peff → ka conversion (cm)
_R_INTESTINE_CM = 0.5


def bcs_class_ka(bcs_class: str) -> float:
    """Return typical ka (1/h) for a given BCS class.

    Parameters
    ----------
    bcs_class:
        One of "I", "II", "III", "IV".

    Returns
    -------
    float
        Typical first-order absorption rate constant in 1/h.
    """
    bcs_class = bcs_class.upper()
    if bcs_class not in _BCS_KA:
        raise ValueError(f"bcs_class must be one of {list(_BCS_KA)}, got {bcs_class!r}")
    return _BCS_KA[bcs_class]


def predict_ka(
    mw: float,
    logP: float,
    psa_A2: float,
    pka: float = 7.0,
    compound_type: str = "neutral",
    particle_radius_um: float = 50.0,
    solubility_mg_mL: float = 1.0,
) -> KaPredictionResult:
    """Predict oral absorption rate constant from physicochemical properties.

    Parameters
    ----------
    mw:
        Molecular weight (Da). Must be > 0.
    logP:
        Octanol-water partition coefficient.
    psa_A2:
        Polar surface area in Angstrom². Must be >= 0.
    pka:
        Ionisation pKa. Default 7.0 (neutral-like).
    compound_type:
        "neutral", "acid", or "base". Affects ionisation correction.
    particle_radius_um:
        Primary particle radius in µm. Affects dissolution. Must be > 0.
    solubility_mg_mL:
        Aqueous solubility in mg/mL. Must be > 0.

    Returns
    -------
    KaPredictionResult
    """
    # --- Input validation ---
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if psa_A2 < 0:
        raise ValueError(f"psa_A2 must be >= 0, got {psa_A2}")
    if particle_radius_um <= 0:
        raise ValueError(f"particle_radius_um must be > 0, got {particle_radius_um}")
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be > 0, got {solubility_mg_mL}")

    # --- 1. Intestinal permeability (Amidon empirical model) ---
    # Peff (cm/s): log10(Peff) = -4.28 - 0.011*PSA + 0.278*logP
    log10_peff = -4.28 - 0.011 * psa_A2 + 0.278 * logP
    peff_cm_s = 10.0**log10_peff

    # Ionisation correction at intestinal pH ~6.5
    ph_intestine = 6.5
    if compound_type.lower() == "acid":
        # Henderson-Hasselbalch: fraction unionised = 1/(1+10^(pH-pKa))
        f_unionised = 1.0 / (1.0 + 10.0 ** (ph_intestine - pka))
        peff_cm_s *= max(f_unionised, 0.05)
    elif compound_type.lower() == "base":
        f_unionised = 1.0 / (1.0 + 10.0 ** (pka - ph_intestine))
        peff_cm_s *= max(f_unionised, 0.05)

    # Clamp to physiological range
    peff_cm_s = max(peff_cm_s, 1e-7)

    # --- 2. ka from Peff (cylindrical intestine approximation) ---
    # ka (1/s) = 2 * Peff / r;  convert to 1/h
    ka_perm_per_s = 2.0 * peff_cm_s / _R_INTESTINE_CM
    ka_permeability_per_h = ka_perm_per_s * 3600.0

    # --- 3. Dissolution rate constant (Noyes-Whitney) ---
    # Empirical calibration: a 50-µm particle of a drug with solubility 1 mg/mL
    # dissolves in ~15 min in GI → k_diss ≈ 4/h (reference point).
    # k_diss scales with: D/(r²) (shrinking sphere surface area) and sqrt(solubility)
    # Normalise to reference: r_ref=50 µm, sol_ref=1 mg/mL → ka_ref=4 /h
    _r_ref_um = 50.0
    _sol_ref = 1.0
    _ka_diss_ref = 4.0  # 1/h — calibrated reference
    r_ratio = (_r_ref_um / particle_radius_um) ** 2
    sol_factor = min(1.0, solubility_mg_mL / _sol_ref) ** 0.5
    ka_dissolution_per_h = _ka_diss_ref * r_ratio * sol_factor

    # --- 4. Effective ka = rate-limiting step ---
    if ka_permeability_per_h <= ka_dissolution_per_h:
        ka_effective_per_h = ka_permeability_per_h
        rate_limiting_step = "permeability"
    else:
        ka_effective_per_h = ka_dissolution_per_h
        rate_limiting_step = "dissolution"

    # --- 5. BCS classification ---
    high_perm = peff_cm_s > 1e-4  # cm/s threshold (human Peff)
    high_sol = solubility_mg_mL > 1.0  # mg/mL threshold

    if high_perm and high_sol:
        bcs_class = "I"
    elif high_perm and not high_sol:
        bcs_class = "II"
    elif not high_perm and high_sol:
        bcs_class = "III"
    else:
        bcs_class = "IV"

    # --- 6. Absorption category ---
    if ka_effective_per_h > 1.0:
        absorption_category = "rapid"
    elif ka_effective_per_h > 0.3:
        absorption_category = "moderate"
    else:
        absorption_category = "slow"

    # --- Notes ---
    notes: list[str] = []
    notes.append(
        f"Peff = {peff_cm_s:.2e} cm/s; BCS class {bcs_class}; rate-limited by {rate_limiting_step}"
    )
    if psa_A2 > 140:
        notes.append(f"Very high PSA ({psa_A2:.0f} A2) — poor membrane permeability expected")
    if logP > 5.0:
        notes.append("High logP may improve permeability but reduce aqueous solubility")
    if particle_radius_um > 100:
        notes.append(
            f"Large particles ({particle_radius_um:.0f} µm) — consider micronization "
        )
    if mw > 500:
        notes.append(f"High MW ({mw:.0f} Da) — may reduce permeability (Lipinski violation)")

    return KaPredictionResult(
        mw=mw,
        logP=logP,
        psa_A2=psa_A2,
        pka=pka,
        compound_type=compound_type,
        peff_cm_s=peff_cm_s,
        ka_permeability_per_h=ka_permeability_per_h,
        ka_dissolution_per_h=ka_dissolution_per_h,
        ka_effective_per_h=ka_effective_per_h,
        rate_limiting_step=rate_limiting_step,
        bcs_class_predicted=bcs_class,
        absorption_category=absorption_category,
        notes=notes,
    )
