"""Volume of distribution prediction from physicochemical descriptors.

Simplified Poulin-Theil mechanistic approach for predicting Vss from
logP, pKa, ionization type, and free fraction in plasma (fup).

Reference: Poulin & Theil (2000) J Pharm Sci.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_ION_TYPES = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class VolumeDistributionResult:
    """Result of volume of distribution prediction."""

    drug_name: str
    logp: float
    pka: float
    ionization_type: str
    fup: float
    vd_ss_L_per_kg: float
    vd_total_L: float
    vd_class: str
    tissue_binding_factor: float
    predicted_t_half_h: float
    distribution_dominant_tissue: str
    notes: str


def predict_volume_of_distribution(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str = "neutral",
    fup: float = 0.5,
    weight_kg: float = 70.0,
) -> VolumeDistributionResult:
    """Predict volume of distribution using simplified Poulin-Theil method.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    logp:
        Octanol-water partition coefficient (log scale).
    pka:
        Acid dissociation constant (0-14).
    ionization_type:
        One of "acid", "base", or "neutral".
    fup:
        Unbound fraction in plasma (0 < fup <= 1).
    weight_kg:
        Body weight in kg (default 70 kg).

    Returns
    -------
    VolumeDistributionResult
    """
    # --- Validation ---
    if not (0.0 < fup <= 1.0):
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if weight_kg <= 0.0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if ionization_type not in _VALID_ION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {_VALID_ION_TYPES}, got {ionization_type!r}"
        )
    if not (0.0 <= pka <= 14.0):
        raise ValueError(f"pka must be in [0, 14], got {pka}")

    # --- Tissue composition (Poulin & Theil 2000 averages) ---
    # Keys: tissue -> (f_phospholipid, f_neutral_lipid, volume_fraction_L_per_kg)
    tissues = {
        "muscle": (0.016, 0.016, 0.40),
        "fat": (0.002, 0.853, 0.19),
        "liver": (0.025, 0.038, 0.026),
    }
    v_plasma_L_per_kg = 0.043

    # --- Partition coefficients ---
    # Neutral lipid partition
    ka_nl = 10.0 ** (0.9 * logp - 0.4)

    # Phospholipid partition depends on ionization type
    if ionization_type == "neutral":
        ka_pl = 10.0 ** (0.5 * logp)
    elif ionization_type == "base":
        ka_pl = 10.0 ** (0.5 * logp + 0.5)
    else:  # acid
        ka_pl = 10.0 ** (0.5 * logp - 0.5)

    # Simplified non-specific binding in tissue
    fu_tissue = 1.0 / (1.0 + ka_nl * 0.003)

    # --- Tissue Kp calculation ---
    kp_values: dict[str, float] = {}
    vkp_values: dict[str, float] = {}  # Vtissue * Kp contribution

    for tissue_name, (f_pl, f_nl, v_frac) in tissues.items():
        kp = (f_pl * ka_pl + f_nl * ka_nl + fu_tissue) / fup
        kp_values[tissue_name] = kp
        vkp_values[tissue_name] = v_frac * kp

    # --- Vss calculation ---
    vss_L_per_kg = v_plasma_L_per_kg + sum(vkp_values.values())

    # Cap and floor
    vss_L_per_kg = max(0.04, min(100.0, vss_L_per_kg))

    vd_total_L = vss_L_per_kg * weight_kg

    # --- Vd class ---
    if vss_L_per_kg < 0.6:
        vd_class = "low"
    elif vss_L_per_kg <= 3.0:
        vd_class = "moderate"
    else:
        vd_class = "high"

    # --- Dominant distribution tissue ---
    dominant_tissue = max(vkp_values, key=lambda k: vkp_values[k])
    # Check if plasma is dominant
    if v_plasma_L_per_kg > max(vkp_values.values()):
        dominant_tissue = "plasma"

    # --- Tissue binding factor: ratio of tissue to plasma binding ---
    # Average Kp across tissues weighted by volume vs plasma contribution
    avg_kp = sum(vkp_values.values()) / sum(v for _, _, v in tissues.values())
    tissue_binding_factor = avg_kp * fup  # dimensionless, relative to plasma

    # --- Predicted t½ for reference CL = 1 L/h ---
    # t½ = 0.693 * Vd / CL  with CL = 1 L/h
    predicted_t_half_h = 0.693 * vd_total_L / 1.0

    notes = (
        f"Simplified Poulin-Theil Vss. "
        f"Ka_NL={ka_nl:.3f}, Ka_PL={ka_pl:.3f}, fu_tissue={fu_tissue:.3f}. "
        f"Dominant tissue: {dominant_tissue}. "
        f"Reference CL=1 L/h used for t½."
    )

    return VolumeDistributionResult(
        drug_name=drug_name,
        logp=logp,
        pka=pka,
        ionization_type=ionization_type,
        fup=fup,
        vd_ss_L_per_kg=vss_L_per_kg,
        vd_total_L=vd_total_L,
        vd_class=vd_class,
        tissue_binding_factor=tissue_binding_factor,
        predicted_t_half_h=predicted_t_half_h,
        distribution_dominant_tissue=dominant_tissue,
        notes=notes,
    )


def screen_vd(compounds: list[dict]) -> list[VolumeDistributionResult]:
    """Screen a list of compounds for volume of distribution.

    Parameters
    ----------
    compounds:
        List of dicts with keys: drug_name, logp, pka, and optionally
        ionization_type, fup, weight_kg.

    Returns
    -------
    List of VolumeDistributionResult sorted by vd_ss_L_per_kg descending.
    """
    results = []
    for c in compounds:
        result = predict_volume_of_distribution(
            drug_name=c["drug_name"],
            logp=c["logp"],
            pka=c["pka"],
            ionization_type=c.get("ionization_type", "neutral"),
            fup=c.get("fup", 0.5),
            weight_kg=c.get("weight_kg", 70.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.vd_ss_L_per_kg, reverse=True)
    return results
