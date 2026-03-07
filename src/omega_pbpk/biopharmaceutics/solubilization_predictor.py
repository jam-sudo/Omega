"""Drug solubilization prediction in GI fluids — Phase 465.

Predicts drug solubilization in fasted/fed state intestinal fluids
using micellar solubilization theory.

References
----------
- Dressman JB et al., Pharm Res. 1998;15(1):11-22.
- Sugano K et al., J Pharm Sci. 2010;99(10):4360-4371.
- Sun D et al., Pharm Res. 2002;19(10):1460-1466.
"""

from __future__ import annotations

from dataclasses import dataclass

# GI segment pH values
_SEGMENT_PH: dict[str, float] = {
    "stomach": 1.5,
    "duodenum": 6.0,
    "jejunum": 6.5,
    "ileum": 7.2,
    "colon": 7.4,
}

# Fasted/fed bile salt and lecithin concentrations (mM)
_STATE_BILE: dict[str, float] = {"fasted": 3.0, "fed": 15.0}
_STATE_LECITHIN: dict[str, float] = {"fasted": 0.8, "fed": 3.75}


@dataclass
class GISolubilityResult:
    """Result of GI solubility prediction."""

    gi_segment: str
    ph: float
    intrinsic_solubility_mg_mL: float
    micellar_solubility_mg_mL: float
    total_solubility_mg_mL: float
    dose_number: float
    solubility_limited: bool
    notes: str


def _intrinsic_solubility_from_logp(drug_logP: float, drug_mw: float) -> float:
    """Estimate intrinsic aqueous solubility (mg/mL) from logP and MW.

    Uses simplified ESOL-like relationship:
        log(S_w, mol/L) = 0.5 - 0.01 * MW - 0.5 * logP
    Converted to mg/mL.
    """
    log_s_mol_L = 0.5 - 0.01 * drug_mw - 0.5 * drug_logP
    # clamp to reasonable range [-8, 2]
    log_s_mol_L = max(-8.0, min(2.0, log_s_mol_L))
    s_mol_L = 10.0**log_s_mol_L
    s_mg_mL = s_mol_L * drug_mw  # g/mol * mol/L = g/L = mg/mL (×1000/1000)
    return max(s_mg_mL, 1e-6)


def _ph_solubility_factor(intrinsic_sol: float, drug_pka: float | None, ph: float) -> float:
    """Apply Henderson-Hasselbalch correction for ionizable drugs.

    For basic drugs  (pKa represents conjugate acid):
        S(pH) = S0 * (1 + 10^(pKa - pH))   — ionized form more soluble
    For acidic drugs (pKa < 7):
        S(pH) = S0 * (1 + 10^(pH - pKa))

    Simple heuristic: if pKa < 7 treat as acid, else as base.
    """
    if drug_pka is None:
        return intrinsic_sol
    if drug_pka < 7.0:
        # acidic drug — ionized at high pH
        factor = 1.0 + 10.0 ** (ph - drug_pka)
    else:
        # basic drug — ionized at low pH
        factor = 1.0 + 10.0 ** (drug_pka - ph)
    return intrinsic_sol * factor


def micellar_solubilization(
    drug_logP: float,
    drug_mw: float,
    bile_salt_mM: float = 5.0,
    lecithin_mM: float = 1.5,
    state: str = "fasted",
) -> float:
    """Predict total drug solubility (mg/mL) via micellar solubilization.

    Uses the partition-into-micelle model:
        S_total = S_w + Kmc * [MC]_effective
    where Kmc is the micellar partition coefficient (estimated from logP)
    and [MC] is effective micellar concentration from bile salts + lecithin.

    Parameters
    ----------
    drug_logP:
        Octanol-water log P of the drug.
    drug_mw:
        Molecular weight (g/mol).
    bile_salt_mM:
        Bile salt concentration in mM (default 5.0 mM fasted).
    lecithin_mM:
        Lecithin (phospholipid) concentration in mM (default 1.5 mM fasted).
    state:
        'fasted' or 'fed' — used only for notes; actual concentrations supplied
        by caller.

    Returns
    -------
    float
        Total solubility in mg/mL.
    """
    if drug_mw <= 0:
        raise ValueError("drug_mw must be positive")
    if bile_salt_mM < 0 or lecithin_mM < 0:
        raise ValueError("Bile salt and lecithin concentrations must be non-negative")
    valid_states = {"fasted", "fed"}
    if state not in valid_states:
        raise ValueError(f"state must be one of {valid_states}, got '{state}'")

    # Intrinsic aqueous solubility
    s_w = _intrinsic_solubility_from_logp(drug_logP, drug_mw)

    # Micellar partition coefficient (log Kmc ~ 0.6 * logP + offset)
    # Dressman model: log Kmc ~ 0.6 * logP + 1.0
    log_kmc = 0.6 * drug_logP + 1.0
    kmc = 10.0**log_kmc  # dimensionless (mL_micellar / mL_aqueous equivalent)

    # Critical micellar concentration (CMC) for bile salts ~ 2 mM
    cmc_bile = 2.0
    cmc_lecithin = 0.5
    effective_bile = max(0.0, bile_salt_mM - cmc_bile)
    effective_lecithin = max(0.0, lecithin_mM - cmc_lecithin)

    # Effective micellar volume fraction
    # Bile salt contribution (mM → approximate volume fraction ~ mM * 1e-3 * V_micelle_mL/mM)
    # Simplified: effective_concentration_mM * molar_volume_factor
    # Typical mixed micelle molar volume ~600 mL/mmol
    mv_bile = 600.0  # mL/mmol — mixed micelle molar volume
    mv_lecithin = 700.0  # mL/mmol
    phi_micelle = (effective_bile * mv_bile + effective_lecithin * mv_lecithin) * 1e-3

    # Micellar enhancement (mg/mL)
    # S_total = S_w * (1 + Kmc * phi_micelle)
    s_total = s_w * (1.0 + kmc * phi_micelle)
    return s_total


def solubilization_ratio(fasted_solubility: float, fed_solubility: float) -> float:
    """Compute the fed-to-fasted solubilization ratio.

    Parameters
    ----------
    fasted_solubility:
        Solubility in fasted state (mg/mL).
    fed_solubility:
        Solubility in fed state (mg/mL).

    Returns
    -------
    float
        Ratio fed/fasted (>1 means fed state enhances solubility).
    """
    if fasted_solubility <= 0:
        raise ValueError("fasted_solubility must be positive")
    if fed_solubility < 0:
        raise ValueError("fed_solubility must be non-negative")
    return fed_solubility / fasted_solubility


def predict_gi_solubility(
    drug_logP: float,
    drug_mw: float,
    drug_pka: float | None = None,
    drug_dose_mg: float = 100.0,
    gi_segment: str = "jejunum",
) -> GISolubilityResult:
    """Predict drug solubility in a specific GI segment.

    Parameters
    ----------
    drug_logP:
        Octanol-water log P.
    drug_mw:
        Molecular weight (g/mol).
    drug_pka:
        pKa of ionizable group (None for non-ionizable drugs).
    drug_dose_mg:
        Oral dose in mg.
    gi_segment:
        One of 'stomach', 'duodenum', 'jejunum', 'ileum', 'colon'.

    Returns
    -------
    GISolubilityResult
    """
    if drug_mw <= 0:
        raise ValueError("drug_mw must be positive")
    if drug_dose_mg <= 0:
        raise ValueError("drug_dose_mg must be positive")
    seg = gi_segment.lower()
    if seg not in _SEGMENT_PH:
        raise ValueError(
            f"gi_segment must be one of {list(_SEGMENT_PH.keys())}, got '{gi_segment}'"
        )

    ph = _SEGMENT_PH[seg]

    # Intrinsic solubility
    s_intrinsic = _intrinsic_solubility_from_logp(drug_logP, drug_mw)

    # pH-corrected aqueous solubility
    s_ph = _ph_solubility_factor(s_intrinsic, drug_pka, ph)

    # Micellar solubility using segment-appropriate bile/lecithin
    # Stomach has negligible micellar solubilization
    if seg == "stomach":
        bile_mM = 0.0
        lecithin_mM = 0.0
    else:
        # Use fasted state defaults scaled by segment position
        bile_mM = _STATE_BILE["fasted"]
        lecithin_mM = _STATE_LECITHIN["fasted"]

    s_micellar = micellar_solubilization(drug_logP, drug_mw, bile_mM, lecithin_mM, state="fasted")
    # Micellar contribution above intrinsic
    s_micellar_contribution = max(0.0, s_micellar - s_intrinsic)

    # Total = pH-corrected + micellar enhancement
    s_total = s_ph + s_micellar_contribution

    # Dose number (volume = 250 mL)
    dn = dose_number(drug_dose_mg, s_total, volume_mL=250.0)
    sol_limited = dn > 1.0

    notes_parts = [f"pH={ph:.1f} at {gi_segment}"]
    if drug_pka is not None:
        notes_parts.append(f"pKa={drug_pka:.1f} applied")
    if seg == "stomach":
        notes_parts.append("No micellar solubilization in stomach")
    if sol_limited:
        notes_parts.append(f"Solubility-limited (Dn={dn:.2f}>1)")
    else:
        notes_parts.append(f"Not solubility-limited (Dn={dn:.2f})")

    return GISolubilityResult(
        gi_segment=gi_segment,
        ph=ph,
        intrinsic_solubility_mg_mL=s_intrinsic,
        micellar_solubility_mg_mL=s_micellar_contribution,
        total_solubility_mg_mL=s_total,
        dose_number=dn,
        solubility_limited=sol_limited,
        notes="; ".join(notes_parts),
    )


def dose_number(dose_mg: float, solubility_mg_mL: float, volume_mL: float = 250.0) -> float:
    """Compute BCS Dose Number.

    Dn = dose_mg / (solubility_mg_mL * volume_mL)

    Dn > 1 → solubility-limited absorption.
    Dn <= 1 → dose can be dissolved in the given volume.

    Parameters
    ----------
    dose_mg:
        Drug dose in mg.
    solubility_mg_mL:
        Drug solubility in mg/mL.
    volume_mL:
        GI fluid volume (default 250 mL, standard BCS luminal volume).

    Returns
    -------
    float
        Dimensionless dose number.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be positive")
    if volume_mL <= 0:
        raise ValueError("volume_mL must be positive")
    return dose_mg / (solubility_mg_mL * volume_mL)


def absorption_limited_by(dose_number_val: float, papp_cm_s: float) -> str:
    """Classify the primary rate-limiting step for absorption.

    Parameters
    ----------
    dose_number_val:
        BCS Dose Number (Dn). Dn > 1 → solubility issue.
    papp_cm_s:
        Apparent permeability (cm/s). Papp < 1e-6 cm/s → low permeability.

    Returns
    -------
    str
        One of 'solubility', 'permeability', 'both', 'neither'.
    """
    if dose_number_val < 0:
        raise ValueError("dose_number_val must be non-negative")
    if papp_cm_s < 0:
        raise ValueError("papp_cm_s must be non-negative")

    # BCS permeability threshold: ~1×10^-6 cm/s (BCS Class I/III boundary)
    _PAPP_THRESHOLD = 1.0e-6
    sol_limited = dose_number_val > 1.0
    perm_limited = papp_cm_s < _PAPP_THRESHOLD

    if sol_limited and perm_limited:
        return "both"
    elif sol_limited:
        return "solubility"
    elif perm_limited:
        return "permeability"
    else:
        return "neither"
