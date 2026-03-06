"""High-concentration mAb formulation viscosity prediction.

Models viscosity of protein solutions as a function of concentration,
temperature, excipients, and molecular properties.

References
----------
- Shire SJ et al., J Pharm Sci. 2004;93(6):1390-402 (high-conc mAb)
- Connolly BD et al., Biophys J. 2012;103(1):69-78 (electrostatic viscosity)
- Sharma VK et al., AAPS PharmSciTech. 2015;16(4):784-98 (excipient effects)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


__all__ = [
    "ViscosityResult",
    "predict_viscosity",
    "screen_concentrations",
]

# Subcutaneous delivery limit (cP / mPa·s)
_SC_VISCOSITY_LIMIT_CP = 20.0
_SC_CONC_LIMIT_MG_ML = 150.0  # typical SC injectability limit


@dataclass(frozen=True)
class ViscosityResult:
    """High-concentration mAb viscosity prediction result.

    Attributes
    ----------
    protein_name : Name of the mAb / protein.
    concentration_mg_mL : Protein concentration (mg/mL).
    temperature_C : Temperature (°C).
    viscosity_cP : Predicted solution viscosity (cP = mPa·s).
    viscosity_category : 'low' (<5), 'moderate' (5-20), 'high' (>20 cP).
    sc_injectable : True if viscosity <= 20 cP (suitable for SC injection).
    relative_viscosity : Ratio to solvent viscosity.
    self_association_index : Estimated inter-molecular interaction parameter kD.
    excipient_effect : Net viscosity reduction fraction from excipients (0-1).
    concentration_for_20cP : Estimated max concentration for SC injectability.
    notes : Summary text.
    """

    protein_name: str
    concentration_mg_mL: float
    temperature_C: float
    viscosity_cP: float
    viscosity_category: str
    sc_injectable: bool
    relative_viscosity: float
    self_association_index: float
    excipient_effect: float
    concentration_for_20cP: float
    notes: str


def _solvent_viscosity(temperature_C: float) -> float:
    """Water viscosity (cP) as a function of temperature (Vogel equation)."""
    T_K = temperature_C + 273.15
    # Simplified: viscosity of water ~ 1.002 cP at 20°C, decreasing with T
    eta_20 = 1.002
    return eta_20 * math.exp(1.7 * (293.15 - T_K) / T_K)


def _self_association_index(
    mw_kDa: float,
    pi_pH: float,
    ph: float,
    logP: float,
    net_charge: float,
) -> float:
    """Estimate kD-like self-association index.

    Higher values indicate stronger attractive interactions → higher viscosity.
    Repulsive electrostatics (high net charge at working pH) reduce kD.
    """
    # Base: MW-dependent (larger = more surface for interaction)
    kd = math.log10(max(1.0, mw_kDa)) * 0.5

    # Electrostatic: repulsion when charge differs from zero
    charge_effect = abs(net_charge) * 0.05
    kd -= charge_effect

    # Hydrophobicity: higher logP → more hydrophobic patches → higher viscosity
    kd += max(0.0, logP) * 0.1

    # pI proximity to working pH → minimum net charge → maximum attractive interactions
    delta_pi = abs(pi_pH - ph)
    if delta_pi < 1.0:
        kd += (1.0 - delta_pi) * 0.3  # near pI: less charge, more aggregation tendency

    return float(max(0.0, kd))


def _excipient_viscosity_factor(
    arginine_mM: float = 0.0,
    nacl_mM: float = 0.0,
    sucrose_pct: float = 0.0,
    polysorbate_pct: float = 0.0,
) -> float:
    """Compute fractional viscosity reduction from common excipients.

    Returns a factor in [0, 1]: 0 = no excipient, 1 = maximum 40% reduction.
    Arginine is most effective (hydrophobic patch masking),
    NaCl provides ionic shielding (moderate),
    sucrose slightly increases viscosity at high levels.
    """
    reduction = 0.0

    # Arginine: up to 25% reduction at 150 mM
    reduction += min(0.25, arginine_mM / 600.0)

    # NaCl (ionic screening): up to 10% reduction at 150 mM
    reduction += min(0.10, nacl_mM / 1500.0)

    # Sucrose: slight increase at >5% (raises osmolyte viscosity), set slight penalty
    if sucrose_pct > 5.0:
        reduction -= 0.02 * (sucrose_pct - 5.0) / 5.0

    # Polysorbate: minor, up to 2%
    reduction += min(0.02, polysorbate_pct * 0.02)

    return float(max(0.0, min(0.40, reduction)))


def predict_viscosity(
    protein_name: str,
    concentration_mg_mL: float,
    mw_kDa: float = 150.0,
    temperature_C: float = 25.0,
    ph: float = 6.0,
    pi_pH: float = 7.5,
    logP: float = 0.0,
    net_charge: float = 0.0,
    arginine_mM: float = 0.0,
    nacl_mM: float = 0.0,
    sucrose_pct: float = 0.0,
    polysorbate_pct: float = 0.0,
) -> ViscosityResult:
    """Predict high-concentration mAb viscosity.

    Parameters
    ----------
    protein_name : Protein/mAb name.
    concentration_mg_mL : Protein concentration (mg/mL). Must be > 0.
    mw_kDa : Molecular weight (kDa).
    temperature_C : Temperature (°C). Must be > 0.
    ph : Formulation pH.
    pi_pH : Isoelectric point (pI).
    logP : Hydrophobicity (octanol-water log P).
    net_charge : Net charge at formulation pH (e).
    arginine_mM : Arginine concentration (mM) — viscosity reducer.
    nacl_mM : NaCl concentration (mM).
    sucrose_pct : Sucrose weight fraction (%).
    polysorbate_pct : Polysorbate 80 weight fraction (%).

    Returns
    -------
    ViscosityResult

    Raises
    ------
    ValueError : If parameters are invalid.
    """
    if concentration_mg_mL <= 0:
        raise ValueError("concentration_mg_mL must be > 0")
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be > 0")
    if temperature_C <= 0:
        raise ValueError("temperature_C must be > 0 (Celsius)")

    eta_s = _solvent_viscosity(temperature_C)
    kd = _self_association_index(mw_kDa, pi_pH, ph, logP, net_charge)
    excipient_red = _excipient_viscosity_factor(arginine_mM, nacl_mM, sucrose_pct, polysorbate_pct)

    # Ross-Minton / Mooney equation for protein viscosity
    # log(eta_rel) = k * phi / (1 - s * phi)
    # phi (volume fraction) = conc_mg_mL / (1000 * rho_protein)
    # rho_protein ~ 1.35 g/mL for typical proteins
    rho_protein = 1.35  # g/mL
    phi = concentration_mg_mL / (rho_protein * 1000.0)

    # Self-crowding factor: k depends on protein shape + interactions
    k_crowding = 2.5 + kd * 2.0   # Einstein coeff 2.5 for spheres
    s_crowding = 0.7 + kd * 0.3   # crowding/interaction factor

    denom = 1.0 - s_crowding * phi
    if denom <= 0:
        # Very high conc / strong self-association → extremely high viscosity
        relative_viscosity = 1000.0
    else:
        ln_eta_rel = k_crowding * phi / denom
        relative_viscosity = math.exp(min(ln_eta_rel, 7.0))  # cap at e^7 ≈ 1097

    # Apply excipient reduction
    viscosity_cP = eta_s * relative_viscosity * (1.0 - excipient_red)

    # Category
    if viscosity_cP < 5.0:
        category = "low"
    elif viscosity_cP <= 20.0:
        category = "moderate"
    else:
        category = "high"

    sc_injectable = viscosity_cP <= _SC_VISCOSITY_LIMIT_CP

    # Estimate concentration for 20 cP (SC limit) by simple inversion
    # Find phi_max such that eta_s * exp(k*phi/(1-s*phi)) * (1 - red) = 20
    target_rel = _SC_VISCOSITY_LIMIT_CP / (eta_s * (1.0 - excipient_red))
    target_ln = math.log(max(1.001, target_rel))
    # ln_eta_rel = k*phi/(1-s*phi) => phi = target_ln / (k + s*target_ln)
    denom_phi = k_crowding + s_crowding * target_ln
    if denom_phi <= 0:
        c_for_20cP = 0.0
    else:
        phi_max = target_ln / denom_phi
        c_for_20cP = float(phi_max * rho_protein * 1000.0)

    notes = (
        f"Viscosity of {protein_name} at {concentration_mg_mL:.0f} mg/mL, "
        f"T={temperature_C:.0f}°C: {viscosity_cP:.1f} cP ({category}). "
        f"SC-injectable: {'yes' if sc_injectable else 'no'}. "
        f"Est. max SC conc: {c_for_20cP:.0f} mg/mL."
    )

    return ViscosityResult(
        protein_name=protein_name,
        concentration_mg_mL=concentration_mg_mL,
        temperature_C=temperature_C,
        viscosity_cP=viscosity_cP,
        viscosity_category=category,
        sc_injectable=sc_injectable,
        relative_viscosity=relative_viscosity,
        self_association_index=kd,
        excipient_effect=excipient_red,
        concentration_for_20cP=c_for_20cP,
        notes=notes,
    )


def screen_concentrations(
    protein_name: str,
    concentrations_mg_mL: list[float],
    **kwargs,
) -> list[ViscosityResult]:
    """Predict viscosity across a range of concentrations.

    Parameters
    ----------
    protein_name : Protein name.
    concentrations_mg_mL : List of concentrations to screen.
    **kwargs : Additional keyword arguments for predict_viscosity.

    Returns
    -------
    List of ViscosityResult, one per concentration.
    """
    return [
        predict_viscosity(protein_name, c, **kwargs)
        for c in concentrations_mg_mL
    ]
