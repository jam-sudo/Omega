"""Drug mucus penetration prediction for GI, pulmonary, and cervical mucus.

Predicts effective diffusivity of drugs through mucus layers based on
physicochemical properties (molecular weight, charge, logP) and any
PEG coating.  Uses a Stokes-Einstein free-diffusion baseline reduced by
mucus-specific hindrance factors.

References
----------
- Lai SK et al., Adv Drug Deliv Rev. 2009
- Cone RA, Adv Drug Deliv Rev. 2009
- Witten J & Ribbeck K, Nanoscale. 2017
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_KB = 1.38e-23  # Boltzmann constant, J/K
_T = 310.0  # Body temperature, K
_ETA_WATER = 0.001  # Dynamic viscosity of water, Pa·s
_PI = 3.14159265358979

# Mucus layer thicknesses (µm) by type
_MUCUS_THICKNESS_UM: dict = {
    "gi": 300.0,
    "pulmonary": 10.0,
    "cervical": 100.0,
}

_VALID_CHARGES = ("cationic", "anionic", "neutral", "zwitterionic")
_VALID_MUCUS_TYPES = ("gi", "pulmonary", "cervical")


@dataclass(frozen=True)
class MucusPenetrationResult:
    """Prediction of drug diffusion through a mucus layer."""

    drug_name: str
    mucus_type: str
    mw_Da: float
    charge: str
    d_free_cm2_s: float
    d_mucus_cm2_s: float
    d_ratio: float
    mucus_thickness_um: float
    permeation_time_s: float
    flux_relative: float
    penetration_class: str
    mucoadhesion_risk: str
    notes: str


def predict_mucus_penetration(
    drug_name: str,
    mw_Da: float,
    charge: str,
    mucus_type: str = "gi",
    logP: float = 0.0,
    peg_coated: bool = False,
) -> MucusPenetrationResult:
    """Predict drug mucus penetration characteristics.

    Parameters
    ----------
    drug_name:
        Identifier for the drug/formulation.
    mw_Da:
        Molecular weight (Da).
    charge:
        Ionic character: "cationic", "anionic", "neutral", or "zwitterionic".
    mucus_type:
        Mucus compartment: "gi", "pulmonary", or "cervical".
    logP:
        Octanol-water partition coefficient.
    peg_coated:
        Whether the particle/drug is PEG-coated (improves mucus penetration).

    Returns
    -------
    MucusPenetrationResult
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if mucus_type not in _VALID_MUCUS_TYPES:
        raise ValueError(f"mucus_type must be one of {_VALID_MUCUS_TYPES}, got '{mucus_type}'")
    if charge not in _VALID_CHARGES:
        raise ValueError(f"charge must be one of {_VALID_CHARGES}, got '{charge}'")

    # ------------------------------------------------------------------
    # Stokes-Einstein free diffusion coefficient
    # ------------------------------------------------------------------
    # Hydrodynamic radius (m) — empirical for small molecules
    r_m = 0.052e-9 * (mw_Da ** (1.0 / 3.0))
    D_free_m2_s = _KB * _T / (6.0 * _PI * _ETA_WATER * r_m)
    D_free_cm2_s = D_free_m2_s * 1e4  # m² → cm²

    # ------------------------------------------------------------------
    # Mucus hindrance factors
    # ------------------------------------------------------------------
    # MW effect: larger molecules diffuse more slowly
    mw_factor = math.exp(-mw_Da / 1000.0)

    # Charge effect: cationic molecules are electrostatically trapped by
    # anionic mucin fibres
    if charge == "cationic":
        charge_factor = 0.1
    elif charge == "zwitterionic":
        charge_factor = 0.5
    else:  # neutral or anionic
        charge_factor = 1.0

    # logP effect: very lipophilic molecules partition into hydrophobic
    # mucin domains and are retarded
    if logP < 1.0:
        logP_factor = 1.0
    else:
        logP_factor = max(0.1, 1.0 - (logP - 1.0) * 0.1)

    # PEG coating: dense PEG brush prevents mucoadhesive interactions
    peg_factor = 3.0 if peg_coated else 1.0

    # Effective mucus diffusivity (clamp to D_free as upper bound)
    D_mucus_cm2_s = min(
        D_free_cm2_s,
        D_free_cm2_s * mw_factor * charge_factor * logP_factor * peg_factor,
    )

    d_ratio = D_mucus_cm2_s / D_free_cm2_s

    # ------------------------------------------------------------------
    # Permeation time (diffusion through the mucus layer)
    # ------------------------------------------------------------------
    thickness_um = _MUCUS_THICKNESS_UM.get(mucus_type, 50.0)
    thickness_m = thickness_um * 1e-6
    D_mucus_m2_s = D_mucus_cm2_s * 1e-4  # cm² → m²
    permeation_time_s = (thickness_m**2) / (2.0 * D_mucus_m2_s)

    # ------------------------------------------------------------------
    # Derived classifications
    # ------------------------------------------------------------------
    flux_relative = d_ratio

    if permeation_time_s < 60.0:
        penetration_class = "rapid"
    elif permeation_time_s < 600.0:
        penetration_class = "moderate"
    else:
        penetration_class = "slow"

    mucoadhesion_risk = "high" if charge == "cationic" else "low"

    notes = (
        f"mw_factor={mw_factor:.4f}, charge_factor={charge_factor}, "
        f"logP_factor={logP_factor:.4f}, peg_factor={peg_factor}; "
        f"D_free={D_free_cm2_s:.3e} cm2/s, D_mucus={D_mucus_cm2_s:.3e} cm2/s"
    )

    return MucusPenetrationResult(
        drug_name=drug_name,
        mucus_type=mucus_type,
        mw_Da=mw_Da,
        charge=charge,
        d_free_cm2_s=D_free_cm2_s,
        d_mucus_cm2_s=D_mucus_cm2_s,
        d_ratio=d_ratio,
        mucus_thickness_um=thickness_um,
        permeation_time_s=permeation_time_s,
        flux_relative=flux_relative,
        penetration_class=penetration_class,
        mucoadhesion_risk=mucoadhesion_risk,
        notes=notes,
    )


def compare_mucus_types(
    drug_name: str,
    mw_Da: float,
    charge: str,
    logP: float = 0.0,
) -> list:
    """Compare drug mucus penetration across all three mucus types.

    Returns a list of :class:`MucusPenetrationResult` objects sorted by
    ``d_ratio`` in descending order (best penetration first).

    Parameters
    ----------
    drug_name, mw_Da, charge, logP:
        Passed directly to :func:`predict_mucus_penetration`.
    """
    results = [
        predict_mucus_penetration(
            drug_name=drug_name,
            mw_Da=mw_Da,
            charge=charge,
            mucus_type=mt,
            logP=logP,
        )
        for mt in _VALID_MUCUS_TYPES
    ]
    results.sort(key=lambda r: r.d_ratio, reverse=True)
    return results
