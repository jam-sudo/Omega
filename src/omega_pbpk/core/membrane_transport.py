"""Membrane transport simulation -- Phase 459.

Models drug transport across biological membranes using passive diffusion
(PAMPA-like prediction) and active transport (Michaelis-Menten kinetics).

Passive permeability is estimated from physicochemical descriptors using a
correlation inspired by the Hou (2004) and Palm (1997) models:

    log(Papp) = a*logP - b*PSA - c*n_hbd - d*log(MW) + intercept

Active (carrier-mediated) transport is modelled with standard Michaelis-Menten:

    J_active = Vmax * C / (Km + C)     [pmol/cm2/min]

Efflux ratio (ER = Papp_BA / Papp_AB):
    ER >= 2.0  ->  likely P-gp / BCRP substrate
    ER < 2.0   ->  predominantly passive permeability

References
----------
Hou T et al. J Chem Inf Model. 2004;44(6):1585-600.
Palm K et al. J Pharmacol Exp Ther. 1997;283(2):727-32.
FDA Guidance: In Vitro Drug Interaction Studies. 2020.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Passive permeability thresholds (cm/s) for Caco-2-like assays
_HIGH_PERM_CM_S: float = 20.0e-6    # > 20e-6 cm/s -> high
_LOW_PERM_CM_S: float = 2.0e-6     # < 2e-6  cm/s -> low

# Efflux ratio threshold for P-gp substrate classification
_PGP_ER_THRESHOLD: float = 2.0

# Fraction absorbed (fa) from Papp using sigmoidal relationship calibrated to
# human jejunal data (Lennernas 1998).
# fa = 1 - exp(-alpha * Papp_cm_s)  with alpha tuned so that 10e-6 cm/s -> ~90%
_FA_ALPHA: float = 1.0e5  # empirical scale factor

# P-gp efflux multiplier applied to BA direction when pgp_efflux=True
_PGP_EFFLUX_MULTIPLIER: float = 3.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class Caco2Result:
    """Results from a simulated Caco-2 bidirectional permeability experiment.

    Attributes
    ----------
    drug_name:
        Identifier for the compound.
    papp_ab_cm_s:
        Apparent apical-to-basolateral permeability (cm/s).
    papp_ba_cm_s:
        Apparent basolateral-to-apical permeability (cm/s).  Includes efflux
        if pgp_efflux is True or active-transport terms are provided.
    efflux_ratio:
        Papp_BA / Papp_AB.  Values >= 2.0 are flagged as P-gp substrate.
    permeability_class:
        'high', 'medium', or 'low' based on Papp_AB thresholds.
    pgp_substrate:
        True if efflux_ratio >= 2.0.
    predicted_fa_pct:
        Predicted fraction absorbed (%) estimated from Papp_AB.
    notes:
        Human-readable summary including key flags and suggestions.
    """

    drug_name: str
    papp_ab_cm_s: float
    papp_ba_cm_s: float
    efflux_ratio: float
    permeability_class: str
    pgp_substrate: bool
    predicted_fa_pct: float
    notes: str


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def passive_permeability(
    logP: float,
    mw: float,
    psa_A2: float,
    n_hbd: int,
    charge_state: str = "neutral",
) -> float:
    """Predict passive membrane permeability (PAMPA / Caco-2 A->B).

    Uses a physicochemical QSAR model calibrated to literature Caco-2 data:

        log10(Papp / cm/s) = 0.30*logP - 0.013*PSA - 0.40*n_hbd
                             - 0.60*log10(MW) - 3.20

    Charge corrections (rough approximation):
        cationic  : +0.20 log units (membrane partitioning favoured)
        anionic   : -0.30 log units (repelled by negative membrane)
        zwitterion: -0.10 log units

    Parameters
    ----------
    logP:
        Octanol-water partition coefficient (can be negative).
    mw:
        Molecular weight (Da).  Must be > 0.
    psa_A2:
        Topological polar surface area (Angstrom^2).  Must be >= 0.
    n_hbd:
        Number of hydrogen bond donors.  Must be >= 0.
    charge_state:
        One of 'neutral', 'cationic', 'anionic', 'zwitterion'.

    Returns
    -------
    float
        Predicted Papp in cm/s (not multiplied by 1e-6).

    Raises
    ------
    ValueError
        If mw <= 0, psa_A2 < 0, n_hbd < 0, or charge_state is unrecognised.
    """
    _validate_passive_inputs(logP, mw, psa_A2, n_hbd, charge_state)

    charge_corrections: dict[str, float] = {
        "neutral": 0.0,
        "cationic": 0.20,
        "anionic": -0.30,
        "zwitterion": -0.10,
    }
    corr = charge_corrections[charge_state]

    log_papp = (
        0.30 * logP
        - 0.013 * psa_A2
        - 0.40 * n_hbd
        - 0.60 * math.log10(max(mw, 1.0))
        - 3.20
        + corr
    )

    # Clamp to physically reasonable range [1e-9, 1e-4] cm/s
    papp = 10.0 ** log_papp
    return float(min(max(papp, 1.0e-9), 1.0e-4))


def active_transport_rate(
    substrate_conc_uM: float,
    vmax_pmol_cm2_min: float,
    km_uM: float,
) -> float:
    """Compute Michaelis-Menten active transport flux.

    Parameters
    ----------
    substrate_conc_uM:
        Substrate (drug) concentration on the donor side (uM).
    vmax_pmol_cm2_min:
        Maximum transport rate (pmol/cm^2/min).
    km_uM:
        Michaelis constant (uM).

    Returns
    -------
    float
        Active flux in pmol/cm^2/min.

    Raises
    ------
    ValueError
        If any parameter is negative, or km_uM <= 0.
    """
    if substrate_conc_uM < 0.0:
        raise ValueError("substrate_conc_uM must be >= 0")
    if vmax_pmol_cm2_min < 0.0:
        raise ValueError("vmax_pmol_cm2_min must be >= 0")
    if km_uM <= 0.0:
        raise ValueError("km_uM must be > 0")

    return float(vmax_pmol_cm2_min * substrate_conc_uM / (km_uM + substrate_conc_uM))


def efflux_ratio(papp_ab_cm_s: float, papp_ba_cm_s: float) -> float:
    """Compute the efflux ratio (ER = Papp_BA / Papp_AB).

    Parameters
    ----------
    papp_ab_cm_s:
        A-to-B permeability (cm/s).
    papp_ba_cm_s:
        B-to-A permeability (cm/s).

    Returns
    -------
    float
        Efflux ratio (dimensionless).

    Raises
    ------
    ValueError
        If either permeability is negative or papp_ab_cm_s is zero.
    """
    if papp_ab_cm_s <= 0.0:
        raise ValueError("papp_ab_cm_s must be > 0")
    if papp_ba_cm_s < 0.0:
        raise ValueError("papp_ba_cm_s must be >= 0")
    return float(papp_ba_cm_s / papp_ab_cm_s)


def simulate_caco2(
    drug_name: str,
    logP: float,
    mw: float,
    psa_A2: float,
    n_hbd: int,
    conc_uM: float = 10.0,
    km_uM: float | None = None,
    vmax: float | None = None,
    pgp_efflux: bool = False,
) -> Caco2Result:
    """Simulate a Caco-2 bidirectional transport experiment.

    Computes passive permeability (AB direction) from physicochemical
    descriptors, adds active carrier-mediated transport if Km/Vmax are
    provided, and applies a P-gp efflux multiplier to the BA direction if
    pgp_efflux is True.

    Parameters
    ----------
    drug_name:
        Compound identifier (used in result and notes).
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight (Da).
    psa_A2:
        Topological polar surface area (Angstrom^2).
    n_hbd:
        Number of hydrogen bond donors.
    conc_uM:
        Donor-side drug concentration (uM).  Default 10 uM.
    km_uM:
        Michaelis constant for active uptake transporter (uM).  None = no
        active transport.
    vmax:
        Vmax for active transporter (pmol/cm^2/min).  Ignored if km_uM is
        None.
    pgp_efflux:
        If True, multiply BA permeability by _PGP_EFFLUX_MULTIPLIER to
        simulate P-gp-mediated efflux.

    Returns
    -------
    Caco2Result
    """
    papp_ab = passive_permeability(logP, mw, psa_A2, n_hbd)

    # Active transport contribution converts flux (pmol/cm2/min) to cm/s
    active_contrib_cm_s = 0.0
    if km_uM is not None and vmax is not None:
        if conc_uM > 0.0:
            flux = active_transport_rate(conc_uM, vmax, km_uM)
            # flux [pmol/cm2/min] / conc [pmol/mL * 1e6/1e12] -> cm/s
            # conc_uM uM = conc_uM * 1e-6 mol/L = conc_uM * 1e-6 * 1e12 pmol/L
            #            = conc_uM * 1e6 pmol/L = conc_uM * 1e3 pmol/mL
            conc_pmol_mL = conc_uM * 1e3
            # Papp (cm/min) = flux (pmol/cm2/min) / conc (pmol/mL = pmol/cm3)
            papp_active_cm_min = flux / conc_pmol_mL if conc_pmol_mL > 0 else 0.0
            active_contrib_cm_s = papp_active_cm_min / 60.0

    papp_ab_total = papp_ab + active_contrib_cm_s

    # BA direction: passive (same as AB for passive) + efflux
    papp_ba = papp_ab  # passive component is symmetric
    if pgp_efflux:
        papp_ba = papp_ab * _PGP_EFFLUX_MULTIPLIER

    er = efflux_ratio(max(papp_ab_total, 1.0e-12), papp_ba)

    perm_class = classify_permeability(papp_ab_total)
    pgp_sub = er >= _PGP_ER_THRESHOLD

    # Predicted fraction absorbed from Papp_AB (sigmoidal approximation)
    fa = 1.0 - math.exp(-_FA_ALPHA * papp_ab_total)
    fa_pct = round(min(max(fa * 100.0, 0.0), 100.0), 2)

    # Build notes string
    notes_parts: list[str] = [
        f"Permeability class: {perm_class}.",
        f"Efflux ratio: {er:.2f} ({'P-gp substrate' if pgp_sub else 'passive'}).",
        f"Predicted fa: {fa_pct}%.",
    ]
    if pgp_efflux:
        notes_parts.append("P-gp efflux simulated (BA multiplier applied).")
    if km_uM is not None:
        notes_parts.append("Carrier-mediated uptake included.")
    if perm_class == "low":
        notes_parts.append("Consider permeation enhancers or prodrug strategy.")

    return Caco2Result(
        drug_name=drug_name,
        papp_ab_cm_s=round(papp_ab_total, 12),
        papp_ba_cm_s=round(papp_ba, 12),
        efflux_ratio=round(er, 4),
        permeability_class=perm_class,
        pgp_substrate=pgp_sub,
        predicted_fa_pct=fa_pct,
        notes=" ".join(notes_parts),
    )


def classify_permeability(papp_cm_s: float) -> str:
    """Classify permeability into 'high', 'medium', or 'low'.

    Thresholds are based on typical Caco-2 acceptance criteria:
        high   : Papp >= 20e-6 cm/s
        medium : 2e-6 <= Papp < 20e-6 cm/s
        low    : Papp < 2e-6 cm/s

    Parameters
    ----------
    papp_cm_s:
        Apparent permeability in cm/s.

    Returns
    -------
    str
        One of 'high', 'medium', 'low'.

    Raises
    ------
    ValueError
        If papp_cm_s is negative.
    """
    if papp_cm_s < 0.0:
        raise ValueError("papp_cm_s must be >= 0")
    if papp_cm_s >= _HIGH_PERM_CM_S:
        return "high"
    if papp_cm_s >= _LOW_PERM_CM_S:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_passive_inputs(
    logP: float,
    mw: float,
    psa_A2: float,
    n_hbd: int,
    charge_state: str,
) -> None:
    """Validate inputs for passive_permeability."""
    if mw <= 0.0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if psa_A2 < 0.0:
        raise ValueError(f"psa_A2 must be >= 0, got {psa_A2}")
    if n_hbd < 0:
        raise ValueError(f"n_hbd must be >= 0, got {n_hbd}")
    valid_charges = {"neutral", "cationic", "anionic", "zwitterion"}
    if charge_state not in valid_charges:
        raise ValueError(
            f"charge_state must be one of {valid_charges}, got '{charge_state}'"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "Caco2Result",
    "passive_permeability",
    "active_transport_rate",
    "efflux_ratio",
    "simulate_caco2",
    "classify_permeability",
]
