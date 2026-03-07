"""Mucoadhesive drug delivery PK model — Phase 268.

Models drug release from a mucoadhesive formulation at a mucosal surface,
with local mucosal absorption and partial systemic bypass.
Relevant for buccal, nasal, vaginal, and GI mucoadhesive systems.

References
----------
- Longer MA & Robinson JR, Pharm Int. 1986
- Chickering D & Mathiowitz E, J Control Release. 1995
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MucoadhesivePKResult:
    """Result from mucoadhesive drug delivery simulation."""

    drug_name: str
    dose_mg: float
    formulation_type: str  # "tablet", "gel", "microsphere"
    adhesion_site: str  # "buccal", "nasal", "gastric", "vaginal"
    times_h: list[float]
    c_plasma_mg_L: list[float]
    m_mucosal_mg: list[float]  # Drug in mucosal tissue
    cmax: float
    tmax_h: float
    auc: float
    t_half_adhesion_h: float  # Time for 50% of formulation to release/detach
    f_mucosal_absorption: float  # Fraction absorbed via mucosa vs. swallowed
    notes: str


_ADHESION_PARAMS: dict[str, dict] = {
    "buccal": {"k_release": 0.1, "k_absorb": 0.3, "k_swallow": 0.05, "bypass_fraction": 0.8},
    "nasal": {"k_release": 0.2, "k_absorb": 0.5, "k_swallow": 0.1, "bypass_fraction": 0.9},
    "gastric": {
        "k_release": 0.05,
        "k_absorb": 0.15,
        "k_swallow": 0.02,
        "bypass_fraction": 0.2,
    },
    "vaginal": {
        "k_release": 0.03,
        "k_absorb": 0.1,
        "k_swallow": 0.001,
        "bypass_fraction": 0.95,
    },
}

_FORMULATION_RELEASE_MULT: dict[str, float] = {
    "tablet": 1.0,
    "gel": 1.5,
    "microsphere": 0.5,  # slower controlled release
}


def simulate_mucoadhesive_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    adhesion_site: str = "buccal",
    formulation_type: str = "tablet",
    t_end_h: float = 12.0,
    dt_h: float = 0.1,
) -> MucoadhesivePKResult:
    """Simulate mucoadhesive drug delivery PK.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose (mg). Must be > 0.
    cl_L_per_h : Systemic clearance (L/h). Must be > 0.
    vd_L : Volume of distribution (L). Must be > 0.
    adhesion_site : 'buccal', 'nasal', 'gastric', or 'vaginal'.
    formulation_type : 'tablet', 'gel', or 'microsphere'.
    t_end_h : Simulation end time (h).
    dt_h : Time step (h).

    Returns
    -------
    MucoadhesivePKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if adhesion_site not in _ADHESION_PARAMS:
        raise ValueError(f"adhesion_site must be one of {list(_ADHESION_PARAMS.keys())}")
    if formulation_type not in _FORMULATION_RELEASE_MULT:
        raise ValueError(
            f"formulation_type must be one of {list(_FORMULATION_RELEASE_MULT.keys())}"
        )

    params = _ADHESION_PARAMS[adhesion_site]
    rel_mult = _FORMULATION_RELEASE_MULT[formulation_type]
    k_rel = params["k_release"] * rel_mult
    k_abs = params["k_absorb"]
    k_sw = params["k_swallow"]
    bypass = params["bypass_fraction"]

    ke = cl_L_per_h / vd_L

    n_steps = int(t_end_h / dt_h) + 1
    times = [i * dt_h for i in range(n_steps)]
    c_plasma = [0.0] * n_steps
    m_form = [0.0] * n_steps  # Drug in formulation (unreleased)
    m_mucosa = [0.0] * n_steps  # Drug in mucosal tissue

    m_form[0] = dose_mg
    absorbed_mucosal = 0.0

    for i in range(1, n_steps):
        mf = m_form[i - 1]
        mm = m_mucosa[i - 1]
        cp = c_plasma[i - 1]

        # Release from formulation to mucosal tissue
        release = k_rel * mf

        # Absorption from mucosa: fraction bypass goes to systemic directly
        mucosal_abs = k_abs * mm * bypass

        # Loss to swallowing / non-absorbed
        swallow_loss = k_sw * mf

        # Systemic elimination
        elim = ke * cp

        dm_form = -(release + swallow_loss) * dt_h
        dm_mucosa = (release - k_abs * mm) * dt_h
        dc_plasma = (mucosal_abs - elim * vd_L) / vd_L * dt_h

        m_form[i] = max(0.0, mf + dm_form)
        m_mucosa[i] = max(0.0, mm + dm_mucosa)
        c_plasma[i] = max(0.0, cp + dc_plasma)

        absorbed_mucosal += mucosal_abs * dt_h

    cmax = max(c_plasma)
    tmax = times[c_plasma.index(cmax)]
    auc = sum(0.5 * (c_plasma[j - 1] + c_plasma[j]) * dt_h for j in range(1, n_steps))

    # Half-adhesion time: when formulation releases 50% of dose
    t_half_adh = math.log(2) / k_rel if k_rel > 0 else t_end_h
    f_mucosa = min(1.0, absorbed_mucosal / dose_mg)

    notes = (
        f"{drug_name} mucoadhesive {formulation_type} at {adhesion_site}: "
        f"k_rel={k_rel:.3f}/h, t\u00bdadh={t_half_adh:.1f}h. "
        f"Cmax={cmax:.4f} mg/L, AUC={auc:.3f} mg*h/L. "
        f"f_mucosal={f_mucosa:.2f}."
    )

    return MucoadhesivePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation_type=formulation_type,
        adhesion_site=adhesion_site,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        m_mucosal_mg=m_mucosa,
        cmax=cmax,
        tmax_h=tmax,
        auc=auc,
        t_half_adhesion_h=t_half_adh,
        f_mucosal_absorption=f_mucosa,
        notes=notes,
    )


__all__ = ["MucoadhesivePKResult", "simulate_mucoadhesive_pk"]
