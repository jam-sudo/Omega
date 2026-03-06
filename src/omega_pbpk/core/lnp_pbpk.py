"""Lipid Nanoparticle (LNP) PBPK model for mRNA/siRNA and small-molecule delivery.

LNPs are the delivery platform for COVID-19 mRNA vaccines and siRNA therapeutics.
Models ionizable-lipid pH-dependent charge, PEG shielding, ApoE-mediated
hepatocyte targeting, and endosomal escape for cargo release.

Compartments
------------
1. Plasma -- LNP circulates, cleared to liver/spleen
2. Liver -- ApoE/LDLR-mediated uptake, cargo release via endosomal escape
3. Spleen -- immune cell uptake
4. Cargo (liver/spleen) -- released mRNA/siRNA/small-molecule

References
----------
- Kulkarni JA et al., Nanoscale. 2018 (LNP design)
- Maier MA et al., Mol Ther. 2013 (ionizable lipid pKa)
- Rosenblum D et al., Nat Comms. 2020 (LNP biodistribution)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class LNPPBPKResult:
    """Results from LNP PBPK simulation."""

    lnp_name: str
    cargo_type: str
    dose_mg_lipid: float
    pka_apparent: float
    peg_mol_pct: float
    size_nm: float
    times_h: list[float]
    a_plasma_mg: list[float]
    a_liver_pct: list[float]
    a_spleen_pct: list[float]
    cargo_liver_mg: list[float]
    cargo_spleen_mg: list[float]
    cmax_plasma_mg_L: float
    auc_plasma: float
    liver_pct_at_24h: float
    liver_pct_at_72h: float
    spleen_pct_at_24h: float
    peak_cargo_liver_mg: float
    peak_cargo_spleen_mg: float
    liver_to_spleen_ratio: float
    t_half_plasma_h: float
    endosomal_escape_efficiency: float
    notes: str


# Compartment volumes (L)
_V_PLASMA = 3.0
_V_LIVER = 1.5
_V_SPLEEN = 0.15


def _cargo_half_life(cargo_type: str, mrna_half_life_h: float) -> float:
    """Return cargo degradation half-life (h) based on cargo type."""
    if cargo_type == "mrna":
        return mrna_half_life_h
    if cargo_type == "sirna":
        return 24.0
    # small_molecule default
    return 12.0


def simulate_lnp_pk(
    lnp_name: str,
    dose_mg_lipid: float,
    cargo_type: str = "mrna",
    cargo_loading_pct: float = 10.0,
    pka_apparent: float = 6.5,
    size_nm: float = 80.0,
    peg_mol_pct: float = 1.5,
    apoe_adsorption: float = 1.0,
    mrna_half_life_h: float = 48.0,
    cl_liver_L_per_h: float = 0.002,
    cl_spleen_L_per_h: float = 0.02,
    t_end_h: float = 168.0,
    dt_h: float = 0.5,
) -> LNPPBPKResult:
    """Simulate LNP PBPK after IV administration.

    Parameters
    ----------
    lnp_name : Formulation name.
    dose_mg_lipid : Total lipid dose (mg).
    cargo_type : "mrna", "sirna", or "small_molecule".
    cargo_loading_pct : Percent of LNP mass that is cargo (0, 100).
    pka_apparent : Apparent pKa of ionizable lipid (typically 6.2-6.8).
    size_nm : LNP diameter (nm).
    peg_mol_pct : PEG-lipid mole percent (1.5-2.0 typical).
    apoe_adsorption : ApoE corona factor (>1 = more hepatic targeting).
    mrna_half_life_h : mRNA cargo half-life in tissue (h).
    cl_liver_L_per_h : Hepatic LNP clearance (L/h).
    cl_spleen_L_per_h : Splenic LNP clearance (L/h).
    t_end_h : Simulation duration (h).
    dt_h : Time step (h).

    Returns
    -------
    LNPPBPKResult with full time-course and PK metrics.

    Raises
    ------
    ValueError : If parameters are invalid.
    """
    if dose_mg_lipid <= 0:
        raise ValueError("dose_mg_lipid must be > 0")
    if cargo_loading_pct <= 0 or cargo_loading_pct >= 100:
        raise ValueError("cargo_loading_pct must be in (0, 100)")
    if size_nm <= 0:
        raise ValueError("size_nm must be > 0")

    # Ionizable lipid endosomal escape efficiency at pH 5.5
    eff = 1.0 / (1.0 + 10.0 ** (5.5 - pka_apparent))

    # Stability at physiologic pH 7.4: high pKa → charged at 7.4 → destabilised
    frac_charged_7_4 = 1.0 / (1.0 + 10.0 ** (7.4 - pka_apparent))
    stability = 1.0 - frac_charged_7_4

    # PEG reduces MPS uptake
    peg_factor = max(0.3, 1.0 - peg_mol_pct * 0.3)

    # Rate constants (stability modulates effective liver/spleen uptake)
    k_liver = 0.4 * apoe_adsorption * peg_factor * stability
    k_spleen = 0.1 * peg_factor * stability
    k_plasma_elim = 0.02 + (1.0 - peg_factor) * 0.1
    k_release_liver = eff * 0.1
    k_release_spleen = eff * 0.05

    cargo_hl = _cargo_half_life(cargo_type, mrna_half_life_h)
    k_cargo_deg = math.log(2) / cargo_hl

    # ODE integration (forward Euler)
    n_steps = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n_steps + 1)

    a_plasma = np.zeros(n_steps + 1)
    a_liver = np.zeros(n_steps + 1)
    a_spleen = np.zeros(n_steps + 1)
    a_cargo_liver = np.zeros(n_steps + 1)
    a_cargo_spleen = np.zeros(n_steps + 1)

    # Initial condition: entire lipid dose in plasma
    a_plasma[0] = dose_mg_lipid

    cargo_frac = cargo_loading_pct / 100.0

    for i in range(n_steps):
        da_plasma = -(k_liver + k_spleen + k_plasma_elim) * a_plasma[i] * dt_h
        da_liver = (k_liver * a_plasma[i] - cl_liver_L_per_h * a_liver[i]) * dt_h
        da_spleen = (k_spleen * a_plasma[i] - cl_spleen_L_per_h * a_spleen[i]) * dt_h

        da_cargo_liver = (
            k_release_liver * cargo_frac * a_liver[i] - k_cargo_deg * a_cargo_liver[i]
        ) * dt_h
        da_cargo_spleen = (
            k_release_spleen * cargo_frac * a_spleen[i] - k_cargo_deg * a_cargo_spleen[i]
        ) * dt_h

        a_plasma[i + 1] = max(0.0, a_plasma[i] + da_plasma)
        a_liver[i + 1] = max(0.0, a_liver[i] + da_liver)
        a_spleen[i + 1] = max(0.0, a_spleen[i] + da_spleen)
        a_cargo_liver[i + 1] = max(0.0, a_cargo_liver[i] + da_cargo_liver)
        a_cargo_spleen[i + 1] = max(0.0, a_cargo_spleen[i] + da_cargo_spleen)

    # Derived quantities
    c_plasma = a_plasma / _V_PLASMA
    liver_pct = a_liver / dose_mg_lipid * 100.0
    spleen_pct = a_spleen / dose_mg_lipid * 100.0

    # Summary metrics
    cmax = float(np.max(c_plasma))
    auc = float(np_trapz(c_plasma, t))

    idx_24 = int(np.argmin(np.abs(t - 24.0)))
    idx_72 = int(np.argmin(np.abs(t - 72.0)))

    liver_24 = float(liver_pct[idx_24])
    liver_72 = float(liver_pct[idx_72])
    spleen_24 = float(spleen_pct[idx_24])

    peak_cargo_liver = float(np.max(a_cargo_liver))
    peak_cargo_spleen = float(np.max(a_cargo_spleen))

    liver_to_spleen = (
        peak_cargo_liver / peak_cargo_spleen if peak_cargo_spleen > 0 else float("inf")
    )

    # Plasma half-life
    c0 = c_plasma[0]
    half_idx = np.where(c_plasma <= c0 * 0.5)[0]
    t_half = float(t[half_idx[0]]) if len(half_idx) > 0 else float(t_end_h)

    notes = (
        f"LNP PBPK: {lnp_name}, cargo={cargo_type}, "
        f"dose={dose_mg_lipid:.1f} mg lipid. "
        f"pKa={pka_apparent:.1f}, PEG={peg_mol_pct:.1f}%. "
        f"Liver={liver_24:.1f}% at 24h, t1/2={t_half:.1f} h."
    )

    return LNPPBPKResult(
        lnp_name=lnp_name,
        cargo_type=cargo_type,
        dose_mg_lipid=dose_mg_lipid,
        pka_apparent=pka_apparent,
        peg_mol_pct=peg_mol_pct,
        size_nm=size_nm,
        times_h=t.tolist(),
        a_plasma_mg=a_plasma.tolist(),
        a_liver_pct=liver_pct.tolist(),
        a_spleen_pct=spleen_pct.tolist(),
        cargo_liver_mg=a_cargo_liver.tolist(),
        cargo_spleen_mg=a_cargo_spleen.tolist(),
        cmax_plasma_mg_L=cmax,
        auc_plasma=auc,
        liver_pct_at_24h=liver_24,
        liver_pct_at_72h=liver_72,
        spleen_pct_at_24h=spleen_24,
        peak_cargo_liver_mg=peak_cargo_liver,
        peak_cargo_spleen_mg=peak_cargo_spleen,
        liver_to_spleen_ratio=liver_to_spleen,
        t_half_plasma_h=t_half,
        endosomal_escape_efficiency=eff,
        notes=notes,
    )


def optimize_pka(
    lnp_name: str,
    dose_mg_lipid: float,
    pka_range: list[float] | None = None,
    **kwargs,
) -> list[LNPPBPKResult]:
    """Test multiple pKa values and return results sorted by peak cargo delivery.

    Parameters
    ----------
    lnp_name : Formulation name.
    dose_mg_lipid : Total lipid dose (mg).
    pka_range : List of pKa values to test. Defaults to [5.5, 6.0, 6.2, 6.5, 6.8, 7.0].
    **kwargs : Additional keyword arguments passed to simulate_lnp_pk.

    Returns
    -------
    List of LNPPBPKResult sorted by peak_cargo_liver_mg descending (highest first).
    """
    if pka_range is None:
        pka_range = [5.5, 6.0, 6.2, 6.5, 6.8, 7.0]

    results = []
    for pka in pka_range:
        r = simulate_lnp_pk(
            lnp_name=lnp_name,
            dose_mg_lipid=dose_mg_lipid,
            pka_apparent=pka,
            **kwargs,
        )
        results.append(r)

    results.sort(key=lambda r: r.peak_cargo_liver_mg, reverse=True)
    return results


__all__ = [
    "LNPPBPKResult",
    "simulate_lnp_pk",
    "optimize_pka",
]
